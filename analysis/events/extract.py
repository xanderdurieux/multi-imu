"""Extract interpretable event candidates from dual-IMU derived signals."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    parse_section_folder_name,
    recording_dir,
    recordings_root,
    sections_root,
)


@dataclass
class EventConfig:
    """Threshold-driven configuration for candidate detectors."""

    event_window_s: float = 2.0
    min_separation_s: float = 0.8
    diagnostic_plot_limit: int = 6

    bump_vertical_z_thr: float = 2.8
    bump_shock_gain_thr: float = 1.2

    braking_longitudinal_thr: float = -2.5
    braking_pitch_rate_thr: float = 0.45

    swerve_roll_rate_thr: float = 0.9
    swerve_lateral_thr: float = 2.0

    divergence_residual_thr: float = 2.2
    divergence_transmission_thr: float = 1.45

    fall_roll_rate_thr: float = 1.8
    fall_vertical_thr: float = -4.5
    fall_residual_thr: float = 3.0

    ambiguous_confidence_floor: float = 0.35


@dataclass
class Candidate:
    event_type: str
    timestamp: float
    time_s: float
    confidence: float
    key_trigger_signals: str
    trigger_value: float
    trigger_threshold: float
    event_window_start_idx: int
    event_window_end_idx: int
    section: str
    recording_id: str
    section_id: str
    ambiguous_flag: int = 0
    failure_flags: str = "ok"


def _to_float_series(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _normalized_score(value: float, threshold: float, direction: str = "high") -> float:
    if not np.isfinite(value) or not np.isfinite(threshold):
        return 0.0
    if direction == "low":
        val = -value
        thr = -threshold
    else:
        val = value
        thr = threshold
    if val <= thr:
        return 0.0
    scale = max(abs(thr), 1e-6)
    return float(min(1.0, (val - thr) / (1.5 * scale)))


def _build_candidate(
    *,
    event_type: str,
    idx: int,
    timestamp: float,
    time_s: float,
    trigger_name: str,
    trigger_value: float,
    trigger_threshold: float,
    confidence: float,
    n: int,
    samples_half_window: int,
    section: str,
    recording_id: str,
    section_id: str,
    failure_flags: str = "ok",
    ambiguous: bool = False,
) -> Candidate:
    lo = max(0, idx - samples_half_window)
    hi = min(n - 1, idx + samples_half_window)
    return Candidate(
        event_type=event_type,
        timestamp=timestamp,
        time_s=time_s,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        key_trigger_signals=trigger_name,
        trigger_value=float(trigger_value) if np.isfinite(trigger_value) else np.nan,
        trigger_threshold=float(trigger_threshold) if np.isfinite(trigger_threshold) else np.nan,
        event_window_start_idx=int(lo),
        event_window_end_idx=int(hi),
        section=section,
        recording_id=recording_id,
        section_id=section_id,
        ambiguous_flag=1 if ambiguous else 0,
        failure_flags=failure_flags,
    )


def _suppressed(cands: list[Candidate], min_separation_s: float) -> list[Candidate]:
    if not cands:
        return []
    kept: list[Candidate] = []
    for event_type in sorted({c.event_type for c in cands}):
        subset = sorted(
            (c for c in cands if c.event_type == event_type),
            key=lambda x: x.confidence,
            reverse=True,
        )
        chosen: list[Candidate] = []
        for c in subset:
            if any(abs(c.time_s - other.time_s) < min_separation_s for other in chosen):
                continue
            chosen.append(c)
        kept.extend(chosen)
    kept.sort(key=lambda c: c.time_s)
    return kept


def _render_diagnostics(
    out_dir: Path,
    events: list[Candidate],
    bike: pd.DataFrame,
    rider: pd.DataFrame,
    cross: pd.DataFrame,
    limit: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not events:
        return

    t = _to_float_series(cross, "time_s")
    bike_vert = _to_float_series(bike, "acc_vertical_m_s2")
    rider_vert = _to_float_series(rider, "acc_vertical_m_s2")
    bike_roll = _to_float_series(bike, "tilt_roll_rate_rad_s")
    rider_roll = _to_float_series(rider, "tilt_roll_rate_rad_s")
    residual = _to_float_series(cross, "residual_vertical_m_s2")

    for i, ev in enumerate(sorted(events, key=lambda c: c.confidence, reverse=True)[:limit]):
        t0 = ev.time_s
        m = (t >= t0 - 2.0) & (t <= t0 + 2.0)
        if np.sum(m) < 4:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t[m], bike_vert[m], label="bike vertical", color="tab:blue")
        axes[0].plot(t[m], rider_vert[m], label="rider vertical", color="tab:orange")
        axes[0].axvline(t0, color="k", linestyle="--", alpha=0.7)
        axes[0].legend(loc="upper right")
        axes[0].set_ylabel("m/s²")

        axes[1].plot(t[m], bike_roll[m], label="bike roll rate", color="tab:green")
        axes[1].plot(t[m], rider_roll[m], label="rider roll rate", color="tab:red")
        axes[1].axvline(t0, color="k", linestyle="--", alpha=0.7)
        axes[1].legend(loc="upper right")
        axes[1].set_ylabel("rad/s")

        axes[2].plot(t[m], residual[m], label="vertical residual rider-bike", color="tab:purple")
        axes[2].axvline(t0, color="k", linestyle="--", alpha=0.7)
        axes[2].legend(loc="upper right")
        axes[2].set_ylabel("m/s²")
        axes[2].set_xlabel("time in section (s)")

        fig.suptitle(f"{ev.event_type} @ {ev.time_s:.2f}s | conf={ev.confidence:.2f}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"event_{i+1:02d}_{ev.event_type}.png", dpi=150)
        plt.close(fig)


def extract_event_candidates_section(
    section_path: Path,
    *,
    section_name: str | None = None,
    config: EventConfig | None = None,
    config_override: dict[str, Any] | None = None,
    write_plots: bool = True,
) -> pd.DataFrame:
    """Extract event candidates for one section and persist them."""
    cfg = config or EventConfig()
    if config_override:
        for k, v in config_override.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    section_path = Path(section_path)
    section_id = section_path.name
    section_name = section_name or section_id
    try:
        recording_id, _ = parse_section_folder_name(section_id)
    except Exception:
        recording_id = section_id

    derived_dir = section_path / "derived"
    bike_path = derived_dir / "sporsa_signals.csv"
    rider_path = derived_dir / "arduino_signals.csv"
    cross_path = derived_dir / "cross_sensor_signals.csv"

    out_dir = section_path / "events"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bike_path.exists() or not rider_path.exists() or not cross_path.exists():
        empty = pd.DataFrame(columns=list(Candidate.__dataclass_fields__.keys()))
        empty.to_csv(out_dir / "event_candidates.csv", index=False)
        (out_dir / "event_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        return empty

    bike = pd.read_csv(bike_path)
    rider = pd.read_csv(rider_path)
    cross = pd.read_csv(cross_path)
    n = min(len(cross), len(bike), len(rider))
    if n == 0:
        empty = pd.DataFrame(columns=list(Candidate.__dataclass_fields__.keys()))
        empty.to_csv(out_dir / "event_candidates.csv", index=False)
        (out_dir / "event_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
        return empty

    t = _to_float_series(cross, "time_s")[:n]
    ts = _to_float_series(cross, "timestamp")[:n]
    if len(t) < 2:
        dt = 0.01
    else:
        dt = float(np.nanmedian(np.diff(t)))
        dt = dt if np.isfinite(dt) and dt > 1e-6 else 0.01

    samples_half_window = max(1, int(round((cfg.event_window_s / 2.0) / dt)))

    bike_vert = _to_float_series(bike, "acc_vertical_m_s2")[:n]
    rider_vert = _to_float_series(rider, "acc_vertical_m_s2")[:n]
    bike_long = _to_float_series(bike, "acc_longitudinal_m_s2")[:n]
    rider_long = _to_float_series(rider, "acc_longitudinal_m_s2")[:n]
    bike_pitch_rate = np.abs(_to_float_series(bike, "tilt_pitch_rate_rad_s")[:n])
    bike_roll_rate = np.abs(_to_float_series(bike, "tilt_roll_rate_rad_s")[:n])
    rider_roll_rate = np.abs(_to_float_series(rider, "tilt_roll_rate_rad_s")[:n])
    bike_lat = np.abs(_to_float_series(bike, "acc_lateral_m_s2")[:n])
    rider_lat = np.abs(_to_float_series(rider, "acc_lateral_m_s2")[:n])

    residual_vert = np.abs(_to_float_series(cross, "residual_vertical_m_s2")[:n])
    residual_long = np.abs(_to_float_series(cross, "residual_longitudinal_m_s2")[:n])
    shock_gain = _to_float_series(cross, "shock_transmission_gain")[:n]

    vert_diff = np.diff(np.nan_to_num(bike_vert, nan=0.0), prepend=np.nan_to_num(bike_vert[:1], nan=0.0))
    vert_z = (vert_diff - np.nanmedian(vert_diff)) / max(np.nanmedian(np.abs(vert_diff - np.nanmedian(vert_diff))) * 1.4826, 1e-6)

    cands: list[Candidate] = []
    for i in range(n):
        if not np.isfinite(t[i]):
            continue
        failure_flags: list[str] = []

        bump_score = max(
            _normalized_score(abs(vert_z[i]), cfg.bump_vertical_z_thr, "high"),
            _normalized_score(shock_gain[i], cfg.bump_shock_gain_thr, "high"),
        )
        if bump_score > 0:
            if not np.isfinite(shock_gain[i]):
                failure_flags.append("missing_shock_gain")
            cands.append(
                _build_candidate(
                    event_type="bump_shock_candidate",
                    idx=i,
                    timestamp=ts[i],
                    time_s=t[i],
                    trigger_name="|d(acc_vertical)_robust_z|, shock_transmission_gain",
                    trigger_value=max(abs(vert_z[i]), shock_gain[i] if np.isfinite(shock_gain[i]) else np.nan),
                    trigger_threshold=max(cfg.bump_vertical_z_thr, cfg.bump_shock_gain_thr),
                    confidence=bump_score,
                    n=n,
                    samples_half_window=samples_half_window,
                    section=section_name,
                    recording_id=recording_id,
                    section_id=section_id,
                    failure_flags=";".join(failure_flags) if failure_flags else "ok",
                )
            )

        brake_signal = np.nanmin([bike_long[i], rider_long[i]])
        brake_score = max(
            _normalized_score(brake_signal, cfg.braking_longitudinal_thr, "low"),
            _normalized_score(bike_pitch_rate[i], cfg.braking_pitch_rate_thr, "high"),
        )
        if brake_score > 0:
            ambig = np.isfinite(brake_signal) and brake_signal > -1.2
            cands.append(
                _build_candidate(
                    event_type="braking_burst",
                    idx=i,
                    timestamp=ts[i],
                    time_s=t[i],
                    trigger_name="longitudinal decel, pitch_rate",
                    trigger_value=brake_signal,
                    trigger_threshold=cfg.braking_longitudinal_thr,
                    confidence=brake_score,
                    n=n,
                    samples_half_window=samples_half_window,
                    section=section_name,
                    recording_id=recording_id,
                    section_id=section_id,
                    ambiguous=ambig,
                    failure_flags="weak_decel_only" if ambig else "ok",
                )
            )

        swerve_signal = np.nanmax([bike_roll_rate[i], rider_roll_rate[i]])
        swerve_lat = np.nanmax([bike_lat[i], rider_lat[i]])
        swerve_score = max(
            _normalized_score(swerve_signal, cfg.swerve_roll_rate_thr, "high"),
            _normalized_score(swerve_lat, cfg.swerve_lateral_thr, "high"),
        )
        if swerve_score > 0:
            cands.append(
                _build_candidate(
                    event_type="swerve_roll_rate_candidate",
                    idx=i,
                    timestamp=ts[i],
                    time_s=t[i],
                    trigger_name="|roll_rate|, |lateral_acc|",
                    trigger_value=max(swerve_signal, swerve_lat),
                    trigger_threshold=max(cfg.swerve_roll_rate_thr, cfg.swerve_lateral_thr),
                    confidence=swerve_score,
                    n=n,
                    samples_half_window=samples_half_window,
                    section=section_name,
                    recording_id=recording_id,
                    section_id=section_id,
                )
            )

        div_score = max(
            _normalized_score(residual_vert[i], cfg.divergence_residual_thr, "high"),
            _normalized_score(residual_long[i], cfg.divergence_residual_thr, "high"),
            _normalized_score(shock_gain[i], cfg.divergence_transmission_thr, "high"),
        )
        if div_score > 0:
            ambig = div_score < cfg.ambiguous_confidence_floor
            cands.append(
                _build_candidate(
                    event_type="rider_bicycle_divergence",
                    idx=i,
                    timestamp=ts[i],
                    time_s=t[i],
                    trigger_name="residual_vertical/longitudinal, shock_gain",
                    trigger_value=max(residual_vert[i], residual_long[i], shock_gain[i] if np.isfinite(shock_gain[i]) else np.nan),
                    trigger_threshold=max(cfg.divergence_residual_thr, cfg.divergence_transmission_thr),
                    confidence=div_score,
                    n=n,
                    samples_half_window=samples_half_window,
                    section=section_name,
                    recording_id=recording_id,
                    section_id=section_id,
                    ambiguous=ambig,
                    failure_flags="low_confidence_overlap" if ambig else "ok",
                )
            )

        fall_score = np.nanmean([
            _normalized_score(swerve_signal, cfg.fall_roll_rate_thr, "high"),
            _normalized_score(np.nanmin([bike_vert[i], rider_vert[i]]), cfg.fall_vertical_thr, "low"),
            _normalized_score(max(residual_vert[i], residual_long[i]), cfg.fall_residual_thr, "high"),
        ])
        if np.isfinite(fall_score) and fall_score > 0.45:
            cands.append(
                _build_candidate(
                    event_type="fall_or_bicycle_drop_candidate",
                    idx=i,
                    timestamp=ts[i],
                    time_s=t[i],
                    trigger_name="roll_rate + vertical drop + residual spike",
                    trigger_value=fall_score,
                    trigger_threshold=0.45,
                    confidence=fall_score,
                    n=n,
                    samples_half_window=samples_half_window,
                    section=section_name,
                    recording_id=recording_id,
                    section_id=section_id,
                )
            )

    dedup = _suppressed(cands, cfg.min_separation_s)
    out_df = pd.DataFrame([asdict(c) for c in dedup])

    qmeta_path = section_path / "quality_metadata.json"
    if qmeta_path.exists() and len(out_df):
        try:
            qmeta = json.loads(qmeta_path.read_text(encoding="utf-8"))
        except Exception:
            qmeta = {}
        if isinstance(qmeta, dict):
            out_df["quality_schema_version"] = qmeta.get("schema_version", "quality_metadata.v1")
            out_df["section_quality_score"] = qmeta.get("overall_quality_score")
            out_df["section_usability_category"] = qmeta.get("overall_usability_category")
            out_df["section_sync_confidence"] = qmeta.get("sync_confidence")

    out_df.to_csv(out_dir / "event_candidates.csv", index=False)
    (out_dir / "event_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    summary = {
        "section": section_name,
        "recording_id": recording_id,
        "n_candidates": int(len(out_df)),
        "counts_by_type": out_df["event_type"].value_counts().to_dict() if len(out_df) else {},
        "notes": {
            "bump_shock_candidate": "Vertical acceleration jerk/shock gain spikes suggest obstacle impact or hard road bump.",
            "braking_burst": "Short negative longitudinal acceleration bursts with pitch-rate change reflect rider+bike deceleration dynamics.",
            "swerve_roll_rate_candidate": "Rapid roll-rate/lateral acceleration surges reflect steering correction or swerve.",
            "rider_bicycle_divergence": "Large rider-bike residual motion indicates body/bike mismatch, often destabilizing maneuvers.",
            "fall_or_bicycle_drop_candidate": "Combined high roll-rate, vertical drop, and residual surge resembles loss-of-balance/drop behavior.",
        },
    }
    (out_dir / "event_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if write_plots:
        _render_diagnostics(out_dir, dedup, bike, rider, cross, limit=cfg.diagnostic_plot_limit)

    return out_df


def load_event_windows(
    section_path: Path,
    *,
    event_types: set[str] | None = None,
    min_confidence: float = 0.0,
) -> pd.DataFrame:
    """Load event candidates and convert into window-center table for feature extraction."""
    path = Path(section_path) / "events" / "event_candidates.csv"
    if not path.exists():
        return pd.DataFrame(columns=["window_center_s", "event_type", "confidence", "event_timestamp"]) 
    df = pd.read_csv(path)
    if event_types:
        df = df[df["event_type"].isin(event_types)]
    df = df[df["confidence"] >= float(min_confidence)]
    if df.empty:
        return pd.DataFrame(columns=["window_center_s", "event_type", "confidence", "event_timestamp"]) 
    out = pd.DataFrame(
        {
            "window_center_s": pd.to_numeric(df["time_s"], errors="coerce"),
            "event_type": df["event_type"].astype(str),
            "confidence": pd.to_numeric(df["confidence"], errors="coerce"),
            "event_timestamp": pd.to_numeric(df["timestamp"], errors="coerce"),
        }
    )
    return out.dropna(subset=["window_center_s"]).sort_values("window_center_s").reset_index(drop=True)


def _resolve_target(name: str) -> tuple[str, Path | None, bool]:
    s = name.strip().rstrip("/").replace("\\", "/")
    if not s:
        raise ValueError("name must be section path/folder, recording name, or session name")

    p = Path(s)
    if p.is_dir():
        return "", p.resolve(), False

    try:
        rec, _idx = parse_section_folder_name(s)
    except Exception:
        rec = ""
    else:
        sec_dir = sections_root() / s
        if sec_dir.is_dir():
            return rec, sec_dir.resolve(), False

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s, None, True
    if re.fullmatch(r".+_r\d+", s):
        return s, None, False
    raise ValueError(f"Unrecognized target {name!r}")


def extract_events_from_args(
    name: str,
    *,
    all_sections: bool = False,
    all_recordings: bool = False,
    config_path: Path | None = None,
    no_plots: bool = False,
) -> list[Path]:
    config_override = None
    if config_path is not None:
        config_override = json.loads(Path(config_path).read_text(encoding="utf-8"))

    rec, section_dir, is_session = _resolve_target(name)

    done: list[Path] = []
    if section_dir is not None:
        extract_event_candidates_section(section_dir, config_override=config_override, write_plots=not no_plots)
        return [section_dir]

    if all_recordings:
        if not is_session:
            raise ValueError("--all requires a session date")
        rec_dirs = sorted(d for d in recordings_root().iterdir() if d.is_dir() and d.name.startswith(rec + "_r"))
    else:
        if not all_sections:
            raise ValueError("Pass a section folder/path, or use --all-sections with a recording, or --all with a session")
        rec_dir = recording_dir(rec)
        if not rec_dir.exists():
            raise FileNotFoundError(f"Recording not found: {rec_dir}")
        rec_dirs = [rec_dir]

    for rdir in rec_dirs:
        for sdir in iter_sections_for_recording(rdir.name):
            extract_event_candidates_section(sdir, config_override=config_override, write_plots=not no_plots)
            done.append(sdir)
    return done


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m events.extract")
    parser.add_argument("name", help="Section path/folder, recording, or session")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument("--all", dest="all_recordings", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config override")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    extract_events_from_args(
        args.name,
        all_sections=args.all_sections,
        all_recordings=args.all_recordings,
        config_path=args.config,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
