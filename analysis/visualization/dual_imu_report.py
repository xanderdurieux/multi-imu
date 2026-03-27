"""Dual-IMU reporting focused on bicycle-versus-rider relationships.

This module builds thesis-ready presentation plots and separate diagnostic plots
from existing derived/event artifacts (no signal recomputation).

CLI supports:
- one section,
- one recording (all sections),
- all good sections in the dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import iter_sections_for_recording, recordings_root, sections_root
from visualization.thesis_style import THESIS_COLORS, apply_matplotlib_thesis_style


@dataclass
class SectionConfidence:
    score: float
    flags: list[str]
    usable: bool


@dataclass
class EventSummary:
    section_id: str
    event_label: str
    event_time_s: float
    lag_ms: float | None
    max_correlation: float | None
    shock_attenuation_pct: float | None
    divergence_peak_time_s: float | None
    bike_peak_vertical_m_s2: float | None
    rider_peak_vertical_m_s2: float | None
    summary_text: str
    confidence: SectionConfidence


def _to_num(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return np.sqrt(s.pow(2).rolling(window=window, min_periods=max(3, window // 3), center=True).mean()).to_numpy()


def _finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) < 5:
        return np.nan
    aa = a[mask] - np.nanmean(a[mask])
    bb = b[mask] - np.nanmean(b[mask])
    den = np.linalg.norm(aa) * np.linalg.norm(bb)
    if den <= 1e-12:
        return np.nan
    return float(np.dot(aa, bb) / den)


def _cross_corr_lag(
    bike: np.ndarray,
    rider: np.ndarray,
    dt_s: float,
    max_lag_s: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    mask = np.isfinite(bike) & np.isfinite(rider)
    if np.sum(mask) < 12:
        return np.array([]), np.array([]), None, None
    xb = bike[mask] - np.nanmean(bike[mask])
    xr = rider[mask] - np.nanmean(rider[mask])
    den = np.linalg.norm(xb) * np.linalg.norm(xr)
    if den <= 1e-12:
        return np.array([]), np.array([]), None, None

    corr_full = np.correlate(xr, xb, mode="full") / den
    lags = np.arange(-len(xb) + 1, len(xb))
    max_lag_samples = max(1, int(round(max_lag_s / max(dt_s, 1e-3))))
    keep = np.abs(lags) <= max_lag_samples
    corr = corr_full[keep]
    lags = lags[keep]
    if corr.size == 0:
        return np.array([]), np.array([]), None, None
    i = int(np.nanargmax(corr))
    best_lag_s = float(lags[i] * dt_s)
    best_corr = float(corr[i])
    return lags * dt_s, corr, best_lag_s, best_corr


def _default_event_candidates(section_path: Path, t: np.ndarray, bike_vert: np.ndarray) -> list[tuple[str, float]]:
    events_csv = section_path / "events" / "event_candidates.csv"
    if events_csv.exists():
        edf = pd.read_csv(events_csv)
        if not edf.empty and {"event_type", "time_s"}.issubset(edf.columns):
            tmp = edf.copy()
            tmp["confidence"] = pd.to_numeric(tmp.get("confidence", 0.0), errors="coerce")
            tmp["time_s"] = pd.to_numeric(tmp["time_s"], errors="coerce")
            tmp = tmp.dropna(subset=["time_s"]).sort_values(["confidence", "time_s"], ascending=[False, True])
            out: list[tuple[str, float]] = []
            for _, row in tmp.head(3).iterrows():
                out.append((str(row["event_type"]), float(row["time_s"])))
            if out:
                return out

    if len(t) == 0 or len(bike_vert) == 0:
        return []
    idx = int(np.nanargmax(np.abs(bike_vert))) if np.isfinite(np.nanmax(np.abs(bike_vert))) else len(t) // 2
    return [("dominant_bike_disturbance", float(t[idx]))]


def summarize_bike_rider_event(
    *,
    section_id: str,
    event_label: str,
    event_time_s: float,
    t: np.ndarray,
    bike_vert: np.ndarray,
    rider_vert: np.ndarray,
    residual_vert: np.ndarray,
    dt_s: float,
    window_s: float = 1.8,
) -> EventSummary:
    """Return a compact event summary sentence and reliability flags.

    Text format example:
    "bike disturbance precedes rider response by X ms; vertical shock attenuated by
    Y%; rider-bicycle divergence peaked at time Z".
    """
    mask = np.isfinite(t) & (t >= (event_time_s - window_s)) & (t <= (event_time_s + window_s))
    flags: list[str] = []
    if np.sum(mask) < 20:
        flags.append("insufficient_samples")

    tb = t[mask]
    b = bike_vert[mask]
    r = rider_vert[mask]
    d = residual_vert[mask]

    lags_s, corr, lag_s, best_corr = _cross_corr_lag(b, r, dt_s)

    bike_peak = float(np.nanmax(np.abs(b))) if b.size else np.nan
    rider_peak = float(np.nanmax(np.abs(r))) if r.size else np.nan
    if not np.isfinite(bike_peak) or bike_peak < 1.0:
        flags.append("weak_bike_disturbance")

    attenuation = None
    if np.isfinite(bike_peak) and bike_peak > 1e-6 and np.isfinite(rider_peak):
        attenuation = float(np.clip(100.0 * (1.0 - (rider_peak / bike_peak)), -300.0, 300.0))

    if best_corr is None or not np.isfinite(best_corr):
        flags.append("lag_unreliable")
    elif best_corr < 0.2:
        flags.append("weak_cross_correlation")

    if lag_s is not None and abs(lag_s) >= (0.79):
        flags.append("lag_hit_search_limit")

    div_peak_t = None
    if np.any(np.isfinite(d)):
        j = int(np.nanargmax(np.abs(d)))
        div_peak_t = float(tb[j]) if tb.size else None
    else:
        flags.append("no_divergence_signal")

    lag_ms = None if lag_s is None else float(lag_s * 1000.0)
    lag_txt = "unknown"
    if lag_ms is not None:
        lag_txt = f"{abs(lag_ms):.0f} ms"
    atten_txt = "unknown"
    if attenuation is not None:
        atten_txt = f"{attenuation:+.1f}%"
    div_txt = "unknown"
    if div_peak_t is not None:
        div_txt = f"{div_peak_t:.2f}s"

    if lag_ms is None:
        timing_sentence = "bike-rider timing is unresolved"
    elif lag_ms > 0:
        timing_sentence = f"bike disturbance precedes rider response by {lag_txt}"
    elif lag_ms < 0:
        timing_sentence = f"rider response precedes bike disturbance by {lag_txt}"
    else:
        timing_sentence = "bike and rider disturbances are near-simultaneous"

    summary_text = (
        f"{timing_sentence}; vertical shock attenuated by {atten_txt}; "
        f"rider-bicycle divergence peaked at time {div_txt}."
    )

    score = 1.0
    score -= 0.28 * ("insufficient_samples" in flags)
    score -= 0.25 * ("weak_bike_disturbance" in flags)
    score -= 0.20 * ("weak_cross_correlation" in flags)
    score -= 0.35 * ("lag_unreliable" in flags)
    score -= 0.10 * ("lag_hit_search_limit" in flags)
    score -= 0.12 * ("no_divergence_signal" in flags)
    score = float(np.clip(score, 0.0, 1.0))

    conf = SectionConfidence(score=score, flags=flags, usable=score >= 0.45)
    return EventSummary(
        section_id=section_id,
        event_label=event_label,
        event_time_s=float(event_time_s),
        lag_ms=lag_ms,
        max_correlation=best_corr,
        shock_attenuation_pct=attenuation,
        divergence_peak_time_s=div_peak_t,
        bike_peak_vertical_m_s2=bike_peak if np.isfinite(bike_peak) else None,
        rider_peak_vertical_m_s2=rider_peak if np.isfinite(rider_peak) else None,
        summary_text=summary_text,
        confidence=conf,
    )


def _resolve_section_input(target: str, mode: str) -> list[Path]:
    p = Path(target)
    if mode == "section":
        if p.is_dir():
            return [p.resolve()]
        cand = sections_root() / target
        if cand.is_dir():
            return [cand.resolve()]
        raise FileNotFoundError(f"Section not found: {target}")

    if mode == "recording":
        if p.is_dir():
            recording_name = p.name
        else:
            recording_name = target
        return [s.resolve() for s in iter_sections_for_recording(recording_name)]

    # mode == all-good
    out: list[Path] = []
    root = sections_root()
    if not root.exists():
        return out
    for sec in sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name):
        if (sec / "derived" / "cross_sensor_signals.csv").exists():
            out.append(sec.resolve())
    return out


def _is_good_section(section_path: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    ddir = section_path / "derived"
    bike = ddir / "sporsa_signals.csv"
    rider = ddir / "arduino_signals.csv"
    cross = ddir / "cross_sensor_signals.csv"
    if not all(p.exists() for p in (bike, rider, cross)):
        reasons.append("missing_derived_artifacts")
        return False, reasons

    cdf = pd.read_csv(cross)
    if len(cdf) < 50:
        reasons.append("too_short")
    q = pd.to_numeric(cdf.get("quality_shock_valid", pd.Series([0] * len(cdf))), errors="coerce")
    if np.nanmean(q.to_numpy(dtype=float)) < 0.5:
        reasons.append("shock_quality_low")

    return len(reasons) == 0, reasons


def _plot_presentation(
    out_path: Path,
    section_id: str,
    t: np.ndarray,
    bike_vert: np.ndarray,
    rider_vert: np.ndarray,
    residual_vert: np.ndarray,
    shock_gain: np.ndarray,
    event: EventSummary,
    lags_s: np.ndarray,
    corr: np.ndarray,
) -> None:
    apply_matplotlib_thesis_style()
    fig, axs = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=False)

    m = np.isfinite(t) & (t >= event.event_time_s - 2.5) & (t <= event.event_time_s + 2.5)
    if np.sum(m) < 10:
        m = np.isfinite(t)

    axs[0].plot(t[m], bike_vert[m], lw=1.3, color=THESIS_COLORS[0], label="Bike vertical acc")
    axs[0].plot(t[m], rider_vert[m], lw=1.3, color=THESIS_COLORS[1], label="Rider vertical acc")
    axs[0].axvline(event.event_time_s, color="#333333", ls="--", lw=1.0, alpha=0.8)
    axs[0].set_ylabel("m/s²")
    axs[0].set_title("Aligned bike vs rider vertical response")
    axs[0].legend(loc="upper right")

    if lags_s.size and corr.size:
        axs[1].plot(lags_s * 1000.0, corr, color=THESIS_COLORS[2], lw=1.4)
        if event.lag_ms is not None:
            axs[1].axvline(event.lag_ms, color=THESIS_COLORS[3], ls="--", lw=1.1, label=f"best lag={event.lag_ms:.0f} ms")
            axs[1].legend(loc="upper right")
    axs[1].axhline(0.0, color="#666666", lw=0.8, alpha=0.6)
    axs[1].set_xlabel("Lag (ms), + means bike leads rider")
    axs[1].set_ylabel("Normalized xcorr")
    axs[1].set_title("Timing relationship: bike↔rider cross-correlation")

    axs[2].plot(t[m], residual_vert[m], color=THESIS_COLORS[4], lw=1.2, label="Rider-bike vertical divergence")
    axs[2].plot(t[m], shock_gain[m], color=THESIS_COLORS[5], lw=1.1, alpha=0.9, label="Shock transmission gain")
    axs[2].axhline(0.0, color="#666666", lw=0.8, alpha=0.5)
    axs[2].axvline(event.event_time_s, color="#333333", ls="--", lw=1.0, alpha=0.8)
    axs[2].set_xlabel("Section time (s)")
    axs[2].set_ylabel("Residual / gain")
    axs[2].set_title("Agreement-disagreement and magnitude relationship")
    axs[2].legend(loc="upper right")

    subtitle = (
        f"{event.event_label} @ {event.event_time_s:.2f}s | "
        f"confidence={event.confidence.score:.2f}"
    )
    fig.suptitle(f"Dual-IMU section report — {section_id}\n{subtitle}", fontsize=12)
    fig.text(0.01, 0.01, event.summary_text, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_diagnostics(
    out_path: Path,
    section_id: str,
    t: np.ndarray,
    bike_df: pd.DataFrame,
    rider_df: pd.DataFrame,
    cross_df: pd.DataFrame,
    summaries: list[EventSummary],
) -> None:
    apply_matplotlib_thesis_style()
    fig, axs = plt.subplots(4, 1, figsize=(11.5, 9.5), sharex=True)

    bike_long = _to_num(bike_df, "acc_longitudinal_m_s2")
    rider_long = _to_num(rider_df, "acc_longitudinal_m_s2")
    bike_roll = _to_num(bike_df, "tilt_roll_rate_rad_s")
    rider_roll = _to_num(rider_df, "tilt_roll_rate_rad_s")

    axs[0].plot(t, _to_num(bike_df, "acc_vertical_m_s2"), lw=1.0, label="bike vertical", color=THESIS_COLORS[0])
    axs[0].plot(t, _to_num(rider_df, "acc_vertical_m_s2"), lw=1.0, label="rider vertical", color=THESIS_COLORS[1])
    axs[0].legend(loc="upper right")
    axs[0].set_ylabel("m/s²")

    axs[1].plot(t, bike_long, lw=1.0, label="bike longitudinal", color=THESIS_COLORS[2])
    axs[1].plot(t, rider_long, lw=1.0, label="rider longitudinal", color=THESIS_COLORS[3])
    axs[1].legend(loc="upper right")
    axs[1].set_ylabel("m/s²")

    axs[2].plot(t, _to_num(cross_df, "residual_vertical_m_s2"), lw=1.0, label="vertical residual", color=THESIS_COLORS[4])
    axs[2].plot(t, _to_num(cross_df, "residual_longitudinal_m_s2"), lw=1.0, label="long residual", color=THESIS_COLORS[5])
    axs[2].legend(loc="upper right")
    axs[2].set_ylabel("m/s²")

    axs[3].plot(t, bike_roll, lw=1.0, label="bike roll rate", color=THESIS_COLORS[6])
    axs[3].plot(t, rider_roll, lw=1.0, label="rider roll rate", color=THESIS_COLORS[7])
    axs[3].plot(t, _to_num(cross_df, "shock_transmission_gain"), lw=1.0, label="shock gain", color=THESIS_COLORS[8])
    axs[3].legend(loc="upper right", ncol=3)
    axs[3].set_ylabel("rad/s, gain")
    axs[3].set_xlabel("Section time (s)")

    for ev in summaries:
        for ax in axs:
            ax.axvline(ev.event_time_s, color="#555555", ls="--", lw=0.8, alpha=0.6)

    fig.suptitle(f"Dual-IMU diagnostics — {section_id}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_section_report(section_path: Path, *, max_events: int = 3) -> dict[str, Any] | None:
    section_path = Path(section_path)
    good, bad_reasons = _is_good_section(section_path)
    if not good:
        return {
            "section": section_path.name,
            "status": "skipped",
            "reasons": bad_reasons,
        }

    ddir = section_path / "derived"
    bike = pd.read_csv(ddir / "sporsa_signals.csv")
    rider = pd.read_csv(ddir / "arduino_signals.csv")
    cross = pd.read_csv(ddir / "cross_sensor_signals.csv")

    n = min(len(bike), len(rider), len(cross))
    bike = bike.iloc[:n].copy()
    rider = rider.iloc[:n].copy()
    cross = cross.iloc[:n].copy()

    t = _to_num(cross, "time_s")
    bike_vert = _to_num(bike, "acc_vertical_m_s2")
    rider_vert = _to_num(rider, "acc_vertical_m_s2")
    residual_vert = _to_num(cross, "residual_vertical_m_s2")
    shock_gain = _to_num(cross, "shock_transmission_gain")

    dt_s = float(np.nanmedian(np.diff(t))) if len(t) > 2 else 0.01
    if (not np.isfinite(dt_s)) or dt_s <= 0:
        dt_s = 0.01

    events = _default_event_candidates(section_path, t, bike_vert)
    summaries: list[EventSummary] = []
    for label, t0 in events[:max_events]:
        summaries.append(
            summarize_bike_rider_event(
                section_id=section_path.name,
                event_label=label,
                event_time_s=t0,
                t=t,
                bike_vert=bike_vert,
                rider_vert=rider_vert,
                residual_vert=residual_vert,
                dt_s=dt_s,
            )
        )

    if not summaries:
        return {
            "section": section_path.name,
            "status": "skipped",
            "reasons": ["no_events_detected"],
        }

    main_event = max(summaries, key=lambda s: s.confidence.score)
    m = np.isfinite(t) & (t >= main_event.event_time_s - 1.8) & (t <= main_event.event_time_s + 1.8)
    lags_s, corr, _, _ = _cross_corr_lag(bike_vert[m], rider_vert[m], dt_s)

    out_root = section_path / "reporting" / "dual_imu"
    pres_path = out_root / "presentation" / "dual_imu_summary.png"
    diag_path = out_root / "diagnostics" / "dual_imu_diagnostics.png"
    _plot_presentation(pres_path, section_path.name, t, bike_vert, rider_vert, residual_vert, shock_gain, main_event, lags_s, corr)
    _plot_diagnostics(diag_path, section_path.name, t, bike, rider, cross, summaries)

    event_rows = []
    for s in summaries:
        d = asdict(s)
        d["confidence"] = asdict(s.confidence)
        event_rows.append(d)

    compact = {
        "section": section_path.name,
        "status": "ok",
        "main_event": event_rows[0] if event_rows else None,
        "events": event_rows,
        "section_summary": {
            "mean_shock_gain": float(np.nanmean(shock_gain)) if np.any(np.isfinite(shock_gain)) else None,
            "peak_divergence_m_s2": float(np.nanmax(np.abs(residual_vert))) if np.any(np.isfinite(residual_vert)) else None,
            "agreement_corr_vertical": _finite_corr(bike_vert, rider_vert),
        },
        "outputs": {
            "presentation_plot": str(pres_path),
            "diagnostic_plot": str(diag_path),
        },
    }

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "dual_imu_section_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")

    pd.DataFrame(
        {
            "event_label": [s.event_label for s in summaries],
            "event_time_s": [s.event_time_s for s in summaries],
            "lag_ms": [s.lag_ms for s in summaries],
            "max_correlation": [s.max_correlation for s in summaries],
            "shock_attenuation_pct": [s.shock_attenuation_pct for s in summaries],
            "divergence_peak_time_s": [s.divergence_peak_time_s for s in summaries],
            "confidence_score": [s.confidence.score for s in summaries],
            "confidence_flags": ["|".join(s.confidence.flags) if s.confidence.flags else "ok" for s in summaries],
            "summary_text": [s.summary_text for s in summaries],
        }
    ).to_csv(out_root / "event_summaries.csv", index=False)

    return compact


def _write_gallery_and_recommendations(processed: list[dict[str, Any]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    best = [p for p in processed if p.get("status") == "ok"]
    best = sorted(best, key=lambda x: x.get("events", [{}])[0].get("confidence", {}).get("score", 0.0), reverse=True)

    gallery_lines = [
        "# Dual-IMU Figure Gallery",
        "",
        "Auto-generated examples of presentation and diagnostic plots.",
        "",
    ]

    for entry in best[:8]:
        sec = entry["section"]
        pplot = entry.get("outputs", {}).get("presentation_plot", "")
        dplot = entry.get("outputs", {}).get("diagnostic_plot", "")
        txt = entry.get("events", [{}])[0].get("summary_text", "")
        gallery_lines.extend(
            [
                f"## {sec}",
                "",
                f"- Summary: {txt}",
                f"- Presentation: `{pplot}`",
                f"- Diagnostic: `{dplot}`",
                "",
            ]
        )

    (destination / "figure_gallery.md").write_text("\n".join(gallery_lines), encoding="utf-8")

    recs = [
        "1) Aligned bike-vs-rider vertical response around a high-confidence bump event.",
        "2) Cross-correlation lag curve showing bike-leading-rider timing and confidence annotations.",
        "3) Shock attenuation comparison across event types (bike peak vs rider peak vertical acceleration).",
        "4) Rider-bicycle divergence timeline (residual vertical + longitudinal) with event markers.",
        "5) Agreement/disagreement phase plot (bike vertical vs rider vertical) colored by time around events.",
        "6) Compact per-section dual-IMU summary heatmap (lag, attenuation, divergence, confidence).",
        "7) Multi-event rider response report panel (top 3 events in one recording).",
        "8) Diagnostic integrity panel (quality flags, missing alignment/orientation indicators) in appendix.",
    ]
    (destination / "thesis_figure_recommendations.md").write_text(
        "# Recommended Thesis Figures (Dual-IMU Added Value)\n\n" + "\n".join(f"- {r}" for r in recs),
        encoding="utf-8",
    )


def run(target: str, *, mode: str = "section", max_events: int = 3) -> dict[str, Any]:
    section_paths = _resolve_section_input(target, mode)
    if mode == "all-good":
        section_paths = [s for s in section_paths if _is_good_section(s)[0]]

    processed: list[dict[str, Any]] = []
    for sec in section_paths:
        res = build_section_report(sec, max_events=max_events)
        if res is not None:
            processed.append(res)

    out_csv = None
    if processed:
        rows = []
        for p in processed:
            if p.get("status") != "ok":
                rows.append({"section": p.get("section"), "status": p.get("status"), "reason": "|".join(p.get("reasons", []))})
                continue
            m = p.get("events", [{}])[0]
            c = m.get("confidence", {})
            rows.append(
                {
                    "section": p.get("section"),
                    "status": "ok",
                    "lag_ms": m.get("lag_ms"),
                    "shock_attenuation_pct": m.get("shock_attenuation_pct"),
                    "divergence_peak_time_s": m.get("divergence_peak_time_s"),
                    "confidence_score": c.get("score"),
                    "confidence_flags": "|".join(c.get("flags", [])) if c.get("flags") else "ok",
                    "summary_text": m.get("summary_text"),
                }
            )
        if mode == "section":
            out_root = section_paths[0] / "reporting" / "dual_imu"
        elif mode == "recording":
            out_root = recordings_root() / target / "reporting" / "dual_imu"
        else:
            out_root = sections_root() / "_dual_imu_reporting"
        out_root.mkdir(parents=True, exist_ok=True)
        out_csv = out_root / "dual_imu_section_summaries.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        _write_gallery_and_recommendations(processed, out_root)

    return {
        "mode": mode,
        "target": target,
        "sections_considered": len(section_paths),
        "sections_processed": len(processed),
        "summary_csv": str(out_csv) if out_csv else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m visualization.dual_imu_report")
    ap.add_argument("target", help="Section id/path, recording id/path, or any token for --all-good")
    ap.add_argument("--mode", choices=["section", "recording", "all-good"], default="section")
    ap.add_argument("--max-events", type=int, default=3)
    args = ap.parse_args()

    result = run(args.target, mode=args.mode, max_events=args.max_events)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
