"""Generate compact per-section summary artifacts for rapid inspection.

Outputs are intended to be thesis-friendly and supervisor-friendly: a single,
consistent document per section with identifiers, quality/confidence signals,
derived signal highlights, event candidates, dual-IMU comparison metrics,
warning flags, a short narrative, and small interpretable plots.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    parse_section_folder_name,
    recordings_root,
    sections_root,
)


@dataclass
class SectionSummaryData:
    recording_id: str
    section_id: str
    duration_s: float
    quality_category: str
    quality_label: str
    sync_confidence: float
    calibration_confidence: float
    orientation_confidence: float
    main_derived_signals: list[str]
    top_events: list[dict[str, Any]]
    salient_features: list[dict[str, Any]]
    dual_imu_summary: dict[str, Any]
    warnings: list[str]
    narrative: str


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _duration_seconds(section_path: Path) -> float:
    durations: list[float] = []
    for sensor in ("sporsa", "arduino"):
        df = _load_csv(section_path / f"{sensor}.csv")
        if df.empty or "timestamp" not in df.columns:
            continue
        ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        durations.append(float((ts.max() - ts.min()) / 1000.0))
    return float(np.nanmean(durations)) if durations else float("nan")


def _main_derived_signals(section_path: Path) -> list[str]:
    meta = _read_json(section_path / "derived" / "derived_signals_meta.json")
    signals = meta.get("signals", {}) if isinstance(meta, dict) else {}
    out: list[str] = []
    for name, block in signals.items():
        trust = bool((block or {}).get("trustworthy", False)) if isinstance(block, dict) else False
        if trust:
            out.append(name)
    if not out and signals:
        out = list(signals.keys())[:4]
    return out[:6]


def _top_events(section_path: Path, limit: int = 5) -> list[dict[str, Any]]:
    df = _load_csv(section_path / "events" / "event_candidates.csv")
    if df.empty:
        return []
    for col in ("confidence", "time_s", "trigger_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("confidence", ascending=False, na_position="last").head(limit)
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "event_type": str(r.get("event_type", "unknown")),
                "time_s": _safe_float(r.get("time_s")),
                "confidence": _safe_float(r.get("confidence")),
                "ambiguous": int(_safe_float(r.get("ambiguous_flag"), 0.0)) == 1,
                "trigger": str(r.get("key_trigger_signals", "")),
                "trigger_value": _safe_float(r.get("trigger_value")),
            }
        )
    return rows


def _salient_features(section_path: Path, limit: int = 8) -> list[dict[str, Any]]:
    df = _load_csv(section_path / "features" / "features.csv")
    if df.empty:
        return []

    preferred = [
        "sporsa__acc_norm_mean",
        "arduino__acc_norm_mean",
        "cross_sensor__acc_norm_mean_absdiff",
        "cross_sensor__jerk_norm_mean_absdiff",
        "sporsa__gyro_norm_mean",
        "arduino__gyro_norm_mean",
        "feature_confidence__cross_sensor",
        "upstream_confidence_score",
    ]

    picked = [c for c in preferred if c in df.columns]
    if len(picked) < limit:
        numeric_candidates = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and "timestamp" not in c and c not in picked
        ]
        # prioritize columns with strongest absolute median deviation from zero
        ranked = sorted(
            numeric_candidates,
            key=lambda c: abs(_safe_float(pd.to_numeric(df[c], errors="coerce").median(), 0.0)),
            reverse=True,
        )
        picked.extend([c for c in ranked if c not in picked])

    rows: list[dict[str, Any]] = []
    for c in picked[:limit]:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append(
            {
                "feature": c,
                "median": _safe_float(s.median()),
                "p95": _safe_float(s.quantile(0.95)),
            }
        )
    return rows


def _dual_imu_summary(section_path: Path) -> dict[str, Any]:
    cross = _load_csv(section_path / "derived" / "cross_sensor_signals.csv")
    out = {
        "residual_vertical_rms": float("nan"),
        "residual_longitudinal_rms": float("nan"),
        "shock_transmission_gain_median": float("nan"),
        "shock_transmission_gain_p95": float("nan"),
    }
    if cross.empty:
        return out

    def rms(col: str) -> float:
        if col not in cross.columns:
            return float("nan")
        x = pd.to_numeric(cross[col], errors="coerce").to_numpy(dtype=float)
        if len(x) == 0:
            return float("nan")
        return float(np.sqrt(np.nanmean(np.square(x))))

    out["residual_vertical_rms"] = rms("residual_vertical_m_s2")
    out["residual_longitudinal_rms"] = rms("residual_longitudinal_m_s2")
    if "shock_transmission_gain" in cross.columns:
        s = pd.to_numeric(cross["shock_transmission_gain"], errors="coerce")
        out["shock_transmission_gain_median"] = _safe_float(s.median())
        out["shock_transmission_gain_p95"] = _safe_float(s.quantile(0.95))
    return out


def _quality_and_confidence(section_path: Path) -> tuple[str, str, float, float, float, list[str]]:
    qc = _read_json(section_path / "qc_section.json")
    qmeta = _read_json(section_path / "quality_metadata.json")

    quality_category = str(qc.get("quality_tier") or qmeta.get("overall_usability_category") or "unknown")
    quality_label = str(qmeta.get("overall_quality_label") or "unknown")

    sync_conf = _safe_float(qmeta.get("sync_confidence"))
    calib_conf = _safe_float(qmeta.get("calibration_quality_score"))
    orient_conf = _safe_float(qmeta.get("orientation_quality_score"))

    warnings = [str(x) for x in qc.get("reasons", []) if str(x).strip()]
    flags = qmeta.get("quality_flags", [])
    if isinstance(flags, list):
        warnings.extend([f"quality_flag:{f}" for f in flags])

    for c_name, c_val in (
        ("sync", sync_conf),
        ("calibration", calib_conf),
        ("orientation", orient_conf),
    ):
        if math.isfinite(c_val) and c_val < 0.45:
            warnings.append(f"low_{c_name}_confidence:{c_val:.2f}")

    return quality_category, quality_label, sync_conf, calib_conf, orient_conf, sorted(set(warnings))


def _compose_narrative(data: SectionSummaryData) -> str:
    if not data.top_events:
        event_txt = "No strong event candidates were detected"
    else:
        top = data.top_events[0]
        ev = top.get("event_type", "event")
        conf = _safe_float(top.get("confidence"))
        t_s = _safe_float(top.get("time_s"))
        event_txt = f"The most likely event is '{ev}' around {t_s:.1f}s (confidence {conf:.2f})"

    quality_note = f"Section quality is '{data.quality_category}' (label: {data.quality_label})"

    dual = data.dual_imu_summary
    rv = _safe_float(dual.get("residual_vertical_rms"))
    sg = _safe_float(dual.get("shock_transmission_gain_median"))
    dual_note = (
        f"Dual-IMU coupling shows vertical residual RMS ≈ {rv:.2f} m/s² and median shock transmission gain ≈ {sg:.2f}."
        if math.isfinite(rv) and math.isfinite(sg)
        else "Dual-IMU coupling metrics are partially unavailable, so interpretation should be cautious."
    )

    warn = " "
    if data.warnings:
        warn = f" Warning flags: {', '.join(data.warnings[:3])}."

    return f"{quality_note}. {event_txt}. {dual_note}{warn}"


def _plot_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _build_plots(section_path: Path, plot_dir: Path) -> tuple[list[str], list[Path]]:
    sp = _load_csv(section_path / "derived" / "sporsa_signals.csv")
    ar = _load_csv(section_path / "derived" / "arduino_signals.csv")
    cross = _load_csv(section_path / "derived" / "cross_sensor_signals.csv")
    events = _load_csv(section_path / "events" / "event_candidates.csv")

    inline: list[str] = []
    files: list[Path] = []

    if not sp.empty and not ar.empty and "time_s" in sp.columns and "acc_vertical_m_s2" in sp.columns and "acc_vertical_m_s2" in ar.columns:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.plot(pd.to_numeric(sp["time_s"], errors="coerce"), pd.to_numeric(sp["acc_vertical_m_s2"], errors="coerce"), label="bike(sporsa)", lw=1.0)
        ax.plot(pd.to_numeric(ar["time_s"], errors="coerce"), pd.to_numeric(ar["acc_vertical_m_s2"], errors="coerce"), label="rider(arduino)", lw=1.0, alpha=0.85)
        ax.set_title("Vertical acceleration")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m/s²")
        ax.legend(loc="upper right", fontsize=8)
        inline.append(_plot_to_base64(fig))

        fig, ax = plt.subplots(figsize=(7, 2.6))
        if "shock_transmission_gain" in cross.columns and "time_s" in cross.columns:
            ax.plot(pd.to_numeric(cross["time_s"], errors="coerce"), pd.to_numeric(cross["shock_transmission_gain"], errors="coerce"), color="tab:purple", lw=1.0)
            ax.set_title("Shock transmission gain")
            ax.set_ylabel("gain")
        elif "residual_vertical_m_s2" in cross.columns and "time_s" in cross.columns:
            ax.plot(pd.to_numeric(cross["time_s"], errors="coerce"), pd.to_numeric(cross["residual_vertical_m_s2"], errors="coerce"), color="tab:purple", lw=1.0)
            ax.set_title("Vertical residual (rider - bike)")
            ax.set_ylabel("m/s²")
        ax.set_xlabel("time (s)")
        inline.append(_plot_to_base64(fig))

    if not events.empty and "time_s" in events.columns and "confidence" in events.columns:
        ev = events.copy()
        ev["confidence"] = pd.to_numeric(ev["confidence"], errors="coerce")
        ev["time_s"] = pd.to_numeric(ev["time_s"], errors="coerce")
        ev = ev.dropna(subset=["time_s", "confidence"]).sort_values("confidence", ascending=False).head(10)
        if not ev.empty:
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.scatter(ev["time_s"], ev["confidence"], c=ev["confidence"], cmap="viridis", s=35)
            ax.set_title("Top event candidates")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("confidence")
            ax.set_ylim(0, 1.05)
            inline.append(_plot_to_base64(fig))

    for idx, b64 in enumerate(inline, start=1):
        png_path = plot_dir / f"plot_{idx:02d}.png"
        img_bytes = base64.b64decode(b64.encode("ascii"))
        png_path.parent.mkdir(parents=True, exist_ok=True)
        png_path.write_bytes(img_bytes)
        files.append(png_path)

    return inline, files


def build_section_summary(section_path: Path) -> SectionSummaryData:
    section_path = Path(section_path)
    section_id = section_path.name
    recording_id, _ = parse_section_folder_name(section_id)

    quality_category, quality_label, sync_conf, calib_conf, orient_conf, warnings = _quality_and_confidence(section_path)

    data = SectionSummaryData(
        recording_id=recording_id,
        section_id=section_id,
        duration_s=_duration_seconds(section_path),
        quality_category=quality_category,
        quality_label=quality_label,
        sync_confidence=sync_conf,
        calibration_confidence=calib_conf,
        orientation_confidence=orient_conf,
        main_derived_signals=_main_derived_signals(section_path),
        top_events=_top_events(section_path),
        salient_features=_salient_features(section_path),
        dual_imu_summary=_dual_imu_summary(section_path),
        warnings=warnings,
        narrative="",
    )
    data.narrative = _compose_narrative(data)
    return data


def _fmt(v: Any, digits: int = 2) -> str:
    x = _safe_float(v)
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def render_markdown(data: SectionSummaryData, *, plot_paths: list[Path]) -> str:
    warnings = "\n".join([f"> ⚠️ {w}" for w in data.warnings]) if data.warnings else "> ✅ No major warnings."
    events = "\n".join(
        [
            f"- `{e['event_type']}` @ { _fmt(e['time_s'],1)}s, conf={_fmt(e['confidence'])}, ambiguous={e['ambiguous']}"
            for e in data.top_events
        ]
    ) or "- none"
    features = "\n".join(
        [f"- `{f['feature']}`: median={_fmt(f['median'])}, p95={_fmt(f['p95'])}" for f in data.salient_features]
    ) or "- none"

    plots_md = "\n".join([f"![{p.stem}]({p.name})" for p in plot_paths]) if plot_paths else "_No plots available._"

    return f"""# Section Summary — {data.section_id}

## Snapshot
- Recording: `{data.recording_id}`
- Section: `{data.section_id}`
- Duration: {_fmt(data.duration_s, 1)} s
- Quality category: **{data.quality_category}** (label: `{data.quality_label}`)
- Confidence (sync/calibration/orientation): `{_fmt(data.sync_confidence)}` / `{_fmt(data.calibration_confidence)}` / `{_fmt(data.orientation_confidence)}`

## Warnings
{warnings}

## Main derived signals
{"; ".join(data.main_derived_signals) if data.main_derived_signals else "none"}

## Top event candidates
{events}

## Salient feature values
{features}

## Dual-IMU comparison summary
- residual_vertical_rms: {_fmt(data.dual_imu_summary.get('residual_vertical_rms'))} m/s²
- residual_longitudinal_rms: {_fmt(data.dual_imu_summary.get('residual_longitudinal_rms'))} m/s²
- shock_transmission_gain_median: {_fmt(data.dual_imu_summary.get('shock_transmission_gain_median'))}
- shock_transmission_gain_p95: {_fmt(data.dual_imu_summary.get('shock_transmission_gain_p95'))}

## Narrative
{data.narrative}

## Quick plots
{plots_md}
"""


def render_html(data: SectionSummaryData, *, inline_plots: list[str]) -> str:
    warning_items = "".join([f"<li>⚠️ {w}</li>" for w in data.warnings]) or "<li>✅ No major warnings.</li>"
    event_rows = "".join(
        [
            "<tr>"
            f"<td>{e['event_type']}</td><td>{_fmt(e['time_s'],1)} s</td><td>{_fmt(e['confidence'])}</td><td>{e['ambiguous']}</td>"
            "</tr>"
            for e in data.top_events
        ]
    ) or "<tr><td colspan='4'>none</td></tr>"
    feature_rows = "".join(
        [
            "<tr>"
            f"<td>{f['feature']}</td><td>{_fmt(f['median'])}</td><td>{_fmt(f['p95'])}</td>"
            "</tr>"
            for f in data.salient_features
        ]
    ) or "<tr><td colspan='3'>none</td></tr>"
    images = "".join([f"<img src='data:image/png;base64,{b}' style='max-width:32%;margin-right:8px;border:1px solid #ddd;'/>" for b in inline_plots])

    return f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'/>
<title>Section Summary {data.section_id}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1,h2 {{ margin-bottom: 8px; }}
.grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.card {{ border:1px solid #ddd; border-radius:8px; padding:10px; }}
table {{ width:100%; border-collapse: collapse; font-size: 14px; }}
th,td {{ border:1px solid #ddd; padding:6px; text-align:left; }}
.warn {{ background:#fff7e6; border-left:4px solid #f59f00; padding:8px; }}
</style>
</head>
<body>
<h1>Section Summary — {data.section_id}</h1>
<div class='grid'>
  <div class='card'>
    <h2>Snapshot</h2>
    <ul>
      <li>Recording: <code>{data.recording_id}</code></li>
      <li>Duration: {_fmt(data.duration_s, 1)} s</li>
      <li>Quality: <b>{data.quality_category}</b> ({data.quality_label})</li>
      <li>Sync/Calibration/Orientation: {_fmt(data.sync_confidence)} / {_fmt(data.calibration_confidence)} / {_fmt(data.orientation_confidence)}</li>
    </ul>
  </div>
  <div class='card warn'>
    <h2>Warnings</h2>
    <ul>{warning_items}</ul>
  </div>
</div>

<div class='card'>
<h2>Main derived signals</h2>
<p>{'; '.join(data.main_derived_signals) if data.main_derived_signals else 'none'}</p>
</div>

<div class='grid'>
<div class='card'>
<h2>Top event candidates</h2>
<table><thead><tr><th>type</th><th>time</th><th>conf</th><th>ambig</th></tr></thead><tbody>{event_rows}</tbody></table>
</div>
<div class='card'>
<h2>Salient features</h2>
<table><thead><tr><th>feature</th><th>median</th><th>p95</th></tr></thead><tbody>{feature_rows}</tbody></table>
</div>
</div>

<div class='card'>
<h2>Dual-IMU comparison</h2>
<ul>
<li>vertical residual RMS: {_fmt(data.dual_imu_summary.get('residual_vertical_rms'))} m/s²</li>
<li>longitudinal residual RMS: {_fmt(data.dual_imu_summary.get('residual_longitudinal_rms'))} m/s²</li>
<li>shock gain (median / p95): {_fmt(data.dual_imu_summary.get('shock_transmission_gain_median'))} / {_fmt(data.dual_imu_summary.get('shock_transmission_gain_p95'))}</li>
</ul>
</div>

<div class='card'>
<h2>Automatic narrative</h2>
<p>{data.narrative}</p>
</div>

<div class='card'>
<h2>Quick plots</h2>
<div>{images or '<p>No plots available.</p>'}</div>
</div>
</body>
</html>
"""


def _resolve_sections(target: str | None, *, all_sections: bool, all_recordings: bool) -> list[Path]:
    root = sections_root()
    if not root.exists():
        return []

    if all_recordings:
        return sorted([d for d in root.iterdir() if d.is_dir()])

    if not target:
        raise ValueError("target is required unless --all is used")

    p = Path(target)
    if p.is_dir() and (p / "qc_section.json").exists():
        return [p]

    sec_candidate = root / target
    if sec_candidate.is_dir():
        return [sec_candidate]

    # recording id
    rec_dir = recordings_root() / target
    if rec_dir.is_dir() or all_sections:
        return iter_sections_for_recording(target)

    raise FileNotFoundError(f"Could not resolve target={target!r} to section(s)")


def write_section_summary(
    section_path: Path,
    *,
    output_root: Path,
    fmt: str,
) -> Path:
    data = build_section_summary(section_path)
    out_dir = output_root / data.recording_id / data.section_id
    out_dir.mkdir(parents=True, exist_ok=True)
    inline_plots, plot_paths = _build_plots(section_path, out_dir / "plots")

    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(data.__dict__, indent=2), encoding="utf-8")

    if fmt == "html":
        html_path = out_dir / "summary.html"
        html_path.write_text(render_html(data, inline_plots=inline_plots), encoding="utf-8")
        return html_path

    md_path = out_dir / "summary.md"
    rel_paths = [p.relative_to(out_dir) for p in plot_paths]
    md_path.write_text(render_markdown(data, plot_paths=rel_paths), encoding="utf-8")
    return md_path


def write_example_summaries(examples_dir: Path) -> list[Path]:
    examples_dir.mkdir(parents=True, exist_ok=True)

    good = SectionSummaryData(
        recording_id="2026-02-26_r2",
        section_id="2026-02-26_r2s3",
        duration_s=22.4,
        quality_category="good",
        quality_label="high",
        sync_confidence=0.91,
        calibration_confidence=0.88,
        orientation_confidence=0.84,
        main_derived_signals=[
            "gravity_compensated_linear_acceleration",
            "longitudinal_lateral_axes",
            "tilt_rates",
            "shock_transmission_vertical",
        ],
        top_events=[
            {"event_type": "bump_shock_candidate", "time_s": 8.7, "confidence": 0.82, "ambiguous": False},
            {"event_type": "braking_burst", "time_s": 14.1, "confidence": 0.73, "ambiguous": False},
        ],
        salient_features=[
            {"feature": "sporsa__acc_norm_mean", "median": 10.52, "p95": 13.21},
            {"feature": "cross_sensor__acc_norm_mean_absdiff", "median": 1.34, "p95": 2.44},
        ],
        dual_imu_summary={
            "residual_vertical_rms": 1.12,
            "residual_longitudinal_rms": 0.87,
            "shock_transmission_gain_median": 1.08,
            "shock_transmission_gain_p95": 1.41,
        },
        warnings=[],
        narrative="Section quality is good with stable sensor agreement. A moderate bump is followed by a short braking burst, consistent with obstacle negotiation and controlled deceleration.",
    )

    marginal = SectionSummaryData(
        recording_id="2026-02-26_r4",
        section_id="2026-02-26_r4s1",
        duration_s=9.3,
        quality_category="marginal",
        quality_label="low",
        sync_confidence=0.43,
        calibration_confidence=0.51,
        orientation_confidence=0.39,
        main_derived_signals=["gravity_compensated_linear_acceleration", "shock_transmission_vertical"],
        top_events=[
            {"event_type": "rider_bicycle_divergence", "time_s": 3.2, "confidence": 0.41, "ambiguous": True},
            {"event_type": "swerve_roll_rate_candidate", "time_s": 6.0, "confidence": 0.36, "ambiguous": True},
        ],
        salient_features=[
            {"feature": "feature_confidence__cross_sensor", "median": 0.35, "p95": 0.52},
            {"feature": "upstream_confidence_score", "median": 0.42, "p95": 0.58},
        ],
        dual_imu_summary={
            "residual_vertical_rms": 2.86,
            "residual_longitudinal_rms": float("nan"),
            "shock_transmission_gain_median": 1.39,
            "shock_transmission_gain_p95": 2.24,
        },
        warnings=[
            "sporsa orientation quality marginal",
            "low_sync_confidence:0.43",
            "quality_flag:sync_corr_missing",
        ],
        narrative="This section appears short and ambiguous: rider-bike divergence and swerve-like activity are present, but confidence is low because sync and orientation quality are limited.",
    )

    good_md = examples_dir / "example_good_section_summary.md"
    marginal_md = examples_dir / "example_marginal_section_summary.md"
    good_md.write_text(render_markdown(good, plot_paths=[]), encoding="utf-8")
    marginal_md.write_text(render_markdown(marginal, plot_paths=[]), encoding="utf-8")
    return [good_md, marginal_md]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.section_summary",
        description="Generate compact per-section summary artifacts (Markdown or HTML).",
    )
    parser.add_argument("target", nargs="?", help="Section folder/path or recording id")
    parser.add_argument("--all-sections", action="store_true", help="Interpret target as recording id and summarize all its sections")
    parser.add_argument("--all", dest="all_recordings", action="store_true", help="Summarize every available section")
    parser.add_argument("--format", choices=("markdown", "html"), default="markdown")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/section_summaries"),
        help="Folder root for generated artifacts (default: data/section_summaries)",
    )
    parser.add_argument(
        "--write-examples",
        action="store_true",
        help="Also write example good/marginal summary markdown files.",
    )
    args = parser.parse_args(argv)

    sections: list[Path] = []
    if args.target or args.all_recordings:
        sections = _resolve_sections(args.target, all_sections=args.all_sections, all_recordings=args.all_recordings)
    if not sections and not args.write_examples:
        raise SystemExit("No sections found to summarize.")

    written: list[Path] = []
    for section in sections:
        out = write_section_summary(section, output_root=args.output_root, fmt=args.format)
        written.append(out)
        print(out)

    if args.write_examples:
        examples = write_example_summaries(args.output_root / "examples")
        for p in examples:
            print(p)


if __name__ == "__main__":
    main()
