"""Compare ``synced_lida`` and ``synced_cal`` synchronization results.

Loads ``sync_info.json`` from both stage directories for a recording and prints
a side-by-side comparison of the fitted models and quality metrics.  Optionally
generates an overlay plot of the ``acc_norm`` signals to give a visual sense of
alignment quality for each method.

CLI::

    python -m sync.compare_sync <recording_name>
    python -m sync.compare_sync 2026-02-26_5 --plot
    python -m sync.compare_sync 2026-02-26_5 --all-2026-02-26

Key fields compared
-------------------
- ``offset_seconds``        — absolute timestamp alignment (sporsa epoch − arduino boot)
- ``drift_seconds_per_second`` — estimated clock drift between the two sensors
- ``correlation.offset_only``  — acc_norm Pearson r after offset-only correction
- ``correlation.offset_and_drift`` — acc_norm Pearson r after full drift correction
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import recording_stage_dir, recordings_root

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_sync_info(recording_name: str, stage: str) -> Optional[dict]:
    """Load ``sync_info.json`` from *stage* directory, or return ``None``."""
    path = recording_stage_dir(recording_name, stage) / "sync_info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sensor_csv(recording_name: str, stage: str, sensor: str) -> Optional[pd.DataFrame]:
    """Load a sensor CSV from *stage*, returning ``None`` if absent."""
    csv_path = recording_stage_dir(recording_name, stage) / f"{sensor}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return None
    return df


# ---------------------------------------------------------------------------
# Comparison data structure
# ---------------------------------------------------------------------------

def compare_sync_models(
    recording_name: str,
    *,
    stage_lida: str = "synced_lida",
    stage_cal: str = "synced_cal",
) -> dict:
    """Load and compare sync models from both methods for one recording.

    Returns a dict with keys:
    - ``"recording"``     — recording name
    - ``"lida"``          — raw dict from ``synced_lida/sync_info.json`` (or None)
    - ``"cal"``           — raw dict from ``synced_cal/sync_info.json``  (or None)
    - ``"delta_offset_s"``  — cal_offset − lida_offset (None if either missing)
    - ``"delta_drift_ppm"`` — (cal_drift − lida_drift) × 1e6 (None if either missing)
    """
    lida = _load_sync_info(recording_name, stage_lida)
    cal = _load_sync_info(recording_name, stage_cal)

    delta_offset = None
    delta_drift_ppm = None
    if lida is not None and cal is not None:
        delta_offset = cal["offset_seconds"] - lida["offset_seconds"]
        delta_drift_ppm = (
            cal["drift_seconds_per_second"] - lida["drift_seconds_per_second"]
        ) * 1e6

    return {
        "recording": recording_name,
        "lida": lida,
        "cal": cal,
        "delta_offset_s": delta_offset,
        "delta_drift_ppm": delta_drift_ppm,
    }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _fmt_none(value, fmt: str, suffix: str = "") -> str:
    if value is None:
        return "  N/A"
    return f"{value:{fmt}}{suffix}"


def _corr_block(info: Optional[dict]) -> tuple[str, str]:
    """Return (offset_only, offset_and_drift) correlation strings."""
    if info is None:
        return "  N/A", "  N/A"
    corr = info.get("correlation", {})
    r_off = corr.get("offset_only")
    r_drift = corr.get("offset_and_drift")
    return _fmt_none(r_off, ".4f"), _fmt_none(r_drift, ".4f")


def print_comparison(result: dict) -> None:
    """Pretty-print a comparison result to stdout."""
    rec = result["recording"]
    lida = result["lida"]
    cal = result["cal"]

    print(f"\n{'─' * 60}")
    print(f"  Recording : {rec}")
    print(f"{'─' * 60}")

    header = f"  {'Metric':<34} {'SDA+LIDA':>12}  {'Cal-Sync':>12}"
    print(header)
    print(f"  {'-' * 58}")

    # Offset
    lida_off = lida["offset_seconds"] if lida else None
    cal_off = cal["offset_seconds"] if cal else None
    print(
        f"  {'Offset (s)':<34} "
        f"{_fmt_none(lida_off, '18.3f'):>12}  "
        f"{_fmt_none(cal_off, '18.3f'):>12}"
    )

    # Delta offset
    d_off = result["delta_offset_s"]
    print(
        f"  {'  Δ offset (cal − lida, s)':<34} "
        f"{'':>12}  "
        f"{_fmt_none(d_off, '+.3f'):>12}"
    )

    # Drift ppm
    lida_drift = lida["drift_seconds_per_second"] * 1e6 if lida else None
    cal_drift = cal["drift_seconds_per_second"] * 1e6 if cal else None
    print(
        f"  {'Drift (ppm)':<34} "
        f"{_fmt_none(lida_drift, '.1f'):>12}  "
        f"{_fmt_none(cal_drift, '.1f'):>12}"
    )

    d_drift = result["delta_drift_ppm"]
    print(
        f"  {'  Δ drift (cal − lida, ppm)':<34} "
        f"{'':>12}  "
        f"{_fmt_none(d_drift, '+.1f'):>12}"
    )

    # Correlations
    lida_r_off, lida_r_drift = _corr_block(lida)
    cal_r_off, cal_r_drift = _corr_block(cal)
    print(
        f"  {'Corr (offset only)':<34} "
        f"{lida_r_off:>12}  "
        f"{cal_r_off:>12}"
    )
    print(
        f"  {'Corr (offset + drift)':<34} "
        f"{lida_r_drift:>12}  "
        f"{cal_r_drift:>12}"
    )

    # Calibration details (cal method only)
    if cal is not None and "calibration" in cal:
        cal_detail = cal["calibration"]
        span = cal_detail.get("calibration_span_s")
        opening_score = cal_detail.get("opening", {}).get("score")
        closing_score = cal_detail.get("closing", {}).get("score")
        print(
            f"  {'Cal span (s)':<34} "
            f"{'':>12}  "
            f"{_fmt_none(span, '.1f'):>12}"
        )
        print(
            f"  {'Cal score  opening / closing':<34} "
            f"{'':>12}  "
            f"  {_fmt_none(opening_score, '.3f')} / {_fmt_none(closing_score, '.3f')}"
        )

    print(f"{'─' * 60}")


# ---------------------------------------------------------------------------
# Overlay plot
# ---------------------------------------------------------------------------

def plot_sync_comparison(
    recording_name: str,
    *,
    stage_lida: str = "synced_lida",
    stage_cal: str = "synced_cal",
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = 10.0,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Generate an acc_norm overlay plot comparing the two sync methods.

    Loads ``sporsa.csv`` and ``arduino.csv`` from both stage directories,
    resamples each to *sample_rate_hz*, and draws four sub-panels:

    - **Top row**: acc_norm overlay for SDA+LIDA-synced streams
    - **Bottom row**: acc_norm overlay for calibration-synced streams

    The left column shows the full recording; the right column zooms into the
    first 60 s (calibration region) to reveal fine timing differences.

    Returns the path to the saved PNG, or ``None`` if neither stage exists.
    """
    from .common import add_vector_norms, resample_stream

    def _load_resample(recording_name: str, stage: str, sensor: str) -> Optional[pd.DataFrame]:
        df = _load_sensor_csv(recording_name, stage, sensor)
        if df is None or df.empty:
            return None
        df = add_vector_norms(df)
        return resample_stream(df, sample_rate_hz)

    lida_ref = _load_resample(recording_name, stage_lida, reference_sensor)
    lida_tgt = _load_resample(recording_name, stage_lida, target_sensor)
    cal_ref = _load_resample(recording_name, stage_cal, reference_sensor)
    cal_tgt = _load_resample(recording_name, stage_cal, target_sensor)

    if lida_ref is None and cal_ref is None:
        log.warning("No data found for either sync method – skipping plot.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharey="row")
    fig.suptitle(f"{recording_name}: acc_norm alignment comparison", fontsize=12)

    panels = [
        (0, "SDA + LIDA", lida_ref, lida_tgt),
        (1, "Calibration-sync", cal_ref, cal_tgt),
    ]
    zoom_s = 90.0  # seconds shown in right column

    for row, label, ref_df, tgt_df in panels:
        for col, zoom in enumerate([False, True]):
            ax = axes[row, col]

            def _plot_stream(df: Optional[pd.DataFrame], name: str, color: str) -> None:
                if df is None or df.empty:
                    return
                ts = df["timestamp"].to_numpy(dtype=float)
                # Convert to seconds relative to start of reference
                if ref_df is not None and not ref_df.empty:
                    t0 = float(ref_df["timestamp"].iloc[0])
                else:
                    t0 = ts[0]
                t_s = (ts - t0) / 1000.0 if ts.mean() > 1e9 else ts - t0
                acc = df["acc_norm"].to_numpy(dtype=float)
                ax.plot(t_s, acc, lw=0.6, alpha=0.85, label=name, color=color)

            _plot_stream(ref_df, reference_sensor, "steelblue")
            _plot_stream(tgt_df, target_sensor, "tomato")

            if zoom and ref_df is not None and not ref_df.empty:
                ax.set_xlim(0, zoom_s)

            ax.set_xlabel("Time (s)" if row == 1 else "")
            ax.set_ylabel("acc_norm (m/s²)" if col == 0 else "")
            title = f"{label} — {'first {:.0f} s'.format(zoom_s) if zoom else 'full recording'}"
            ax.set_title(title, fontsize=9)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_dir is None:
        out_dir = recordings_root() / recording_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sync_method_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}] Comparison plot saved → {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def compare_all_recordings(
    session_prefix: str,
    *,
    stage_lida: str = "synced_lida",
    stage_cal: str = "synced_cal",
    plot: bool = False,
) -> list[dict]:
    """Run :func:`compare_sync_models` for every recording matching *session_prefix*.

    Returns a list of result dicts (one per recording).
    """
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith(f"{session_prefix}_")
    )
    if not recordings:
        log.warning("No recordings found matching prefix '%s_'", session_prefix)
        return []

    results = []
    for rec in recordings:
        result = compare_sync_models(rec, stage_lida=stage_lida, stage_cal=stage_cal)
        results.append(result)
        print_comparison(result)
        if plot:
            plot_sync_comparison(rec, stage_lida=stage_lida, stage_cal=stage_cal)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.compare_sync",
        description=(
            "Compare SDA+LIDA (synced_lida) and calibration-based (synced_cal) "
            "synchronization results for one or more recordings."
        ),
    )
    parser.add_argument(
        "recording_name",
        help="Recording name (e.g. '2026-02-26_5') or session prefix when --all is used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Compare all recordings whose name starts with RECORDING_NAME followed by '_'.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate acc_norm overlay PNG for each recording.",
    )
    parser.add_argument(
        "--stage-lida",
        default="synced_lida",
        metavar="STAGE",
        help="Stage name for SDA+LIDA output (default: synced_lida).",
    )
    parser.add_argument(
        "--stage-cal",
        default="synced_cal",
        metavar="STAGE",
        help="Stage name for calibration-sync output (default: synced_cal).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.all_recordings:
        compare_all_recordings(
            args.recording_name,
            stage_lida=args.stage_lida,
            stage_cal=args.stage_cal,
            plot=args.plot,
        )
    else:
        result = compare_sync_models(
            args.recording_name,
            stage_lida=args.stage_lida,
            stage_cal=args.stage_cal,
        )
        print_comparison(result)
        if args.plot:
            plot_sync_comparison(
                args.recording_name,
                stage_lida=args.stage_lida,
                stage_cal=args.stage_cal,
            )


if __name__ == "__main__":
    main()
