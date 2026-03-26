"""Timing quality visualizations for parsed IMU recordings.

Produces three figure types per recording, all saved in the ``parsed/`` stage
directory:

1. **Interval distributions** (``parsed_timing_intervals.png``):
   Overlaid histograms of inter-packet intervals for Sporsa and Arduino,
   revealing timing jitter and the nominal sample period of each device.

2. **Interval timeline** (``parsed_timing_timeline.png``):
   Inter-packet interval as a function of recording time for both sensors,
   with gap markers (intervals exceeding the gap threshold) highlighted in red.
   Useful for spotting burst-loss events and BLE dropout patterns.

3. **Arduino clock drift** (``parsed_clock_drift.png``):
   Scatter of the host-received timestamp against the device timestamp for the
   Arduino stream, with a linear fit to estimate clock drift in ppm and a
   residual subplot to assess fit quality.  The Arduino device clock runs
   independently of the host clock, so this plot reveals the systematic offset
   and drift that must be corrected during synchronisation.

CLI::

    python -m visualization.plot_timing 2026-02-26_5
    python -m visualization.plot_timing 2026-02-26_5 --no-drift
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import find_sensor_csv, load_dataframe, recording_dir
from ._utils import mask_valid_plot_x, nan_mask_invalid_plot_x, time_axis_seconds, resolve_stage_dir, parse_recording_or_section_id

log = logging.getLogger(__name__)

_STAGE = "parsed"
_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_GAP_COLOR = "#d62728"

# Per-sensor nominal gap thresholds (ms) — intervals above this are gaps.
_GAP_THRESHOLDS = {"sporsa": 15.0, "arduino": 25.5}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_timing_stats(recording_name: str) -> dict:
    # For section ids we currently don't have per-section timing stats;
    # just return empty so timing plots are skipped.
    _rec, sec_idx = parse_recording_or_section_id(recording_name)
    if sec_idx is not None:
        return {}

    stats_path = recording_dir(_rec) / "session_stats.json"
    if not stats_path.exists():
        legacy = resolve_stage_dir(_rec, _STAGE) / "session_stats.json"
        stats_path = legacy if legacy.exists() else stats_path
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _packet_intervals(timestamps_ms: pd.Series) -> np.ndarray:
    """Return inter-packet intervals in milliseconds."""
    ts = timestamps_ms.astype(float).to_numpy()
    return np.diff(ts)


# ---------------------------------------------------------------------------
# Plot 1: Interval distribution
# ---------------------------------------------------------------------------

def plot_timing_intervals(
    recording_name: str,
    *,
    max_interval_ms: float = 100.0,
) -> Path | None:
    """Histogram of inter-packet intervals for both sensors.

    Returns the path of the saved PNG, or ``None`` if no CSVs were found.
    """
    stage_dir = resolve_stage_dir(recording_name, _STAGE)
    if not stage_dir.exists():
        log.warning("[%s/%s] stage directory not found — skipping timing plots.", recording_name, _STAGE)
        return None
    stats = _load_timing_stats(recording_name)

    dfs: dict[str, pd.DataFrame] = {}
    for sensor in _SENSORS:
        try:
            csv_path = find_sensor_csv(recording_name, _STAGE, sensor)
            df = load_dataframe(csv_path)
            if not df.empty:
                dfs[sensor] = df
        except FileNotFoundError:
            pass

    if not dfs:
        log.warning("[%s/%s] no sensor CSVs found — skipping timing intervals plot.", recording_name, _STAGE)
        return None

    fig, axes = plt.subplots(
        1, len(dfs),
        figsize=(5 * len(dfs), 4),
        constrained_layout=True,
    )
    if len(dfs) == 1:
        axes = [axes]

    for ax, (sensor, df) in zip(axes, dfs.items()):
        intervals = _packet_intervals(df["timestamp"])
        intervals_clipped = intervals[intervals <= max_interval_ms]
        gap_thresh = _GAP_THRESHOLDS.get(sensor, 30.0)

        stream_stats = stats.get("streams", {}).get(sensor, {})
        timing = stream_stats.get("timing", {})
        median_ms = timing.get("interval_ms", {}).get("median_ms", np.nan)
        std_ms = timing.get("interval_ms", {}).get("std_ms", np.nan)
        gap_count = timing.get("gaps", {}).get("gap_count", "?")

        n_bins = min(200, max(30, len(intervals_clipped) // 50))
        ax.hist(
            intervals_clipped,
            bins=n_bins,
            color=_SENSOR_COLORS.get(sensor, "gray"),
            alpha=0.75,
            edgecolor="none",
        )
        ax.axvline(
            gap_thresh, color=_GAP_COLOR, linestyle="--", linewidth=1.2,
            label=f"gap threshold ({gap_thresh:.0f} ms)",
        )
        if np.isfinite(median_ms):
            ax.axvline(
                median_ms, color="k", linestyle=":", linewidth=1.0,
                label=f"median = {median_ms:.1f} ms",
            )

        info = f"n = {len(intervals_clipped)}\nmedian = {median_ms:.1f} ms\nstd = {std_ms:.1f} ms\ngaps > {gap_thresh:.0f} ms: {gap_count}"
        ax.text(
            0.97, 0.97, info,
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

        ax.set_title(sensor, fontsize=10)
        ax.set_xlabel("Inter-packet interval [ms]", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{recording_name} / {_STAGE} — Packet interval distributions", fontsize=11)
    out_path = stage_dir / "parsed_timing_intervals.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{_STAGE}] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 2: Interval timeline
# ---------------------------------------------------------------------------

def plot_timing_timeline(
    recording_name: str,
    *,
    max_display_ms: float = 200.0,
) -> Path | None:
    """Inter-packet interval over time with gap events highlighted.

    Returns the path of the saved PNG, or ``None`` if no CSVs were found.
    """
    stage_dir = resolve_stage_dir(recording_name, _STAGE)
    if not stage_dir.exists():
        return None

    dfs: dict[str, pd.DataFrame] = {}
    for sensor in _SENSORS:
        try:
            csv_path = find_sensor_csv(recording_name, _STAGE, sensor)
            df = load_dataframe(csv_path)
            if not df.empty:
                dfs[sensor] = df
        except FileNotFoundError:
            pass

    if not dfs:
        log.warning("[%s/%s] no sensor CSVs found — skipping timing timeline.", recording_name, _STAGE)
        return None

    fig, axes = plt.subplots(
        len(dfs), 1,
        figsize=(12, 3 * len(dfs)),
        sharex=False,
        constrained_layout=True,
    )
    if len(dfs) == 1:
        axes = [axes]

    for ax, (sensor, df) in zip(axes, dfs.items()):
        ts = df["timestamp"].astype(float).to_numpy()
        t_mid = ((ts[:-1] + ts[1:]) / 2.0 - ts[0]) / 1000.0  # midpoint time in seconds
        intervals = np.diff(ts)

        gap_thresh = _GAP_THRESHOLDS.get(sensor, 30.0)
        is_gap = intervals > gap_thresh

        display_mask = intervals <= max_display_ms
        xm = mask_valid_plot_x(t_mid)
        plot_mask = display_mask & xm
        ax.plot(
            t_mid[plot_mask],
            intervals[plot_mask],
            color=_SENSOR_COLORS.get(sensor, "gray"),
            linewidth=0.5,
            alpha=0.7,
            label="interval",
        )
        if is_gap.any():
            vmask = is_gap & (intervals <= max_display_ms) & xm
            ax.vlines(
                t_mid[vmask],
                ymin=0, ymax=gap_thresh * 1.5,
                colors=_GAP_COLOR,
                linewidth=0.8,
                alpha=0.6,
                label=f"gaps > {gap_thresh:.0f} ms ({is_gap.sum()})",
            )
        ax.axhline(gap_thresh, color=_GAP_COLOR, linestyle="--", linewidth=0.9, alpha=0.6)

        ax.set_title(sensor, fontsize=10)
        ax.set_ylabel("Interval [ms]", fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"{recording_name} / {_STAGE} — Packet interval timeline", fontsize=11)
    out_path = stage_dir / "parsed_timing_timeline.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{_STAGE}] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Plot 3: Arduino clock drift
# ---------------------------------------------------------------------------

def plot_clock_drift(recording_name: str) -> Path | None:
    """Scatter of device vs host timestamps for the Arduino stream.

    The Arduino BLE device uses an internal clock (ms since boot) while the
    host phone records a wall-clock epoch timestamp when each BLE packet is
    received.  The two clocks drift relative to each other at a rate of tens
    to hundreds of ppm.  This plot visualises the relationship and estimates
    the drift from a linear fit.

    Returns the path of the saved PNG, or ``None`` if the Arduino CSV was not
    found or has no ``timestamp_received`` column.
    """
    stage_dir = resolve_stage_dir(recording_name, _STAGE)
    if not stage_dir.exists():
        return None
    stats = _load_timing_stats(recording_name)

    try:
        csv_path = find_sensor_csv(recording_name, _STAGE, "arduino")
    except FileNotFoundError:
        log.warning("[%s/%s] arduino CSV not found — skipping clock drift plot.", recording_name, _STAGE)
        return None

    df = load_dataframe(csv_path)
    if df.empty or "timestamp_received" not in df.columns:
        log.warning("[%s/%s] no timestamp_received column — skipping clock drift.", recording_name, _STAGE)
        return None

    device_ms = df["timestamp"].astype(float).to_numpy()
    received_ms = df["timestamp_received"].astype(float).to_numpy()
    valid = np.isfinite(device_ms) & np.isfinite(received_ms)
    device_ms = device_ms[valid]
    received_ms = received_ms[valid]

    if len(device_ms) < 10:
        return None

    # Linear fit: received = a * device + b
    coeffs = np.polyfit(device_ms, received_ms, 1)
    slope, intercept = coeffs
    fit_vals = np.polyval(coeffs, device_ms)
    residuals_ms = received_ms - fit_vals

    # Drift in ppm: (slope - 1) * 1e6, but clocks measure the same real time
    # so slope ≈ 1 + drift/1e6.
    drift_ppm_fit = (slope - 1.0) * 1e6

    # Compare with stored drift estimate from session_stats
    stored_drift = stats.get("streams", {}).get("arduino", {}).get(
        "device_to_received_clock", {}
    ).get("drift_ppm", None)
    r2 = stats.get("streams", {}).get("arduino", {}).get(
        "device_to_received_clock", {}
    ).get("fit_r2", None)

    # Convert to seconds for display
    device_s = (device_ms - device_ms[0]) / 1000.0
    offset_s = (received_ms - device_ms) / 1000.0  # offset: received - device (seconds)
    offset_fit_s = (fit_vals - device_ms) / 1000.0
    xm_dev = mask_valid_plot_x(device_s)
    ds_sc = device_s[xm_dev]
    os_sc = offset_s[xm_dev]
    ds_line, ofs_line = nan_mask_invalid_plot_x(device_s, offset_fit_s)
    rs_sc = residuals_ms[xm_dev]

    fig, (ax_main, ax_res) = plt.subplots(
        2, 1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
        constrained_layout=True,
    )

    # Top: offset over device time
    ax_main.scatter(
        ds_sc[::10], os_sc[::10],
        s=2, color=_SENSOR_COLORS["arduino"], alpha=0.4, label="received − device offset",
    )
    ax_main.plot(
        ds_line, ofs_line,
        color="k", linewidth=1.2,
        label=f"linear fit  (drift = {drift_ppm_fit:+.1f} ppm)",
    )

    info_lines = [f"drift (fit) = {drift_ppm_fit:+.1f} ppm"]
    if stored_drift is not None:
        info_lines.append(f"drift (stored) = {stored_drift:+.1f} ppm")
    if r2 is not None:
        info_lines.append(f"R² = {r2:.7f}")
    ax_main.text(
        0.02, 0.97, "\n".join(info_lines),
        transform=ax_main.transAxes, fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
    )
    ax_main.set_ylabel("Clock offset [s]", fontsize=9)
    ax_main.legend(fontsize=8, loc="lower right")
    ax_main.grid(True, alpha=0.25)

    # Bottom: residuals
    ax_res.scatter(
        ds_sc[::10], rs_sc[::10],
        s=2, color=_SENSOR_COLORS["arduino"], alpha=0.5,
    )
    ax_res.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax_res.set_ylabel("Residual [ms]", fontsize=9)
    ax_res.set_xlabel("Device time [s]", fontsize=9)
    ax_res.grid(True, alpha=0.25)

    fig.suptitle(f"{recording_name} / {_STAGE} — Arduino clock drift (device vs host)", fontsize=11)
    out_path = stage_dir / "parsed_clock_drift.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{_STAGE}] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience: run all timing plots for one recording
# ---------------------------------------------------------------------------

def plot_timing_stage(recording_name: str, *, drift: bool = True) -> None:
    """Generate all timing quality plots for *recording_name*.

    Produces:
    - ``parsed_timing_intervals.png``: inter-packet interval histograms.
    - ``parsed_timing_timeline.png``: interval timeline with gap markers.
    - ``parsed_clock_drift.png``: Arduino clock drift (if ``drift=True``).
    """
    plot_timing_intervals(recording_name)
    plot_timing_timeline(recording_name)
    if drift:
        plot_clock_drift(recording_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_timing",
        description="Generate timing quality plots for one or more parsed recordings.",
    )
    parser.add_argument(
        "recording_names",
        nargs="+",
        help="One or more recording names (e.g. 2026-02-26_5).",
    )
    parser.add_argument(
        "--no-drift",
        action="store_true",
        help="Skip the Arduino clock drift plot.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for recording_name in args.recording_names:
        plot_timing_stage(recording_name, drift=not args.no_drift)


if __name__ == "__main__":
    main()
