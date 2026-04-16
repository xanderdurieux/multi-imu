"""Plot calibration segment detection results for a sensor stream."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from parser.calibration_segments import (
    CalibrationSegment,
    cal_segment_kwargs_for_sensor,
    prepare_calibration_detection_arrays,
)
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)

def plot_calibration_segments_from_detection(
    df: pd.DataFrame,
    segments: Sequence[CalibrationSegment],
    out_path: Path,
    *,
    sensor: str,
) -> None:
    """Draw *segments* on *df* and save to *out_path*.

    Two stacked panels share the time axis: **|acc|** on top, **smoothed |acc−g|**
    (same signal as :func:`parser.calibration_segments.find_calibration_segments`)
    below. Segment shading and peak markers are drawn on both.

    Detection and tabular summaries are the caller's responsibility
    (:func:`find_calibration_segments`, :func:`describe_calibration_segments`).
    """
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    ts_s = (ts_ms - ts_ms[0]) / 1000.0 if ts_ms.size > 0 else np.array([])

    seg_list = list(segments)
    cal_kw = cal_segment_kwargs_for_sensor(sensor)
    prep = prepare_calibration_detection_arrays(df, sensor=sensor)
    if prep is None:
        acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
        acc_norm = strict_vector_norm(df, acc_cols) if acc_cols else np.full(len(df), np.nan, dtype=float)
        smooth = np.full(len(df), np.nan, dtype=float)
    else:
        acc_norm = prep.norm
        smooth = prep.dynamic_smooth

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True)

    def _draw_segment_overlays(ax: Axes, *, legend_labels: bool) -> None:
        colors = plt.cm.tab10.colors
        for i, seg in enumerate(seg_list):
            color = colors[i % len(colors)]
            s_time = float(ts_s[seg.start_idx]) if seg.start_idx < len(ts_s) else 0.0
            e_time = float(ts_s[min(seg.end_idx, len(ts_s) - 1)])
            label = f"Cal {i + 1}" if legend_labels else None
            ax.axvspan(s_time, e_time, alpha=0.15, color=color, zorder=0, label=label)
            for p in seg.peak_indices:
                if p < len(ts_s):
                    ax.axvline(ts_s[p], color=color, lw=0.5, alpha=0.6, zorder=3)

    x_plot, y_plot = filter_valid_plot_xy(ts_s, acc_norm)
    ax_top.plot(x_plot, y_plot, lw=0.7, zorder=2, label="|acc|")
    ax_top.axhline(y=9.81, color="blue", lw=0.5, ls="--", alpha=0.5, zorder=1, label="g")

    ax_top.set_ylabel("|acc| (m/s²)")
    ax_top.set_title(f"Calibration segments detected: {len(seg_list)} ({sensor})")
    ax_top.grid(True, alpha=0.25)

    x_s, y_s = filter_valid_plot_xy(ts_s, smooth)
    ax_bot.plot(
        x_s,
        y_s,
        lw=0.7,
        zorder=2,
        label="Smoothed |acc−g|",
    )
    pmh = float(cal_kw["peak_min_height"])
    ax_bot.axhline(
        pmh,
        color="blue",
        lw=0.5,
        ls="--",
        alpha=0.5,
        zorder=1,
        label=f"Peak min height ({pmh:g})",
    )

    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("Smoothed |acc−g| (m/s²)")
    ax_bot.grid(True, alpha=0.25)

    _draw_segment_overlays(ax_top, legend_labels=True)
    _draw_segment_overlays(ax_bot, legend_labels=False)

    def _dedupe_legend(ax: Axes, **kw: object) -> None:
        handles, labels = ax.get_legend_handles_labels()
        seen: dict[str, object] = {}
        for h, lab in zip(handles, labels):
            if not lab:
                continue
            seen[lab] = h
        if seen:
            ax.legend(list(seen.values()), list(seen.keys()), **kw)

    _dedupe_legend(
        ax_top,
        loc="upper right",
        fontsize=8,
        framealpha=0.92,
        ncol=min(4, max(1, len(seg_list) + 2)),
    )
    _dedupe_legend(
        ax_bot,
        loc="upper right",
        fontsize=8,
        framealpha=0.92,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    log.debug("Saved calibration segments plot → %s", out_path)
