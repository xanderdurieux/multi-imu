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

from common.signals import smooth_moving_average
from parser.calibration_segments import CalibrationSegment
from visualization._utils import filter_valid_plot_xy, save_figure

log = logging.getLogger(__name__)

_NOMINAL_GRAVITY_MS2 = 9.81


def _compute_plot_signals(
    df: pd.DataFrame,
    ts_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (acc_norm, dynamic_smooth) arrays aligned to *ts_ms*."""
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if "acc_norm" in df.columns:
        acc_norm = df["acc_norm"].to_numpy(dtype=float)
    elif acc_cols:
        acc_norm = np.sqrt(
            np.nansum(
                np.column_stack([df[c].to_numpy(dtype=float) ** 2 for c in acc_cols]),
                axis=1,
            )
        )
    else:
        acc_norm = np.full(len(df), np.nan, dtype=float)

    g_med = float(np.nanmedian(acc_norm)) if np.any(np.isfinite(acc_norm)) else _NOMINAL_GRAVITY_MS2
    g = g_med if 8.0 <= g_med <= 12.5 else _NOMINAL_GRAVITY_MS2

    dts = np.diff(ts_ms)
    dts = dts[np.isfinite(dts) & (dts > 0)]
    median_dt = float(np.median(dts)) if dts.size > 0 else 10.0
    smooth_win = max(3, int(round(100.0 / max(median_dt, 1e-6))))
    dynamic_smooth = np.abs(smooth_moving_average(acc_norm, smooth_win) - g)

    return acc_norm, dynamic_smooth


def plot_calibration_segments_from_detection(
    df: pd.DataFrame,
    segments: Sequence[CalibrationSegment],
    out_path: Path,
    *,
    sensor: str,
) -> None:
    """Draw *segments* on *df* and save to *out_path*.

    Two stacked panels share the time axis: **|acc|** on top, **smoothed |acc−g|**
    below. Segment shading and peak markers are drawn on both.
    """
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    ts_s = (ts_ms - ts_ms[0]) / 1000.0 if ts_ms.size > 0 else np.array([])

    seg_list = list(segments)
    acc_norm, dynamic_smooth = _compute_plot_signals(df, ts_ms)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True)

    def _draw_segment_overlays(ax: Axes, *, legend_labels: bool) -> None:
        colors = plt.cm.tab10.colors
        t0_ms = ts_ms[0] if ts_ms.size > 0 else 0.0
        for i, seg in enumerate(seg_list):
            color = colors[i % len(colors)]
            s_time = (seg.start_ms - t0_ms) / 1000.0
            e_time = (seg.end_ms - t0_ms) / 1000.0
            label = f"Cal {i + 1}" if legend_labels else None
            ax.axvspan(s_time, e_time, alpha=0.15, color=color, zorder=0, label=label)
            for p_ms in (seg.peak_ms or []):
                ax.axvline((p_ms - t0_ms) / 1000.0, color=color, lw=0.5, alpha=0.6, zorder=3)

    x_plot, y_plot = filter_valid_plot_xy(ts_s, acc_norm)
    ax_top.plot(x_plot, y_plot, lw=0.7, zorder=2, label="|acc|")
    ax_top.axhline(y=_NOMINAL_GRAVITY_MS2, color="blue", lw=0.5, ls="--", alpha=0.5, zorder=1, label="g")
    ax_top.set_ylabel("|acc| (m/s²)")
    ax_top.set_title(f"Calibration segments detected: {len(seg_list)} ({sensor})")
    ax_top.grid(True, alpha=0.25)

    x_s, y_s = filter_valid_plot_xy(ts_s, dynamic_smooth)
    ax_bot.plot(x_s, y_s, lw=0.7, zorder=2, label="Smoothed |acc−g|")
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
    _dedupe_legend(ax_bot, loc="upper right", fontsize=8, framealpha=0.92)

    save_figure(fig, out_path)
