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
    """Plot calibration segments from detection."""
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    ts_s = (ts_ms - ts_ms[0]) / 1000.0 if ts_ms.size > 0 else np.array([])

    seg_list = list(segments)
    acc_norm, dynamic_smooth = _compute_plot_signals(df, ts_ms)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(16, 6.5), sharex=True)

    def _draw_segment_overlays(ax: Axes, *, legend_labels: bool) -> None:
        """Draw segment overlays."""
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
        """Return dedupe legend."""
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


def _dedupe_legend(ax: Axes, **kw: object) -> None:
    """Draw a legend with duplicate labels removed."""
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if not label or label == "_":
            continue
        seen.setdefault(label, handle)
    if seen:
        ax.legend(list(seen.values()), list(seen.keys()), **kw)


def _selected_segment(
    segments: Sequence[CalibrationSegment],
    segment_index: int | None,
) -> tuple[int, CalibrationSegment] | None:
    """Return the requested segment, or the strongest segment by default."""
    seg_list = list(segments)
    if not seg_list:
        return None
    if segment_index is not None:
        if 0 <= segment_index < len(seg_list):
            return segment_index, seg_list[segment_index]
        log.warning("Calibration segment index %d out of range (%d segment(s))", segment_index, len(seg_list))
    return max(
        enumerate(seg_list),
        key=lambda item: (
            item[1].peak_strength,
            len(item[1].peak_ms or []),
            item[1].end_ms - item[1].start_ms,
        ),
    )


def plot_zoomed_calibration_sequence_from_detection(
    df: pd.DataFrame,
    segments: Sequence[CalibrationSegment],
    out_path: Path,
    *,
    sensor: str,
    segment_index: int | None = None,
    flank_s: float = 5.0,
) -> Path | None:
    """Plot a presentation-friendly zoom around one detected calibration sequence."""
    selected = _selected_segment(segments, segment_index)
    if selected is None:
        log.info("No calibration segments for %s — skipping zoom plot", sensor)
        return None

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return None

    ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    finite_ts = ts_ms[np.isfinite(ts_ms)]
    if finite_ts.size == 0:
        return None

    selected_index, seg = selected
    peaks = [float(p) for p in (seg.peak_ms or []) if np.isfinite(p)]
    if peaks:
        burst_start_ms = min(peaks)
        burst_end_ms = max(peaks)
    else:
        burst_start_ms = seg.start_ms + 0.5 * (seg.end_ms - seg.start_ms)
        burst_end_ms = burst_start_ms

    flank_ms = max(0.0, flank_s) * 1000.0
    zoom_start_ms = max(float(finite_ts[0]), burst_start_ms - flank_ms)
    zoom_end_ms = min(float(finite_ts[-1]), burst_end_ms + flank_ms)
    if zoom_end_ms <= zoom_start_ms:
        zoom_start_ms = max(float(finite_ts[0]), seg.start_ms)
        zoom_end_ms = min(float(finite_ts[-1]), seg.end_ms)
    if zoom_end_ms <= zoom_start_ms:
        return None

    mask = np.isfinite(ts_ms) & (ts_ms >= zoom_start_ms) & (ts_ms <= zoom_end_ms)
    if not mask.any():
        return None

    ts_s = (ts_ms[mask] - zoom_start_ms) / 1000.0
    acc_norm, dynamic_smooth = _compute_plot_signals(df, ts_ms)
    acc_zoom = acc_norm[mask]
    dyn_zoom = dynamic_smooth[mask]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    def _span(ax: Axes, start_ms: float, end_ms: float, color: str, label: str, alpha: float) -> None:
        start_ms = max(start_ms, zoom_start_ms)
        end_ms = min(end_ms, zoom_end_ms)
        if end_ms <= start_ms:
            return
        ax.axvspan(
            (start_ms - zoom_start_ms) / 1000.0,
            (end_ms - zoom_start_ms) / 1000.0,
            color=color,
            alpha=alpha,
            lw=0,
            zorder=0,
            label=label,
        )

    pre_end_ms = seg.start_ms + max(0.0, seg.static_pre_ms)
    post_start_ms = seg.end_ms - max(0.0, seg.static_post_ms)
    for ax in (ax_top, ax_bot):
        _span(ax, seg.start_ms, pre_end_ms, "green", "pre-static", 0.16)
        _span(ax, burst_start_ms, burst_end_ms, "#d62728", "tap sequence", 0.14)
        _span(ax, post_start_ms, seg.end_ms, "lime", "post-static", 0.16)
        for idx, peak_ms in enumerate(peaks):
            if zoom_start_ms <= peak_ms <= zoom_end_ms:
                ax.axvline(
                    (peak_ms - zoom_start_ms) / 1000.0,
                    color="#d62728",
                    lw=0.8,
                    alpha=0.75,
                    zorder=3,
                    label="tap peak" if idx == 0 else "_",
                )

    x_plot, y_plot = filter_valid_plot_xy(ts_s, acc_zoom)
    ax_top.plot(x_plot, y_plot, lw=1.1, color="#1f77b4", zorder=2, label="|acc|")
    ax_top.axhline(
        y=_NOMINAL_GRAVITY_MS2,
        color="gray",
        lw=0.8,
        ls="--",
        alpha=0.7,
        zorder=1,
        label="g",
    )
    ax_top.set_ylabel("|acc| (m/s²)")
    ax_top.set_title(f"{sensor} calibration sequence {selected_index + 1}")
    ax_top.grid(True, alpha=0.25)

    x_s, y_s = filter_valid_plot_xy(ts_s, dyn_zoom)
    ax_bot.plot(x_s, y_s, lw=1.1, color="#4c78a8", zorder=2, label="Smoothed |acc-g|")
    ax_bot.set_xlabel("Time in zoom window (s)")
    ax_bot.set_ylabel("Smoothed |acc-g| (m/s²)")
    ax_bot.grid(True, alpha=0.25)
    ax_bot.set_xlim(0.0, (zoom_end_ms - zoom_start_ms) / 1000.0)

    _dedupe_legend(ax_top, loc="upper right", fontsize=8, framealpha=0.92, ncol=3)
    _dedupe_legend(ax_bot, loc="upper right", fontsize=8, framealpha=0.92)

    fig.tight_layout()
    return save_figure(fig, out_path)
