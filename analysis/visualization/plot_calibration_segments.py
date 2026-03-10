from __future__ import annotations

"""Diagnostic plots for calibration-segment detection.

These helpers visualise:

- The dynamic acceleration magnitude ``|acc_norm - g|`` used for detection.
- Detected calibration segments as coloured bands.
- Detected peak locations within each segment.
"""

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calibration.segments import (
    CalibrationSegment,
    _acc_norm,
    _smooth,
    describe_calibration_segments,
    find_calibration_segments,
)


def _time_axis(df: pd.DataFrame, sample_rate_hz: float) -> np.ndarray:
    """Return time in seconds starting at 0."""
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    if "timestamp" in df.columns:
        ts = df["timestamp"].astype(float).to_numpy()
        t0 = ts[0]
        return (ts - t0) / 1000.0
    return np.arange(n, dtype=float) / float(sample_rate_hz)


def plot_calibration_segments(
    df: pd.DataFrame,
    segments: Iterable[CalibrationSegment],
    *,
    sample_rate_hz: float = 100.0,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Tuple[plt.Figure, pd.DataFrame, Path | None]:
    """Plot dynamic acceleration and overlay calibration segments + peaks.

    Parameters
    ----------
    df:
        IMU DataFrame containing at least ``ax, ay, az`` (and optionally
        ``timestamp``).
    segments:
        Iterable of :class:`CalibrationSegment` produced by
        :func:`calibration.segments.find_calibration_segments`.
    sample_rate_hz:
        Sampling rate used for peak detection, only used when no timestamp
        column is available.
    title:
        Optional figure title.  When omitted, a default is used.
    out_path:
        Optional path where the figure will be saved as a PNG.  When ``None``,
        the figure is not written to disk.

    Returns
    -------
    (fig, info_df, out_path)
        *fig* is the created Matplotlib figure,
        *info_df* is the segment summary from
        :func:`calibration.segments.describe_calibration_segments`,
        *out_path* is the resolved Path where the figure was saved (or
        ``None`` if *out_path* was not given).
    """
    seg_list = list(segments)

    # Build timing summary (used both in legend/text and for downstream checks).
    info_df = describe_calibration_segments(df, seg_list, sample_rate_hz=sample_rate_hz)

    time_s = _time_axis(df, sample_rate_hz=sample_rate_hz)
    if time_s.size == 0:
        raise ValueError("Input DataFrame is empty; cannot plot calibration segments.")

    # Recompute the detector's dynamic signal for visualisation.
    norm = _acc_norm(df)
    g = float(np.nanmedian(norm)) if norm.size else 0.0
    is_dropout = norm < 0.1 * g
    # Mask dropout positions as NaN in the raw trace so the ~g spikes from
    # near-zero dropout packets don't dominate the display.
    dynamic_raw = np.abs(norm - g)
    dynamic_raw_display = dynamic_raw.copy().astype(float)
    dynamic_raw_display[is_dropout] = np.nan
    norm_for_smooth = norm.copy()
    norm_for_smooth[is_dropout] = g
    smooth_win = max(3, int(sample_rate_hz * 0.1))
    dynamic_smooth = np.abs(_smooth(norm_for_smooth, smooth_win) - g)

    fig, (ax_signal, ax_seg) = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        constrained_layout=True,
    )

    # Top panel: raw and smoothed dynamic acceleration magnitude.
    ax_signal.plot(time_s, dynamic_raw_display, color="#cccccc", linewidth=0.6, label="|acc_norm - g| (raw)")
    ax_signal.plot(
        time_s,
        dynamic_smooth,
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.9,
        label="|acc_norm - g| (smoothed)",
    )
    ax_signal.set_ylabel("m/s²")
    ax_signal.grid(True, alpha=0.25)
    ax_signal.legend(fontsize=8, loc="upper right")

    # Bottom panel: segment bands + peak markers in a single "track".
    ax_seg.set_ylim(0.0, 1.0)
    ax_seg.set_yticks([])
    ax_seg.set_xlabel("Time [s]")
    ax_seg.set_ylabel("Calibration segments")
    ax_seg.grid(True, axis="x", alpha=0.2)

    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for _, row in info_df.iterrows():
        color = colors[int(row["segment_index"]) % len(colors)]
        start_t = float(row["start_time_s"])
        end_t = float(row["end_time_s"])
        peaks_t = row["peak_times_s"] or []

        # Draw a horizontal band for the full calibration segment.
        ax_seg.axvspan(
            start_t,
            end_t,
            ymin=0.25,
            ymax=0.75,
            facecolor=color,
            edgecolor="none",
            alpha=0.25,
        )

        # Mark the peak times as vertical lines across the band.
        if peaks_t:
            ax_seg.vlines(
                peaks_t,
                ymin=0.25,
                ymax=0.75,
                colors=color,
                linewidth=1.2,
            )

        # Label segment index above the band centre.
        centre_t = 0.5 * (start_t + end_t)
        ax_seg.text(
            centre_t,
            0.8,
            f"seg {int(row['segment_index'])}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=color,
        )

    if title is None:
        title = "Calibration segment detection diagnostics"
    fig.suptitle(title, fontsize=11)

    saved_path: Path | None = None
    if out_path is not None:
        saved_path = Path(out_path).with_suffix(".png")
        fig.savefig(saved_path, dpi=120, bbox_inches="tight")

    return fig, info_df, saved_path


def plot_calibration_segments_from_detection(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_count: int | None = None,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    title: str | None = None,
    out_path: str | Path | None = None,
) -> Tuple[list[CalibrationSegment], pd.DataFrame, Path | None]:
    """Detect calibration segments and immediately plot them.

    This is a convenience wrapper that runs
    :func:`calibration.segments.find_calibration_segments`, builds the segment
    summary via :func:`describe_calibration_segments`, and generates the
    diagnostic figure with :func:`plot_calibration_segments`.
    """
    segments = find_calibration_segments(
        df,
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_count=peak_max_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
    )

    fig, info_df, saved_path = plot_calibration_segments(
        df,
        segments,
        sample_rate_hz=sample_rate_hz,
        title=title,
        out_path=out_path,
    )

    # Explicitly close the figure when saving to disk to avoid accumulating GUI
    # windows in interactive environments; the caller still receives the
    # Figure object for further customisation if desired.
    if saved_path is not None:
        plt.close(fig)

    return segments, info_df, saved_path

