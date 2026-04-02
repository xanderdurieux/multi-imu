"""Plot calibration segment detection results for a sensor stream."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from parser.calibration_segments import find_calibration_segments
from visualization._utils import filter_valid_plot_xy, strict_vector_norm

log = logging.getLogger(__name__)


def plot_calibration_segments_from_detection(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    static_min_s: float = 2.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 2.5,
    peak_min_count: int = 5,
    peak_max_count: int = 20,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 8.0,
    out_path: Optional[Path] = None,
) -> tuple[plt.Figure, pd.DataFrame, list]:
    """Detect calibration segments in *df* and plot them.

    Returns
    -------
    fig : matplotlib.Figure
    info_df : pd.DataFrame
        One row per detected segment with columns:
        segment_index, start_time_s, end_time_s, num_peaks, peak_times_s.
    segments : list[CalibrationSegment]
        Raw detected segments.
    """
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts_ms = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    ts_s = (ts_ms - ts_ms[0]) / 1000.0 if ts_ms.size > 0 else np.array([])

    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    acc_norm = np.full(len(df), np.nan, dtype=float)
    if acc_cols:
        acc_norm = strict_vector_norm(df, acc_cols)

    segments = find_calibration_segments(
        df,
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
    )

    # Build info DataFrame.
    rows_info: list[dict] = []
    for i, seg in enumerate(segments):
        s_time = float(ts_s[seg.start_idx]) if seg.start_idx < len(ts_s) else 0.0
        e_time = float(ts_s[min(seg.end_idx, len(ts_s) - 1)])
        peak_times = [float(ts_s[p]) for p in seg.peak_indices if p < len(ts_s)]
        rows_info.append({
            "segment_index": i,
            "start_time_s": s_time,
            "end_time_s": e_time,
            "num_peaks": len(seg.peak_indices),
            "peak_times_s": peak_times,
        })
    info_df = pd.DataFrame(rows_info)

    # Plot.
    fig, ax = plt.subplots(figsize=(14, 4))

    if ts_s.size > 0:
        x_plot, y_plot = filter_valid_plot_xy(ts_s, acc_norm)
        ax.plot(x_plot, y_plot, lw=0.7, color="#555", label="|acc|")
        ax.axhline(y=9.81, color="blue", lw=0.5, ls="--", alpha=0.5, label="g")

    colors = plt.cm.tab10.colors
    for i, seg in enumerate(segments):
        color = colors[i % len(colors)]
        s_time = float(ts_s[seg.start_idx]) if seg.start_idx < len(ts_s) else 0.0
        e_time = float(ts_s[min(seg.end_idx, len(ts_s) - 1)])
        ax.axvspan(s_time, e_time, alpha=0.15, color=color, label=f"cal {i}")
        for p in seg.peak_indices:
            if p < len(ts_s):
                ax.axvline(ts_s[p], color=color, lw=0.5, alpha=0.6)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|acc| (m/s²)")
    ax.set_title(f"Calibration segments detected: {len(segments)}")
    if segments:
        ax.legend(loc="upper right", fontsize=7, ncol=min(4, len(segments) + 2))
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        log.debug("Saved calibration segments plot → %s", out_path)
        plt.close(fig)

    return fig, info_df, segments
