"""Feature timeline plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.thesis_style import THESIS_COLORS, apply_matplotlib_thesis_style
from visualization._utils import nan_mask_invalid_plot_x


def _apply_plot_style() -> None:
    apply_matplotlib_thesis_style()
    mpl.rcParams.update(
        {
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 1.0,
            "legend.frameon": False,
        }
    )


def plot_features_timeline(features_dir: Path, df: pd.DataFrame) -> None:
    """Top 4 discriminative features (by std/mean ratio) as timeline heatmap-style."""
    _apply_plot_style()
    plots_dir = features_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Numeric columns only, exclude metadata
    meta = {
        "section",
        "recording_id",
        "section_id",
        "window_start_s",
        "window_end_s",
        "window_center_s",
        "scenario_label",
        "sync_method",
        "orientation_method",
        "calibration_quality",
        "label_source",
    }
    feat_cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        return

    # Discriminative = high std/mean ratio (avoid div by zero)
    ratios = []
    for c in feat_cols:
        vals = df[c].dropna()
        if len(vals) < 2:
            ratios.append(0.0)
        else:
            mean_abs = np.abs(vals.mean())
            ratios.append(vals.std() / max(mean_abs, 1e-9))
    top_idx = np.argsort(ratios)[::-1][:4]
    top_cols = [feat_cols[i] for i in top_idx]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    t = df["window_center_s"].to_numpy()

    for ax, col in zip(axes, top_cols):
        v = df[col].to_numpy(dtype=float)
        v_valid = v[np.isfinite(v)]
        tp, vp = nan_mask_invalid_plot_x(t, v)
        ax.fill_between(tp, vp, alpha=0.5)
        ax.plot(tp, vp, color=THESIS_COLORS[0], linewidth=1)
        ax.set_ylabel(col)
        if len(v_valid) > 0:
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            margin = max(0.1 * (np.nanstd(v) or 1e-9), 1e-9)
            ax.set_ylim(vmin - margin, vmax + margin)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Top 4 discriminative features over time")
    fig.savefig(plots_dir / "features_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
