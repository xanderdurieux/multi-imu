"""Feature timeline plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#d4d4d4",
            "axes.labelcolor": "#262626",
            "axes.titlecolor": "#171717",
            "text.color": "#171717",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
            "grid.color": "#e5e5e5",
            "axes.grid": True,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )


def plot_features_timeline(features_dir: Path, df: pd.DataFrame) -> None:
    """Top 4 discriminative features (by std/mean ratio) as timeline heatmap-style."""
    _apply_plot_style()
    plots_dir = features_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Numeric columns only, exclude metadata
    meta = {"section", "window_start_s", "window_end_s", "window_center_s", "scenario_label"}
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
        ax.fill_between(t, v, alpha=0.5)
        ax.plot(t, v, color="#2563eb", linewidth=1)
        ax.set_ylabel(col)
        if len(v_valid) > 0:
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            margin = max(0.1 * (np.nanstd(v) or 1e-9), 1e-9)
            ax.set_ylim(vmin - margin, vmax + margin)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Top 4 discriminative features over time")
    fig.savefig(plots_dir / "features_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
