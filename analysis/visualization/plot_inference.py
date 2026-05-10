"""Inference stage visualizations: timeline + confidence strip + class distribution."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from common.paths import read_csv
from labels.parser import LabelRow, load_labels
from visualization._utils import (
    QUALITATIVE_PALETTE,
    UNKNOWN_LABEL_COLOR,
    save_figure,
)

log = logging.getLogger(__name__)

_LABEL_ALPHA = 0.22
_AMBIGUITY_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_colors(labels: list[LabelRow]) -> dict[str, str]:
    unique = sorted({lr.label for lr in labels if lr.label})
    return {name: QUALITATIVE_PALETTE[i % len(QUALITATIVE_PALETTE)] for i, name in enumerate(unique)}


def _draw_spans(axes: list[plt.Axes], labels: list[LabelRow], t0_ms: float, colors: dict[str, str]) -> None:
    for lr in labels:
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, UNKNOWN_LABEL_COLOR)
        alpha = _LABEL_ALPHA * max(0.3, lr.confidence)
        for ax in axes:
            ax.axvspan(x0, x1, color=color, alpha=alpha, lw=0, zorder=0)


def _step_xy(df: plt.Axes, col: str, t0_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_seconds, values) for a window-level feature column as a step series."""
    t = ((df["window_start_ms"] + df["window_end_ms"]) / 2.0 - t0_ms) / 1000.0
    v = df[col].to_numpy(dtype=float)
    return t.to_numpy(dtype=float), v


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_inference_section(section_dir: Path, label_col: str) -> Path | None:
    """Generate a 3-panel timeline + confidence plot for one section and label scheme."""
    labels_path = section_dir / "labels" / f"labels_inferred_{label_col}.csv"
    features_path = section_dir / "features" / "features.csv"

    if not labels_path.exists():
        log.debug("No inferred labels for %s (%s) — skipping plot", section_dir.name, label_col)
        return None

    labels = load_labels(labels_path)
    if not labels:
        return None

    if not features_path.exists():
        log.debug("No features.csv for %s — skipping inference plot", section_dir.name)
        return None

    df = read_csv(features_path)
    if "window_type" in df.columns:
        df = df[df["window_type"] == "sliding"].copy()
    df = df.sort_values("window_start_ms").reset_index(drop=True)
    if df.empty:
        return None

    t0_ms = float(df["window_start_ms"].min())
    colors = _label_colors(labels)

    # Pick signal columns: prefer bike, fall back to rider
    acc_col = next((c for c in ("bike_acc_norm_mean", "rider_acc_norm_mean") if c in df.columns), None)
    gyro_col = next((c for c in ("bike_gyro_norm_mean", "rider_gyro_norm_mean") if c in df.columns), None)

    n_signal_panels = sum(c is not None for c in (acc_col, gyro_col))
    n_panels = n_signal_panels + 1  # +1 for confidence strip
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 2.2 * n_panels),
        sharex=True,
        gridspec_kw={"height_ratios": [2] * n_signal_panels + [1]},
    )
    if n_panels == 1:
        axes = [axes]

    signal_axes = axes[:n_signal_panels]
    conf_ax = axes[-1]

    # --- Signal panels ---
    panel_idx = 0
    for col, ylabel in ((acc_col, "acc (m/s²)"), (gyro_col, "gyro (rad/s)")):
        if col is None:
            continue
        ax = signal_axes[panel_idx]
        t, v = _step_xy(df, col, t0_ms)
        mask = np.isfinite(t) & np.isfinite(v)
        ax.plot(t[mask], v[mask], lw=0.8, color="#1f77b4", alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(alpha=0.18, lw=0.4)
        panel_idx += 1

    _draw_spans(signal_axes, labels, t0_ms, colors)

    # --- Confidence strip ---
    conf_ax.set_ylabel("confidence", fontsize=8)
    conf_ax.set_ylim(0, 1.05)
    conf_ax.axhline(_AMBIGUITY_THRESHOLD, color="red", lw=0.6, ls="--", alpha=0.5, zorder=2)
    conf_ax.grid(alpha=0.18, lw=0.4)

    for lr in labels:
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, UNKNOWN_LABEL_COLOR)
        conf_ax.fill_between([x0, x1], 0, lr.confidence, color=color, alpha=0.55, lw=0, step="pre")
        if lr.ambiguous:
            conf_ax.axvspan(x0, x1, color="red", alpha=0.08, lw=0, zorder=1)

    axes[-1].set_xlabel("time (s)", fontsize=8)

    # --- Legend ---
    patches = [mpatches.Patch(color=c, alpha=0.7, label=lbl) for lbl, c in colors.items()]
    if patches:
        axes[0].legend(
            handles=patches,
            loc="upper right",
            fontsize=7,
            framealpha=0.85,
            ncol=min(len(patches), 4),
        )

    section_id = section_dir.name
    fig.suptitle(f"{section_id}  ·  {label_col}", fontsize=9, y=1.01)
    fig.tight_layout()

    out = section_dir / "labels" / f"labels_inferred_{label_col}_timeline.png"
    return save_figure(fig, out)


def plot_inference_class_distribution(section_dir: Path, label_col: str) -> Path | None:
    """Bar chart: total predicted duration and mean confidence per class."""
    labels_path = section_dir / "labels" / f"labels_inferred_{label_col}.csv"
    if not labels_path.exists():
        return None

    labels = load_labels(labels_path)
    if not labels:
        return None

    colors = _label_colors(labels)

    from collections import defaultdict
    durations: dict[str, float] = defaultdict(float)
    conf_sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for lr in labels:
        dur = (lr.end_ms - lr.start_ms) / 1000.0
        durations[lr.label] += dur
        conf_sums[lr.label] += lr.confidence
        counts[lr.label] += 1

    classes = sorted(durations)
    dur_vals = [durations[c] for c in classes]
    conf_vals = [conf_sums[c] / counts[c] for c in classes]
    bar_colors = [colors.get(c, UNKNOWN_LABEL_COLOR) for c in classes]

    fig, (ax_dur, ax_conf) = plt.subplots(1, 2, figsize=(max(6, len(classes) * 1.2), 4))

    x = np.arange(len(classes))
    ax_dur.bar(x, dur_vals, color=bar_colors, alpha=0.75, edgecolor="white", lw=0.5)
    ax_dur.set_xticks(x)
    ax_dur.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax_dur.set_ylabel("total duration (s)", fontsize=8)
    ax_dur.set_title("Predicted duration per class", fontsize=9)
    ax_dur.grid(axis="y", alpha=0.18, lw=0.4)

    ax_conf.bar(x, conf_vals, color=bar_colors, alpha=0.75, edgecolor="white", lw=0.5)
    ax_conf.axhline(_AMBIGUITY_THRESHOLD, color="red", lw=0.7, ls="--", alpha=0.6)
    ax_conf.set_xticks(x)
    ax_conf.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax_conf.set_ylabel("mean confidence", fontsize=8)
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_title("Mean confidence per class", fontsize=9)
    ax_conf.grid(axis="y", alpha=0.18, lw=0.4)

    section_id = section_dir.name
    fig.suptitle(f"{section_id}  ·  {label_col}", fontsize=9)
    fig.tight_layout()

    out = section_dir / "labels" / f"labels_inferred_{label_col}_distribution.png"
    return save_figure(fig, out)
