"""Session-level parsed-stage integrity plots for thesis reporting."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._utils import SENSOR_COLORS, save_figure


_DPI = 200


def _quality_badge(category: str) -> str:
    return {
        "good": "G",
        "usable": "U",
        "limited": "L",
    }.get(str(category).strip().lower(), "?")


def plot_parsed_session_bar_chart(summary_df: pd.DataFrame, output_path: Path) -> Path:
    """Grouped bar chart of recording durations and sample counts for one session."""
    if summary_df.empty:
        raise ValueError("summary_df must not be empty")

    df = summary_df.copy().reset_index(drop=True)
    labels = df["recording_name"].tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    duration_ax = axes[0]
    duration_sporsa = pd.to_numeric(df["sporsa_duration_s"], errors="coerce").to_numpy(dtype=float)
    duration_arduino = pd.to_numeric(df["arduino_duration_s"], errors="coerce").to_numpy(dtype=float)
    duration_ax.bar(
        x - width / 2,
        np.nan_to_num(duration_sporsa, nan=0.0),
        width=width,
        color=SENSOR_COLORS["sporsa"],
        alpha=0.9,
        label="SPORSA",
    )
    duration_ax.bar(
        x + width / 2,
        np.nan_to_num(duration_arduino, nan=0.0),
        width=width,
        color=SENSOR_COLORS["arduino"],
        alpha=0.9,
        label="Arduino",
    )
    duration_ax.set_ylabel("Duration (s)")
    duration_ax.set_title("Parsed session integrity — recording durations")
    duration_ax.grid(axis="y", alpha=0.25, lw=0.5)
    duration_ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    duration_ax.spines["top"].set_visible(False)
    duration_ax.spines["right"].set_visible(False)

    count_ax = axes[1]
    sporsa_samples = pd.to_numeric(df["sporsa_num_samples"], errors="coerce").to_numpy(dtype=float)
    arduino_samples = pd.to_numeric(df["arduino_num_samples"], errors="coerce").to_numpy(dtype=float)
    count_ax.bar(
        x - width / 2,
        np.nan_to_num(sporsa_samples, nan=0.0),
        width=width,
        color=SENSOR_COLORS["sporsa"],
        alpha=0.9,
        label="SPORSA",
    )
    count_ax.bar(
        x + width / 2,
        np.nan_to_num(arduino_samples, nan=0.0),
        width=width,
        color=SENSOR_COLORS["arduino"],
        alpha=0.9,
        label="Arduino",
    )
    count_ax.set_ylabel("Samples")
    count_ax.set_title("Parsed session integrity — sample counts")
    count_ax.grid(axis="y", alpha=0.25, lw=0.5)
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)

    for idx, category in enumerate(df["quality_category"].tolist()):
        badge = _quality_badge(str(category))
        count_ax.text(
            x[idx],
            0.98,
            badge,
            transform=count_ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.9},
        )

    count_ax.set_xticks(x)
    count_ax.set_xticklabels(labels, rotation=20, ha="right")
    count_ax.set_xlabel("Recording")

    fig.tight_layout()
    saved = save_figure(fig, output_path, dpi=_DPI)
    return saved
