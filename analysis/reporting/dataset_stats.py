"""Dataset overview plots and summary tables for the thesis reporting stage."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from common.paths import project_relative_path

log = logging.getLogger(__name__)

_DPI = 200

# Class family groupings for colour coding
_CLASS_FAMILIES: dict[str, str] = {
    "riding": "#2196F3",
    "riding_standing": "#1565C0",
    "sprint_standing": "#0D47A1",
    "accelerating": "#4CAF50",
    "braking": "#F44336",
    "hard_braking": "#B71C1C",
    "cornering": "#FF9800",
    "stationary": "#9E9E9E",
    "stops": "#757575",
    "grounded": "#607D8B",
    "fall": "#E91E63",
    "wheelie": "#9C27B0",
    "head_movement": "#00BCD4",
    "shoulder_check": "#009688",
    "helmet_move": "#26A69A",
    "calibration_sequence": "#795548",
    "forest": "#8BC34A",
}

_QUALITY_COLORS: dict[str, str] = {
    "poor": "#ef5350",
    "marginal": "#FFA726",
    "good": "#66BB6A",
}


def _class_color(label: str) -> str:
    """Return class color."""
    return _CLASS_FAMILIES.get(label, "#90A4AE")


# ---------------------------------------------------------------------------
# Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> Path:
    """Horizontal bar chart showing window count per scenario class, sorted by count."""
    if "scenario_label" not in df.columns:
        log.warning("No 'scenario_label' column; skipping class distribution plot")
        return output_path

    counts = df["scenario_label"].value_counts().sort_values()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colors = [_class_color(lbl) for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(labels) + 1.5)))

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.4)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Number of windows")
    ax.set_title("Dataset: window count per scenario class")
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote class distribution → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Quality breakdown per recording
# ---------------------------------------------------------------------------

def plot_quality_breakdown(df: pd.DataFrame, output_path: Path) -> Path:
    """Stacked horizontal bar chart: quality tiers per recording."""
    required = {"section_id", "overall_quality_label"}
    if not required.issubset(df.columns):
        log.warning("Missing columns for quality breakdown; skipping")
        return output_path

    # Derive recording name from section_id (strip section suffix 's<N>')
    df = df.copy()
    df["recording"] = df["section_id"].str.replace(r"s\d+$", "", regex=True)

    pivot = (
        df.groupby(["recording", "overall_quality_label"])
        .size()
        .unstack(fill_value=0)
    )
    quality_order = [q for q in ("poor", "marginal", "good") if q in pivot.columns]
    pivot = pivot[quality_order]

    recordings = pivot.index.tolist()
    n = len(recordings)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * n + 1.5)))

    left = np.zeros(n)
    for q in quality_order:
        vals = pivot[q].values
        color = _QUALITY_COLORS.get(q, "#90A4AE")
        ax.barh(recordings, vals, left=left, color=color, label=q.capitalize(),
                edgecolor="white", linewidth=0.4)
        # Label non-zero bars
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0:
                ax.text(l + v / 2, i, str(v), ha="center", va="center",
                        fontsize=7, color="white" if v > 5 else "black")
        left += vals

    ax.set_xlabel("Number of windows")
    ax.set_title("Window quality tier per recording")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote quality breakdown → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Session timeline
# ---------------------------------------------------------------------------

def plot_session_timeline(df: pd.DataFrame, output_path: Path) -> Path:
    """Horizontal timeline: each window as a thin rectangle coloured by scenario_label."""
    required = {"section_id", "window_start_ms", "window_end_ms", "scenario_label"}
    if not required.issubset(df.columns):
        log.warning("Missing columns for session timeline; skipping")
        return output_path

    df = df.copy()
    df["recording"] = df["section_id"].str.replace(r"s\d+$", "", regex=True)

    recordings = sorted(df["recording"].unique())
    n_rec = len(recordings)
    rec_index = {r: i for i, r in enumerate(recordings)}

    # Build unique class → color mapping
    classes = sorted(df["scenario_label"].dropna().unique())
    color_map = {c: _class_color(c) for c in classes}

    fig, ax = plt.subplots(figsize=(14, max(3, 0.5 * n_rec + 1.5)))

    for _, row in df.iterrows():
        rec = row["recording"]
        y = rec_index[rec]
        t_start_ms = float(row["window_start_ms"])
        t_end_ms = float(row["window_end_ms"])
        label = str(row["scenario_label"]) if pd.notna(row["scenario_label"]) else "unlabeled"
        color = color_map.get(label, "#ccc")

        # Each recording's timeline starts at its own zero
        # Group start per recording
        rec_mask = df["recording"] == rec
        rec_t0 = float(df.loc[rec_mask, "window_start_ms"].min())
        x0 = (t_start_ms - rec_t0) / 1000.0
        width = max((t_end_ms - t_start_ms) / 1000.0, 0.5)

        ax.barh(y, width, left=x0, height=0.6, color=color, alpha=0.85, linewidth=0)

    ax.set_yticks(range(n_rec))
    ax.set_yticklabels(recordings, fontsize=8)
    ax.set_xlabel("Time from section start (s)")
    ax.set_title("Session timeline — windows coloured by scenario label")
    ax.grid(axis="x", alpha=0.2, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend — only classes present
    handles = [
        mpatches.Patch(color=color_map[c], label=c)
        for c in classes
        if c in df["scenario_label"].values
    ]
    if handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=6,
            framealpha=0.85,
            ncol=max(1, len(handles) // 10),
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote session timeline → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Recording summary table
# ---------------------------------------------------------------------------

def plot_recording_summary_table(df: pd.DataFrame, output_path: Path) -> Path:
    """Render a per-recording statistics table as a matplotlib figure."""
    if "section_id" not in df.columns:
        log.warning("Missing 'section_id' column; skipping recording summary table")
        return output_path

    df = df.copy()
    df["recording"] = df["section_id"].str.replace(r"s\d+$", "", regex=True)

    rows = []
    for rec, grp in df.groupby("recording"):
        n_sections = grp["section_id"].nunique()
        n_windows = len(grp)
        n_classes = grp["scenario_label"].nunique() if "scenario_label" in grp.columns else "—"
        if "window_duration_s" in grp.columns:
            total_s = float(grp["window_duration_s"].sum())
            duration_str = f"{total_s / 60:.1f} min"
        else:
            duration_str = "—"
        rows.append([rec, n_sections, n_windows, duration_str, n_classes])

    if not rows:
        log.warning("No data for recording summary table")
        return output_path

    col_labels = ["Recording", "Sections", "Windows", "Total duration", "Classes"]
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * n_rows + 1.0)))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#37474F")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, n_rows + 1):
        bg = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    ax.set_title("Recording summary", fontsize=11, pad=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote recording summary table → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------

def generate_dataset_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Write CSV summary tables to output_dir/tables/."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Class counts
    if "scenario_label" in df.columns:
        counts = (
            df["scenario_label"]
            .value_counts()
            .reset_index()
            .rename(columns={"scenario_label": "scenario_label", "count": "n_windows"})
        )
        counts.to_csv(tables_dir / "class_counts.csv", index=False)

    # 2. Per-recording stats
    if "section_id" in df.columns:
        rdf = df.copy()
        rdf["recording"] = rdf["section_id"].str.replace(r"s\d+$", "", regex=True)
        rec_stats = (
            rdf.groupby("recording")
            .agg(
                n_sections=("section_id", "nunique"),
                n_windows=("section_id", "count"),
                n_classes=("scenario_label", "nunique"),
                total_window_s=("window_duration_s", "sum"),
            )
            .reset_index()
        )
        if "window_duration_s" in df.columns:
            rec_stats["total_window_s"] = rec_stats["total_window_s"].round(1)
        rec_stats.to_csv(tables_dir / "recording_stats.csv", index=False)

    # 3. Quality stats
    if "overall_quality_label" in df.columns:
        qual_counts = (
            df["overall_quality_label"]
            .value_counts()
            .reset_index()
            .rename(columns={"overall_quality_label": "quality_label", "count": "n_windows"})
        )
        qual_counts.to_csv(tables_dir / "quality_counts.csv", index=False)

    log.info("Wrote dataset tables to %s", project_relative_path(tables_dir))
