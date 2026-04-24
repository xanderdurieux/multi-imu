"""EDA figures derived from the aggregated per-window feature table."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common.paths import project_relative_path
from visualization._exports_common import (
    DPI,
    META_COLS,
    available,
    label_colors,
    labeled_only,
)
from visualization._utils import save_figure

log = logging.getLogger(__name__)

_KEY_FEATURES: dict[str, list[str]] = {
    "bike": [
        "bike_acc_norm_mean",
        "bike_acc_norm_std",
        "bike_acc_norm_max",
        "bike_gyro_norm_mean",
        "bike_gyro_norm_std",
        "bike_jerk_norm_mean",
        "bike_acc_hf_mean",
        "bike_acc_hf_std",
    ],
    "rider": [
        "rider_acc_norm_mean",
        "rider_acc_norm_std",
        "rider_acc_norm_max",
        "rider_gyro_norm_mean",
        "rider_gyro_norm_std",
        "rider_jerk_norm_mean",
        "rider_acc_hf_mean",
    ],
    "cross": [
        "cross_acc_correlation_mean",
        "cross_gyro_diff_mean",
        "cross_acc_diff_mean",
        "cross_disagree_score_mean",
        "cross_disagree_score_max",
        "cross_acc_energy_ratio",
    ],
}

_GROUP_TITLES = {
    "bike": "Bike sensor features by scenario label",
    "rider": "Rider sensor features by scenario label",
    "cross": "Cross-sensor features by scenario label",
}


def plot_label_distribution(df: pd.DataFrame, output_dir: Path) -> Path:
    """Horizontal bar chart showing window count per scenario label."""
    out_path = output_dir / "label_distribution.png"

    if "scenario_label" not in df.columns:
        log.warning("No scenario_label column; skipping label distribution plot")
        return out_path

    counts = (
        df["scenario_label"]
        .fillna("unlabeled")
        .value_counts()
        .sort_values(ascending=True)
    )

    labels = counts.index.tolist()
    colors = label_colors(labels)
    bar_colors = [colors.get(lbl, "gray") for lbl in labels]
    total = int(counts.sum())

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(labels) + 1.2)))

    bars = ax.barh(labels, counts.values, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, cnt in zip(bars, counts.values):
        pct = 100.0 * cnt / total
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax.set_xlabel("Number of windows")
    ax.set_title(f"Label distribution — {total} total windows")
    ax.set_xlim(right=counts.max() * 1.30)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def _violin_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    title: str,
    output_path: Path,
) -> Path | None:
    """Violin plots for feature_cols, one subplot per feature, grouped by label."""
    labeled = labeled_only(df)
    if labeled.empty or not feature_cols:
        return None

    labels = sorted(labeled["scenario_label"].unique())
    color_map = label_colors(labels)

    plot_cols = [
        col
        for col in feature_cols
        if pd.to_numeric(labeled[col], errors="coerce").notna().any()
    ]
    if not plot_cols:
        return None

    n_features = len(plot_cols)
    n_cols = min(3, max(1, int(math.ceil(math.sqrt(n_features)))))
    n_rows = int(math.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.4 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    flat_axes: list[plt.Axes] = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

    for idx, col in enumerate(plot_cols):
        ax = flat_axes[idx]
        data_by_label = [
            pd.to_numeric(labeled.loc[labeled["scenario_label"] == lbl, col], errors="coerce")
            .dropna()
            .values
            for lbl in labels
        ]

        valid = [(lbl, d) for lbl, d in zip(labels, data_by_label) if len(d) > 0]
        if not valid:
            ax.set_visible(False)
            continue

        vl, vd = zip(*valid)

        try:
            parts = ax.violinplot(
                list(vd),
                positions=range(len(vl)),
                showmedians=True,
                showextrema=True,
            )
            for body, lbl in zip(parts["bodies"], vl):
                body.set_facecolor(color_map.get(lbl, "gray"))
                body.set_alpha(0.72)
        except (ValueError, np.linalg.LinAlgError):
            ax.boxplot(list(vd), positions=range(len(vl)), patch_artist=True,
                       boxprops=dict(facecolor="steelblue", alpha=0.7))

        ax.set_xticks(range(len(vl)))
        ax.set_xticklabels(vl, rotation=30, ha="right", fontsize=7)
        short = col.replace("bike_", "").replace("rider_", "").replace("cross_", "")
        ax.set_ylabel(short, fontsize=7)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(plot_cols), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), pad=0.45, w_pad=0.35, h_pad=0.45)
    return save_figure(fig, output_path, dpi=DPI)


def plot_feature_distributions_by_label(
    df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate violin plots for bike, rider, and cross-sensor features by label."""
    paths: list[Path] = []
    for group, key_cols in _KEY_FEATURES.items():
        cols = available(key_cols, df)
        if not cols:
            log.debug("No columns for group '%s'", group)
            continue
        out_path = output_dir / f"feature_distributions_{group}.png"
        result = _violin_by_label(df, cols, _GROUP_TITLES[group], out_path)
        if result:
            paths.append(result)
    return paths


def plot_feature_correlation(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int = 30,
) -> Path | None:
    """Correlation heatmap for the most variable numeric features."""
    out_path = output_dir / "feature_correlation.png"

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and not df[c].isna().all()
    ]
    if len(feature_cols) < 2:
        log.debug("Not enough numeric feature columns for correlation heatmap")
        return None

    variances = df[feature_cols].apply(pd.to_numeric, errors="coerce").var(skipna=True)
    top_cols = variances.sort_values(ascending=False).dropna().index[:top_n].tolist()
    if len(top_cols) < 2:
        return None

    corr = df[top_cols].apply(pd.to_numeric, errors="coerce").corr()

    n = len(top_cols)
    size = max(7, n * 0.38)
    fig, ax = plt.subplots(figsize=(size + 1.5, size))

    im = ax.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)

    short = [
        c.replace("bike_", "B:")
        .replace("rider_", "R:")
        .replace("cross_", "X:")
        for c in top_cols
    ]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=max(5, 8 - n // 10))
    ax.set_yticklabels(short, fontsize=max(5, 8 - n // 10))

    plt.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title(f"Feature correlation (top {n} by variance)", fontsize=11)

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_pca_by_label(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """2D PCA scatter colored by scenario label."""
    out_path = output_dir / "pca_by_label.png"

    labeled = labeled_only(df)
    if len(labeled) < 5:
        log.debug("Too few labeled rows for PCA (%d)", len(labeled))
        return None

    feature_cols = [
        c for c in labeled.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(labeled[c])
    ]
    if len(feature_cols) < 2:
        return None

    X = labeled[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    unique_labels = sorted(labeled["scenario_label"].unique())
    color_map = label_colors(unique_labels)
    label_vals = labeled["scenario_label"].values

    fig, ax = plt.subplots(figsize=(8, 6))

    for lbl in unique_labels:
        mask = label_vals == lbl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=[color_map[lbl]],
            label=lbl,
            alpha=0.75,
            s=45,
            edgecolors="white",
            linewidths=0.3,
        )

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f}% explained variance)")
    ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}% explained variance)")
    ax.set_title("PCA — feature space by scenario label")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    ax.grid(alpha=0.2, lw=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_quality_distribution(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Bar chart showing quality tier distribution of windows."""
    out_path = output_dir / "quality_distribution.png"

    if "quality_tier" not in df.columns:
        log.debug("No quality_tier column; skipping quality distribution")
        return None

    tier_order = ["A", "B", "C"]
    tier_colors = {"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}
    counts = df["quality_tier"].fillna("?").value_counts()
    present = [t for t in tier_order if t in counts]
    if not present:
        return None

    vals = [int(counts[t]) for t in present]
    colors = [tier_colors.get(t, "#95a5a6") for t in present]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(present, vals, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, vals):
        pct = 100 * val / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("Number of windows")
    ax.set_title(f"Window quality distribution — {total} total")
    ax.set_ylim(top=max(vals) * 1.3)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_section_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Bar chart showing number of windows per section, colored by dominant label."""
    out_path = output_dir / "section_overview.png"

    if "section_id" not in df.columns:
        return None

    sections = sorted(df["section_id"].unique())
    if not sections:
        return None

    def dominant_label(sub: pd.DataFrame) -> str:
        if "scenario_label" not in sub.columns:
            return "unlabeled"
        labeled = sub[sub["scenario_label"].notna() & (sub["scenario_label"] != "unlabeled")]
        if labeled.empty:
            return "unlabeled"
        return labeled["scenario_label"].value_counts().index[0]

    section_counts = df.groupby("section_id").size()
    section_labels = {s: dominant_label(df[df["section_id"] == s]) for s in sections}

    all_labels = sorted(set(section_labels.values()))
    color_map = label_colors(all_labels)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(sections) + 2), 4))

    for i, sec in enumerate(sections):
        cnt = int(section_counts.get(sec, 0))
        lbl = section_labels[sec]
        color = color_map.get(lbl, "gray")
        ax.bar(i, cnt, color=color, edgecolor="white", linewidth=0.5)
        ax.text(i, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(sections)))
    ax.set_xticklabels(
        [s.replace("2026-02-26_", "") for s in sections],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Number of windows")
    ax.set_title("Windows per section (color = dominant scenario label)")

    patches = [
        plt.Rectangle((0, 0), 1, 1, color=color_map[lbl], alpha=0.8, label=lbl)
        for lbl in all_labels
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=7, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def run_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Run all feature-EDA figure generation on the aggregated feature DataFrame."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures" / "features"
    figures_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Running EDA on %d rows, %d columns → %s",
        len(df),
        len(df.columns),
        project_relative_path(figures_dir),
    )

    generated: list[Path] = []

    def _add(result: Path | list[Path] | None) -> None:
        if result is None:
            return
        if isinstance(result, list):
            generated.extend(result)
        else:
            generated.append(result)

    _add(plot_label_distribution(df, figures_dir))
    _add(plot_quality_distribution(df, figures_dir))
    _add(plot_section_overview(df, figures_dir))
    _add(plot_feature_distributions_by_label(df, figures_dir))
    _add(plot_feature_correlation(df, figures_dir))
    _add(plot_pca_by_label(df, figures_dir))

    log.info(
        "EDA complete: %d figures written to %s",
        len(generated),
        project_relative_path(figures_dir),
    )
    return generated
