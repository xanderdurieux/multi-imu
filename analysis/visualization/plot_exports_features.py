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
from features.labels import ensure_resolved_labels
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

_FEATURE_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("bike", "Bike only", "#3498db"),
    ("rider", "Rider only", "#e74c3c"),
    ("cross", "Cross-sensor", "#2ecc71"),
)

_FEATURE_FAMILY_ORDER: tuple[str, ...] = (
    "Statistical",
    "Temporal",
    "Frequency",
    "Peaks",
    "Orientation",
    "Cross-sensor",
)

_FEATURE_FAMILY_COLORS: dict[str, str] = {
    "Statistical": "#4c78a8",
    "Temporal": "#f58518",
    "Frequency": "#54a24b",
    "Peaks": "#e45756",
    "Orientation": "#b279a2",
    "Cross-sensor": "#72b7b2",
}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric exported feature columns grouped by sensor prefixes."""
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c.startswith(("bike_", "rider_", "cross_"))
        and not df[c].isna().all()
    ]


def _feature_family(feature: str) -> str:
    """Return a presentation-friendly feature family for an exported column."""
    if feature.startswith("cross_"):
        return "Cross-sensor"

    name = feature.replace("bike_", "", 1).replace("rider_", "", 1)
    if "_temporal_" in name:
        return "Temporal"
    if "_spectral_" in name or "_dominant_freq" in name or "_bandpower_" in name:
        return "Frequency"
    if "_peak_" in name:
        return "Peaks"
    if any(token in name for token in ("pitch", "roll", "yaw_rate")):
        return "Orientation"
    return "Statistical"


def _feature_group(feature: str) -> str:
    """Return bike/rider/cross group for an exported feature column."""
    return feature.split("_", 1)[0]


def _feature_family_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return feature counts indexed by sensor group and feature family."""
    cols = _feature_columns(df)
    rows = [
        {
            "group": _feature_group(col),
            "family": _feature_family(col),
            "feature": col,
        }
        for col in cols
    ]
    if not rows:
        return pd.DataFrame(index=[g for g, _, _ in _FEATURE_GROUPS], columns=_FEATURE_FAMILY_ORDER).fillna(0)

    counts = (
        pd.DataFrame(rows)
        .groupby(["group", "family"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=[g for g, _, _ in _FEATURE_GROUPS], fill_value=0)
        .reindex(columns=_FEATURE_FAMILY_ORDER, fill_value=0)
    )
    return counts.astype(int)


def _feature_set_totals(counts: pd.DataFrame) -> dict[str, int]:
    """Return canonical model feature-set sizes from group/family counts."""
    bike = int(counts.loc["bike"].sum()) if "bike" in counts.index else 0
    rider = int(counts.loc["rider"].sum()) if "rider" in counts.index else 0
    cross = int(counts.loc["cross"].sum()) if "cross" in counts.index else 0
    return {
        "Bike": bike,
        "Rider": rider,
        "Fused\n(no cross)": bike + rider,
        "Fused": bike + rider + cross,
    }


def _format_int(value: float | int) -> str:
    """Return integer with thin presentation-friendly thousands grouping."""
    return f"{int(value):,}"


def _pct(value: float, total: float) -> str:
    """Return percent string."""
    if total <= 0:
        return "0.0%"
    return f"{100.0 * float(value) / float(total):.1f}%"


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


def plot_feature_family_summary(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Stacked feature-family chart for the exported model inputs."""
    out_path = output_dir / "feature_family_summary.png"
    counts = _feature_family_counts(df)
    if int(counts.to_numpy().sum()) == 0:
        log.debug("No exported feature columns found; skipping feature-family summary")
        return None

    group_labels = {key: label for key, label, _ in _FEATURE_GROUPS}
    group_colors = {key: color for key, _, color in _FEATURE_GROUPS}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        gridspec_kw={"width_ratios": [1.6, 1.0]},
    )
    ax_stack, ax_totals = axes

    y = np.arange(len(counts.index))
    left = np.zeros(len(counts.index), dtype=float)
    for family in _FEATURE_FAMILY_ORDER:
        vals = counts[family].to_numpy(dtype=float)
        if not np.any(vals):
            continue
        ax_stack.barh(
            y,
            vals,
            left=left,
            height=0.62,
            color=_FEATURE_FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.7,
            label=family,
        )
        for i, val in enumerate(vals):
            if val >= 8:
                ax_stack.text(
                    left[i] + val / 2,
                    y[i],
                    str(int(val)),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if family in {"Statistical", "Peaks", "Orientation"} else "black",
                )
        left += vals

    ax_stack.set_yticks(y)
    ax_stack.set_yticklabels([group_labels.get(g, g) for g in counts.index], fontsize=9)
    ax_stack.invert_yaxis()
    ax_stack.set_title("Feature families by source")
    ax_stack.grid(axis="x", alpha=0.25, lw=0.6)
    ax_stack.spines["top"].set_visible(False)
    ax_stack.spines["right"].set_visible(False)
    ax_stack.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=3,
        fontsize=8,
        frameon=False,
    )

    totals = _feature_set_totals(counts)
    total_labels = list(totals)
    total_vals = list(totals.values())
    total_colors = [
        group_colors["bike"],
        group_colors["rider"],
        "#f39c12",
        group_colors["cross"],
    ]
    bars = ax_totals.bar(total_labels, total_vals, color=total_colors, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, total_vals):
        ax_totals.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(total_vals) * 0.025,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_totals.set_ylabel("Features available to model")
    ax_totals.set_title("Feature-set sizes")
    ax_totals.set_ylim(top=max(total_vals) * 1.18)
    ax_totals.grid(axis="y", alpha=0.25, lw=0.6)
    ax_totals.spines["top"].set_visible(False)
    ax_totals.spines["right"].set_visible(False)

    fig.suptitle("Feature extraction overview", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    return save_figure(fig, out_path, dpi=DPI)


def plot_export_feature_summary(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Full slide-style summary of exported feature extraction outputs."""
    out_path = output_dir / "feature_extraction_summary.png"
    feature_counts = _feature_family_counts(df)
    if df.empty or int(feature_counts.to_numpy().sum()) == 0:
        log.debug("No feature rows or columns found; skipping export feature summary")
        return None

    n_windows = len(df)
    n_sessions = int(df["session"].nunique()) if "session" in df.columns else 0
    n_recordings = int(df["recording_name"].nunique()) if "recording_name" in df.columns else 0
    n_sections = int(df["section_id"].nunique()) if "section_id" in df.columns else 0
    duration_min = (
        float(df.groupby("section_id")["window_end_ms"].max().sub(df.groupby("section_id")["window_start_ms"].min()).sum() / 60000.0)
        if "section_id" in df.columns and {"window_start_ms", "window_end_ms"}.issubset(df.columns)
        else float("nan")
    )

    window_counts = (
        df["window_type"].fillna("unknown").value_counts()
        if "window_type" in df.columns
        else pd.Series({"windows": n_windows})
    )
    quality_counts = (
        df["quality_tier"].fillna("?").value_counts()
        if "quality_tier" in df.columns
        else pd.Series(dtype=int)
    )
    feature_set_totals = _feature_set_totals(feature_counts)

    mean_samples = {
        "Sporsa": float(pd.to_numeric(df.get("window_n_samples_sporsa"), errors="coerce").mean())
        if "window_n_samples_sporsa" in df.columns else float("nan"),
        "Arduino": float(pd.to_numeric(df.get("window_n_samples_arduino"), errors="coerce").mean())
        if "window_n_samples_arduino" in df.columns else float("nan"),
    }
    valid_ratio = {
        "Sporsa": float(pd.to_numeric(df.get("window_valid_ratio_sporsa"), errors="coerce").mean())
        if "window_valid_ratio_sporsa" in df.columns else float("nan"),
        "Arduino": float(pd.to_numeric(df.get("window_valid_ratio_arduino"), errors="coerce").mean())
        if "window_valid_ratio_arduino" in df.columns else float("nan"),
    }

    fig = plt.figure(figsize=(13.5, 8.3))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.72, 1.25, 1.15], hspace=0.88, wspace=0.50)

    ax_cards = fig.add_subplot(gs[0, :])
    ax_cards.axis("off")
    cards = [
        ("Windows", _format_int(n_windows), "1.0 s windows / 0.25 s hop"),
        ("Sections", str(n_sections), f"{n_recordings} recordings / {n_sessions} sessions"),
        ("Duration", f"{duration_min:.1f} min" if np.isfinite(duration_min) else "n/a", "section-level processed data"),
        ("Model inputs", str(int(feature_counts.to_numpy().sum())), "numeric fused features"),
    ]
    card_colors = ("#ecf5ff", "#fff4e5", "#ecf8f1", "#f5f0ff")
    for i, (title, value, subtitle) in enumerate(cards):
        x0 = 0.02 + i * 0.245
        ax_cards.add_patch(
            plt.Rectangle(
                (x0, 0.12),
                0.225,
                0.76,
                facecolor=card_colors[i],
                edgecolor="#d6dce5",
                linewidth=0.8,
            )
        )
        ax_cards.text(x0 + 0.018, 0.70, title, fontsize=9, color="#4a5568", ha="left", va="center")
        ax_cards.text(x0 + 0.018, 0.43, value, fontsize=18, fontweight="bold", ha="left", va="center")
        ax_cards.text(x0 + 0.018, 0.22, subtitle, fontsize=8, color="#4a5568", ha="left", va="center")

    ax_family = fig.add_subplot(gs[1, :2])
    x = np.arange(len(feature_counts.index))
    bottom = np.zeros(len(feature_counts.index), dtype=float)
    group_labels = {key: label for key, label, _ in _FEATURE_GROUPS}
    for family in _FEATURE_FAMILY_ORDER:
        vals = feature_counts[family].to_numpy(dtype=float)
        if not np.any(vals):
            continue
        ax_family.bar(
            x,
            vals,
            bottom=bottom,
            color=_FEATURE_FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.7,
            label=family,
        )
        bottom += vals
    ax_family.set_xticks(x)
    ax_family.set_xticklabels([group_labels.get(g, g) for g in feature_counts.index], fontsize=8)
    ax_family.set_ylabel("Feature count")
    ax_family.set_title("Feature families by source")
    ax_family.grid(axis="y", alpha=0.25, lw=0.6)
    ax_family.spines["top"].set_visible(False)
    ax_family.spines["right"].set_visible(False)
    ax_family.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.40),
        ncol=3,
        fontsize=7,
        frameon=False,
    )

    ax_sets = fig.add_subplot(gs[1, 2:])
    set_labels = list(feature_set_totals)
    set_vals = list(feature_set_totals.values())
    set_colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax_sets.bar(set_labels, set_vals, color=set_colors, edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars, set_vals):
        ax_sets.text(bar.get_x() + bar.get_width() / 2, val + max(set_vals) * 0.02, str(val), ha="center", va="bottom", fontsize=9)
    ax_sets.set_title("Feature sets used in evaluation")
    ax_sets.set_ylabel("Features")
    ax_sets.set_ylim(top=max(set_vals) * 1.18)
    ax_sets.grid(axis="y", alpha=0.25, lw=0.6)
    ax_sets.spines["top"].set_visible(False)
    ax_sets.spines["right"].set_visible(False)

    ax_windows = fig.add_subplot(gs[2, 0])
    labels = window_counts.index.astype(str).tolist()
    vals = window_counts.to_numpy(dtype=float)
    colors = ["#4c78a8", "#f58518", "#9ca3af"][: len(vals)]
    ax_windows.pie(vals, labels=labels, colors=colors, autopct=lambda p: f"{p:.1f}%", textprops={"fontsize": 8})
    ax_windows.set_title("Window types")

    ax_quality = fig.add_subplot(gs[2, 1])
    if not quality_counts.empty:
        order = [q for q in ("A", "B", "C", "?") if q in quality_counts.index]
        vals_q = [int(quality_counts[q]) for q in order]
        colors_q = [{"A": "#2ecc71", "B": "#f39c12", "C": "#e74c3c"}.get(q, "#95a5a6") for q in order]
        bars_q = ax_quality.bar(order, vals_q, color=colors_q, edgecolor="white", linewidth=0.7)
        for bar, val in zip(bars_q, vals_q):
            ax_quality.text(bar.get_x() + bar.get_width() / 2, val + max(vals_q) * 0.02, _pct(val, n_windows), ha="center", va="bottom", fontsize=8)
    ax_quality.set_title("Window quality")
    ax_quality.set_ylabel("Windows")
    ax_quality.grid(axis="y", alpha=0.25, lw=0.6)
    ax_quality.spines["top"].set_visible(False)
    ax_quality.spines["right"].set_visible(False)

    ax_samples = fig.add_subplot(gs[2, 2])
    sample_labels = list(mean_samples)
    sample_vals = [mean_samples[k] for k in sample_labels]
    bars_s = ax_samples.bar(sample_labels, sample_vals, color=["#3498db", "#e74c3c"], edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars_s, sample_vals):
        if np.isfinite(val):
            ax_samples.text(bar.get_x() + bar.get_width() / 2, val + max(sample_vals) * 0.03, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax_samples.set_title("Mean samples / window")
    ax_samples.set_ylabel("Samples")
    ax_samples.grid(axis="y", alpha=0.25, lw=0.6)
    ax_samples.spines["top"].set_visible(False)
    ax_samples.spines["right"].set_visible(False)

    ax_valid = fig.add_subplot(gs[2, 3])
    valid_labels = list(valid_ratio)
    valid_vals = [valid_ratio[k] * 100.0 for k in valid_labels]
    bars_v = ax_valid.bar(valid_labels, valid_vals, color=["#3498db", "#e74c3c"], edgecolor="white", linewidth=0.7)
    for bar, val in zip(bars_v, valid_vals):
        if np.isfinite(val):
            ax_valid.text(bar.get_x() + bar.get_width() / 2, min(100, val) + 2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax_valid.set_title("Mean valid ratio")
    ax_valid.set_ylabel("Valid samples")
    ax_valid.set_ylim(0, 108)
    ax_valid.grid(axis="y", alpha=0.25, lw=0.6)
    ax_valid.spines["top"].set_visible(False)
    ax_valid.spines["right"].set_visible(False)

    fig.suptitle("Exports stage: feature extraction summary", fontsize=14, y=0.985)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.06, right=0.97)
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
        """Return dominant label."""
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
    df = ensure_resolved_labels(df)
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
        """Add."""
        if result is None:
            return
        if isinstance(result, list):
            generated.extend(result)
        else:
            generated.append(result)

    _add(plot_label_distribution(df, figures_dir))
    _add(plot_quality_distribution(df, figures_dir))
    _add(plot_section_overview(df, figures_dir))
    _add(plot_feature_family_summary(df, figures_dir))
    _add(plot_export_feature_summary(df, figures_dir))
    _add(plot_feature_distributions_by_label(df, figures_dir))
    _add(plot_feature_correlation(df, figures_dir))
    _add(plot_pca_by_label(df, figures_dir))

    log.info(
        "EDA complete: %d figures written to %s",
        len(generated),
        project_relative_path(figures_dir),
    )
    return generated
