"""EDA and thesis-ready figures generated from the aggregated feature export."""

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

log = logging.getLogger(__name__)

_DPI = 150
_LABEL_CMAP = "tab10"
_SENSORS = ("sporsa", "arduino")
_SENSOR_COLORS = {"sporsa": "#2980b9", "arduino": "#e67e22"}
_QUALITY_COLORS = {"good": "#2ecc71", "marginal": "#f39c12", "poor": "#e74c3c", "": "#95a5a6"}
_ALL_SYNC_METHODS = ["sda", "lida", "calibration", "online", "adaptive"]
_METHOD_COLORS = {
    "sda": "#3498db",
    "lida": "#9b59b6",
    "calibration": "#2ecc71",
    "online": "#e67e22",
    "adaptive": "#e74c3c",
}

_META_COLS = frozenset({
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "scenario_label",
    "overall_quality_label",
    "overall_quality_score",
    "quality_tier",
    "calibration_quality",
    "sync_confidence",
    "window_n_samples_sporsa",
    "window_n_samples_arduino",
    "window_valid_ratio_sporsa",
})

# Key features to use for per-group violin plots (use first N that exist in df)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_colors(labels: list[str]) -> dict[str, tuple]:
    cmap = plt.get_cmap(_LABEL_CMAP)
    return {lbl: cmap(i % 10) for i, lbl in enumerate(sorted(labels))}


def _available(cols: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in cols if c in df.columns]


def _short_section(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else name


def _short_recording(name: str) -> str:
    parts = name.split("_")
    return parts[-1] if parts else name


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(path))
    return path


def _labeled_only(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario_label" not in df.columns:
        return pd.DataFrame()
    return df[df["scenario_label"].notna() & (df["scenario_label"] != "unlabeled")].copy()


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

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
    colors = _label_colors(labels)
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
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(out_path))
    return out_path


def _violin_by_label(
    df: pd.DataFrame,
    feature_cols: list[str],
    title: str,
    output_path: Path,
) -> Path | None:
    """Violin plots for feature_cols, one subplot per feature, grouped by label."""
    labeled = _labeled_only(df)
    if labeled.empty or not feature_cols:
        return None

    labels = sorted(labeled["scenario_label"].unique())
    if len(labels) < 2:
        # With only one class, box plots are trivial — still produce for completeness
        pass
    color_map = _label_colors(labels)

    # Keep only columns that have at least one numeric labeled value.
    # This avoids allocating subplot slots that are guaranteed to be empty.
    plot_cols = [
        col
        for col in feature_cols
        if pd.to_numeric(labeled[col], errors="coerce").notna().any()
    ]
    if not plot_cols:
        return None

    n_features = len(plot_cols)
    # Simple compact layout: near-square grid, capped to 3 columns.
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
        except Exception:
            # Fallback to box plot if violinplot fails (e.g., single-value groups)
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
    # Reserve a little room for the title and tighten panel spacing to reduce
    # unused white margins in the exported PNG.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97), pad=0.45, w_pad=0.35, h_pad=0.45)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(output_path))
    return output_path


def plot_feature_distributions_by_label(
    df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Generate violin plots for bike, rider, and cross-sensor features by label."""
    paths: list[Path] = []
    for group, key_cols in _KEY_FEATURES.items():
        cols = _available(key_cols, df)
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
        if c not in _META_COLS
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
        .replace("events_", "E:")
        for c in top_cols
    ]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=90, fontsize=max(5, 8 - n // 10))
    ax.set_yticklabels(short, fontsize=max(5, 8 - n // 10))

    plt.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title(f"Feature correlation (top {n} by variance)", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(out_path))
    return out_path


def plot_pca_by_label(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """2D PCA scatter colored by scenario label."""
    out_path = output_dir / "pca_by_label.png"

    labeled = _labeled_only(df)
    if len(labeled) < 5:
        log.debug("Too few labeled rows for PCA (%d)", len(labeled))
        return None

    feature_cols = [
        c for c in labeled.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(labeled[c])
    ]
    if len(feature_cols) < 2:
        return None

    X = labeled[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    unique_labels = sorted(labeled["scenario_label"].unique())
    color_map = _label_colors(unique_labels)
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
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(out_path))
    return out_path


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
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(out_path))
    return out_path


def plot_section_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Bar chart showing number of windows per section, colored by dominant label."""
    out_path = output_dir / "section_overview.png"

    if "section_id" not in df.columns:
        return None

    sections = sorted(df["section_id"].unique())
    if not sections:
        return None

    # Dominant label per section (most frequent non-unlabeled, or unlabeled)
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
    color_map = _label_colors(all_labels)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(sections) + 2), 4))

    for i, sec in enumerate(sections):
        cnt = int(section_counts.get(sec, 0))
        lbl = section_labels[sec]
        color = color_map.get(lbl, "gray")
        bar = ax.bar(i, cnt, color=color, edgecolor="white", linewidth=0.5)
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
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", project_relative_path(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Run all EDA figure generation on the aggregated feature DataFrame.

    Parameters
    ----------
    df:
        Combined feature DataFrame (output of :func:`exports.pipeline.aggregate_features`).
    output_dir:
        Parent exports directory; figures are written to ``output_dir/figures/``.

    Returns
    -------
    List of generated figure paths.
    """
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


def plot_calibration_quality_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "cal_quality_overview.png"
    if df.empty or "section_id" not in df.columns:
        return None
    sections = df["section_id"].tolist()
    n = len(sections)
    if n == 0:
        return None
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55 + 2), 4))
    for i, sensor in enumerate(_SENSORS):
        col = f"{sensor}_quality"
        if col not in df.columns:
            continue
        qualities = df[col].fillna("").tolist()
        colors = [_QUALITY_COLORS.get(q, "#95a5a6") for q in qualities]
        offset = (i - 0.5) * width
        ax.bar(x + offset, [1] * n, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_title("Calibration quality per section")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_gravity_residuals(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "cal_gravity_residuals.png"
    if df.empty or "section_id" not in df.columns:
        return None
    sections = df["section_id"].tolist()
    n = len(sections)
    if n == 0:
        return None
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55 + 2), 4))
    for i, sensor in enumerate(_SENSORS):
        col = f"{sensor}_gravity_residual_ms2"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).tolist()
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=_SENSOR_COLORS[sensor], label=sensor, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Gravity residual (m/s²)")
    ax.set_title("Calibration gravity residual per section")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sensor_biases(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    if df.empty or "section_id" not in df.columns:
        return paths
    sections = df["section_id"].tolist()
    n = len(sections)
    if n == 0:
        return paths
    x = np.arange(n)
    axes_labels = ["x", "y", "z"]
    for bias_type in ("acc_bias", "gyro_bias"):
        ylabel = "Acc bias (m/s²)" if bias_type == "acc_bias" else "Gyro bias (rad/s)"
        out_path = output_dir / f"cal_{bias_type}.png"
        fig, axes = plt.subplots(1, len(_SENSORS), figsize=(max(8, n * 0.45 + 2) * len(_SENSORS) / 2, 4), squeeze=False)
        for col_idx, sensor in enumerate(_SENSORS):
            ax = axes[0][col_idx]
            width = 0.25
            offsets = [-width, 0, width]
            for i, axis_label in enumerate(axes_labels):
                col = f"{sensor}_{bias_type}_{axis_label}"
                if col not in df.columns:
                    continue
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).tolist()
                ax.bar(x + offsets[i], vals, width, label=axis_label, edgecolor="white", linewidth=0.5)

            # Overlay static calibration reference for Arduino when available.
            if sensor == "arduino":
                if bias_type == "acc_bias":
                    static_prefix = "static_acc_bias_"
                else:
                    static_prefix = "static_gyro_bias_deg_s_"
                for i, axis_label in enumerate(axes_labels):
                    static_col = f"{static_prefix}{axis_label}"
                    if static_col not in df.columns:
                        continue
                    static_vals = pd.to_numeric(df[static_col], errors="coerce").dropna()
                    if static_vals.empty:
                        continue
                    ref = float(static_vals.median())
                    ax.axhline(
                        ref,
                        color="black",
                        linestyle="--",
                        linewidth=0.8,
                        alpha=0.75,
                    )
            ax.set_xticks(x)
            ax.set_xticklabels([_short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{sensor} — {bias_type.replace('_', ' ')}")
            ax.legend(fontsize=8, framealpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.grid(axis="y", alpha=0.3, lw=0.5)
        fig.suptitle(f"{bias_type.replace('_', ' ').title()} across sections", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        paths.append(_save(fig, out_path))
    return paths


def plot_forward_confidence(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "cal_forward_confidence.png"
    if df.empty or "section_id" not in df.columns:
        return None
    sections = df["section_id"].tolist()
    n = len(sections)
    if n == 0:
        return None
    if not any(f"{s}_forward_confidence" in df.columns for s in _SENSORS):
        return None
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55 + 2), 4))
    for i, sensor in enumerate(_SENSORS):
        col = f"{sensor}_forward_confidence"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).tolist()
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=_SENSOR_COLORS[sensor], label=sensor, edgecolor="white", linewidth=0.5)
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Forward confidence")
    ax.set_title("Forward direction estimation confidence per section")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_static_calibration_reference(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot static calibration reference values (bias + scale) if present."""
    needed = [
        "static_acc_bias_x", "static_acc_bias_y", "static_acc_bias_z",
        "static_acc_scale_x", "static_acc_scale_y", "static_acc_scale_z",
        "static_gyro_bias_deg_s_x", "static_gyro_bias_deg_s_y", "static_gyro_bias_deg_s_z",
    ]
    if df.empty or not any(col in df.columns for col in needed):
        return None

    out_path = output_dir / "cal_static_reference.png"
    ref_vals: dict[str, float] = {}
    for col in needed:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if not vals.empty:
            ref_vals[col] = float(vals.median())
    if not ref_vals:
        return None

    acc_bias = [ref_vals.get(f"static_acc_bias_{a}", 0.0) for a in ("x", "y", "z")]
    acc_scale = [ref_vals.get(f"static_acc_scale_{a}", 1.0) for a in ("x", "y", "z")]
    gyro_bias = [ref_vals.get(f"static_gyro_bias_deg_s_{a}", 0.0) for a in ("x", "y", "z")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), squeeze=False)
    ax0, ax1, ax2 = axes[0]
    idx = np.arange(3)
    labels = ["x", "y", "z"]

    ax0.bar(idx, acc_bias, color="#3498db", edgecolor="white", linewidth=0.5)
    ax0.axhline(0, color="black", linewidth=0.5)
    ax0.set_xticks(idx)
    ax0.set_xticklabels(labels)
    ax0.set_title("Static acc bias")
    ax0.set_ylabel("m/s²")

    ax1.bar(idx, acc_scale, color="#2ecc71", edgecolor="white", linewidth=0.5)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(labels)
    ax1.set_title("Static acc scale")

    ax2.bar(idx, gyro_bias, color="#e67e22", edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(labels)
    ax2.set_title("Static gyro bias")
    ax2.set_ylabel("deg/s")

    fig.suptitle("Static calibration reference (Arduino)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


def plot_sync_method_selection(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_method_selection.png"
    if df.empty or "recording_name" not in df.columns or "selected_method" not in df.columns:
        return None
    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None
    methods = df["selected_method"].fillna("unknown").tolist()
    colors = [_METHOD_COLORS.get(m, "#95a5a6") for m in methods]
    fig, ax = plt.subplots(figsize=(max(7, n * 0.7 + 2), 4))
    ax.bar(range(n), [1] * n, color=colors, edgecolor="white", linewidth=0.5)
    for i, method in enumerate(methods):
        ax.text(i, 0.5, method, ha="center", va="center", fontsize=8, rotation=90, color="white", fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_title("Selected sync method per recording")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sync_method_availability(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_method_availability.png"
    if df.empty or "recording_name" not in df.columns:
        return None
    available_cols = [f"{m}_available" for m in _ALL_SYNC_METHODS if f"{m}_available" in df.columns]
    if not available_cols:
        return None
    recordings = df["recording_name"].tolist()
    n_rec = len(recordings)
    method_names = [c.replace("_available", "") for c in available_cols]
    n_methods = len(method_names)
    matrix = np.zeros((n_methods, n_rec), dtype=float)
    for j, col in enumerate(available_cols):
        avail = df[col].fillna(False).tolist()
        for i, val in enumerate(avail):
            matrix[j, i] = 1.0 if bool(val) else 0.0
    fig, ax = plt.subplots(figsize=(max(7, n_rec * 0.6 + 2), max(3, n_methods * 0.6 + 1.5)))
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_rec))
    ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(method_names, fontsize=9)
    ax.set_title("Sync method availability per recording")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sync_correlation_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_correlation_overview.png"
    if df.empty or "recording_name" not in df.columns:
        return None
    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None
    available_methods = [m for m in _ALL_SYNC_METHODS if f"{m}_corr_offset_and_drift" in df.columns]
    if not available_methods and "corr_offset_and_drift" not in df.columns:
        return None
    x = np.arange(n)
    width = 0.8 / max(len(available_methods), 1)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7 + 2), 4))
    if available_methods:
        legend_handles: list[plt.Rectangle] = []
        for i, method in enumerate(available_methods):
            col = f"{method}_corr_offset_and_drift"
            vals = pd.to_numeric(df[col], errors="coerce").tolist()
            offset = (i - len(available_methods) / 2 + 0.5) * width
            color = _METHOD_COLORS.get(method, "gray")
            ax.bar(
                x + offset,
                [v if v is not None else 0 for v in vals],
                width,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.85,
            )
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=method))
    else:
        vals = pd.to_numeric(df["corr_offset_and_drift"], errors="coerce").tolist()
        ax.bar(x, [v if v is not None else 0 for v in vals], 0.6, color="#3498db", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correlation (offset + drift)")
    ax.set_title("Sync correlation scores per recording and method")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    if available_methods:
        ax.legend(
            handles=legend_handles,
            title="Method",
            loc="upper right",
            fontsize=8,
            title_fontsize=8,
            framealpha=0.8,
        )
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sync_drift_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_drift_overview.png"
    if df.empty or "recording_name" not in df.columns:
        return None

    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None

    available_methods = [
        m for m in _ALL_SYNC_METHODS
        if f"{m}_drift_ppm" in df.columns
    ]

    # Fallback to selected drift if per-method columns are unavailable.
    if not available_methods:
        drift_col = "drift_ppm"
        if drift_col not in df.columns:
            if "drift_seconds_per_second" in df.columns:
                df = df.copy()
                df[drift_col] = pd.to_numeric(df["drift_seconds_per_second"], errors="coerce") * 1e6
            else:
                return None
        drift_vals = pd.to_numeric(df[drift_col], errors="coerce").tolist()
        fig, ax = plt.subplots(figsize=(max(7, n * 0.7 + 2), 4))
        colors = ["#e74c3c" if (v is not None and abs(v) > 1000) else "#2980b9" for v in drift_vals]
        ax.bar(range(n), [v if pd.notna(v) else 0.0 for v in drift_vals], color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Drift (ppm)")
        ax.set_title("Selected sync drift per recording")
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        fig.tight_layout()
        return _save(fig, out_path)

    x = np.arange(n)
    width = 0.8 / len(available_methods)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.7 + 2), 4))
    legend_handles: list[plt.Rectangle] = []

    for i, method in enumerate(available_methods):
        col = f"{method}_drift_ppm"
        vals = pd.to_numeric(df[col], errors="coerce").tolist()
        offset = (i - len(available_methods) / 2 + 0.5) * width
        color = _METHOD_COLORS.get(method, "gray")
        ax.bar(
            x + offset,
            [v if pd.notna(v) else 0.0 for v in vals],
            width,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=method))

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(400, color="gray", linestyle="--", linewidth=0.7)
    ax.axhline(-400, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Drift (ppm)")
    ax.set_title("Sync drift per recording and method")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(handles=legend_handles, title="Method", loc="upper right", fontsize=8, title_fontsize=8, framealpha=0.8)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_sync_calibration_anchor_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_calibration_anchors_overview.png"
    if df.empty or "recording_name" not in df.columns:
        return None
    if "calibration_cal_n_windows" not in df.columns and "calibration_cal_fit_r2" not in df.columns:
        return None

    recordings = df["recording_name"].tolist()
    n = len(recordings)
    if n == 0:
        return None

    n_windows = pd.to_numeric(df.get("calibration_cal_n_windows"), errors="coerce")
    fit_r2 = pd.to_numeric(df.get("calibration_cal_fit_r2"), errors="coerce")
    if (n_windows is None or n_windows.isna().all()) and (fit_r2 is None or fit_r2.isna().all()):
        return None

    x = np.arange(n)
    fig, axes = plt.subplots(2, 1, figsize=(max(8, n * 0.7 + 2), 6), sharex=True)

    ax = axes[0]
    vals = n_windows.fillna(0.0).tolist() if n_windows is not None else [0.0] * n
    ax.bar(x, vals, color="#2ca02c", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("Count")
    ax.set_title("Calibration anchors used per recording")
    ax.grid(axis="y", alpha=0.3, lw=0.5)

    ax = axes[1]
    vals = fit_r2.fillna(0.0).tolist() if fit_r2 is not None else [0.0] * n
    colors = ["#2ca02c" if v >= 0.2 else "#d62728" for v in vals]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.axhline(0.2, color="orange", linestyle="--", linewidth=0.9)
    ax.set_ylabel("R²")
    ax.set_title("Calibration all-anchor fit quality (R²)")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    return _save(fig, out_path)


def plot_sync_offset_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "sync_offset_overview.png"
    if df.empty or "recording_name" not in df.columns or "offset_seconds" not in df.columns:
        return None
    recordings = df["recording_name"].tolist()
    offset_vals = pd.to_numeric(df["offset_seconds"], errors="coerce").tolist()
    valid_offsets = [float(v) for v in offset_vals if pd.notna(v)]
    if not valid_offsets:
        return None

    fig, ax = plt.subplots(figsize=(max(7, len(recordings) * 0.7 + 2), 4))
    ax.bar(range(len(recordings)), [v or 0 for v in offset_vals], color="#9b59b6", edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(recordings)))
    ax.set_xticklabels([_short_recording(r) for r in recordings], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Offset (s)")
    ax.set_title("Selected sync time offset per recording")
    # Zoom y-axis to the observed offset range so small differences are visible.
    min_v = min(valid_offsets)
    max_v = max(valid_offsets)
    span = max_v - min_v
    pad = max(1e-3, span * 0.12)
    if span == 0:
        ax.set_ylim(min_v - pad, max_v + pad)
    else:
        ax.set_ylim(min_v - pad, max_v + pad)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return _save(fig, out_path)


def run_calibration_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if df.empty:
        log.warning("Calibration params DataFrame is empty; skipping calibration EDA")
        return []
    figures_dir = Path(output_dir) / "figures" / "calibration"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for result in (
        plot_calibration_quality_overview(df, figures_dir),
        plot_gravity_residuals(df, figures_dir),
        plot_sensor_biases(df, figures_dir),
        plot_forward_confidence(df, figures_dir),
        plot_static_calibration_reference(df, figures_dir),
    ):
        if result is not None:
            if isinstance(result, list):
                generated.extend(result)
            else:
                generated.append(result)
    log.info("Calibration EDA complete: %d figures", len(generated))
    return generated


def run_sync_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if df.empty:
        log.warning("Sync params DataFrame is empty; skipping sync EDA")
        return []
    figures_dir = Path(output_dir) / "figures" / "sync"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for result in (
        plot_sync_method_selection(df, figures_dir),
        plot_sync_method_availability(df, figures_dir),
        plot_sync_correlation_overview(df, figures_dir),
        plot_sync_drift_overview(df, figures_dir),
        plot_sync_calibration_anchor_overview(df, figures_dir),
    ):
        if result is not None:
            generated.append(result)
    log.info("Sync EDA complete: %d figures", len(generated))
    return generated
