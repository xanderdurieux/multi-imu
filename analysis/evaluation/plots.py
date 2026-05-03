"""Thesis-ready evaluation figures: confusion matrices, feature importance, model comparison."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import (
    parse_section_folder_name,
    project_relative_path,
    read_csv,
    recording_labels_csv,
    section_dir,
    section_labels_csv,
)
from common.signals import vector_norm
from labels.parser import load_labels
from visualization._utils import filter_valid_plot_xy
from visualization.plot_labels import _label_colors, _label_patches

log = logging.getLogger(__name__)

_DPI = 150

_MODEL_DISPLAY: dict[str, str] = {
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist. Gradient Boosting",
    "logistic_regression": "Logistic Regression",
}

_MODEL_COLORS: dict[str, str] = {
    "random_forest": "#3498db",
    "hist_gradient_boosting": "#2ecc71",
    "logistic_regression": "#e74c3c",
}

_FEATURE_SET_COLORS: dict[str, str] = {
    "bike": "#3498db",
    "rider": "#e74c3c",
    "fused_no_cross": "#f39c12",
    "fused": "#2ecc71",
}

# Matches _FS_DISPLAY in experiments.py — kept local to avoid cross-module import.
_FS_DISPLAY: dict[str, str] = {
    "bike": "Bike only",
    "rider": "Rider only",
    "fused_no_cross": "Fused\n(no cross)",
    "fused": "Fused",
}

# Canonical x-axis order for comparison figures (simple → complex).
_FS_ORDER: list[str] = ["bike", "rider", "fused_no_cross", "fused"]

# Color coding for feature groups in importance plots
_GROUP_COLORS: dict[str, str] = {
    "bike_": "#3498db",
    "rider_": "#e74c3c",
    "cross_": "#2ecc71",
    "events_": "#f39c12",
    "spectral_": "#9b59b6",
    "orientation_": "#1abc9c",
}


def _group_color(feature: str) -> str:
    """Return group color."""
    for prefix, color in _GROUP_COLORS.items():
        if feature.startswith(prefix):
            return color
    return "#95a5a6"


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> Path:
    """Plot confusion matrix."""
    classes = cm_df.index.tolist()
    cm = cm_df.values.astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)
    else:
        cm_norm = cm

    n = len(classes)
    cell_size = max(0.75, 5.5 / n)
    fig, ax = plt.subplots(figsize=(n * cell_size + 1.5, n * cell_size))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            raw = int(cm[i, j])
            cell_text = f"{val:.2f}\n({raw})" if normalize else f"{raw}"
            text_color = "white" if val > 0.55 else "black"
            fs = max(6, min(10, 11 - n))
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=fs, color=text_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Recall" if normalize else "Count")
    ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("Wrote confusion matrix → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    fi_df: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 20,
    title: str = "Feature importance",
) -> Path:
    """Plot feature importance."""
    fi_df = fi_df.copy().sort_values("importance", ascending=False).head(top_n)
    if fi_df.empty:
        log.warning("No feature importance data to plot; skipping")
        return output_path

    # Reverse so most important is at the top
    features = fi_df["feature"].tolist()[::-1]
    importances = fi_df["importance"].tolist()[::-1]
    colors = [_group_color(f) for f in features]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(features) + 1.2)))

    ax.barh(features, importances, color=colors, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend for feature groups present in this chart
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=prefix.rstrip("_").replace("_", " ").capitalize())
        for prefix, color in _GROUP_COLORS.items()
        if any(f.startswith(prefix) for f in features)
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("Wrote feature importance → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Model comparison",
) -> Path:
    """Plot model comparison."""
    if metrics_df.empty:
        log.warning("Empty metrics DataFrame; skipping model comparison plot")
        return output_path

    # Sort feature sets in canonical ablation order; unknowns go last.
    present_fs = metrics_df["feature_set"].unique().tolist()
    feature_sets = [fs for fs in _FS_ORDER if fs in present_fs] + sorted(
        fs for fs in present_fs if fs not in _FS_ORDER
    )
    models = list(dict.fromkeys(metrics_df["model"].tolist()))  # preserve insertion order
    n_models = len(models)

    x = np.arange(len(feature_sets))
    width = 0.20
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    has_std = "accuracy_std" in metrics_df.columns and "macro_f1_std" in metrics_df.columns

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, (metric_col, std_col, ylabel) in zip(
        axes,
        [
            ("accuracy", "accuracy_std", "Accuracy"),
            ("macro_f1", "macro_f1_std", "Macro-F1"),
        ],
    ):
        for model, offset in zip(models, offsets):
            vals, errs = [], []
            for fs in feature_sets:
                row = metrics_df[
                    (metrics_df["feature_set"] == fs) & (metrics_df["model"] == model)
                ]
                if row.empty:
                    vals.append(0.0)
                    errs.append(0.0)
                else:
                    vals.append(float(row[metric_col].iloc[0]))
                    errs.append(float(row[std_col].iloc[0]) if has_std else 0.0)

            color = _MODEL_COLORS.get(model, "#7f8c8d")
            bars = ax.bar(
                x + offset,
                vals,
                width,
                yerr=errs if has_std else None,
                error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#555"},
                label=_MODEL_DISPLAY.get(model, model),
                color=color,
                alpha=0.85,
                edgecolor="white",
            )

            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(errs) if has_std else 0) + 0.012,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

        x_labels = [_FS_DISPLAY.get(fs, fs.replace("_", " ").title()) for fs in feature_sets]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.18)
        ax.set_title(ylabel)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels_text = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        loc="lower center",
        ncol=n_models,
        fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote model comparison → %s", project_relative_path(output_path))
    return output_path


def plot_per_class_f1(
    metrics_df: pd.DataFrame,
    all_results: dict,
    classes: list[str],
    output_path: Path,
    *,
    feature_set: str = "fused",
    title: str = "",
) -> Path | None:
    """Plot per class f1."""
    models = [m for m in _MODEL_DISPLAY if f"{feature_set}__{m}" in all_results]
    if not models:
        log.debug("No results for feature set '%s'; skipping per-class F1 plot", feature_set)
        return None

    x = np.arange(len(classes))
    width = 0.22
    n_models = len(models)
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(max(7, 0.8 * len(classes) + 2), 4))

    for model, offset in zip(models, offsets):
        key = f"{feature_set}__{model}"
        per_class = all_results[key].get("per_class", {})
        f1_vals = [float(per_class.get(cls, {}).get("f1-score", 0.0)) for cls in classes]

        color = _MODEL_COLORS.get(model, "#7f8c8d")
        ax.bar(
            x + offset,
            f1_vals,
            width,
            label=_MODEL_DISPLAY.get(model, model),
            color=color,
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.1)
    ax.set_title(title or f"Per-class F1-score — {feature_set} features")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote per-class F1 → %s", project_relative_path(output_path))
    return output_path


def plot_per_class_f1_feature_set_comparison(
    all_results: dict,
    classes: list[str],
    output_path: Path,
    *,
    feature_sets: tuple[str, ...] = ("bike", "rider", "fused"),
    title: str = "Per-class F1-score by feature set",
) -> Path | None:
    """Plot per-class F1 with feature sets grouped side by side for each model."""
    models = [
        model_name
        for model_name in _MODEL_DISPLAY
        if any(f"{feature_set}__{model_name}" in all_results for feature_set in feature_sets)
    ]
    if not models:
        log.debug("No per-class results available for feature-set comparison plot")
        return None

    available_feature_sets = [
        feature_set
        for feature_set in feature_sets
        if any(f"{feature_set}__{model_name}" in all_results for model_name in models)
    ]
    if not available_feature_sets:
        log.debug("Requested feature sets are not available for comparison plot")
        return None

    x = np.arange(len(classes))
    n_feature_sets = len(available_feature_sets)
    width = min(0.24, 0.78 / max(n_feature_sets, 1))
    offsets = np.linspace(
        -(n_feature_sets - 1) / 2,
        (n_feature_sets - 1) / 2,
        n_feature_sets,
    ) * width

    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(max(8, 0.9 * len(classes) + 2), max(3.6 * len(models), 4)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, model_name in zip(axes_flat, models):
        for feature_set, offset in zip(available_feature_sets, offsets):
            key = f"{feature_set}__{model_name}"
            per_class = all_results.get(key, {}).get("per_class", {})
            f1_vals = [float(per_class.get(cls, {}).get("f1-score", 0.0)) for cls in classes]

            ax.bar(
                x + offset,
                f1_vals,
                width,
                label=_FS_DISPLAY.get(feature_set, feature_set.replace("_", " ").title()),
                color=_FEATURE_SET_COLORS.get(feature_set, "#7f8c8d"),
                alpha=0.88,
                edgecolor="white",
            )

        ax.set_ylim(0, 1.08)
        ax.set_ylabel("F1-score")
        ax.set_title(_MODEL_DISPLAY.get(model_name, model_name), fontsize=10)
        ax.grid(axis="y", alpha=0.3, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes_flat[-1].set_xticks(x)
    axes_flat[-1].set_xticklabels(classes, rotation=30, ha="right", fontsize=8)

    handles, labels_text = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_text,
        loc="lower center",
        ncol=len(available_feature_sets),
        fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "Wrote per-class F1 feature-set comparison → %s",
        project_relative_path(output_path),
    )
    return output_path


# ---------------------------------------------------------------------------
# Batch generation from saved artefacts
# ---------------------------------------------------------------------------

def generate_evaluation_figures(output_dir: Path) -> list[Path]:
    """Generate evaluation figures."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # ---- model comparison (cross-model, top-level figures/) ----
    metrics_path = output_dir / "metrics_table.csv"
    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            out = figures_dir / "model_comparison.png"
            plot_model_comparison(metrics_df, out)
            generated.append(out)
        except Exception as exc:
            log.warning("Failed to generate model comparison figure: %s", exc)
    else:
        log.warning(
            "metrics_table.csv not found in %s — skipping model comparison",
            project_relative_path(output_dir),
        )

    # ---- per-model figures: confusion matrices and feature importances ----
    all_results_disk: dict[str, dict] = {}
    model_dirs = sorted(d for d in output_dir.iterdir() if d.is_dir() and d.name != "figures")

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_figures_dir = model_dir / "figures"
        model_figures_dir.mkdir(exist_ok=True)

        for cm_path in sorted(model_dir.glob("confusion_matrix_*.csv")):
            try:
                cm_df = pd.read_csv(cm_path, index_col=0)
                _, _, fs_name = cm_path.stem.partition("confusion_matrix_")
                human = (
                    f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                    f"{fs_name.replace('_', ' ').title()}"
                )
                out = model_figures_dir / f"{cm_path.stem}.png"
                plot_confusion_matrix(cm_df, out, title=f"Confusion matrix — {human}")
                generated.append(out)
            except Exception as exc:
                log.warning("Failed to plot confusion matrix %s: %s", cm_path.name, exc)

        for fi_path in sorted(model_dir.glob("feature_importance_*.csv")):
            try:
                fi_df = pd.read_csv(fi_path)
                _, _, fs_name = fi_path.stem.partition("feature_importance_")
                human = (
                    f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                    f"{fs_name.replace('_', ' ').title()}"
                )
                out = model_figures_dir / f"{fi_path.stem}.png"
                plot_feature_importance(fi_df, out, title=f"Feature importance — {human}")
                generated.append(out)
            except Exception as exc:
                log.warning("Failed to plot feature importance %s: %s", fi_path.name, exc)

        for pc_path in sorted(model_dir.glob("per_class_report_*.json")):
            try:
                _, _, fs_name = pc_path.stem.partition("per_class_report_")
                data = json.loads(pc_path.read_text(encoding="utf-8"))
                all_results_disk[f"{fs_name}__{model_name}"] = {"per_class": data}
            except Exception as exc:
                log.warning("Failed to load per-class report %s: %s", pc_path.name, exc)

    # ---- per-class F1 (cross-model, top-level figures/, grouped by feature set) ----
    if all_results_disk:
        _aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
        classes_set: set[str] = set()
        for v in all_results_disk.values():
            classes_set.update(k for k in v.get("per_class", {}) if k not in _aggregate_keys)
        classes_disk = sorted(classes_set)
        feature_sets_disk = sorted({k.split("__")[0] for k in all_results_disk})
        metrics_df_disk = pd.DataFrame(columns=["feature_set", "model"])

        for fs_name in feature_sets_disk:
            out = figures_dir / f"per_class_f1_{fs_name}.png"
            try:
                result = plot_per_class_f1(
                    metrics_df_disk,
                    all_results_disk,
                    classes_disk,
                    out,
                    feature_set=fs_name,
                )
                if result:
                    generated.append(result)
            except Exception as exc:
                log.warning("Failed to plot per-class F1 for %s: %s", fs_name, exc)

        out = figures_dir / "per_class_f1_feature_sets.png"
        try:
            result = plot_per_class_f1_feature_set_comparison(
                all_results_disk,
                classes_disk,
                out,
            )
            if result:
                generated.append(result)
        except Exception as exc:
            log.warning("Failed to plot per-class F1 feature-set comparison: %s", exc)

    log.info(
        "Evaluation figures: %d written to %s",
        len(generated),
        project_relative_path(output_dir),
    )
    return generated


# ---------------------------------------------------------------------------
# Permutation feature importance + sensor-group contribution
# ---------------------------------------------------------------------------

# Sensor-group → color (used for both per-feature bars and the group totals).
_SENSOR_GROUP_COLORS: dict[str, str] = {
    "bike": "#3498db",
    "rider": "#e74c3c",
    "cross": "#2ecc71",
    "other": "#95a5a6",
}


def plot_permutation_importance(
    perm_df: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 25,
    title: str = "Permutation importance",
) -> Path | None:
    """Plot permutation importance."""
    if perm_df.empty:
        log.warning("Empty permutation importance DataFrame; skipping plot")
        return None

    df = perm_df.head(top_n).copy()
    # Reverse so the most important feature ends up at the top of the figure.
    df = df.iloc[::-1].reset_index(drop=True)

    colors = [
        _SENSOR_GROUP_COLORS.get(g, _SENSOR_GROUP_COLORS["other"])
        for g in df["sensor_group"]
    ]

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.36 * len(df) + 1.0)))
    ax.barh(
        df["feature"],
        df["perm_importance_mean"],
        xerr=df["perm_importance_std"].fillna(0).clip(lower=0),
        color=colors,
        edgecolor="white",
        linewidth=0.3,
        error_kw={"elinewidth": 0.8, "capsize": 2.0, "ecolor": "#444"},
    )
    ax.set_xlabel("Permutation importance (Δ macro-F1)")
    ax.set_title(title)
    ax.axvline(0, color="#333", lw=0.6)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_SENSOR_GROUP_COLORS[g], label=g)
        for g in ("bike", "rider", "cross", "other")
        if g in set(df["sensor_group"])
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("Wrote permutation importance → %s", project_relative_path(output_path))
    return output_path


def plot_sensor_group_contribution(
    grouped_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Sensor-group contribution",
) -> Path | None:
    """Plot sensor group contribution."""
    if grouped_df.empty:
        return None

    df = grouped_df.copy().sort_values("total_importance", ascending=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    colors = [
        _SENSOR_GROUP_COLORS.get(g, _SENSOR_GROUP_COLORS["other"])
        for g in df["sensor_group"]
    ]
    bars = ax.bar(
        df["sensor_group"],
        df["total_importance"],
        color=colors,
        edgecolor="white",
    )
    for bar, n_feat in zip(bars, df["n_features"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={int(n_feat)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Σ permutation importance")
    ax.set_title(title)
    ax.axhline(0, color="#333", lw=0.6)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("Wrote sensor-group contribution → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# IMU contribution (paired feature-set deltas)
# ---------------------------------------------------------------------------


def plot_imu_contribution(
    contribution_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
    title: str | None = None,
) -> Path | None:
    """Plot imu contribution."""
    if contribution_df.empty:
        return None

    df = contribution_df[contribution_df["metric"] == metric].copy()
    if df.empty:
        return None

    df["pair"] = df["better"] + "\nvs\n" + df["baseline"]
    pairs = df["pair"].drop_duplicates().tolist()
    models = df["model"].drop_duplicates().tolist()

    x = np.arange(len(pairs))
    width = min(0.24, 0.78 / max(len(models), 1))
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(max(6.5, 1.4 * len(pairs) + 2.5), 4.2))

    for model, offset in zip(models, offsets):
        sub = df[df["model"] == model].set_index("pair").reindex(pairs)
        means = sub["delta_mean"].astype(float).values
        stds = sub["delta_std"].astype(float).fillna(0).values
        pvals = sub["wilcoxon_p_one_sided"].values

        color = _MODEL_COLORS.get(model, "#7f8c8d")
        ax.bar(
            x + offset, means, width,
            yerr=stds,
            color=color,
            edgecolor="white",
            error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#444"},
            label=_MODEL_DISPLAY.get(model, model),
            alpha=0.9,
        )

        for xi, mean, p in zip(x + offset, means, pvals):
            star = ""
            if p is not None and not pd.isna(p):
                if float(p) < 0.01:
                    star = "**"
                elif float(p) < 0.05:
                    star = "*"
            label_y = mean + (max(stds) if len(stds) else 0) + 0.005
            txt = f"{mean:+.3f}{star}"
            ax.text(xi, label_y, txt, ha="center", va="bottom", fontsize=6.5)

    ax.axhline(0, color="#333", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel(f"Δ {metric}")
    ax.set_title(title or f"IMU contribution — paired Δ {metric}")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("Wrote IMU contribution plot → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Sweep figures (label_col × min_quality)
# ---------------------------------------------------------------------------


def plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
    feature_set: str = "fused",
    model: str | None = None,
) -> Path | None:
    """Plot sweep heatmap."""
    if sweep_df.empty:
        return None

    df = sweep_df[sweep_df["feature_set"] == feature_set].copy()
    if model is not None:
        df = df[df["model"] == model]
    if df.empty:
        return None

    if model is None:
        # Pick best model per cell (max metric).
        df = (
            df.sort_values(metric, ascending=False)
            .groupby(["label_col", "min_quality"], as_index=False)
            .first()
        )

    pivot = df.pivot_table(
        index="label_col",
        columns="min_quality",
        values=metric,
        aggfunc="max",
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * pivot.shape[1] + 3.5),
                                     max(3.5, 0.55 * pivot.shape[0] + 1.5)))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Min quality filter")
    ax.set_ylabel("Label scheme")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            ax.text(
                j, i, f"{v:.3f}",
                ha="center", va="center",
                fontsize=8,
                color="white" if v > 0.55 else "black",
            )

    plt.colorbar(im, ax=ax, shrink=0.8, label=metric)
    title_model = "best model" if model is None else _MODEL_DISPLAY.get(model, model)
    ax.set_title(f"{metric} sweep — {feature_set} ({title_model})", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote sweep heatmap → %s", project_relative_path(output_path))
    return output_path


def plot_sweep_label_quality_grid(
    sweep_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
) -> Path | None:
    """Plot sweep label quality grid."""
    if sweep_df.empty:
        return None

    feature_sets = [fs for fs in _FS_ORDER if fs in sweep_df["feature_set"].unique()]
    if not feature_sets:
        return None

    fig, axes = plt.subplots(
        1, len(feature_sets),
        figsize=(max(6.0, 3.0 * len(feature_sets) + 1.5), 3.6),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    last_im = None
    for ax, fs in zip(axes_flat, feature_sets):
        sub = sweep_df[sweep_df["feature_set"] == fs]
        # Take best model per (label_col, min_quality) cell.
        sub = (
            sub.sort_values(metric, ascending=False)
            .groupby(["label_col", "min_quality"], as_index=False)
            .first()
        )
        pivot = sub.pivot_table(
            index="label_col", columns="min_quality", values=metric, aggfunc="max"
        )
        if pivot.empty:
            ax.set_visible(False)
            continue
        last_im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, fontsize=8)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(_FS_DISPLAY.get(fs, fs), fontsize=10)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.isna(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                    color="white" if v > 0.55 else "black",
                )

    if last_im is not None:
        fig.colorbar(last_im, ax=axes_flat.tolist(), shrink=0.85, label=metric)
    fig.suptitle(f"{metric} across label scheme × quality filter", fontsize=11)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote sweep grid → %s", project_relative_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Misclassified-window overlay (binary objectives)
# ---------------------------------------------------------------------------

# False positive  = predicted positive class (riding) but actually negative.
# False negative  = predicted negative class (non_riding) but actually positive.
# Distinct hues so the two error modes pop against the (lighter) annotation
# strip in the background.
_FP_COLOR = "#d62728"  # red
_FN_COLOR = "#ff7f0e"  # orange
_MISCLS_ALPHA = 0.45
_MISCLS_EDGE_LW = 1.2


def _load_section_signal(sec_dir: Path, sensor: str) -> pd.DataFrame | None:
    """Load section signal."""
    derived = sec_dir / "derived" / f"{sensor}_signals.csv"
    if derived.exists():
        df = read_csv(derived)
        if "timestamp" in df.columns:
            return df

    calibrated = sec_dir / "calibrated" / f"{sensor}.csv"
    if calibrated.exists():
        df = read_csv(calibrated)
        if "timestamp" in df.columns:
            for col, axes in (("acc_norm", ("ax", "ay", "az")),
                              ("gyro_norm", ("gx", "gy", "gz"))):
                if col not in df.columns and all(a in df.columns for a in axes):
                    df[col] = vector_norm(df, list(axes))
            return df

    return None


def _draw_misclassified_spans(
    axes: list[plt.Axes],
    annotation_ax: plt.Axes,
    rows: pd.DataFrame,
    t0_ms: float,
) -> tuple[int, int]:
    """Draw misclassified spans."""
    n_fp = n_fn = 0
    for _, r in rows.iterrows():
        x0 = (float(r["window_start_ms"]) - t0_ms) / 1000.0
        x1 = (float(r["window_end_ms"]) - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        is_fp = str(r.get("error_type", "")).startswith("FP")
        color = _FP_COLOR if is_fp else _FN_COLOR
        n_fp += int(is_fp)
        n_fn += int(not is_fp)

        for ax in axes:
            ax.axvspan(
                x0, x1,
                facecolor=color,
                edgecolor=color,
                alpha=_MISCLS_ALPHA,
                lw=_MISCLS_EDGE_LW,
                zorder=4,
            )

        win_idx = r.get("window_idx")
        proba = r.get("pred_proba")
        parts: list[str] = []
        if pd.notna(win_idx):
            parts.append(f"#{int(win_idx)}")
        if pd.notna(proba):
            parts.append(f"{float(proba):.2f}")
        if parts:
            annotation_ax.text(
                (x0 + x1) / 2.0,
                0.5,
                "\n".join(parts),
                fontsize=6,
                ha="center",
                va="center",
                color=color,
                zorder=5,
                fontweight="bold",
            )
    return n_fp, n_fn


def _draw_label_strip(
    ax: plt.Axes,
    labels,
    t0_ms: float,
    colors: dict,
) -> None:
    """Draw label strip."""
    for lr in labels:
        if not lr.label:
            continue
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, "#90A4AE")
        ax.axvspan(x0, x1, facecolor=color, edgecolor="none",
                   alpha=0.85, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(0, 1)
    ax.set_ylabel("labels", fontsize=7, rotation=0, ha="right", va="center", labelpad=18)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelbottom=False)


def plot_misclassified_overlay_for_section(
    section_id: str,
    miscls_rows: pd.DataFrame,
    output_path: Path,
    *,
    title_suffix: str = "",
) -> Path | None:
    """Plot misclassified overlay for section."""
    sec_dir = section_dir(section_id)
    if not sec_dir.is_dir():
        log.warning("Section directory missing for %s — skipping overlay", section_id)
        return None

    sporsa = _load_section_signal(sec_dir, "sporsa")
    arduino = _load_section_signal(sec_dir, "arduino")
    if sporsa is None and arduino is None:
        log.warning("No signal CSVs for %s — skipping overlay", section_id)
        return None

    t0_ms = min(
        float(d["timestamp"].iloc[0])
        for d in (sporsa, arduino)
        if d is not None and not d.empty and "timestamp" in d.columns
    )

    # Layout: thin annotation row (window numbers + probabilities), thin
    # label strip, then three signal panels. Stripping labels off the
    # signal axes keeps the IMU traces readable when label density is high.
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs = fig.add_gridspec(
        5, 1,
        height_ratios=[0.25, 0.20, 1.0, 1.0, 1.0],
        hspace=0.18,
    )
    ax_anno = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_anno)
    ax_acc_bike = fig.add_subplot(gs[2], sharex=ax_anno)
    ax_acc_rider = fig.add_subplot(gs[3], sharex=ax_anno)
    ax_gyro = fig.add_subplot(gs[4], sharex=ax_anno)
    signal_axes = [ax_acc_bike, ax_acc_rider, ax_gyro]

    # Annotation row holds the per-window text labels. No data, no spines.
    ax_anno.set_yticks([])
    ax_anno.set_ylim(0, 1)
    for spine in ("top", "right", "left", "bottom"):
        ax_anno.spines[spine].set_visible(False)
    ax_anno.tick_params(left=False, bottom=False, labelbottom=False)

    for sensor_name, df, ax in (
        ("bike (sporsa)", sporsa, ax_acc_bike),
        ("rider (arduino)", arduino, ax_acc_rider),
    ):
        if df is None or df.empty or "acc_norm" not in df.columns:
            ax.text(0.5, 0.5, f"no acc_norm for {sensor_name}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)
            continue
        ts_s = (pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float) - t0_ms) / 1000.0
        x, y = filter_valid_plot_xy(ts_s, df["acc_norm"].to_numpy(dtype=float))
        ax.plot(x, y, lw=0.7, color="#1f77b4")
        ax.set_ylabel(f"{sensor_name}\n|acc| (m/s²)", fontsize=8)
        ax.grid(True, alpha=0.3, lw=0.4)

    for sensor_name, df, color in (
        ("bike", sporsa, "#1f77b4"),
        ("rider", arduino, "#ff7f0e"),
    ):
        if df is None or df.empty or "gyro_norm" not in df.columns:
            continue
        ts_s = (pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float) - t0_ms) / 1000.0
        x, y = filter_valid_plot_xy(ts_s, df["gyro_norm"].to_numpy(dtype=float))
        ax_gyro.plot(x, y, lw=0.7, color=color, label=sensor_name)
    ax_gyro.set_ylabel("|gyro|", fontsize=8)
    ax_gyro.set_xlabel("Time since section start (s)")
    ax_gyro.grid(True, alpha=0.3, lw=0.4)
    ax_gyro.legend(loc="upper right", fontsize=7)

    # Per-section labels live at <sec>/labels/labels.csv (preferred — already
    # trimmed to the section); fallback is the recording-level intervals file
    # for sections that haven't been event-labeled yet.
    label_colors: dict = {}
    section_labels: list = []
    section_labels_path = section_labels_csv(sec_dir)
    if section_labels_path.exists():
        try:
            section_labels = load_labels(section_labels_path)
        except Exception as exc:
            log.warning("Could not load section labels for %s: %s", section_id, exc)
    else:
        try:
            recording_name, _sec_idx = parse_section_folder_name(section_id)
            rec_labels_path = recording_labels_csv(recording_name)
            if rec_labels_path.exists():
                section_labels = load_labels(rec_labels_path)
        except Exception as exc:
            log.warning("Could not load recording labels for %s: %s", section_id, exc)
    if section_labels:
        label_colors = _label_colors(section_labels)
        _draw_label_strip(ax_strip, section_labels, t0_ms, label_colors)
    else:
        _draw_label_strip(ax_strip, [], t0_ms, {})

    # Misclassified windows on the signal panels; per-window text on the
    # dedicated annotation row above.
    n_fp, n_fn = _draw_misclassified_spans(signal_axes, ax_anno, miscls_rows, t0_ms)

    legend_handles = [
        mpatches.Patch(color=_FP_COLOR, alpha=_MISCLS_ALPHA,
                       label=f"FP — predicted riding (n={n_fp})"),
        mpatches.Patch(color=_FN_COLOR, alpha=_MISCLS_ALPHA,
                       label=f"FN — predicted non_riding (n={n_fn})"),
    ] + _label_patches(label_colors)
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=min(6, len(legend_handles)),
               fontsize=7, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    title = f"{section_id} — misclassified windows"
    if title_suffix:
        title += f"  ({title_suffix})"
    fig.suptitle(title, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote misclassified overlay → %s", project_relative_path(output_path))
    return output_path


def plot_misclassified_overlay(
    miscls_csv: Path,
    output_dir: Path,
    *,
    title_suffix: str = "",
) -> list[Path]:
    """Plot misclassified overlay."""
    miscls_csv = Path(miscls_csv)
    if not miscls_csv.exists():
        log.warning("Misclassified CSV missing: %s", project_relative_path(miscls_csv))
        return []

    df = pd.read_csv(miscls_csv)
    required = {"section_id", "window_start_ms", "window_end_ms"}
    missing = required - set(df.columns)
    if missing:
        log.warning("Misclassified CSV %s missing columns: %s", miscls_csv.name, missing)
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for sec_id, rows in df.groupby("section_id"):
        out = output_dir / f"{sec_id}.png"
        try:
            saved = plot_misclassified_overlay_for_section(
                str(sec_id), rows, out, title_suffix=title_suffix,
            )
        except Exception as exc:
            log.warning("Misclassified overlay failed for %s: %s", sec_id, exc)
            continue
        if saved is not None:
            written.append(saved)
    log.info(
        "Misclassified overlays: %d sections → %s",
        len(written),
        project_relative_path(output_dir),
    )
    return written
