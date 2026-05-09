"""Evaluation figures for scenario classification and label-grid runs."""

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
    read_csv,
    recording_labels_csv,
    section_dir,
    section_labels_csv,
)
from common.signals import vector_norm
from labels.parser import load_labels
from visualization._eval_common import (
    _DPI,
    _FEATURE_SET_COLORS,
    _FS_DISPLAY,
    _FS_ORDER,
    _GROUP_COLORS,
    _MODEL_COLORS,
    _MODEL_DISPLAY,
    _display_path,
    _group_color,
    _ordered_feature_sets,
    plot_imu_contribution,
    plot_permutation_importance,
    plot_sensor_group_contribution,
)
from visualization._utils import filter_valid_plot_xy
from visualization.plot_labels import _label_colors, _label_patches

log = logging.getLogger(__name__)

_LABEL_GRID_CMAP = "RdYlGn"
_LABEL_GRID_QUALITY_ORDER = ("poor", "marginal", "good")
_LABEL_GRID_QUALITY_FEATURE_SETS = ("bike", "rider", "fused_no_cross", "fused")


def _save_confusion_matrix_figure(fig: matplotlib.figure.Figure, output_path: Path) -> None:
    """Save a confusion matrix figure, retrying without tight bbox on font-copy bugs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    except AttributeError as exc:
        msg = str(exc)
        if "FontProperties" not in msg or "__newobj__" not in msg:
            raise
        log.warning(
            "Tight-bbox save failed for confusion matrix (%s); retrying without tight bbox",
            exc,
        )
        fig.savefig(output_path, dpi=_DPI)


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
    """Plot a row-normalized confusion matrix."""
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

    try:
        fig.tight_layout()
        _save_confusion_matrix_figure(fig, output_path)
    finally:
        plt.close(fig)
    log.debug("Wrote confusion matrix → %s", _display_path(output_path))
    return output_path


def _label_grid_color_limits(values: np.ndarray) -> tuple[float, float]:
    """Return compact score limits for label-grid heatmaps."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lower = float(np.floor(np.nanmin(finite) * 20.0) / 20.0)
    return max(0.0, lower), 1.0


def _label_grid_text_color(value: float, vmin: float, vmax: float) -> str:
    """Choose annotation text color from the normalized heatmap value."""
    if vmax <= vmin:
        return "black"
    rel = (float(value) - vmin) / (vmax - vmin)
    return "white" if rel < 0.28 else "black"


def _ordered_quality_filters(values: pd.Series) -> list[str]:
    """Return quality filters in pipeline quality order."""
    present = list(dict.fromkeys(values.dropna().astype(str)))
    ordered = [q for q in _LABEL_GRID_QUALITY_ORDER if q in present]
    ordered.extend(sorted(q for q in present if q not in _LABEL_GRID_QUALITY_ORDER))
    return ordered


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
    """Plot top-N feature importances as a horizontal bar chart."""
    fi_df = fi_df.copy().sort_values("importance", ascending=False).head(top_n)
    if fi_df.empty:
        log.warning("No feature importance data to plot; skipping")
        return output_path

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

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            color=color,
            label=prefix.rstrip("_").replace("_", " ").capitalize(),
        )
        for prefix, color in _GROUP_COLORS.items()
        if any(f.startswith(prefix) for f in features)
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.debug("Wrote feature importance → %s", _display_path(output_path))
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
    """Side-by-side accuracy and macro-F1 bars grouped by feature set."""
    if metrics_df.empty:
        log.warning("Empty metrics DataFrame; skipping model comparison plot")
        return output_path

    present_fs = metrics_df["feature_set"].unique().tolist()
    feature_sets = [fs for fs in _FS_ORDER if fs in present_fs] + sorted(
        fs for fs in present_fs if fs not in _FS_ORDER
    )
    registry_order = list(_MODEL_COLORS.keys())
    raw_models = metrics_df["model"].unique().tolist()
    models = sorted(
        raw_models,
        key=lambda m: (registry_order.index(m) if m in registry_order else len(registry_order), m),
    )
    n_models = len(models)
    if n_models == 0:
        log.warning("No models in metrics table; skipping model comparison plot")
        return output_path

    x = np.arange(len(feature_sets))
    width = min(0.22, 0.78 / max(n_models, 1))
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
            vals: list[float] = []
            errs: list[float] = []
            for fs in feature_sets:
                row = metrics_df[
                    (metrics_df["feature_set"] == fs) & (metrics_df["model"] == model)
                ]
                if row.empty:
                    vals.append(float("nan"))
                    errs.append(float("nan"))
                else:
                    vals.append(float(row[metric_col].iloc[0]))
                    errs.append(float(row[std_col].iloc[0]) if has_std else 0.0)

            color = _MODEL_COLORS.get(model, "#7f8c8d")
            yerr_plot = np.asarray(errs, dtype=float) if has_std else None
            bars = ax.bar(
                x + offset,
                vals,
                width,
                yerr=yerr_plot,
                error_kw={"elinewidth": 1.0, "capsize": 2.5, "ecolor": "#555"},
                label=_MODEL_DISPLAY.get(model, model),
                color=color,
                alpha=0.85,
                edgecolor="white",
            )

            err_cap = float(np.nanmax(errs)) if has_std and errs else 0.0
            for bar, v in zip(bars, vals):
                if np.isfinite(v) and v > 0:
                    h = bar.get_height()
                    if not np.isfinite(h):
                        continue
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + err_cap + 0.012,
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
    log.info("Wrote model comparison → %s", _display_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Per-class F1
# ---------------------------------------------------------------------------


def plot_per_class_f1(
    metrics_df: pd.DataFrame,
    all_results: dict,
    classes: list[str],
    output_path: Path,
    *,
    feature_set: str = "fused",
    title: str = "",
) -> Path | None:
    """Bar chart of per-class F1 for each model at a given feature set."""
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
    log.info("Wrote per-class F1 → %s", _display_path(output_path))
    return output_path


def plot_per_class_f1_feature_set_comparison(
    all_results: dict,
    classes: list[str],
    output_path: Path,
    *,
    feature_sets: tuple[str, ...] = ("bike", "rider", "fused_no_cross", "fused"),
    title: str = "Per-class F1-score by feature set",
) -> Path | None:
    """Per-class F1 with feature sets side by side for each model."""
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
    log.info("Wrote per-class F1 feature-set comparison → %s", _display_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Batch generation from saved artefacts
# ---------------------------------------------------------------------------


def generate_evaluation_figures(output_dir: Path) -> list[Path]:
    """Generate all scenario evaluation figures from saved CSV/JSON artefacts."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

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
            _display_path(output_dir),
        )

    all_results_disk: dict[str, dict] = {}
    ignored_dirs = {"figures", "per_class_reports", "saved_models"}
    model_dirs = sorted(
        d
        for d in output_dir.iterdir()
        if d.is_dir()
        and d.name not in ignored_dirs
        and (
            (d / "confusion_matrix.csv").exists()
            or (d / "feature_importance.csv").exists()
            or (d / "feature_importance_by_group.csv").exists()
            or any(d.glob("per_class_report_*.json"))
        )
    )

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_figures_dir = model_dir / "figures"
        model_figures_dir.mkdir(exist_ok=True)

        cm_path = model_dir / "confusion_matrix.csv"
        if cm_path.exists():
            try:
                cm_long = pd.read_csv(cm_path)
            except pd.errors.EmptyDataError:
                log.warning("Confusion matrix CSV is empty: %s", _display_path(cm_path))
            except Exception as exc:
                log.warning("Failed to load confusion matrix CSV %s: %s", cm_path.name, exc)
            else:
                required_cols = {"feature_set", "true", "predicted", "count"}
                missing = required_cols - set(cm_long.columns)
                if cm_long.empty:
                    log.warning("Confusion matrix CSV has no rows: %s", _display_path(cm_path))
                elif missing:
                    log.warning(
                        "Confusion matrix CSV %s missing columns: %s",
                        _display_path(cm_path),
                        sorted(missing),
                    )
                else:
                    for fs_name, sub in cm_long.groupby("feature_set", sort=False):
                        try:
                            cm_wide = sub.pivot_table(
                                index="true", columns="predicted", values="count",
                                aggfunc="sum", fill_value=0,
                            )
                            cm_wide.columns.name = None
                            cm_wide.index.name = None
                            human = (
                                f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                                f"{str(fs_name).replace('_', ' ').title()}"
                            )
                            out = model_figures_dir / f"confusion_matrix_{fs_name}.png"
                            plot_confusion_matrix(
                                cm_wide,
                                out,
                                title=f"Confusion matrix — {human}",
                            )
                            generated.append(out)
                        except Exception as exc:
                            log.warning(
                                "Failed to plot confusion matrix for %s / %s: %s",
                                model_name,
                                fs_name,
                                exc,
                                exc_info=log.isEnabledFor(logging.DEBUG),
                            )

        fi_path = model_dir / "feature_importance.csv"
        if fi_path.exists():
            try:
                fi_all = pd.read_csv(fi_path)
                for fs_name, fi_df in fi_all.groupby("feature_set", sort=False):
                    human = (
                        f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                        f"{fs_name.replace('_', ' ').title()}"
                    )
                    out = model_figures_dir / f"feature_importance_{fs_name}.png"
                    plot_feature_importance(
                        fi_df.drop(columns=["feature_set"]),
                        out,
                        title=f"Feature importance — {human}",
                    )
                    generated.append(out)
            except Exception as exc:
                log.warning("Failed to plot feature importance %s: %s", fi_path.name, exc)

        fi_grp_path = model_dir / "feature_importance_by_group.csv"
        if fi_grp_path.exists():
            try:
                fi_grp_all = pd.read_csv(fi_grp_path)
                for fs_name, grp_df in fi_grp_all.groupby("feature_set", sort=False):
                    human = (
                        f"{_MODEL_DISPLAY.get(model_name, model_name)} / "
                        f"{fs_name.replace('_', ' ').title()}"
                    )
                    out = model_figures_dir / f"feature_importance_by_group_{fs_name}.png"
                    plot_sensor_group_contribution(
                        grp_df.drop(columns=["feature_set"]),
                        out,
                        title=f"Feature importance by sensor group — {human}",
                    )
                    generated.append(out)
            except Exception as exc:
                log.warning("Failed to plot feature importance by group %s: %s", fi_grp_path.name, exc)

        model_report_dir = model_dir / "per_class_report"
        report_paths = (
            sorted(model_report_dir.glob("per_class_report_*.json"))
            if model_report_dir.exists()
            else []
        )
        legacy_flat_report_paths = sorted(model_dir.glob("per_class_report_*.json"))
        legacy_root_report_dir = output_dir / "per_class_reports" / model_name
        legacy_root_report_paths = (
            sorted(legacy_root_report_dir.glob("per_class_report_*.json"))
            if legacy_root_report_dir.exists()
            else []
        )
        for pc_path in [*report_paths, *legacy_flat_report_paths, *legacy_root_report_paths]:
            try:
                _, _, fs_name = pc_path.stem.partition("per_class_report_")
                data = json.loads(pc_path.read_text(encoding="utf-8"))
                all_results_disk[f"{fs_name}__{model_name}"] = {"per_class": data}
            except Exception as exc:
                log.warning("Failed to load per-class report %s: %s", pc_path.name, exc)

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

        out = figures_dir / "feature_set_comparison_per_class_f1.png"
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
        _display_path(output_dir),
    )
    return generated


# ---------------------------------------------------------------------------
# Label-grid figures (label_col × min_quality)
# ---------------------------------------------------------------------------


def plot_label_grid_heatmap(
    grid_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
    feature_set: str = "fused",
    model: str | None = None,
) -> Path | None:
    """Heatmap of metric values across the label × quality grid."""
    if grid_df.empty:
        return None

    df = grid_df[grid_df["feature_set"] == feature_set].copy()
    if model is not None:
        df = df[df["model"] == model]
    if df.empty:
        return None

    if model is None:
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

    fig, ax = plt.subplots(
        figsize=(max(5.5, 0.9 * pivot.shape[1] + 3.5), max(3.5, 0.55 * pivot.shape[0] + 1.5))
    )
    vmin, vmax = _label_grid_color_limits(pivot.values)
    im = ax.imshow(pivot.values, cmap=_LABEL_GRID_CMAP, aspect="auto", vmin=vmin, vmax=vmax)

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
                color=_label_grid_text_color(v, vmin, vmax),
            )

    plt.colorbar(im, ax=ax, shrink=0.8, label=metric)
    title_model = "best model" if model is None else _MODEL_DISPLAY.get(model, model)
    ax.set_title(f"{metric} — {feature_set} ({title_model})", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote label-grid heatmap → %s", _display_path(output_path))
    return output_path


def plot_label_grid_quality_grid(
    grid_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
) -> Path | None:
    """Grouped bars comparing feature sets across label schemes and quality filters."""
    if grid_df.empty:
        return None

    present_feature_sets = grid_df["feature_set"].dropna().astype(str).unique().tolist()
    feature_sets = [fs for fs in _LABEL_GRID_QUALITY_FEATURE_SETS if fs in present_feature_sets]
    if not feature_sets:
        log.debug("Label-grid quality grid: no known feature sets; skipping")
        return None

    label_cols = list(dict.fromkeys(grid_df["label_col"].dropna().astype(str)))
    qualities = _ordered_quality_filters(grid_df["min_quality"])
    if not label_cols or not qualities:
        log.debug("Label-grid quality grid: no label columns or quality filters; skipping")
        return None

    df = (
        grid_df.sort_values(metric, ascending=False)
        .groupby(["label_col", "min_quality", "feature_set"], as_index=False, sort=False)
        .first()
    )
    vmin, vmax = _label_grid_color_limits(df[metric].to_numpy(dtype=float))

    n_rows = min(2, len(qualities))
    n_cols = int(np.ceil(len(qualities) / n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(10.5, 1.05 * len(label_cols) * n_cols + 2.0), max(4.2, 3.2 * n_rows)),
        layout="constrained",
        squeeze=False,
    )
    axes_flat = axes.ravel()

    plotted = False
    x = np.arange(len(label_cols))
    width = min(0.18, 0.82 / max(1, len(feature_sets)))
    offsets = np.linspace(
        -width * (len(feature_sets) - 1) / 2,
        width * (len(feature_sets) - 1) / 2,
        len(feature_sets),
    )

    label_display = [label.replace("scenario_label_", "") for label in label_cols]

    for panel_idx, (ax, quality) in enumerate(zip(axes_flat, qualities)):
        sub = df[df["min_quality"].astype(str) == quality]
        pivot = sub.pivot_table(
            index="label_col",
            columns="feature_set",
            values=metric,
            aggfunc="max",
        ).reindex(index=label_cols, columns=feature_sets)
        if pivot.empty:
            ax.set_visible(False)
            continue

        for fs, offset in zip(feature_sets, offsets):
            values = pd.to_numeric(pivot[fs], errors="coerce").to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=_FEATURE_SET_COLORS.get(fs, "#7f8c8d"),
                edgecolor="white",
                linewidth=0.6,
                label=_FS_DISPLAY.get(fs, fs).replace("\n", " "),
            )
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90 if len(label_cols) > 6 else 0,
                )
        plotted = True
        ax.set_xticks(x)
        ax.set_xticklabels(label_display, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        if len(qualities) > 1:
            ax.set_title(f"min_quality = {quality}", fontsize=10)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes_flat[len(qualities):]:
        ax.set_visible(False)

    if not plotted:
        plt.close(fig)
        log.debug("Label-grid feature-set comparison: no plottable cells; skipping %s", output_path.name)
        return None

    visible_axes = [ax for ax in axes_flat if ax.get_visible()]
    handles, labels = visible_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(feature_sets), fontsize=8, frameon=False)
    fig.suptitle(f"{metric}: feature-set comparison across label scheme × quality filter", fontsize=11)
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    log.info("Wrote label-grid feature-set comparison → %s", _display_path(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Misclassified-window overlays (binary objectives)
# ---------------------------------------------------------------------------

_FP_COLOR = "#d62728"
_FN_COLOR = "#ff7f0e"
_MISCLS_ALPHA = 0.45
_MISCLS_EDGE_LW = 1.2


def _load_section_signal(sec_dir: Path, sensor: str) -> pd.DataFrame | None:
    derived = sec_dir / "derived" / f"{sensor}_signals.csv"
    if derived.exists():
        df = read_csv(derived)
        if "timestamp" in df.columns:
            return df

    calibrated = sec_dir / "calibrated" / f"{sensor}.csv"
    if calibrated.exists():
        df = read_csv(calibrated)
        if "timestamp" in df.columns:
            for col, axes in (
                ("acc_norm", ("ax", "ay", "az")),
                ("gyro_norm", ("gx", "gy", "gz")),
            ):
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
    for lr in labels:
        if not lr.label:
            continue
        x0 = (lr.start_ms - t0_ms) / 1000.0
        x1 = (lr.end_ms - t0_ms) / 1000.0
        if x1 <= x0:
            continue
        color = colors.get(lr.label, "#90A4AE")
        ax.axvspan(x0, x1, facecolor=color, edgecolor="none", alpha=0.85, zorder=1)
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
    positive_class: str = "riding",
    negative_class: str = "non_riding",
) -> Path | None:
    """IMU signal overlay with highlighted FP/FN windows for one section."""
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

    n_fp, n_fn = _draw_misclassified_spans(signal_axes, ax_anno, miscls_rows, t0_ms)

    legend_handles = [
        mpatches.Patch(color=_FP_COLOR, alpha=_MISCLS_ALPHA,
                       label=f"FP — predicted {positive_class} (n={n_fp})"),
        mpatches.Patch(color=_FN_COLOR, alpha=_MISCLS_ALPHA,
                       label=f"FN — predicted {negative_class} (n={n_fn})"),
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
    log.info("Wrote misclassified overlay → %s", _display_path(output_path))
    return output_path


def plot_misclassified_overlay(
    miscls_csv: Path,
    output_dir: Path,
    *,
    title_suffix: str = "",
    positive_class: str = "riding",
    negative_class: str = "non_riding",
) -> list[Path]:
    """Generate per-section misclassified window overlays from a misclassified CSV."""
    from common.paths import project_relative_path

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
                str(sec_id), rows, out,
                title_suffix=title_suffix,
                positive_class=positive_class,
                negative_class=negative_class,
            )
        except Exception as exc:
            log.warning("Misclassified overlay failed for %s: %s", sec_id, exc)
            continue
        if saved is not None:
            written.append(saved)
    log.info("Misclassified overlays: %d sections → %s", len(written), _display_path(output_dir))
    return written


__all__ = [
    "generate_evaluation_figures",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_imu_contribution",
    "plot_label_grid_heatmap",
    "plot_label_grid_quality_grid",
    "plot_misclassified_overlay",
    "plot_misclassified_overlay_for_section",
    "plot_model_comparison",
    "plot_per_class_f1",
    "plot_per_class_f1_feature_set_comparison",
    "plot_permutation_importance",
    "plot_sensor_group_contribution",
]
