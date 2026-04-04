"""Thesis-ready evaluation figures: confusion matrices, feature importance, model comparison."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import project_relative_path

log = logging.getLogger(__name__)

_DPI = 150

_MODEL_DISPLAY: dict[str, str] = {
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "logistic_regression": "Logistic Regression",
}

_MODEL_COLORS: dict[str, str] = {
    "random_forest": "#3498db",
    "gradient_boosting": "#2ecc71",
    "logistic_regression": "#e74c3c",
}

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
    """Annotated heatmap of a square confusion matrix DataFrame.

    Parameters
    ----------
    cm_df:
        Square DataFrame where index = true labels, columns = predicted labels.
    output_path:
        Where to save the PNG.
    normalize:
        Row-normalize so cells show recall per class.
    """
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
    """Horizontal bar chart of top-N features by importance, color-coded by group.

    Parameters
    ----------
    fi_df:
        DataFrame with columns ``feature`` and ``importance``.
    top_n:
        Maximum number of features to display.
    """
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
    """Grouped bar chart comparing accuracy and macro-F1 across feature sets and models.

    Parameters
    ----------
    metrics_df:
        DataFrame with columns ``feature_set``, ``model``, ``accuracy``, ``macro_f1``.
    """
    if metrics_df.empty:
        log.warning("Empty metrics DataFrame; skipping model comparison plot")
        return output_path

    feature_sets = sorted(metrics_df["feature_set"].unique())
    models = metrics_df["model"].unique().tolist()
    n_models = len(models)

    x = np.arange(len(feature_sets))
    width = 0.20
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)

    for ax, (metric_col, ylabel) in zip(
        axes,
        [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1")],
    ):
        for model, offset in zip(models, offsets):
            vals = []
            for fs in feature_sets:
                row = metrics_df[
                    (metrics_df["feature_set"] == fs) & (metrics_df["model"] == model)
                ]
                vals.append(float(row[metric_col].iloc[0]) if not row.empty else 0.0)

            color = _MODEL_COLORS.get(model, "#7f8c8d")
            bars = ax.bar(
                x + offset,
                vals,
                width,
                label=_MODEL_DISPLAY.get(model, model),
                color=color,
                alpha=0.85,
                edgecolor="white",
            )

            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.5,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([fs.capitalize() for fs in feature_sets], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.15)
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
    """Grouped bar chart of per-class F1-score for a given feature set.

    Parameters
    ----------
    metrics_df:
        Summary metrics table (not used directly, only for model filtering).
    all_results:
        Raw result dict from :func:`evaluation.experiments.run_evaluation`.
        Keys are ``<feature_set>__<model>``, values contain ``per_class`` dict.
    classes:
        Ordered class labels.
    feature_set:
        Which feature set to visualize.
    """
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


# ---------------------------------------------------------------------------
# Batch generation from saved artefacts
# ---------------------------------------------------------------------------

def generate_evaluation_figures(output_dir: Path) -> list[Path]:
    """Read evaluation CSV artefacts and generate all thesis figures.

    Reads from *output_dir*:
    - ``metrics_table.csv``
    - ``confusion_matrix_<set>_<model>.csv``
    - ``feature_importance_<model>_<set>.csv``

    Figures are written to ``output_dir/figures/``.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    # ---- model comparison ----
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

    # ---- confusion matrices ----
    for cm_path in sorted(output_dir.glob("confusion_matrix_*.csv")):
        try:
            cm_df = pd.read_csv(cm_path, index_col=0)
            stem = cm_path.stem
            _, _, rest = stem.partition("confusion_matrix_")
            human = rest.replace("_", " / ").title()
            out = figures_dir / f"{stem}.png"
            plot_confusion_matrix(cm_df, out, title=f"Confusion matrix — {human}")
            generated.append(out)
        except Exception as exc:
            log.warning("Failed to plot confusion matrix %s: %s", cm_path.name, exc)

    # ---- feature importances ----
    for fi_path in sorted(output_dir.glob("feature_importance_*.csv")):
        try:
            fi_df = pd.read_csv(fi_path)
            stem = fi_path.stem
            _, _, rest = stem.partition("feature_importance_")
            human = rest.replace("_", " / ").title()
            out = figures_dir / f"{stem}.png"
            plot_feature_importance(fi_df, out, title=f"Feature importance — {human}")
            generated.append(out)
        except Exception as exc:
            log.warning("Failed to plot feature importance %s: %s", fi_path.name, exc)

    log.info(
        "Evaluation figures: %d written to %s",
        len(generated),
        project_relative_path(figures_dir),
    )
    return generated
