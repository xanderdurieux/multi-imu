"""Evaluation figures for event-contrast and two-stage event runs."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import read_csv
from visualization._eval_common import (
    _DPI,
    _FEATURE_SET_COLORS,
    _FS_DISPLAY,
    _MODEL_DISPLAY,
    _SENSOR_GROUP_COLORS,
    _display_path,
    _ordered_feature_sets,
    _ordered_models,
    plot_imu_contribution,
    plot_sensor_group_contribution,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event-contrast figures
# ---------------------------------------------------------------------------


def plot_event_contrast_metric_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
    title: str | None = None,
) -> Path | None:
    """Grid of bars (contrast × model) comparing feature-set performance."""
    if metrics_df.empty or metric not in metrics_df.columns:
        return None

    contrasts = sorted(metrics_df["contrast"].dropna().unique().tolist())
    models = _ordered_models(metrics_df["model"].dropna().unique().tolist())
    feature_sets = _ordered_feature_sets(metrics_df["feature_set"].dropna().unique().tolist())
    if not contrasts or not models or not feature_sets:
        return None

    fig, axes = plt.subplots(
        len(contrasts),
        len(models),
        figsize=(max(5.0 * len(models), 6.0), max(3.2 * len(contrasts), 3.8)),
        sharey=True,
        squeeze=False,
    )

    for row_idx, contrast in enumerate(contrasts):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            sub = metrics_df[
                (metrics_df["contrast"] == contrast) & (metrics_df["model"] == model)
            ]
            vals = []
            labels = []
            colors = []
            for fs in feature_sets:
                row = sub[sub["feature_set"] == fs]
                if row.empty:
                    continue
                vals.append(float(row[metric].iloc[0]))
                labels.append(_FS_DISPLAY.get(fs, fs.replace("_", " ").title()))
                colors.append(_FEATURE_SET_COLORS.get(fs, "#7f8c8d"))

            ax.bar(labels, vals, color=colors, edgecolor="white", alpha=0.9)
            ax.set_ylim(0, 1.08)
            ax.set_title(
                f"{contrast.replace('_', ' ')}\n{_MODEL_DISPLAY.get(model, model)}",
                fontsize=9,
            )
            ax.grid(axis="y", alpha=0.3, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelrotation=20, labelsize=8)
            if col_idx == 0:
                ax.set_ylabel(metric.replace("_", " ").title())
            for idx, val in enumerate(vals):
                if np.isfinite(val):
                    ax.text(idx, val + 0.015, f"{val:.2f}", ha="center", fontsize=7)

    fig.suptitle(title or f"Event contrast {metric.replace('_', ' ')}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote event-contrast metric plot → %s", _display_path(output_path))
    return output_path


def plot_event_contrast_support(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Event contrast support",
) -> Path | None:
    """Bar chart of normal vs critical window counts per contrast."""
    if metrics_df.empty:
        return None
    cols = {"contrast", "support_normal", "support_critical"}
    if not cols.issubset(metrics_df.columns):
        return None

    support = (
        metrics_df[["contrast", "normal_label", "critical_label", "support_normal", "support_critical"]]
        .drop_duplicates("contrast")
        .sort_values("contrast")
    )
    if support.empty:
        return None

    x = np.arange(len(support))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, 2.1 * len(support)), 4.0))
    ax.bar(x - width / 2, support["support_normal"], width, color="#3498db", label="normal")
    ax.bar(x + width / 2, support["support_critical"], width, color="#e74c3c", label="critical")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", " ") for c in support["contrast"]],
        rotation=20,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Windows")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote event-contrast support plot → %s", _display_path(output_path))
    return output_path


def plot_event_contrast_stability(
    stability_df: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 20,
    title: str = "Event contrast feature stability",
) -> Path | None:
    """Horizontal bar chart of the most stable event-contrast features."""
    if stability_df.empty or "rank_mean" not in stability_df.columns:
        return None

    df = stability_df.copy()
    df = df.sort_values(["top_k_frequency", "rank_mean"], ascending=[False, True]).head(top_n)
    if df.empty:
        return None
    df = df.iloc[::-1].reset_index(drop=True)
    labels = [
        f"{row.feature}\n{row.contrast} / {row.feature_set}"
        for row in df.itertuples(index=False)
    ]
    colors = [
        _SENSOR_GROUP_COLORS.get(g, _SENSOR_GROUP_COLORS["other"])
        for g in df["sensor_group"]
    ]

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.48 * len(df) + 1.0)))
    ax.barh(labels, df["top_k_frequency"], color=colors, edgecolor="white")
    ax.set_xlabel("Top-k frequency across folds")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote event-contrast stability plot → %s", _display_path(output_path))
    return output_path


def generate_event_contrast_figures(output_dir: Path) -> list[Path]:
    """Generate all event-contrast figures from saved CSV artefacts."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    metrics_path = output_dir / "event_contrast_metrics.csv"
    if metrics_path.exists():
        metrics_df = read_csv(metrics_path)
        for metric in ("macro_f1", "balanced_accuracy", "roc_auc"):
            out = figures_dir / f"event_contrast_{metric}.png"
            result = plot_event_contrast_metric_comparison(metrics_df, out, metric=metric)
            if result:
                generated.append(result)
        result = plot_event_contrast_support(
            metrics_df,
            figures_dir / "event_contrast_support.png",
        )
        if result:
            generated.append(result)

    imu_path = output_dir / "event_contrast_imu_contribution.csv"
    if imu_path.exists():
        imu_df = read_csv(imu_path)
        for metric in ("macro_f1", "balanced_accuracy"):
            result = plot_imu_contribution(
                imu_df,
                figures_dir / f"event_contrast_imu_{metric}.png",
                metric=metric,
                title=f"Event contrast IMU contribution — {metric.replace('_', ' ')}",
            )
            if result:
                generated.append(result)

    stability_path = output_dir / "event_contrast_feature_stability.csv"
    if stability_path.exists():
        stability_df = read_csv(stability_path)
        result = plot_event_contrast_stability(
            stability_df,
            figures_dir / "event_contrast_feature_stability.png",
        )
        if result:
            generated.append(result)

    group_path = output_dir / "event_contrast_feature_importance_by_group.csv"
    if group_path.exists():
        grouped_df = read_csv(group_path)
        for (contrast, model, fs), sub in grouped_df.groupby(
            ["contrast", "model", "feature_set"], sort=False
        ):
            stem = f"{contrast}__{model}__{fs}"
            result = plot_sensor_group_contribution(
                sub.drop(columns=["contrast", "model", "feature_set"]),
                figures_dir / f"event_contrast_sensor_contribution_{stem}.png",
                title=f"Event contrast sensor contribution — {contrast.replace('_', ' ')} / {model} / {fs}",
            )
            if result:
                generated.append(result)

    log.info(
        "Event contrast figures: %d written to %s",
        len(generated),
        _display_path(figures_dir),
    )
    return generated


# ---------------------------------------------------------------------------
# Two-stage event figures
# ---------------------------------------------------------------------------


def plot_two_stage_metric_comparison(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    metrics: tuple[str, ...] = (
        "detector_recall",
        "oracle_contrast_macro_f1",
        "predicted_gated_contrast_macro_f1",
        "end_to_end_critical_recall",
    ),
    title: str = "Two-stage event performance",
) -> Path | None:
    """Stage-wise metric bars grouped by task and model."""
    if metrics_df.empty:
        return None
    present_metrics = [m for m in metrics if m in metrics_df.columns]
    if not present_metrics:
        return None

    tasks = sorted(metrics_df["task"].dropna().unique().tolist())
    models = _ordered_models(metrics_df["model"].dropna().unique().tolist())
    feature_sets = _ordered_feature_sets(metrics_df["feature_set"].dropna().unique().tolist())
    if not tasks or not models or not feature_sets:
        return None

    fig, axes = plt.subplots(
        len(tasks),
        len(models),
        figsize=(max(6.2 * len(models), 7.0), max(3.8 * len(tasks), 4.2)),
        sharey=True,
        squeeze=False,
    )
    x = np.arange(len(present_metrics))
    width = min(0.18, 0.8 / max(len(feature_sets), 1))
    offsets = np.linspace(
        -(len(feature_sets) - 1) / 2,
        (len(feature_sets) - 1) / 2,
        len(feature_sets),
    ) * width

    for row_idx, task in enumerate(tasks):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            sub = metrics_df[(metrics_df["task"] == task) & (metrics_df["model"] == model)]
            for fs, offset in zip(feature_sets, offsets):
                fs_row = sub[sub["feature_set"] == fs]
                if fs_row.empty:
                    continue
                vals = [float(fs_row[m].iloc[0]) for m in present_metrics]
                ax.bar(
                    x + offset,
                    vals,
                    width,
                    label=_FS_DISPLAY.get(fs, fs),
                    color=_FEATURE_SET_COLORS.get(fs, "#7f8c8d"),
                    edgecolor="white",
                    alpha=0.9,
                )
            ax.set_ylim(0, 1.08)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [m.replace("_", "\n") for m in present_metrics],
                fontsize=7,
            )
            ax.set_title(
                f"{task.replace('_', ' ')}\n{_MODEL_DISPLAY.get(model, model)}",
                fontsize=9,
            )
            ax.grid(axis="y", alpha=0.3, lw=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col_idx == 0:
                ax.set_ylabel("Score")

    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_text,
            loc="lower center",
            ncol=len(feature_sets),
            fontsize=8,
            framealpha=0.85,
            bbox_to_anchor=(0.5, -0.02),
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote two-stage metric plot → %s", _display_path(output_path))
    return output_path


def plot_two_stage_routing(
    fold_scores_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Two-stage routing outcomes",
) -> Path | None:
    """Stacked bar chart of routed true events, false positives, and misses."""
    if fold_scores_df.empty:
        return None
    needed = {
        "task",
        "feature_set",
        "model",
        "n_routed_true_event",
        "n_routed_false_positive",
        "n_detector_missed_events",
    }
    if not needed.issubset(fold_scores_df.columns):
        return None

    grouped = (
        fold_scores_df.groupby(["task", "feature_set", "model"], as_index=False)[
            ["n_routed_true_event", "n_routed_false_positive", "n_detector_missed_events"]
        ]
        .sum()
        .sort_values(["task", "model", "feature_set"])
    )
    if grouped.empty:
        return None

    labels = [
        f"{r.task}\n{_MODEL_DISPLAY.get(r.model, r.model)}\n{_FS_DISPLAY.get(r.feature_set, r.feature_set)}"
        for r in grouped.itertuples(index=False)
    ]
    x = np.arange(len(grouped))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.72 * len(grouped) + 2.0), 4.6))
    bottom = np.zeros(len(grouped), dtype=float)
    stacks = [
        ("n_routed_true_event", "routed true event", "#2ecc71"),
        ("n_routed_false_positive", "routed false positive", "#f39c12"),
        ("n_detector_missed_events", "missed true event", "#e74c3c"),
    ]
    for col, label, color in stacks:
        vals = grouped[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="white")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Windows")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote two-stage routing plot → %s", _display_path(output_path))
    return output_path


def plot_two_stage_candidates_by_label(
    candidates_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Two-stage candidate labels",
) -> Path | None:
    """Bar chart of candidate-window counts by task and final label."""
    if candidates_df.empty or "final_label" not in candidates_df.columns:
        return None
    grouped = (
        candidates_df.groupby(["task", "final_label"], as_index=False)
        .size()
        .sort_values(["task", "final_label"])
    )
    if grouped.empty:
        return None
    pivot = grouped.pivot(index="task", columns="final_label", values="size").fillna(0)
    labels = pivot.index.tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(pivot.columns), 1)
    fig, ax = plt.subplots(figsize=(max(6.5, 2.2 * len(labels)), 4.0))
    for idx, col in enumerate(pivot.columns):
        offset = (idx - (len(pivot.columns) - 1) / 2) * width
        ax.bar(x + offset, pivot[col], width, label=col, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ") for label in labels], fontsize=8)
    ax.set_ylabel("Candidate windows")
    ax.set_title(title)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote two-stage candidate-label plot → %s", _display_path(output_path))
    return output_path


def generate_two_stage_event_figures(output_dir: Path) -> list[Path]:
    """Generate all two-stage event figures from saved CSV artefacts."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    metrics_path = output_dir / "two_stage_event_metrics.csv"
    if metrics_path.exists():
        metrics_df = read_csv(metrics_path)
        result = plot_two_stage_metric_comparison(
            metrics_df,
            figures_dir / "two_stage_event_performance.png",
        )
        if result:
            generated.append(result)

    folds_path = output_dir / "two_stage_event_fold_scores.csv"
    if folds_path.exists():
        folds_df = read_csv(folds_path)
        result = plot_two_stage_routing(
            folds_df,
            figures_dir / "two_stage_event_routing.png",
        )
        if result:
            generated.append(result)

    candidates_path = output_dir / "two_stage_event_candidates.csv"
    if candidates_path.exists():
        candidates_df = read_csv(candidates_path)
        result = plot_two_stage_candidates_by_label(
            candidates_df,
            figures_dir / "two_stage_event_candidates_by_label.png",
        )
        if result:
            generated.append(result)

    imu_path = output_dir / "two_stage_event_imu_contribution.csv"
    if imu_path.exists():
        imu_df = read_csv(imu_path)
        for metric in ("detector_macro_f1", "end_to_end_critical_f2", "end_to_end_critical_recall"):
            result = plot_imu_contribution(
                imu_df,
                figures_dir / f"two_stage_event_imu_{metric}.png",
                metric=metric,
                title=f"Two-stage IMU contribution — {metric.replace('_', ' ')}",
            )
            if result:
                generated.append(result)

    log.info(
        "Two-stage event figures: %d written to %s",
        len(generated),
        _display_path(figures_dir),
    )
    return generated


__all__ = [
    "generate_event_contrast_figures",
    "generate_two_stage_event_figures",
    "plot_event_contrast_metric_comparison",
    "plot_event_contrast_stability",
    "plot_event_contrast_support",
    "plot_two_stage_candidates_by_label",
    "plot_two_stage_metric_comparison",
    "plot_two_stage_routing",
]
