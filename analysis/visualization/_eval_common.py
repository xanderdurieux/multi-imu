"""Shared constants and plot helpers for evaluation figures."""

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

# ---------------------------------------------------------------------------
# Display names and color palettes
# ---------------------------------------------------------------------------

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

_FS_DISPLAY: dict[str, str] = {
    "bike": "Bike only",
    "rider": "Rider only",
    "fused_no_cross": "Fused\n(no cross)",
    "fused": "Fused",
}

# Simple → rich canonical order for comparison figures.
_FS_ORDER: list[str] = ["bike", "rider", "fused_no_cross", "fused"]

# Feature-group prefix → bar color used in importance/permutation plots.
_GROUP_COLORS: dict[str, str] = {
    "bike_": "#3498db",
    "rider_": "#e74c3c",
    "cross_": "#2ecc71",
    "events_": "#f39c12",
    "spectral_": "#9b59b6",
    "orientation_": "#1abc9c",
}

_SENSOR_GROUP_COLORS: dict[str, str] = {
    "bike": "#3498db",
    "rider": "#e74c3c",
    "cross": "#2ecc71",
    "other": "#95a5a6",
}

# ---------------------------------------------------------------------------
# Ordering and display helpers
# ---------------------------------------------------------------------------


def _display_path(path: Path | str) -> str:
    try:
        return project_relative_path(path)
    except ValueError:
        return str(Path(path))


def _group_color(feature: str) -> str:
    for prefix, color in _GROUP_COLORS.items():
        if feature.startswith(prefix):
            return color
    return "#95a5a6"


def _ordered_feature_sets(values) -> list[str]:
    """Return feature sets in canonical simple-to-rich order."""
    present = list(dict.fromkeys(values))
    return [fs for fs in _FS_ORDER if fs in present] + sorted(
        fs for fs in present if fs not in _FS_ORDER
    )


def _ordered_models(values) -> list[str]:
    """Return models in registry display order."""
    present = list(dict.fromkeys(values))
    registry = list(_MODEL_COLORS)
    return sorted(
        present,
        key=lambda model: (
            registry.index(model) if model in registry else len(registry),
            model,
        ),
    )


# ---------------------------------------------------------------------------
# Shared evaluation plots (used by both scenario and event evaluators)
# ---------------------------------------------------------------------------


def plot_permutation_importance(
    perm_df: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 25,
    title: str = "Permutation importance",
) -> Path | None:
    """Plot permutation importance as a horizontal bar chart."""
    if perm_df.empty:
        log.warning("Empty permutation importance DataFrame; skipping plot")
        return None

    df = perm_df.head(top_n).copy()
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
    log.debug("Wrote permutation importance → %s", _display_path(output_path))
    return output_path


def plot_sensor_group_contribution(
    grouped_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str = "Sensor-group contribution",
) -> Path | None:
    """Plot total permutation importance aggregated by sensor group."""
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
    log.debug("Wrote sensor-group contribution → %s", _display_path(output_path))
    return output_path


def plot_imu_contribution(
    contribution_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "macro_f1",
    title: str | None = None,
) -> Path | None:
    """Plot paired feature-set delta (Δ metric) for each sensor-addition pair."""
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
            ax.text(xi, label_y, f"{mean:+.3f}{star}", ha="center", va="bottom", fontsize=6.5)

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
    log.debug("Wrote IMU contribution → %s", _display_path(output_path))
    return output_path
