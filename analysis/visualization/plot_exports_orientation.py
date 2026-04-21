"""Per-section orientation export audit figures."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._exports_common import (
    ALL_ORIENTATION_METHODS,
    DPI,
    ORIENTATION_METHOD_COLORS,
    QUALITY_COLORS,
    short_section,
)
from visualization._utils import SENSOR_COLORS, SENSORS, save_figure

log = logging.getLogger(__name__)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _residual_values(df: pd.DataFrame, col: str) -> pd.Series:
    vals = _numeric(df[col])
    return vals.replace([np.inf, -np.inf], np.nan)


def _available_orientation_methods(df: pd.DataFrame) -> list[str]:
    methods: list[str] = []
    for method in ALL_ORIENTATION_METHODS:
        if any(f"{sensor}_{method}_residual_ms2" in df.columns for sensor in SENSORS):
            methods.append(method)
    extra = sorted({
        col.split("_", 2)[1]
        for col in df.columns
        if col.startswith(tuple(f"{sensor}_" for sensor in SENSORS))
        and col.endswith("_residual_ms2")
        and not col.endswith("_selected_residual_ms2")
        and col.split("_", 2)[1] not in methods
    })
    return methods + extra


def plot_orientation_method_selection(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "orientation_method_selection.png"
    if df.empty or "selected_method" not in df.columns:
        return None

    counts = df["selected_method"].fillna("unknown").replace("", "unknown").value_counts()
    if counts.empty:
        return None

    methods = counts.index.tolist()
    values = counts.to_numpy(dtype=float)
    colors = [ORIENTATION_METHOD_COLORS.get(method, "#95a5a6") for method in methods]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.55 * len(methods) + 1.5)))
    bars = ax.barh(methods, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, value in zip(bars, values, strict=False):
        ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=9)
    ax.set_xlabel("Sections")
    ax.set_title("Selected orientation method")
    ax.set_xlim(0, max(values) * 1.25 + 0.5)
    ax.grid(axis="x", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_orientation_quality_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "orientation_quality_overview.png"
    if df.empty or "section_id" not in df.columns:
        return None

    quality_cols = [f"{sensor}_quality" for sensor in SENSORS if f"{sensor}_quality" in df.columns]
    if not quality_cols:
        return None

    sections = df["section_id"].astype(str).tolist()
    x = np.arange(len(sections))
    width = 0.8 / len(quality_cols)

    fig, ax = plt.subplots(figsize=(max(8, len(sections) * 0.55 + 2), 4))
    for idx, col in enumerate(quality_cols):
        qualities = df[col].fillna("").astype(str).tolist()
        colors = [QUALITY_COLORS.get(q, "#95a5a6") for q in qualities]
        offset = (idx - (len(quality_cols) - 1) / 2) * width
        hatch = "" if idx == 0 else "///"
        ax.bar(
            x + offset,
            [1] * len(sections),
            width * 0.92,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([short_section(section) for section in sections], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_title("Orientation quality per section")

    quality_handles = [
        mpatches.Patch(color=QUALITY_COLORS.get(q, "#95a5a6"), label=q)
        for q in ("good", "marginal", "poor")
    ]
    sensor_handles = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="" if idx == 0 else "///", label=col.removesuffix("_quality"))
        for idx, col in enumerate(quality_cols)
    ]
    ax.legend(handles=quality_handles + sensor_handles, fontsize=8, ncol=2, framealpha=0.85)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_orientation_selected_residuals(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "orientation_selected_residuals.png"
    if df.empty or "section_id" not in df.columns:
        return None

    residual_cols = [f"{sensor}_selected_residual_ms2" for sensor in SENSORS if f"{sensor}_selected_residual_ms2" in df.columns]
    if not residual_cols:
        return None

    sections = df["section_id"].astype(str).tolist()
    x = np.arange(len(sections))
    width = 0.8 / len(residual_cols)
    finite_max = 0.0
    for col in residual_cols:
        vals = _residual_values(df, col).dropna()
        if not vals.empty:
            finite_max = max(finite_max, float(vals.max()))
    inf_height = max(1.0, finite_max * 1.15)

    fig, ax = plt.subplots(figsize=(max(8, len(sections) * 0.6 + 2), 4.5))
    for idx, col in enumerate(residual_cols):
        sensor = col.removesuffix("_selected_residual_ms2")
        raw = _numeric(df[col])
        finite = raw.replace([np.inf, -np.inf], np.nan)
        vals = finite.fillna(0.0).clip(lower=0.0).to_numpy(dtype=float).copy()
        inf_mask = np.isinf(raw.to_numpy(dtype=float))
        vals[inf_mask] = inf_height
        offset = (idx - (len(residual_cols) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals,
            width * 0.92,
            color=SENSOR_COLORS.get(sensor, "#95a5a6"),
            label=sensor,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
        for bar, is_inf in zip(bars, inf_mask, strict=False):
            if is_inf:
                ax.text(bar.get_x() + bar.get_width() / 2, inf_height, "inf", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([short_section(section) for section in sections], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Gravity residual (m/s2)")
    ax.set_title("Selected orientation residual per section")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_orientation_method_residual_heatmap(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "orientation_method_residual_heatmap.png"
    if df.empty or "section_id" not in df.columns:
        return None

    methods = _available_orientation_methods(df)
    if not methods:
        return None

    sections = df["section_id"].astype(str).tolist()
    fig, axes = plt.subplots(
        1,
        len(SENSORS),
        figsize=(max(7, len(methods) * 1.3 + 2) * len(SENSORS) / 2, max(4, len(sections) * 0.38 + 1.8)),
        squeeze=False,
        sharey=True,
    )

    any_panel = False
    image = None
    for ax, sensor in zip(axes[0], SENSORS, strict=False):
        cols = [f"{sensor}_{method}_residual_ms2" for method in methods]
        available_cols = [col for col in cols if col in df.columns]
        if not available_cols:
            ax.axis("off")
            continue

        mat = np.full((len(df), len(methods)), np.nan, dtype=float)
        for method_idx, method in enumerate(methods):
            col = f"{sensor}_{method}_residual_ms2"
            if col in df.columns:
                mat[:, method_idx] = _residual_values(df, col).to_numpy(dtype=float)

        finite = mat[np.isfinite(mat)]
        vmax = max(1.0, float(np.nanpercentile(finite, 90))) if finite.size else 1.0
        image = ax.imshow(np.clip(mat, 0, vmax), aspect="auto", cmap="viridis_r", vmin=0, vmax=vmax)
        any_panel = True

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(sections)))
        ax.set_yticklabels([short_section(section) for section in sections], fontsize=7)
        ax.set_title(sensor)

        if "selected_method" in df.columns:
            for row_idx, selected in enumerate(df["selected_method"].fillna("").astype(str)):
                if selected in methods:
                    col_idx = methods.index(selected)
                    ax.add_patch(plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.2))

    if not any_panel:
        plt.close(fig)
        return None

    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Gravity residual (m/s2)", shrink=0.8)
    fig.suptitle("Orientation method residuals (box = selected method)", fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.18, top=0.9, wspace=0.18)
    return save_figure(fig, out_path, dpi=DPI)


def plot_orientation_method_quality_counts(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "orientation_method_quality_counts.png"
    if df.empty:
        return None

    methods = _available_orientation_methods(df)
    quality_order = ["good", "marginal", "poor"]
    labels: list[str] = []
    counts: dict[str, list[int]] = {q: [] for q in quality_order}

    for sensor in SENSORS:
        for method in methods:
            col = f"{sensor}_{method}_quality"
            if col not in df.columns:
                continue
            values = df[col].fillna("").astype(str)
            if values.empty:
                continue
            labels.append(f"{sensor}\n{method}")
            for quality in quality_order:
                counts[quality].append(int((values == quality).sum()))

    if not labels:
        return None

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65 + 2), 4))
    for quality in quality_order:
        vals = np.asarray(counts[quality], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=QUALITY_COLORS.get(quality, "#95a5a6"), label=quality, edgecolor="white", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sections")
    ax.set_title("Orientation quality counts by method and sensor")
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def run_orientation_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if df.empty:
        log.warning("Orientation stats DataFrame is empty; skipping orientation EDA")
        return []
    figures_dir = Path(output_dir) / "figures" / "orientation"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for result in (
        plot_orientation_method_selection(df, figures_dir),
        plot_orientation_quality_overview(df, figures_dir),
        plot_orientation_selected_residuals(df, figures_dir),
        plot_orientation_method_residual_heatmap(df, figures_dir),
        plot_orientation_method_quality_counts(df, figures_dir),
    ):
        if result is not None:
            generated.append(result)

    log.info("Orientation EDA complete: %d figures", len(generated))
    return generated
