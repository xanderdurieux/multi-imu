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
    DPI,
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


def plot_orientation_residuals(df: pd.DataFrame, output_dir: Path) -> Path | None:
    out_path = output_dir / "orientation_residuals.png"
    if df.empty or "section_id" not in df.columns:
        return None

    residual_cols = [f"{sensor}_residual_ms2" for sensor in SENSORS if f"{sensor}_residual_ms2" in df.columns]
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
        sensor = col.removesuffix("_residual_ms2")
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
    ax.set_title("Orientation residual per section (Mahony)")
    ax.grid(axis="y", alpha=0.3, lw=0.5)

    # Draw horizontal quality-threshold lines and annotate them.
    thresholds = ((0.5, "good"), (1.5, "marginal"))
    # x position for labels: just to the right of the last bar group
    x_text = float(x[-1]) + 0.5 if len(x) > 0 else 0.5
    for val, q in thresholds:
        ax.axhline(val, color=QUALITY_COLORS.get(q, "#888888"), linestyle="--", linewidth=0.8, alpha=0.9)
        ax.text(x_text, val, f" {q} ({val})", va="center", ha="left", fontsize=7, color=QUALITY_COLORS.get(q, "#444444"))

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
        plot_orientation_quality_overview(df, figures_dir),
        plot_orientation_residuals(df, figures_dir),
    ):
        if result is not None:
            generated.append(result)

    log.info("Orientation EDA complete: %d figures", len(generated))
    return generated
