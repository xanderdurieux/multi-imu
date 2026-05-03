"""Per-section calibration audit figures (quality, biases, scales, residuals)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization._exports_common import DPI, QUALITY_COLORS, short_section
from visualization._utils import SENSOR_COLORS, SENSORS, save_figure

log = logging.getLogger(__name__)


def plot_calibration_quality_overview(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot calibration quality overview."""
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
    for i, sensor in enumerate(SENSORS):
        col = f"{sensor}_quality"
        if col not in df.columns:
            continue
        qualities = df[col].fillna("").tolist()
        colors = [QUALITY_COLORS.get(q, "#95a5a6") for q in qualities]
        offset = (i - 0.5) * width
        ax.bar(x + offset, [1] * n, width, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([])
    ax.set_title("Calibration quality per section")
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_gravity_residuals(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot gravity residuals."""
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
    for i, sensor in enumerate(SENSORS):
        col = f"{sensor}_gravity_residual_ms2"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).tolist()
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=SENSOR_COLORS[sensor], label=sensor, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Gravity residual (m/s²)")
    ax.set_title("Calibration gravity residual per section")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


def plot_sensor_biases(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot sensor biases."""
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
        fig, axes = plt.subplots(1, len(SENSORS), figsize=(max(8, n * 0.45 + 2) * len(SENSORS) / 2, 4), squeeze=False)
        for col_idx, sensor in enumerate(SENSORS):
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
            ax.set_xticklabels([short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{sensor} — {bias_type.replace('_', ' ')}")
            ax.legend(fontsize=8, framealpha=0.8)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.grid(axis="y", alpha=0.3, lw=0.5)
        fig.suptitle(f"{bias_type.replace('_', ' ').title()} across sections", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        paths.append(save_figure(fig, out_path, dpi=DPI))
    return paths


def plot_forward_confidence(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Plot forward confidence."""
    out_path = output_dir / "cal_forward_confidence.png"
    if df.empty or "section_id" not in df.columns:
        return None
    sections = df["section_id"].tolist()
    n = len(sections)
    if n == 0:
        return None
    if not any(f"{s}_forward_confidence" in df.columns for s in SENSORS):
        return None
    x = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n * 0.55 + 2), 4))
    for i, sensor in enumerate(SENSORS):
        col = f"{sensor}_forward_confidence"
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).tolist()
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, color=SENSOR_COLORS[sensor], label=sensor, edgecolor="white", linewidth=0.5)
    ax.axhline(0.3, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_section(s) for s in sections], rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Forward confidence")
    ax.set_title("Forward direction estimation confidence per section")
    ax.legend(fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()
    return save_figure(fig, out_path, dpi=DPI)


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
    return save_figure(fig, out_path, dpi=DPI)


def run_calibration_eda(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Run calibration eda."""
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
