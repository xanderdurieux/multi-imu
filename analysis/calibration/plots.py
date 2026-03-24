"""Diagnostic plots for ride-level calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe

from .calibrate import _acc_norm, ACC_COLS, GRAVITY_M_S2


def _apply_plot_style() -> None:
    """Match sync/plots style."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#d4d4d4",
            "axes.labelcolor": "#262626",
            "axes.titlecolor": "#171717",
            "text.color": "#171717",
            "xtick.color": "#404040",
            "ytick.color": "#404040",
            "grid.color": "#e5e5e5",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )


def plot_calibration_diagnostics(
    section_path: Path,
    calibrated_dir: Path,
    calibration: dict[str, Any],
) -> None:
    """Create two diagnostic plots per sensor: acc_norm with static window, before/after z-acc."""
    _apply_plot_style()
    plots_dir = calibrated_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for sensor, meta in calibration.items():
        raw_path = section_path / f"{sensor}.csv"
        cal_path = calibrated_dir / f"{sensor}.csv"
        if not raw_path.exists() or not cal_path.exists():
            continue

        df_raw = load_dataframe(raw_path)
        df_cal = load_dataframe(cal_path)
        df_raw = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        df_cal = df_cal.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        t0 = float(df_raw["timestamp"].iloc[0])
        t_sec = (df_raw["timestamp"].astype(float) - t0) / 1000.0

        n_static = meta.get("n_static_samples", 0)
        acc_norm_raw = _acc_norm(df_raw)
        acc_norm_cal = _acc_norm(df_cal)

        # Plot 1: acc_norm over full section with static window highlighted
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(t_sec, acc_norm_raw, color="#2563eb", alpha=0.7, label="Raw acc_norm")
        ax.axhline(GRAVITY_M_S2, color="#666", linestyle="--", alpha=0.7, label="g = 9.81")
        if n_static > 0 and len(t_sec) >= n_static:
            ax.axvspan(
                t_sec.iloc[0],
                t_sec.iloc[min(n_static, len(t_sec) - 1)],
                alpha=0.2,
                color="#22c55e",
                label="Static window",
            )
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("acc_norm [m/s²]")
        ax.set_title(f"{sensor.capitalize()} — accelerometer norm (static window highlighted)")
        ax.legend(loc="upper right")
        ax.set_xlim(t_sec.iloc[0], t_sec.iloc[-1])
        fig.savefig(plots_dir / f"{sensor}_acc_norm_timeline.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot 2: Before/after z-axis acceleration (static window centered ~9.81)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        az_raw = df_raw["az"].to_numpy(dtype=float) if "az" in df_raw.columns else np.full(len(df_raw), np.nan)
        az_cal = df_cal["az"].to_numpy(dtype=float) if "az" in df_cal.columns else np.full(len(df_cal), np.nan)

        mask_raw = np.isfinite(t_sec.to_numpy(dtype=float)) & np.isfinite(az_raw)
        if np.any(mask_raw):
            axes[0].plot(t_sec.to_numpy(dtype=float)[mask_raw], az_raw[mask_raw], color="#ca3c3c", alpha=0.7, label="Raw az")
        axes[0].axhline(GRAVITY_M_S2, color="#666", linestyle="--", alpha=0.7)
        axes[0].set_ylabel("az [m/s²]")
        axes[0].set_title("Before calibration")
        axes[0].legend(loc="upper right")

        mask_cal = np.isfinite(t_sec.to_numpy(dtype=float)) & np.isfinite(az_cal)
        if np.any(mask_cal):
            axes[1].plot(t_sec.to_numpy(dtype=float)[mask_cal], az_cal[mask_cal], color="#2563eb", alpha=0.7, label="Calibrated az (world frame)")
        axes[1].axhline(GRAVITY_M_S2, color="#666", linestyle="--", alpha=0.7)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("az [m/s²]")
        axes[1].set_title("After calibration (should center near 9.81 in static window)")
        axes[1].legend(loc="upper right")

        fig.suptitle(f"{sensor.capitalize()} — z-axis acceleration before/after")
        fig.savefig(plots_dir / f"{sensor}_z_acc_before_after.png", dpi=150, bbox_inches="tight")
        plt.close()
