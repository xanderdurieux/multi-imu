"""Orientation diagnostic plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe


def _apply_plot_style() -> None:
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


def plot_orientation_diagnostics(
    section_path: Path,
    orient_dir: Path,
    results: dict[str, Any],
) -> None:
    """Create roll/pitch/yaw over time, gravity error, and cross-sensor pitch plots."""
    _apply_plot_style()
    plots_dir = orient_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sensors = list(results.keys())
    variants = []
    for s in sensors:
        for k in results[s]:
            if k.startswith("__"):
                variants.append(k)

    # Roll, pitch, yaw over time — one plot per variant, both sensors
    for var in set(variants):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        t0_ref = None
        for sensor in sensors:
            csv_path = orient_dir / f"{sensor}{var}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            ts = df["timestamp"].to_numpy(dtype=float)
            if t0_ref is None:
                t0_ref = float(ts[0])
            t_sec = (ts - t0_ref) / 1000.0
            for ax, col in zip(axes, ["roll_deg", "pitch_deg", "yaw_deg"]):
                if col in df.columns:
                    y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                    mask = np.isfinite(t_sec) & np.isfinite(y)
                    if np.any(mask):
                        ax.plot(t_sec[mask], y[mask], label=sensor, alpha=0.8)
        for ax, name in zip(axes, ["Roll [deg]", "Pitch [deg]", "Yaw [deg]"]):
            ax.set_ylabel(name)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time [s]")
        fig.suptitle(f"Orientation — {var[2:]}")
        fig.savefig(plots_dir / f"rpy_{var[2:].replace('.', '_')}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Cross-sensor pitch comparison
    comp_var = "__complementary_orientation"
    dfs = {}
    for sensor in sensors:
        p = orient_dir / f"{sensor}{comp_var}.csv"
        if p.exists():
            dfs[sensor] = pd.read_csv(p)
    if len(dfs) >= 2:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        t0_ref = None
        for sensor, df in dfs.items():
            ts = df["timestamp"].to_numpy(dtype=float)
            if t0_ref is None:
                t0_ref = float(ts[0])
            t_sec = (ts - t0_ref) / 1000.0
            pitch = pd.to_numeric(df["pitch_deg"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(t_sec) & np.isfinite(pitch)
            if np.any(mask):
                ax.plot(t_sec[mask], pitch[mask], label=sensor, alpha=0.8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Pitch [deg]")
        ax.set_title("Cross-sensor pitch comparison (complementary filter)")
        ax.legend()
        fig.savefig(plots_dir / "pitch_cross_sensor.png", dpi=150, bbox_inches="tight")
        plt.close()
