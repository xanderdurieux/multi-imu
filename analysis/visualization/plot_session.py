"""Plot all sensors for a recording stage (acc, gyro, comparison)."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.paths import recording_stage_dir, sections_root

log = logging.getLogger(__name__)

SENSORS = ["sporsa", "arduino"]
COLORS = {"sporsa": "#1f77b4", "arduino": "#ff7f0e"}


def _ts_s(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if ts.size > 0:
        ts = (ts - ts[0]) / 1000.0
    return ts


def _plot_sensor_panels(
    stage_dir: Path,
    sensor_name: str,
    output_path: Path,
) -> None:
    csv = stage_dir / f"{sensor_name}.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    ts = _ts_s(df)

    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]
    rows = sum([bool(acc_cols), bool(gyro_cols)])
    if rows == 0:
        return

    fig, axes = plt.subplots(rows, 1, figsize=(14, 3 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    idx = 0
    if acc_cols:
        for col in acc_cols:
            axes[idx].plot(ts, df[col].to_numpy(dtype=float), lw=0.6, label=col)
        acc_arr = df[acc_cols].to_numpy(dtype=float)
        acc_norm = np.sqrt(np.nansum(acc_arr ** 2, axis=1))
        axes[idx].plot(ts, acc_norm, lw=1.0, color="k", alpha=0.4, label="|acc|")
        axes[idx].set_ylabel("Acc (m/s²)")
        axes[idx].legend(loc="upper right", fontsize=7)
        idx += 1

    if gyro_cols:
        for col in gyro_cols:
            axes[idx].plot(ts, df[col].to_numpy(dtype=float), lw=0.6, label=col)
        axes[idx].set_ylabel("Gyro")
        axes[idx].legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{stage_dir.name} / {sensor_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.debug("Saved %s", output_path)


def _plot_comparison(stage_dir: Path, output_path: Path) -> None:
    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in SENSORS:
        csv = stage_dir / f"{sensor}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            sensor_dfs[sensor] = df

    if not sensor_dfs:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for sensor, df in sensor_dfs.items():
        ts = _ts_s(df)
        color = COLORS.get(sensor)
        acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
        gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]
        if acc_cols:
            norm = np.sqrt(np.nansum(df[acc_cols].to_numpy(dtype=float) ** 2, axis=1))
            axes[0].plot(ts, norm, lw=0.8, color=color, label=f"{sensor} |acc|", alpha=0.8)
        if gyro_cols:
            norm = np.sqrt(np.nansum(df[gyro_cols].to_numpy(dtype=float) ** 2, axis=1))
            axes[1].plot(ts, norm, lw=0.8, color=color, label=f"{sensor} |gyro|", alpha=0.8)

    axes[0].set_ylabel("|acc| (m/s²)")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("|gyro|")
    axes[1].legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{stage_dir.name} — comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.debug("Saved %s", output_path)


def plot_recording(
    recording_name: str,
    *,
    stage_filter: str = "parsed",
    sensors: list[str] | None = None,
) -> None:
    """Generate per-sensor and comparison plots for a recording stage."""
    if sensors is None:
        sensors = list(SENSORS)
    stage_dir = recording_stage_dir(recording_name, stage_filter)
    if not stage_dir.exists():
        log.warning("Stage dir not found: %s", stage_dir)
        return

    for sensor in sensors:
        out = stage_dir / f"{sensor}.png"
        try:
            _plot_sensor_panels(stage_dir, sensor, out)
        except Exception as exc:
            log.warning("Failed to plot %s/%s: %s", recording_name, sensor, exc)

    try:
        _plot_comparison(stage_dir, stage_dir / "comparison.png")
    except Exception as exc:
        log.warning("Failed comparison plot for %s: %s", recording_name, exc)
