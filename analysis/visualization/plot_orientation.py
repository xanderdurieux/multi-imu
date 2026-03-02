"""Plot orientation data and orientation-compensated acceleration from CSV files.

This module operates on orientation CSVs produced by ``orientation.session``.
For each orientation file, it can generate:

- Orientation (yaw, pitch, roll) over time.
- World-frame, gravity-compensated linear acceleration over time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe, recording_stage_dir
from orientation.quaternion import euler_from_quat, quat_rotate


def _load_orientation_csv(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load an orientation CSV and return (DataFrame, time_seconds)."""
    df = load_dataframe(csv_path)
    if df.empty or "timestamp" not in df.columns:
        return df, pd.Series(dtype=float)

    time_ms = df["timestamp"].astype(float)
    time_seconds = (time_ms - float(time_ms.iloc[0])) / 1000.0
    return df, time_seconds


def _compute_euler_deg_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return yaw, pitch, roll arrays in degrees, using columns or quaternions."""
    n = len(df)
    yaw = np.full(n, np.nan, dtype=float)
    pitch = np.full(n, np.nan, dtype=float)
    roll = np.full(n, np.nan, dtype=float)

    # Prefer pre-computed Euler angles if they exist and contain any finite data.
    euler_cols = {"yaw_deg", "pitch_deg", "roll_deg"}
    if euler_cols.issubset(df.columns):
        yaw_col = df["yaw_deg"].to_numpy(dtype=float)
        pitch_col = df["pitch_deg"].to_numpy(dtype=float)
        roll_col = df["roll_deg"].to_numpy(dtype=float)
        if np.any(np.isfinite(yaw_col)) or np.any(np.isfinite(pitch_col)) or np.any(
            np.isfinite(roll_col)
        ):
            return yaw_col, pitch_col, roll_col

    # Fall back to computing from quaternions.
    quat_cols = {"qw", "qx", "qy", "qz"}
    if not quat_cols.issubset(df.columns):
        return yaw, pitch, roll

    quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)
    for i in range(n):
        if not np.all(np.isfinite(quats[i])):
            continue
        y, p, r = euler_from_quat(quats[i])
        yaw[i] = np.degrees(y)
        pitch[i] = np.degrees(p)
        roll[i] = np.degrees(r)

    return yaw, pitch, roll


def _compute_world_linear_acc(
    df: pd.DataFrame,
    gravity: float = 9.81,
) -> np.ndarray:
    """Compute world-frame, gravity-compensated linear acceleration.

    Assumes:
    - Body-frame acceleration: ``ax, ay, az``.
    - Body→world quaternions: ``qw, qx, qy, qz``.

    Returns an ``(N, 3)`` array with NaNs for samples where inputs are invalid.
    """
    required = {"ax", "ay", "az", "qw", "qx", "qy", "qz"}
    if not required.issubset(df.columns):
        return np.zeros((len(df), 3), dtype=float) * np.nan

    acc_body = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)

    n = len(df)
    acc_world = np.full((n, 3), np.nan, dtype=float)

    for i in range(n):
        if not np.all(np.isfinite(acc_body[i])) or not np.all(np.isfinite(quats[i])):
            continue
        acc_world[i] = quat_rotate(quats[i], acc_body[i])

    # Remove gravity (world gravity assumed ~[0, 0, -g]).
    lin_world = acc_world.copy()
    lin_world[:, 2] = lin_world[:, 2] + float(gravity)
    return lin_world


def plot_orientation_over_time(csv_path: Path) -> None:
    """Plot yaw, pitch, and roll over time for a single orientation CSV."""
    df, time_s = _load_orientation_csv(csv_path)
    if df.empty or time_s.empty:
        return

    yaw_deg, pitch_deg, roll_deg = _compute_euler_deg_from_df(df)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    angle_series: Iterable[tuple[str, np.ndarray, plt.Axes]] = (
        ("Yaw [deg]", yaw_deg, axes[0]),
        ("Pitch [deg]", pitch_deg, axes[1]),
        ("Roll [deg]", roll_deg, axes[2]),
    )

    for label, series, ax in angle_series:
        mask = np.isfinite(series)
        if not np.any(mask):
            continue
        ax.plot(time_s[mask], series[mask], label=label.split()[0])
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    for ax in axes:
        ax.legend(loc="upper right")

    fig.suptitle(f"{csv_path.stem} — Orientation (Euler angles)")

    out_path = csv_path.with_suffix("").with_name(f"{csv_path.stem}_orientation_euler.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved orientation plot: {out_path.name}")


def plot_orientation_compensated_acc(csv_path: Path, gravity: float = 9.81) -> None:
    """Plot world-frame, gravity-compensated acceleration over time."""
    df, time_s = _load_orientation_csv(csv_path)
    if df.empty or time_s.empty:
        return

    lin_world = _compute_world_linear_acc(df, gravity=gravity)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )

    components = ("ax_world_lin", "ay_world_lin", "az_world_lin")
    for i, (label, ax) in enumerate(zip(components, axes, strict=False)):
        series = lin_world[:, i]
        mask = np.isfinite(series)
        if not np.any(mask):
            continue
        ax.plot(time_s[mask], series[mask], label=label)
        ax.set_ylabel("m/s²")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{csv_path.stem} — World linear acceleration (gravity-compensated)")

    out_path = csv_path.with_suffix("").with_name(f"{csv_path.stem}_orientation_linacc_world.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved compensated acceleration plot: {out_path.name}")


def plot_orientation_stage(
    recording_name: str,
    stage: str = "orientation",
    gravity: float = 9.81,
) -> None:
    """Generate orientation plots for all relevant CSVs in a recording stage.

    Prefer complementary + calibrated orientation CSVs; if none are found,
    fall back to all ``*_orientation.csv`` files.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    # Prefer the "best" orientation variant if available.
    csv_files = sorted(stage_dir.glob("*__complementary_calib_orientation.csv"))
    if not csv_files:
        csv_files = sorted(stage_dir.glob("*_orientation.csv"))

    for csv_path in csv_files:
        print(f"[plot_orientation] Processing {csv_path.name} ...")
        plot_orientation_over_time(csv_path)
        plot_orientation_compensated_acc(csv_path, gravity=gravity)

