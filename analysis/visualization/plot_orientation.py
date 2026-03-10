"""Plot orientation data and orientation-compensated acceleration from CSV files.

This module operates on orientation CSVs produced by ``orientation.session``.
For each orientation file, it can generate:

- Orientation (yaw, pitch, roll) over time.
- World-frame, gravity-compensated linear acceleration over time.
- Relative orientation between two sensors (head vs handlebar).
- Side-by-side comparison of both sensor orientations.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe, recording_stage_dir
from orientation.quaternion import euler_from_quat, quat_rotate

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")


def _unwrap_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Unwrap a degree-valued angle array to remove ±180° jump discontinuities.

    Only operates on the finite subset; NaN values are left in place.
    """
    out = angles_deg.copy()
    finite = np.isfinite(out)
    if finite.sum() > 1:
        out[finite] = np.degrees(np.unwrap(np.radians(out[finite])))
    return out
_SENSOR_COLORS = {"sporsa": "#e05c44", "arduino": "#4c9be8"}
_SENSOR_LABELS = {"sporsa": "Sporsa (handlebar)", "arduino": "Arduino (helmet)"}


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


def _stage_label(csv_path: Path) -> str:
    """Derive a ``[recording/stage]`` prefix from a CSV path."""
    return f"[{csv_path.parent.parent.name}/{csv_path.parent.name}]"


def plot_orientation_over_time(csv_path: Path) -> None:
    """Plot yaw, pitch, and roll over time for a single orientation CSV."""
    df, time_s = _load_orientation_csv(csv_path)
    if df.empty or time_s.empty:
        return

    yaw_deg, pitch_deg, roll_deg = _compute_euler_deg_from_df(df)
    yaw_deg = _unwrap_deg(yaw_deg)
    pitch_deg = _unwrap_deg(pitch_deg)
    roll_deg = _unwrap_deg(roll_deg)

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
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")

    fig.suptitle(f"{csv_path.stem} — Orientation (Euler angles)")

    out_path = csv_path.with_suffix("").with_name(f"{csv_path.stem}_orientation_euler.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{_stage_label(csv_path)} {out_path.name}")


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
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"{csv_path.stem} — World linear acceleration (gravity-compensated)")

    out_path = csv_path.with_suffix("").with_name(f"{csv_path.stem}_orientation_linacc_world.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{_stage_label(csv_path)} {out_path.name}")


def plot_orientation_stage(
    recording_name: str,
    stage: str = "orientation",
    gravity: float = 9.81,
) -> None:
    """Generate per-file and multi-sensor orientation plots for a recording stage.

    Prefer complementary + calibrated orientation CSVs; if none are found,
    fall back to all ``*_orientation.csv`` files.

    Produces per-file Euler and linear-acceleration plots, plus:
    - ``orientation_comparison.png``: side-by-side overlay of both sensors.
    - ``orientation_relative.png``: relative head-bike orientation angles.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    csv_files = sorted(stage_dir.glob("*__complementary_calib_orientation.csv"))
    if not csv_files:
        csv_files = sorted(stage_dir.glob("*_orientation.csv"))

    for csv_path in csv_files:
        plot_orientation_over_time(csv_path)
        plot_orientation_compensated_acc(csv_path, gravity=gravity)

    plot_orientation_comparison(recording_name, stage=stage)
    plot_relative_orientation(recording_name, stage=stage)


# ---------------------------------------------------------------------------
# Multi-sensor: side-by-side comparison
# ---------------------------------------------------------------------------

def _find_orientation_csv(stage_dir: Path, sensor: str) -> Path | None:
    """Return the best-matching orientation CSV for *sensor* in *stage_dir*."""
    candidates = sorted(stage_dir.glob(f"{sensor}*__complementary_calib_orientation.csv"))
    if not candidates:
        candidates = sorted(stage_dir.glob(f"{sensor}*_orientation.csv"))
    return candidates[0] if candidates else None


def plot_orientation_comparison(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Overlay Sporsa (handlebar) and Arduino (helmet) orientation on shared axes.

    Plots yaw, pitch, and roll for both sensors in the same figure, making it
    easy to compare the two IMUs' perceived orientation during the recording.

    Returns the path of the saved PNG, or ``None`` if fewer than two sensor
    CSVs were found.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.exists():
        return None

    sensor_data: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for sensor in _SENSORS:
        csv_path = _find_orientation_csv(stage_dir, sensor)
        if csv_path is None:
            continue
        df, time_s = _load_orientation_csv(csv_path)
        if not df.empty and not time_s.empty:
            sensor_data[sensor] = (df, time_s)

    if len(sensor_data) < 2:
        log.warning("[%s/%s] need both sensor orientation CSVs — skipping comparison.", recording_name, stage)
        return None

    angle_names = ("Yaw [deg]", "Pitch [deg]", "Roll [deg]")
    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=False,
        constrained_layout=True,
    )

    for sensor, (df, time_s) in sensor_data.items():
        yaw, pitch, roll = _compute_euler_deg_from_df(df)
        yaw = _unwrap_deg(yaw)
        pitch = _unwrap_deg(pitch)
        roll = _unwrap_deg(roll)
        color = _SENSOR_COLORS.get(sensor, "gray")
        label = _SENSOR_LABELS.get(sensor, sensor)
        for ax, series in zip(axes, (yaw, pitch, roll)):
            mask = np.isfinite(series)
            if np.any(mask):
                ax.plot(time_s[mask], series[mask], color=color, linewidth=0.8,
                        alpha=0.85, label=label)

    for ax, name in zip(axes, angle_names):
        ax.set_ylabel(name, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]", fontsize=9)
    fig.suptitle(f"{recording_name} / {stage} — Orientation comparison (Sporsa vs Arduino)", fontsize=11)

    out_path = stage_dir / "orientation_comparison.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Multi-sensor: relative orientation (head vs handlebar)
# ---------------------------------------------------------------------------

def plot_relative_orientation(
    recording_name: str,
    stage: str = "orientation",
) -> Path | None:
    """Plot the orientation of the helmet relative to the handlebar.

    Computes the difference (Δyaw, Δpitch, Δroll) = (Arduino − Sporsa) after
    resampling both streams to a common 100 Hz time grid.  This reveals head
    movements made by the cyclist — looking left/right (Δyaw), nodding (Δpitch),
    and tilting the head (Δroll) — independently of the bicycle orientation.

    Returns the path of the saved PNG, or ``None`` if fewer than two sensor
    CSVs were found.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.exists():
        return None

    sensor_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sensor in _SENSORS:
        csv_path = _find_orientation_csv(stage_dir, sensor)
        if csv_path is None:
            continue
        df, time_s = _load_orientation_csv(csv_path)
        if df.empty or time_s.empty:
            continue
        yaw, pitch, roll = _compute_euler_deg_from_df(df)
        # Unwrap before differencing so per-sensor drift does not create jumps.
        sensor_data[sensor] = (
            time_s.to_numpy(),
            _unwrap_deg(yaw),
            _unwrap_deg(pitch),
            _unwrap_deg(roll),
        )

    if len(sensor_data) < 2:
        log.warning("[%s/%s] need both sensor orientation CSVs — skipping relative orientation.", recording_name, stage)
        return None

    sporsa_t, sporsa_yaw, sporsa_pitch, sporsa_roll = sensor_data["sporsa"]
    arduino_t, arduino_yaw, arduino_pitch, arduino_roll = sensor_data["arduino"]

    # Resample Arduino to Sporsa time grid via linear interpolation
    def _interp(t_src: np.ndarray, v_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        t_overlap_min = max(t_src[0], t_dst[0])
        t_overlap_max = min(t_src[-1], t_dst[-1])
        mask_dst = (t_dst >= t_overlap_min) & (t_dst <= t_overlap_max)
        mask_src = (t_src >= t_overlap_min) & (t_src <= t_overlap_max)
        if not mask_src.any():
            return np.full(len(t_dst), np.nan)
        result = np.full(len(t_dst), np.nan)
        result[mask_dst] = np.interp(t_dst[mask_dst], t_src[mask_src], v_src[mask_src])
        return result

    t_ref = sporsa_t
    delta_yaw = _interp(arduino_t, arduino_yaw, t_ref) - sporsa_yaw
    delta_pitch = _interp(arduino_t, arduino_pitch, t_ref) - sporsa_pitch
    delta_roll = _interp(arduino_t, arduino_roll, t_ref) - sporsa_roll

    # Unwrap yaw difference to avoid 360° jumps
    valid_yaw = np.isfinite(delta_yaw)
    if valid_yaw.sum() > 1:
        delta_yaw[valid_yaw] = np.unwrap(delta_yaw[valid_yaw], period=360.0)

    angle_names = ("ΔYaw [deg]", "ΔPitch [deg]", "ΔRoll [deg]")
    angle_series = (delta_yaw, delta_pitch, delta_roll)
    colors = ("#555555", "#888888", "#aaaaaa")

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    for ax, name, series, color in zip(axes, angle_names, angle_series, colors):
        mask = np.isfinite(series)
        if np.any(mask):
            ax.plot(t_ref[mask], series[mask], color=color, linewidth=0.8, alpha=0.85)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.7, alpha=0.4)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]", fontsize=9)
    fig.suptitle(
        f"{recording_name} / {stage} — Relative orientation  (helmet − handlebar)",
        fontsize=11,
    )

    out_path = stage_dir / "orientation_relative.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}/{stage}] {out_path.name}")
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_orientation",
        description=(
            "Generate orientation plots (Euler angles, linear acceleration, "
            "sensor comparison, relative orientation) for one or more recordings."
        ),
    )
    parser.add_argument(
        "recording_names",
        nargs="+",
        help="One or more recording names (e.g. 2026-02-26_5).",
    )
    parser.add_argument(
        "--stage",
        default="orientation",
        help="Orientation stage directory name (default: orientation).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    for recording_name in args.recording_names:
        plot_orientation_stage(recording_name, stage=args.stage)


if __name__ == "__main__":
    main()

