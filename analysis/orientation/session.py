"""Session-level orientation pipeline.

CLI:
    uv run -m orientation.session <session_name>/<stage>

This command:
- Loads all CSV files in the given session + stage (e.g. parsed, synced).
- For each CSV, estimates orientation using both filters
  (complementary and Madgwick), with and without simple static calibration.
- Writes the resulting orientation CSVs to a new stage directory
  ``<stage>_orientation`` for the same session.
- Computes basic quality statistics for each method and writes a summary CSV.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common import load_dataframe, session_stage_dir, write_dataframe
from .calibration import BiasCalibration, estimate_bias_from_dataframe_static_segment
from .pipeline import (
    run_complementary_on_dataframe,
    run_madgwick_on_dataframe,
)
from .quaternion import quat_rotate


@dataclass
class OrientationMethodResult:
    """Metadata and quality metrics for one (filter, calibration) run on one CSV."""

    session: str
    stage_in: str
    stage_out: str
    sensor_file: str
    filter_type: str  # "complementary" or "madgwick"
    calibrated: bool
    output_csv: str
    g_err_mean: float
    g_err_std: float
    g_err_abs_mean: float
    g_err_abs_p95: float
    static_fraction: float
    pitch_static_std_deg: float
    roll_static_std_deg: float
    num_static_samples: int


def _compute_orientation_stats(df: pd.DataFrame, gravity: float = 9.81) -> dict:
    """Compute basic internal-consistency metrics for an oriented IMU DataFrame.

    Assumes:
    - Body-frame acceleration columns: ax, ay, az
    - Body→world quaternion columns: qw, qx, qy, qz
    - Optional Euler columns in degrees: yaw_deg, pitch_deg, roll_deg
    """
    required_cols = {"ax", "ay", "az", "qw", "qx", "qy", "qz"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

    acc_body = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)

    # Rotate body acceleration into world frame.
    acc_world = np.zeros_like(acc_body)
    for k in range(len(df)):
        acc_world[k] = quat_rotate(quats[k], acc_body[k])

    acc_world_norm = np.linalg.norm(acc_world, axis=1)
    g_err = acc_world_norm - float(gravity)

    # Simple static detector: |a| ≈ g and |ω| small.
    acc_norm_body = np.linalg.norm(acc_body, axis=1)
    gyro_cols = [c for c in ("gx", "gy", "gz") if c in df.columns]
    if gyro_cols:
        gyro = df[gyro_cols].to_numpy(dtype=float)
        gyro_norm = np.linalg.norm(gyro, axis=1)
    else:
        gyro_norm = np.zeros(len(df), dtype=float)

    static_mask = (
        np.isfinite(acc_norm_body)
        & np.isfinite(gyro_norm)
        & (np.abs(acc_norm_body - float(gravity)) < 0.1 * float(gravity))
        & (gyro_norm < 0.1)
    )

    # Prepare metrics.
    def _safe(fn, arr, default=np.nan):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return float(default)
        return float(fn(arr))

    metrics: dict[str, float | int] = {
        "g_err_mean": _safe(np.nanmean, g_err),
        "g_err_std": _safe(np.nanstd, g_err),
        "g_err_abs_mean": _safe(np.nanmean, np.abs(g_err)),
        "g_err_abs_p95": _safe(lambda x: np.nanpercentile(x, 95.0), np.abs(g_err)),
        "static_fraction": float(static_mask.mean() if len(static_mask) else 0.0),
    }

    # Static tilt stability (pitch/roll) if Euler angles are available.
    if {"pitch_deg", "roll_deg"}.issubset(df.columns):
        pitch = df["pitch_deg"].to_numpy(dtype=float)
        roll = df["roll_deg"].to_numpy(dtype=float)
        pitch_static = pitch[static_mask]
        roll_static = roll[static_mask]
        metrics["pitch_static_std_deg"] = _safe(np.nanstd, pitch_static)
        metrics["roll_static_std_deg"] = _safe(np.nanstd, roll_static)
        metrics["num_static_samples"] = int(len(pitch_static))
    else:
        metrics["pitch_static_std_deg"] = float("nan")
        metrics["roll_static_std_deg"] = float("nan")
        metrics["num_static_samples"] = 0

    return metrics


def _process_sensor_csv(
    session_name: str,
    stage_in: str,
    stage_out: str,
    csv_path: Path,
    static_window_ms: float = 5000.0,
    gravity: float = 9.81,
) -> list[OrientationMethodResult]:
    """Run all orientation filter variants on one sensor CSV and collect stats."""
    df = load_dataframe(csv_path)
    if df.empty:
        return []

    # Estimate simple static-segment calibration (first N ms assumed static).
    t0 = float(df["timestamp"].iloc[0])
    calib: Optional[BiasCalibration]
    try:
        calib = estimate_bias_from_dataframe_static_segment(
            df,
            start_time=t0,
            end_time=t0 + static_window_ms,
            expected_gravity_body=[0.0, 0.0, -gravity],
        )
    except Exception:
        calib = None

    results: list[OrientationMethodResult] = []
    # (filter_type, calibrated_flag)
    variants = [
        ("complementary", False),
        ("complementary", True),
        ("madgwick", False),
        ("madgwick", True),
    ]

    out_dir = session_stage_dir(session_name, stage_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for filter_type, use_calib in variants:
        current_calib = calib if (use_calib and calib is not None) else None

        if filter_type == "complementary":
            df_orient = run_complementary_on_dataframe(df, calibration=current_calib)
        elif filter_type == "madgwick":
            df_orient = run_madgwick_on_dataframe(df, calibration=current_calib)
        else:
            continue

        suffix = f"__{filter_type}_{'calib' if use_calib else 'raw'}_orientation.csv"
        out_path = out_dir / f"{csv_path.stem}{suffix}"
        write_dataframe(df_orient, out_path)

        stats = _compute_orientation_stats(df_orient, gravity=gravity)
        result = OrientationMethodResult(
            session=session_name,
            stage_in=stage_in,
            stage_out=stage_out,
            sensor_file=csv_path.name,
            filter_type=filter_type,
            calibrated=bool(current_calib is not None),
            output_csv=str(out_path),
            g_err_mean=float(stats["g_err_mean"]),
            g_err_std=float(stats["g_err_std"]),
            g_err_abs_mean=float(stats["g_err_abs_mean"]),
            g_err_abs_p95=float(stats["g_err_abs_p95"]),
            static_fraction=float(stats["static_fraction"]),
            pitch_static_std_deg=float(stats["pitch_static_std_deg"]),
            roll_static_std_deg=float(stats["roll_static_std_deg"]),
            num_static_samples=int(stats["num_static_samples"]),
        )
        results.append(result)

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for session-level orientation."""
    parser = argparse.ArgumentParser(
        prog="python -m orientation.session",
        description=(
            "Estimate orientation for all CSVs in a session stage using multiple "
            "filters (complementary, Madgwick), with and without calibration."
        ),
    )
    parser.add_argument(
        "session_name_stage",
        type=str,
        help="Session and stage as '<session_name>/<stage>' (e.g. 'test_data_rate/parsed').",
    )
    parser.add_argument(
        "--static-window-ms",
        type=float,
        default=5000.0,
        help="Duration of initial static window used for bias calibration (default: 5000 ms).",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="Assumed gravity magnitude in m/s^2 (default: 9.81).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    parts = args.session_name_stage.split("/", 1)
    if len(parts) != 2:
        parser.error("session_name_stage must be in format '<session_name>/<stage>'")
    session_name, stage_in = parts

    stage_out = f"{stage_in}_orientation"
    in_dir = session_stage_dir(session_name, stage_in)
    if not in_dir.is_dir():
        parser.error(f"Input stage directory not found: {in_dir}")

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        parser.error(f"No CSV files found in {in_dir}")

    all_results: list[OrientationMethodResult] = []
    for csv_path in csv_files:
        print(f"[{session_name}/{stage_in}] processing {csv_path.name}")
        all_results.extend(
            _process_sensor_csv(
                session_name=session_name,
                stage_in=stage_in,
                stage_out=stage_out,
                csv_path=csv_path,
                static_window_ms=float(args.static_window_ms),
                gravity=float(args.gravity),
            )
        )

    if not all_results:
        print("No orientation results produced (empty inputs?).")
        return

    # Export summary stats table to the output stage directory.
    out_dir = session_stage_dir(session_name, stage_out)
    stats_df = pd.DataFrame([r.__dict__ for r in all_results])
    stats_path = out_dir / "orientation_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote orientation stats to {stats_path}")


if __name__ == "__main__":
    main()

