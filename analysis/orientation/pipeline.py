"""Pipelines for running orientation filters on parsed IMU CSV data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from common import load_dataframe, write_dataframe
from .complementary import ComplementaryFilterConfig, ComplementaryOrientationFilter
from .madgwick import MadgwickConfig, MadgwickOrientationFilter
from .calibration import BiasCalibration, apply_calibration_bias, estimate_bias_from_dataframe_static_segment
from .quaternion import euler_from_quat


FilterType = Literal["complementary", "madgwick"]


@dataclass
class OrientationPipelineConfig:
    """High-level configuration for running an orientation filter."""

    filter_type: FilterType = "complementary"
    complementary: ComplementaryFilterConfig = field(default_factory=ComplementaryFilterConfig)
    madgwick: MadgwickConfig = field(default_factory=MadgwickConfig)
    # If True, add yaw/pitch/roll (deg) columns.
    add_euler_degrees: bool = True


def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _run_filter_over_dataframe(
    df: pd.DataFrame,
    pipeline_cfg: OrientationPipelineConfig,
    calibration: Optional[BiasCalibration] = None,
) -> pd.DataFrame:
    df = _ensure_time_sorted(df)
    if calibration is not None:
        df = apply_calibration_bias(
            df,
            accel_bias=calibration.accel_bias,
            gyro_bias=calibration.gyro_bias,
            inplace=False,
        )

    timestamps = df["timestamp"].to_numpy(dtype=float)
    # Convert timestamp units (assume ms) to seconds for dt.
    if len(timestamps) < 2:
        raise ValueError("Need at least two samples to estimate orientation.")

    # Heuristic: if timestamps look like Unix epoch (1e12), assume ms.
    if np.nanmedian(np.diff(timestamps)) > 1.0:
        t_sec = timestamps * 1e-3
    else:
        t_sec = timestamps

    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = dt[1] if len(dt) > 1 else dt[0]

    gyro = df[["gx", "gy", "gz"]].to_numpy(dtype=float)
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)

    if pipeline_cfg.filter_type == "complementary":
        filt = ComplementaryOrientationFilter(pipeline_cfg.complementary)
        step_fn = lambda k: filt.step(dt[k], gyro[k], acc[k])
    elif pipeline_cfg.filter_type == "madgwick":
        filt = MadgwickOrientationFilter(pipeline_cfg.madgwick)
        step_fn = lambda k: filt.step(dt[k], gyro[k], acc[k])
    else:
        raise ValueError(f"Unknown filter_type: {pipeline_cfg.filter_type}")

    quats = np.zeros((len(df), 4), dtype=float)
    for k in range(len(df)):
        q = step_fn(k)
        quats[k, :] = q

    out = df.copy()
    out["qw"] = quats[:, 0]
    out["qx"] = quats[:, 1]
    out["qy"] = quats[:, 2]
    out["qz"] = quats[:, 3]

    if pipeline_cfg.add_euler_degrees:
        ypr = np.array([euler_from_quat(q) for q in quats], dtype=float)
        yaw_deg = np.degrees(ypr[:, 0])
        pitch_deg = np.degrees(ypr[:, 1])
        roll_deg = np.degrees(ypr[:, 2])
        out["yaw_deg"] = yaw_deg
        out["pitch_deg"] = pitch_deg
        out["roll_deg"] = roll_deg

    return out


def run_complementary_on_dataframe(
    df: pd.DataFrame,
    calibration: Optional[BiasCalibration] = None,
    config: Optional[ComplementaryFilterConfig] = None,
) -> pd.DataFrame:
    """Run complementary filter on an in-memory IMU DataFrame."""
    pipeline_cfg = OrientationPipelineConfig(
        filter_type="complementary",
        complementary=config or ComplementaryFilterConfig(),
    )
    return _run_filter_over_dataframe(df, pipeline_cfg, calibration=calibration)


def run_madgwick_on_dataframe(
    df: pd.DataFrame,
    calibration: Optional[BiasCalibration] = None,
    config: Optional[MadgwickConfig] = None,
) -> pd.DataFrame:
    """Run Madgwick filter on an in-memory IMU DataFrame."""
    pipeline_cfg = OrientationPipelineConfig(
        filter_type="madgwick",
        madgwick=config or MadgwickConfig(),
    )
    return _run_filter_over_dataframe(df, pipeline_cfg, calibration=calibration)


def run_orientation_pipeline_on_csv(
    csv_path: str | Path,
    calibration: Optional[BiasCalibration] = None,
    config: Optional[OrientationPipelineConfig] = None,
) -> pd.DataFrame:
    """Convenience: load a parsed CSV file and run the chosen filter."""
    df = load_dataframe(Path(csv_path))
    cfg = config or OrientationPipelineConfig()
    return _run_filter_over_dataframe(df, cfg, calibration=calibration)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the orientation pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m orientation.pipeline",
        description="Estimate IMU orientation from a parsed CSV file.",
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to parsed IMU CSV (e.g. data/<session>/parsed/sensor.csv).",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        help="Optional output CSV path "
        "(default: <input_stem>_orientation.csv in the same directory).",
    )
    parser.add_argument(
        "--filter",
        dest="filter_type",
        choices=["complementary", "madgwick"],
        default="complementary",
        help="Orientation filter to use (default: complementary).",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help=(
            "Estimate constant accelerometer and gyroscope bias from the first "
            "5 seconds of data (assumed static) and apply before filtering."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Run orientation pipeline from the command line."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input_csv
    output_path: Path = (
        args.output_csv
        if args.output_csv is not None
        else input_path.parent / f"{input_path.stem}_orientation.csv"
    )

    # Load data once so we can optionally estimate calibration.
    df = load_dataframe(input_path)

    calibration: Optional[BiasCalibration] = None
    if getattr(args, "calibration", False):
        if len(df) < 2:
            raise SystemExit("Not enough samples to estimate calibration.")
        t0 = float(df["timestamp"].iloc[0])
        # Assume timestamps in milliseconds → use first 5 seconds as static window.
        end_time = t0 + 5000.0
        calibration = estimate_bias_from_dataframe_static_segment(
            df,
            start_time=t0,
            end_time=end_time,
            expected_gravity_body=[0.0, 0.0, -9.81],
        )

    cfg = OrientationPipelineConfig(filter_type=args.filter_type)
    df_orient = _run_filter_over_dataframe(df, cfg, calibration=calibration)

    write_dataframe(df_orient, output_path)
    print(f"Wrote orientation estimates to {output_path}")


if __name__ == "__main__":
    main()

