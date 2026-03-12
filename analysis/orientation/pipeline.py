"""Pipelines for running orientation filters on IMU CSV data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from common import load_dataframe, write_dataframe
from .calibration import apply_gyro_bias
from .complementary import ComplementaryFilterConfig, ComplementaryOrientationFilter
from .madgwick import MadgwickConfig, MadgwickOrientationFilter
from .quaternion import euler_from_quat


FilterType = Literal["complementary", "madgwick"]


@dataclass
class OrientationPipelineConfig:
    """High-level configuration for running an orientation filter."""

    filter_type: FilterType = "complementary"
    complementary: ComplementaryFilterConfig = field(default_factory=ComplementaryFilterConfig)
    madgwick: MadgwickConfig = field(default_factory=MadgwickConfig)
    add_euler_degrees: bool = True


def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")
    return df.sort_values("timestamp").reset_index(drop=True)


def _run_filter_over_dataframe(
    df: pd.DataFrame,
    pipeline_cfg: OrientationPipelineConfig,
    gyro_bias: Optional[np.ndarray] = None,
    initial_q: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run an orientation filter over an IMU DataFrame.

    Parameters
    ----------
    df:
        Parsed body-frame IMU DataFrame with ``timestamp, ax, ay, az, gx, gy, gz``.
    pipeline_cfg:
        Filter type and per-filter configuration.
    gyro_bias:
        Optional shape ``(3,)`` gyro zero-rate bias (same units as CSV columns).
        Subtracted from ``gx/gy/gz`` before filtering.
    initial_q:
        Optional shape ``(4,)`` unit quaternion ``[w, x, y, z]`` for the
        filter's starting pose (e.g. static sensor-to-world rotation from
        ``calibration.json``).  When ``None`` the filter starts at identity.
    """
    df = _ensure_time_sorted(df)
    if gyro_bias is not None:
        df = apply_gyro_bias(df, gyro_bias)

    timestamps = df["timestamp"].to_numpy(dtype=float)
    if len(timestamps) < 2:
        raise ValueError("Need at least two samples to estimate orientation.")

    # Heuristic: if timestamps look like Unix epoch ms (median diff > 1.0), convert to s.
    if np.nanmedian(np.diff(timestamps)) > 1.0:
        t_sec = timestamps * 1e-3
    else:
        t_sec = timestamps

    dt = np.diff(t_sec, prepend=t_sec[0])
    dt[0] = dt[1] if len(dt) > 1 else dt[0]

    # Gyro is stored in deg/s in CSV; filters integrate in rad/s.
    gyro_rad = np.radians(df[["gx", "gy", "gz"]].to_numpy(dtype=float))
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)

    initial_q_arr = np.asarray(initial_q, dtype=float) if initial_q is not None else None

    if pipeline_cfg.filter_type == "complementary":
        filt = ComplementaryOrientationFilter(pipeline_cfg.complementary)
        if initial_q_arr is not None:
            filt.reset(initial_quaternion=initial_q_arr)
        step_fn = lambda k: filt.step(dt[k], gyro_rad[k], acc[k])
    elif pipeline_cfg.filter_type == "madgwick":
        filt = MadgwickOrientationFilter(pipeline_cfg.madgwick)
        if initial_q_arr is not None:
            filt.reset(initial_quaternion=initial_q_arr)
        step_fn = lambda k: filt.step(dt[k], gyro_rad[k], acc[k])
    else:
        raise ValueError(f"Unknown filter_type: {pipeline_cfg.filter_type}")

    from .quaternion import quat_identity
    quats = np.zeros((len(df), 4), dtype=float)
    prev_q = initial_q_arr if initial_q_arr is not None else quat_identity()
    for k in range(len(df)):
        if np.all(np.isfinite(gyro_rad[k])):
            # Normal step: NaN acc is handled by the filter's own gating.
            q = step_fn(k)
            prev_q = q
        else:
            # Gyro NaN: hold previous orientation (dropout packet).
            q = prev_q
        quats[k, :] = q

    out = df.copy()
    out["qw"] = quats[:, 0]
    out["qx"] = quats[:, 1]
    out["qy"] = quats[:, 2]
    out["qz"] = quats[:, 3]

    if pipeline_cfg.add_euler_degrees:
        ypr = np.array([euler_from_quat(q) for q in quats], dtype=float)
        out["yaw_deg"] = np.degrees(ypr[:, 0])
        out["pitch_deg"] = np.degrees(ypr[:, 1])
        out["roll_deg"] = np.degrees(ypr[:, 2])

    return out


def run_complementary_on_dataframe(
    df: pd.DataFrame,
    gyro_bias: Optional[np.ndarray] = None,
    initial_q: Optional[np.ndarray] = None,
    config: Optional[ComplementaryFilterConfig] = None,
) -> pd.DataFrame:
    """Run complementary filter on an in-memory body-frame IMU DataFrame.

    Parameters
    ----------
    df:
        Body-frame IMU DataFrame (``timestamp, ax, ay, az, gx, gy, gz``).
    gyro_bias:
        Optional gyro zero-rate bias subtracted before filtering.
    initial_q:
        Optional starting quaternion (e.g. from ``calibration.json``).
    config:
        Filter configuration.  Uses defaults when ``None``.
    """
    pipeline_cfg = OrientationPipelineConfig(
        filter_type="complementary",
        complementary=config or ComplementaryFilterConfig(),
    )
    return _run_filter_over_dataframe(df, pipeline_cfg, gyro_bias=gyro_bias, initial_q=initial_q)


def run_madgwick_on_dataframe(
    df: pd.DataFrame,
    gyro_bias: Optional[np.ndarray] = None,
    initial_q: Optional[np.ndarray] = None,
    config: Optional[MadgwickConfig] = None,
) -> pd.DataFrame:
    """Run Madgwick filter on an in-memory body-frame IMU DataFrame.

    Parameters
    ----------
    df:
        Body-frame IMU DataFrame (``timestamp, ax, ay, az, gx, gy, gz``).
    gyro_bias:
        Optional gyro zero-rate bias subtracted before filtering.
    initial_q:
        Optional starting quaternion (e.g. from ``calibration.json``).
    config:
        Filter configuration.  Uses defaults when ``None``.
    """
    pipeline_cfg = OrientationPipelineConfig(
        filter_type="madgwick",
        madgwick=config or MadgwickConfig(),
    )
    return _run_filter_over_dataframe(df, pipeline_cfg, gyro_bias=gyro_bias, initial_q=initial_q)


def run_orientation_pipeline_on_csv(
    csv_path: str | Path,
    gyro_bias: Optional[np.ndarray] = None,
    initial_q: Optional[np.ndarray] = None,
    config: Optional[OrientationPipelineConfig] = None,
) -> pd.DataFrame:
    """Convenience: load a body-frame CSV file and run the chosen filter."""
    df = load_dataframe(Path(csv_path))
    cfg = config or OrientationPipelineConfig()
    return _run_filter_over_dataframe(df, cfg, gyro_bias=gyro_bias, initial_q=initial_q)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orientation.pipeline",
        description=(
            "Estimate IMU orientation from a parsed (body-frame) CSV file. "
            "For calibrated initialization use orientation.session instead."
        ),
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to parsed body-frame IMU CSV.",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        nargs="?",
        help="Output CSV path (default: <input_stem>_orientation.csv).",
    )
    parser.add_argument(
        "--filter",
        dest="filter_type",
        choices=["complementary", "madgwick"],
        default="complementary",
        help="Orientation filter to use (default: complementary).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input_csv
    output_path: Path = (
        args.output_csv
        if args.output_csv is not None
        else input_path.parent / f"{input_path.stem}_orientation.csv"
    )

    df = load_dataframe(input_path)
    cfg = OrientationPipelineConfig(filter_type=args.filter_type)
    df_orient = _run_filter_over_dataframe(df, cfg)

    write_dataframe(df_orient, output_path)
    print(f"Wrote orientation estimates to {output_path}")


if __name__ == "__main__":
    main()
