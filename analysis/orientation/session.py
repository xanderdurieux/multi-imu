"""Recording-level orientation pipeline.

CLI::

    uv run -m orientation.session <recording_name>/<stage_in>

Example::

    uv run -m orientation.session 2026-02-26_5/parsed

This command:

- Loads all CSV files in the given recording stage (e.g. ``parsed``, ``synced``).
- For each CSV, estimates orientation using both filters
  (complementary and Madgwick), with and without simple static calibration.
- Writes the resulting orientation CSVs to ``data/recordings/<recording_name>/orientation/``.
- Computes basic quality statistics for each method and writes
  ``orientation_stats.json`` to the same output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common import load_dataframe, recording_stage_dir, write_dataframe
from .calibration import BiasCalibration, estimate_bias_from_dataframe_static_segment
from .pipeline import (
    run_complementary_on_dataframe,
    run_madgwick_on_dataframe,
)
from .quaternion import quat_rotate


@dataclass
class OrientationMethodResult:
    """Metadata and quality metrics for one (filter, calibration) run on one CSV."""

    recording: str
    stage_in: str
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
    """Compute basic internal-consistency metrics for an oriented IMU DataFrame."""
    required_cols = {"ax", "ay", "az", "qw", "qx", "qy", "qz"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame missing required columns: {required_cols - set(df.columns)}")

    acc_body = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    quats = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)

    acc_world = np.zeros_like(acc_body)
    for k in range(len(df)):
        acc_world[k] = quat_rotate(quats[k], acc_body[k])

    acc_world_norm = np.linalg.norm(acc_world, axis=1)
    g_err = acc_world_norm - float(gravity)

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
    recording_name: str,
    stage_in: str,
    csv_path: Path,
    static_window_ms: float = 5000.0,
    gravity: float = 9.81,
) -> list[OrientationMethodResult]:
    """Run all orientation filter variants on one sensor CSV and collect stats."""
    df = load_dataframe(csv_path)
    if df.empty:
        return []

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

    out_dir = recording_stage_dir(recording_name, "orientation")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[OrientationMethodResult] = []
    variants = [
        ("complementary", False),
        ("complementary", True),
        ("madgwick", False),
        ("madgwick", True),
    ]

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
            recording=recording_name,
            stage_in=stage_in,
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
    parser = argparse.ArgumentParser(
        prog="python -m orientation.session",
        description=(
            "Estimate orientation for all CSVs in a recording stage using multiple "
            "filters (complementary, Madgwick), with and without calibration. "
            "Output always goes to the 'orientation' stage directory."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        type=str,
        help="Recording name and input stage as '<recording_name>/<stage>' (e.g. '2026-02-26_5/parsed').",
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

    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        parser.error("recording_name_stage must be in format '<recording_name>/<stage>'")
    recording_name, stage_in = parts

    in_dir = recording_stage_dir(recording_name, stage_in)
    if not in_dir.is_dir():
        parser.error(f"Input stage directory not found: {in_dir}")

    csv_files = sorted(in_dir.glob("*.csv"))
    if not csv_files:
        parser.error(f"No CSV files found in {in_dir}")

    all_results: list[OrientationMethodResult] = []
    for csv_path in csv_files:
        print(f"[{recording_name}/{stage_in}] processing {csv_path.name}")
        all_results.extend(
            _process_sensor_csv(
                recording_name=recording_name,
                stage_in=stage_in,
                csv_path=csv_path,
                static_window_ms=float(args.static_window_ms),
                gravity=float(args.gravity),
            )
        )

    if not all_results:
        print("No orientation results produced (empty inputs?).")
        return

    out_dir = recording_stage_dir(recording_name, "orientation")
    json_path = out_dir / "orientation_stats.json"
    json_path.write_text(
        json.dumps([r.__dict__ for r in all_results], indent=2),
        encoding="utf-8",
    )
    print(f"Orientation stats written to {json_path}")


if __name__ == "__main__":
    main()
