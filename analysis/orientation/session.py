"""Recording-level orientation pipeline.

Reads body-frame sensor CSVs from the ``synced`` stage, loads static
calibration parameters from ``calibrated/calibration.json`` (produced by
:func:`calibration.session.calibrate_recording`), and runs complementary and
Madgwick orientation filters to produce per-sample quaternion tracks.

Why ``synced`` and not ``parsed``?
-----------------------------------
The Sporsa sensor already uses absolute wall-clock timestamps, so its
``parsed`` and ``synced`` CSVs are identical.  The Arduino sensor uses a
local clock that starts from zero and accumulates drift.  The sync stage
corrects that clock to the same absolute epoch as Sporsa, which means:

* The ``dt`` values fed to the orientation filter are accurate (no drift
  from the Arduino's uncorrected ~18 kHz local clock).
* Both sensors share a common time axis, so relative-orientation analysis
  is geometrically correct without any extra time-alignment step.

The sensor readings (acc, gyro, mag columns) are unchanged by syncing.

Pipeline
--------
1. Load gyro bias and static sensor-to-world rotation from
   ``calibrated/calibration.json``.
2. For each sensor CSV in ``data/recordings/<recording>/<stage>/``:

   a. Subtract gyro bias (deg/s) before filtering.
   b. Convert gyro to rad/s.
   c. Initialize each filter with the calibration quaternion.
   d. Run complementary filter → ``orientation/complementary/<sensor>_orientation.csv``.
   e. Run Madgwick filter    → ``orientation/madgwick/<sensor>_orientation.csv``.

3. Compute quality statistics and write ``orientation_stats.json``.

Output layout::

    orientation/
    ├── complementary/
    │   ├── sporsa_orientation.csv
    │   └── arduino_orientation.csv
    ├── madgwick/
    │   ├── sporsa_orientation.csv
    │   └── arduino_orientation.csv
    └── orientation_stats.json

CLI::

    python -m orientation.session 2026-02-26_5
    python -m orientation.session 2026-02-26_5 --stage synced
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common import load_dataframe, recording_stage_dir, write_dataframe
from common.paths import find_sensor_csv
from .calibration import load_calibration_params
from .pipeline import (
    run_complementary_on_dataframe,
    run_madgwick_on_dataframe,
)
from .quaternion import quat_rotate

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")


@dataclass
class OrientationMethodResult:
    """Metadata and quality metrics for one (filter) run on one sensor CSV."""

    recording: str
    stage_in: str
    sensor_name: str
    filter_type: str
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

    g_err = np.linalg.norm(acc_world, axis=1) - float(gravity)

    acc_norm_body = np.linalg.norm(acc_body, axis=1)
    gyro_cols = [c for c in ("gx", "gy", "gz") if c in df.columns]
    gyro_norm = (
        np.linalg.norm(df[gyro_cols].to_numpy(dtype=float), axis=1)
        if gyro_cols
        else np.zeros(len(df), dtype=float)
    )

    # Gyro columns in df_orient are in deg/s (same as the CSV source).
    # 0.1 rad/s ≈ 5.73 deg/s is a reasonable "static" threshold.
    gyro_static_threshold = float(np.degrees(0.1))
    static_mask = (
        np.isfinite(acc_norm_body)
        & np.isfinite(gyro_norm)
        & (np.abs(acc_norm_body - float(gravity)) < 0.1 * float(gravity))
        & (gyro_norm < gyro_static_threshold)
    )

    def _safe(fn, arr, default=np.nan):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or not np.any(np.isfinite(arr)):
            return float(default)
        return float(fn(arr))

    metrics: dict = {
        "g_err_mean": _safe(np.nanmean, g_err),
        "g_err_std": _safe(np.nanstd, g_err),
        "g_err_abs_mean": _safe(np.nanmean, np.abs(g_err)),
        "g_err_abs_p95": _safe(lambda x: np.nanpercentile(x, 95.0), np.abs(g_err)),
        "static_fraction": float(static_mask.mean() if len(static_mask) else 0.0),
    }

    if {"pitch_deg", "roll_deg"}.issubset(df.columns):
        pitch_static = df["pitch_deg"].to_numpy(dtype=float)[static_mask]
        roll_static = df["roll_deg"].to_numpy(dtype=float)[static_mask]
        metrics["pitch_static_std_deg"] = _safe(np.nanstd, pitch_static)
        metrics["roll_static_std_deg"] = _safe(np.nanstd, roll_static)
        metrics["num_static_samples"] = int(len(pitch_static))
    else:
        metrics["pitch_static_std_deg"] = float("nan")
        metrics["roll_static_std_deg"] = float("nan")
        metrics["num_static_samples"] = 0

    return metrics


def _process_sensor(
    recording_name: str,
    stage_in: str,
    sensor_name: str,
    gyro_bias: np.ndarray,
    initial_q: np.ndarray,
    gravity: float = 9.81,
) -> list[OrientationMethodResult]:
    """Run orientation filters for one sensor and return quality stats."""
    try:
        csv_path = find_sensor_csv(recording_name, stage_in, sensor_name)
    except FileNotFoundError:
        # Fall back to 'parsed' if the requested stage has no CSV yet
        # (e.g. single-sensor recordings that were never synced).
        if stage_in != "parsed":
            try:
                csv_path = find_sensor_csv(recording_name, "parsed", sensor_name)
                log.warning(
                    "No CSV for '%s' in stage '%s' — falling back to 'parsed'.",
                    sensor_name, stage_in,
                )
            except FileNotFoundError:
                log.warning(
                    "No CSV for sensor '%s' in '%s' or 'parsed'. Skipping.",
                    sensor_name, recording_name,
                )
                return []
        else:
            log.warning(
                "No CSV for sensor '%s' in stage '%s/%s'. Skipping.",
                sensor_name, recording_name, stage_in,
            )
            return []

    df = load_dataframe(csv_path)
    if df.empty:
        log.warning("Empty CSV for sensor '%s'. Skipping.", sensor_name)
        return []

    orientation_root = recording_stage_dir(recording_name, "orientation")
    results: list[OrientationMethodResult] = []

    for filter_type, run_fn in (
        ("complementary", run_complementary_on_dataframe),
        ("madgwick", run_madgwick_on_dataframe),
    ):
        print(
            f"[{recording_name}/orientation/{filter_type}] {sensor_name} ← {stage_in}"
        )
        df_orient = run_fn(df, gyro_bias=gyro_bias, initial_q=initial_q)

        out_dir = orientation_root / filter_type
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sensor_name}_orientation.csv"
        write_dataframe(df_orient, out_path)

        try:
            stats = _compute_orientation_stats(df_orient, gravity=gravity)
        except Exception as exc:
            log.warning("Stats computation failed for %s/%s: %s", sensor_name, filter_type, exc)
            stats = {
                "g_err_mean": float("nan"),
                "g_err_std": float("nan"),
                "g_err_abs_mean": float("nan"),
                "g_err_abs_p95": float("nan"),
                "static_fraction": float("nan"),
                "pitch_static_std_deg": float("nan"),
                "roll_static_std_deg": float("nan"),
                "num_static_samples": 0,
            }

        results.append(OrientationMethodResult(
            recording=recording_name,
            stage_in=stage_in,
            sensor_name=sensor_name,
            filter_type=filter_type,
            output_csv=str(out_path),
            g_err_mean=float(stats["g_err_mean"]),
            g_err_std=float(stats["g_err_std"]),
            g_err_abs_mean=float(stats["g_err_abs_mean"]),
            g_err_abs_p95=float(stats["g_err_abs_p95"]),
            static_fraction=float(stats["static_fraction"]),
            pitch_static_std_deg=float(stats["pitch_static_std_deg"]),
            roll_static_std_deg=float(stats["roll_static_std_deg"]),
            num_static_samples=int(stats["num_static_samples"]),
        ))

    return results


def run_orientation_for_recording(
    recording_name: str,
    stage_in: str = "synced",
    gravity: float = 9.81,
) -> Path:
    """Run orientation estimation for all sensors in a recording.

    Reads body-frame sensor CSVs from
    ``data/recordings/<recording_name>/<stage_in>/``, loads calibration
    parameters from ``calibrated/calibration.json``, runs complementary and
    Madgwick filters, and writes results to per-method subdirectories under
    ``data/recordings/<recording_name>/orientation/<method>/``.

    Parameters
    ----------
    recording_name:
        Recording identifier, e.g. ``"2026-02-26_5"``.
    stage_in:
        Stage containing the body-frame sensor CSVs (default: ``"synced"``).
        Use ``"parsed"`` only if sync has not been run for this recording.
    gravity:
        Assumed gravity magnitude in m/s² (default: 9.81).

    Returns
    -------
    Path
        Path to the written ``orientation_stats.json``.

    Raises
    ------
    FileNotFoundError
        If ``calibrated/calibration.json`` does not exist.
    """
    all_results: list[OrientationMethodResult] = []

    for sensor_name in _SENSORS:
        try:
            gyro_bias, initial_q = load_calibration_params(recording_name, sensor_name)
        except FileNotFoundError as exc:
            raise
        except KeyError:
            log.warning(
                "Sensor '%s' not in calibration.json for recording '%s'. Skipping.",
                sensor_name, recording_name,
            )
            continue

        results = _process_sensor(
            recording_name=recording_name,
            stage_in=stage_in,
            sensor_name=sensor_name,
            gyro_bias=gyro_bias,
            initial_q=initial_q,
            gravity=gravity,
        )
        all_results.extend(results)

    out_dir = recording_stage_dir(recording_name, "orientation")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "orientation_stats.json"
    json_path.write_text(
        json.dumps([r.__dict__ for r in all_results], indent=2),
        encoding="utf-8",
    )
    print(f"[{recording_name}/orientation] orientation_stats.json")
    return json_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m orientation.session",
        description=(
            "Run complementary and Madgwick orientation filters for all sensors "
            "in a recording.  Reads body-frame CSVs from the given stage and "
            "calibration parameters from calibrated/calibration.json."
        ),
    )
    parser.add_argument(
        "recording_name",
        help="Recording identifier, e.g. '2026-02-26_5'.",
    )
    parser.add_argument(
        "--stage",
        default="synced",
        help="Input stage with body-frame sensor CSVs (default: synced).",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=9.81,
        help="Assumed gravity magnitude in m/s² (default: 9.81).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    json_path = run_orientation_for_recording(
        recording_name=args.recording_name,
        stage_in=args.stage,
        gravity=args.gravity,
    )
    print(f"\norientation stats: {json_path}")


if __name__ == "__main__":
    main()
