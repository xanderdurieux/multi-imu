"""Bridge between calibration/ outputs and orientation/ filters.

This module loads the static calibration parameters produced by
:mod:`calibration.session` (gyro bias and sensor-to-world rotation) and
applies them to raw body-frame IMU DataFrames before running orientation
filters.

The orientation filters (:mod:`orientation.complementary`,
:mod:`orientation.madgwick`) require **body-frame** sensor data and an
optional initial orientation quaternion.  Two things are needed from the
upstream calibration:

1. **Gyro bias** — subtract from ``gx/gy/gz`` before integration so the
   filter does not accumulate a constant angular velocity offset.
2. **Initial orientation** — the static sensor-to-world rotation estimated
   during the calibration sequence, used to seed the filter with the correct
   starting pose rather than identity.

Both are read from ``calibrated/calibration.json`` produced by
:func:`calibration.session.calibrate_recording`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import recording_stage_dir
from .quaternion import quat_from_rotation_matrix


def load_calibration_params(
    recording_name: str,
    sensor_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load gyro bias and initial orientation from ``calibration.json``.

    Parameters
    ----------
    recording_name:
        Recording identifier, e.g. ``"2026-02-26_5"``.
    sensor_name:
        Sensor identifier, e.g. ``"sporsa"`` or ``"arduino"``.

    Returns
    -------
    gyro_bias:
        Shape ``(3,)`` gyro zero-rate bias in the same units as the raw CSV
        ``gx/gy/gz`` columns.  Subtract from raw readings before filtering.
    initial_q:
        Shape ``(4,)`` unit quaternion ``[w, x, y, z]`` representing the
        static sensor-to-world rotation at calibration time.  Use as the
        filter's starting pose.

    Raises
    ------
    FileNotFoundError
        If ``calibrated/calibration.json`` does not exist.  Run
        :func:`calibration.session.calibrate_recording` first.
    KeyError
        If *sensor_name* is not present in ``calibration.json``.
    """
    cal_path = recording_stage_dir(recording_name, "calibrated") / "calibration.json"
    if not cal_path.exists():
        raise FileNotFoundError(
            f"calibration.json not found at {cal_path}. "
            "Run calibration.session first: "
            f"python -m calibration.session {recording_name}"
        )

    cal = json.loads(cal_path.read_text(encoding="utf-8"))

    if sensor_name not in cal:
        available = [k for k in cal if k != "metadata"]
        raise KeyError(
            f"Sensor '{sensor_name}' not found in {cal_path}. "
            f"Available sensors: {available}"
        )

    sensor_cal = cal[sensor_name]
    gyro_bias = np.array(sensor_cal["gyro_bias_deg_per_s"], dtype=float)
    R = np.array(sensor_cal["rotation_sensor_to_world"], dtype=float)
    initial_q = quat_from_rotation_matrix(R)

    return gyro_bias, initial_q


def apply_gyro_bias(
    df: pd.DataFrame,
    gyro_bias: np.ndarray,
) -> pd.DataFrame:
    """Subtract gyroscope bias from ``gx/gy/gz`` columns.

    Parameters
    ----------
    df:
        IMU DataFrame with ``gx``, ``gy``, ``gz`` columns.
    gyro_bias:
        Shape ``(3,)`` bias in the same units as the gyro columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with gyro bias subtracted.
    """
    out = df.copy()
    for col, bias_val in zip(["gx", "gy", "gz"], gyro_bias):
        if col in out.columns:
            out[col] = out[col] - bias_val
    return out
