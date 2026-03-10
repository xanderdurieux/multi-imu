"""Lightweight bias calibration for use inside orientation filters.

This module provides simple, field-deployable helpers to estimate **constant
accelerometer and gyroscope bias** from short static segments. The resulting
``BiasCalibration`` objects are used by the orientation pipelines
(:mod:`orientation.pipeline`, :mod:`orientation.session`) to optionally
de-bias IMU streams before running complementary or Madgwick filters.

For full sensor calibration and world-frame alignment (including gravity
vector, magnetometer hard-iron offset, and sensor-to-world rotation), see
``calibration.per_sensor`` and the recording-level pipeline in
``calibration.session``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class BiasCalibration:
    """Constant sensor bias for accelerometer and gyroscope (sensor/body frame)."""

    accel_bias: np.ndarray  # shape (3,)
    gyro_bias: np.ndarray  # shape (3,)


def _as_array3(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError("Expected 3-vector.")
    return arr


def estimate_gyro_bias_static(gyro_samples_rad: np.ndarray) -> np.ndarray:
    """Estimate constant gyroscope bias from a static recording.

    Parameters
    ----------
    gyro_samples_rad:
        Array of shape (N, 3) with angular velocity in rad/s for a period where
        the sensor is at rest (true angular velocity ≈ 0).
    """
    if gyro_samples_rad.ndim != 2 or gyro_samples_rad.shape[1] != 3:
        raise ValueError("gyro_samples_rad must have shape (N, 3).")
    return np.nanmean(gyro_samples_rad, axis=0)


def estimate_acc_bias_static(
    acc_samples_ms2: np.ndarray,
    expected_gravity_body: Iterable[float] | None = None,
) -> np.ndarray:
    """Estimate constant accelerometer bias from a static pose.

    Parameters
    ----------
    acc_samples_ms2:
        Array of shape (N, 3) with specific force readings in m/s^2 during a
        static period (only gravity is present, no linear acceleration).
    expected_gravity_body:
        Expected gravity vector in the *sensor/body* frame for this pose
        (default assumes sensor z-axis aligned with world -Z, i.e. device lying
        flat with z up: ``[0, 0, -9.81]``).

    Notes
    -----
    - In a more sophisticated multi-pose calibration you would solve for scale
      and axis misalignment as well (e.g. Tedaldi et al. 2014). Here we restrict
      ourselves to a constant bias vector, which is usually the dominant term.
    """
    if expected_gravity_body is None:
        expected_gravity_body = np.array([0.0, 0.0, -9.81], dtype=float)
    g_body = _as_array3(expected_gravity_body)

    if acc_samples_ms2.ndim != 2 or acc_samples_ms2.shape[1] != 3:
        raise ValueError("acc_samples_ms2 must have shape (N, 3).")

    mean_measured = np.nanmean(acc_samples_ms2, axis=0)
    return mean_measured - g_body


def apply_calibration_bias(
    df: pd.DataFrame,
    accel_bias: Iterable[float] | None = None,
    gyro_bias: Iterable[float] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply constant bias correction to accelerometer and gyroscope columns.

    Parameters
    ----------
    df:
        DataFrame with columns ``ax, ay, az, gx, gy, gz`` (as produced by the
        existing parsers).
    accel_bias, gyro_bias:
        3-vectors in sensor/body frame. If ``None``, the corresponding sensor
        is left unchanged.
    inplace:
        If ``True``, modify ``df`` in-place and return it. Otherwise work on a
        copy.
    """
    if not inplace:
        df = df.copy()

    if accel_bias is not None:
        b = _as_array3(accel_bias)
        for col, val in zip(["ax", "ay", "az"], b):
            if col in df.columns:
                df[col] = df[col] - val

    if gyro_bias is not None:
        b = _as_array3(gyro_bias)
        for col, val in zip(["gx", "gy", "gz"], b):
            if col in df.columns:
                df[col] = df[col] - val

    return df


def estimate_bias_from_dataframe_static_segment(
    df: pd.DataFrame,
    start_time: float,
    end_time: float,
    expected_gravity_body: Iterable[float] | None = None,
) -> BiasCalibration:
    """Convenience wrapper: estimate biases from a static segment in a session.

    Parameters
    ----------
    df:
        IMU DataFrame with columns ``timestamp, ax, ay, az, gx, gy, gz``.
    start_time, end_time:
        Timestamps defining a static interval in the same units as ``df["timestamp"]``.
    expected_gravity_body:
        Expected gravity vector for this pose in body frame (see
        :func:`estimate_acc_bias_static`).
    """
    mask = (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)
    segment = df.loc[mask]
    acc = segment[["ax", "ay", "az"]].to_numpy()
    gyro = segment[["gx", "gy", "gz"]].to_numpy()

    accel_bias = estimate_acc_bias_static(acc, expected_gravity_body=expected_gravity_body)
    gyro_bias = estimate_gyro_bias_static(gyro)
    return BiasCalibration(accel_bias=accel_bias, gyro_bias=gyro_bias)

