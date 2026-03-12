"""Orientation estimation utilities for IMU CSV streams.

This package provides:

- Quaternion math utilities.
- Helpers to load calibration parameters from ``calibrated/calibration.json``
  (produced by :mod:`calibration.session`) and apply gyro bias correction.
- Lightweight complementary and Madgwick filters producing body→world quaternions.
- Pipelines that run the filters on standardized CSV data.

Typical usage::

    from orientation.session import run_orientation_for_recording
    run_orientation_for_recording("2026-02-26_5")

The pipeline reads body-frame ``parsed/`` sensor CSVs, loads gyro bias and
the static sensor-to-world rotation from ``calibrated/calibration.json``,
and writes quaternion-augmented orientation CSVs to ``orientation/``.
"""

from .quaternion import (
    quat_identity,
    quat_normalize,
    quat_conjugate,
    quat_multiply,
    quat_from_axis_angle,
    quat_from_gyro,
    quat_rotate,
    quat_slerp,
    euler_from_quat,
    quat_from_euler,
    quat_from_rotation_matrix,
    tilt_quat_from_acc,
)
from .calibration import (
    load_calibration_params,
    apply_gyro_bias,
)
from .complementary import ComplementaryFilterConfig, ComplementaryOrientationFilter
from .madgwick import MadgwickConfig, MadgwickOrientationFilter
from .pipeline import (
    OrientationPipelineConfig,
    run_complementary_on_dataframe,
    run_madgwick_on_dataframe,
    run_orientation_pipeline_on_csv,
)

__all__ = [
    # Quaternion utilities
    "quat_identity",
    "quat_normalize",
    "quat_conjugate",
    "quat_multiply",
    "quat_from_axis_angle",
    "quat_from_gyro",
    "quat_rotate",
    "quat_slerp",
    "euler_from_quat",
    "quat_from_euler",
    "quat_from_rotation_matrix",
    "tilt_quat_from_acc",
    # Calibration bridge
    "load_calibration_params",
    "apply_gyro_bias",
    # Complementary filter
    "ComplementaryFilterConfig",
    "ComplementaryOrientationFilter",
    # Madgwick filter
    "MadgwickConfig",
    "MadgwickOrientationFilter",
    # Pipelines
    "OrientationPipelineConfig",
    "run_complementary_on_dataframe",
    "run_madgwick_on_dataframe",
    "run_orientation_pipeline_on_csv",
]
