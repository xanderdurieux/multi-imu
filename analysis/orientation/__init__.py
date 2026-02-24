"""Orientation estimation utilities for IMU CSV streams.

This package provides:

- Sensor-level calibration helpers (bias estimation).
- Lightweight complementary and Madgwick filters producing body→world quaternions.
- Simple pipelines that run the filters on standardized CSV data from ``analysis.parser``.
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
    tilt_quat_from_acc,
)
from .calibration import (
    BiasCalibration,
    estimate_gyro_bias_static,
    estimate_acc_bias_static,
    estimate_bias_from_dataframe_static_segment,
    apply_calibration_bias,
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
    "tilt_quat_from_acc",
    # Calibration
    "BiasCalibration",
    "estimate_gyro_bias_static",
    "estimate_acc_bias_static",
    "estimate_bias_from_dataframe_static_segment",
    "apply_calibration_bias",
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

