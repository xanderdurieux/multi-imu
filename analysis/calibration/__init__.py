"""IMU calibration and world-frame alignment.

Stages
------
1. Estimate accelerometer and gyroscope biases from static segments.
2. Compute gravity vector in sensor frame.
3. Optionally estimate forward direction from motion segments.
4. Build a rotation matrix (sensor → world frame).
5. Apply calibration to produce ``calibrated/<sensor>.csv`` and
   ``calibrated/calibration.json``.

CLI::

    python -m calibration <section_name>
    python -m calibration <section_name> --frame gravity_plus_forward
    python -m calibration --session 2026-02-26 --all
"""

from .core import (
    CalibrationParams,
    SectionCalibration,
    estimate_calibration,
    apply_calibration,
)
from .pipeline import calibrate_section, calibrate_recording_sections

__all__ = [
    "CalibrationParams",
    "SectionCalibration",
    "estimate_calibration",
    "apply_calibration",
    "calibrate_section",
    "calibrate_recording_sections",
]
