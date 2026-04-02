"""Protocol-aware IMU calibration and world-frame alignment.

Stages
------
1. Detect opening routine (pre-tap static → taps → post-tap static).
2. Estimate sensor intrinsics (gyro bias; acc from static hardware cal if available).
3. Estimate alignment rotation from first post-mount stable window (Arduino)
   or opening static window (Sporsa).
4. Apply calibration: bias-correct raw axes, rotate to world frame.
5. Write ``calibrated/<sensor>.csv`` and ``calibrated/calibration.json``.

CLI::

    python -m calibration <section_name>
    python -m calibration --recording 2026-02-26_r1 --force
"""

from .core import (
    OpeningSequence,
    SensorIntrinsics,
    SensorAlignment,
    SectionCalibration,
    detect_protocol_landmarks,
    estimate_sensor_intrinsics,
    estimate_sensor_alignment,
    apply_calibration,
)
from .pipeline import calibrate_section, calibrate_recording_sections

__all__ = [
    "OpeningSequence",
    "SensorIntrinsics",
    "SensorAlignment",
    "SectionCalibration",
    "detect_protocol_landmarks",
    "estimate_sensor_intrinsics",
    "estimate_sensor_alignment",
    "apply_calibration",
    "calibrate_section",
    "calibrate_recording_sections",
]
