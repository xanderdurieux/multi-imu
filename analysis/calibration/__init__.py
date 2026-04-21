"""Protocol-aware IMU calibration and world-frame alignment.

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
