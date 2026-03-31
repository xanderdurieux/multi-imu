"""Orientation estimation package for dual-IMU cycling pipeline.

Provides Madgwick and complementary filter implementations as well as a
pipeline for processing calibrated section CSVs into orientation outputs.

Usage
-----
::

    from orientation import run_orientation_filters, process_section_orientation
    from orientation import process_recording_orientation, OrientationStats

CLI::

    python -m orientation <section_name>
    python -m orientation --recording <recording_name>
"""

from .pipeline import (
    OrientationStats,
    run_orientation_filters,
    process_section_orientation,
    process_recording_orientation,
)

__all__ = [
    "OrientationStats",
    "run_orientation_filters",
    "process_section_orientation",
    "process_recording_orientation",
]
