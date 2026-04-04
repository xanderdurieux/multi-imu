"""Orientation estimation for the dual-IMU cycling pipeline.

Implements Madgwick (IMU/MARG), complementary, and quaternion EKF filters plus
a section/recording pipeline that scores variants and flattens the best result.

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
    ALL_ORIENTATION_METHODS,
    DEFAULT_CANONICAL_ORIENTATION_METHOD,
    DEFAULT_ORIENTATION_VARIANTS,
    OrientationStats,
    run_orientation_filters,
    process_section_orientation,
    process_recording_orientation,
)

__all__ = [
    "ALL_ORIENTATION_METHODS",
    "DEFAULT_CANONICAL_ORIENTATION_METHOD",
    "DEFAULT_ORIENTATION_VARIANTS",
    "OrientationStats",
    "run_orientation_filters",
    "process_section_orientation",
    "process_recording_orientation",
]
