"""Orientation estimation for the dual-IMU cycling pipeline.

Four AHRS filters (Madgwick, Mahony, Complementary, EKF) are run and scored
using a gravity-residual metric in the known-static opening windows.  The
best-performing method's quaternion output is written as the canonical
orientation for each section.

Usage
-----
::

    from orientation import process_section_orientation, process_recording_orientation

CLI::

    python -m orientation <section_name>
    python -m orientation --recording <recording_name>
"""

from .filters import METHODS, DEFAULT_PARAMS, run_filter
from .pipeline import process_section_orientation, process_recording_orientation

__all__ = [
    "METHODS",
    "DEFAULT_PARAMS",
    "run_filter",
    "process_section_orientation",
    "process_recording_orientation",
]
