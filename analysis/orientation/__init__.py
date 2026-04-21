"""Mahony orientation estimation for the dual-IMU cycling pipeline.

CLI::

    python -m orientation <section_name>
    python -m orientation --recording <recording_name>
"""

from .filters import run_mahony
from .pipeline import process_section_orientation, process_recording_orientation

__all__ = [
    "run_mahony",
    "process_section_orientation",
    "process_recording_orientation",
]
