"""Derived signal computation stage for the dual-IMU cycling pipeline.

Computes physical signals from calibrated IMU data for each sensor and
cross-sensor disagreement signals.

Outputs per section
-------------------
- ``<section_dir>/derived/sporsa_signals.csv``   — per-sensor derived signals
- ``<section_dir>/derived/arduino_signals.csv``  — per-sensor derived signals
- ``<section_dir>/derived/cross_sensor_signals.csv`` — cross-sensor comparison

CLI::

    python -m derived <section_name>
    python -m derived --recording <recording_name>
"""

from .signals import compute_sensor_signals, compute_cross_sensor_signals
from .pipeline import process_section_derived, process_recording_derived

__all__ = [
    "compute_sensor_signals",
    "compute_cross_sensor_signals",
    "process_section_derived",
    "process_recording_derived",
]
