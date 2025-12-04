"""Multi-IMU processing utilities.

Provides modules for loading, synchronizing, aligning, visualizing, and analyzing IMU datasets from multiple sensors.
"""

from .data_models import IMUSensorData, SyncedIMUData
from .io import (
    export_raw_session,
    load_imu_csv,
    parse_arduino_log,
    parse_phone_log,
    parse_sporsa_log,
)
from .preprocessing import resample_signal, remove_gravity
from .synchronization import estimate_time_offset, synchronize_streams
from .alignment import align_axes, compute_alignment_matrix
from .analysis import detect_falls, detect_braking_events, detect_turns
from .visualization import (
    plot_timeseries,
    plot_comparison,
    plot_magnitude,
    plot_event_annotations,
)

__all__ = [
    "IMUSensorData",
    "SyncedIMUData",
    "export_raw_session",
    "load_imu_csv",
    "parse_arduino_log",
    "parse_phone_log",
    "parse_sporsa_log",
    "resample_signal",
    "remove_gravity",
    "estimate_time_offset",
    "synchronize_streams",
    "align_axes",
    "compute_alignment_matrix",
    "detect_falls",
    "detect_braking_events",
    "detect_turns",
    "plot_timeseries",
    "plot_comparison",
    "plot_magnitude",
    "plot_event_annotations",
]
