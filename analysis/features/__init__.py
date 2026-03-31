"""Windowed feature extraction stage for the dual-IMU cycling pipeline.

Computes a flat feature vector for each sliding time window over a section's
calibrated, derived, and orientation data.

Outputs per section
-------------------
- ``<section_dir>/features/features.csv``        — one row per window
- ``<section_dir>/features/features_stats.json``

CLI::

    python -m features <section_name>
    python -m features --recording <recording_name>
"""

from .pipeline import (
    extract_features_for_section,
    process_section_features,
    process_recording_features,
)

__all__ = [
    "extract_features_for_section",
    "process_section_features",
    "process_recording_features",
]
