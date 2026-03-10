"""Feature extraction utilities for IMU recordings and sections.

This package provides two layers:

- Low-level helpers in :mod:`features.window_features` that compute features
  for a single contiguous IMU time series in a :class:`pandas.DataFrame`.
- Recording / section orchestrators in :mod:`features.section_features` that
  load the appropriate CSVs under ``data/recordings/<recording_name>/`` and
  write consolidated feature tables to ``features/``.
"""

from .window_features import compute_time_series_features
from .section_features import (
    extract_recording_features,
    extract_section_features,
)

__all__ = [
    "compute_time_series_features",
    "extract_recording_features",
    "extract_section_features",
]

