"""Extract labelled sliding-window features from section signals."""

from .lag_features import add_lag_features, lag_column_names
from .pipeline import (
    extract_features_for_section,
    process_section_features,
    process_recording_features,
)

__all__ = [
    "extract_features_for_section",
    "process_section_features",
    "process_recording_features",
    "add_lag_features",
    "lag_column_names",
]
