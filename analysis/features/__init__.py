"""Extract labelled sliding-window features from section signals."""

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
