"""Align Arduino timestamps to the Sporsa reference clock."""

from .model import (
    SyncModel,
    apply_linear_time_transform,
    apply_sync_model,
)
from .pipeline import (
    main,
    synchronize_recording_all_methods,
    synchronize_recording_chosen_method,
)
from .selection import (
    SYNC_METHODS,
    SyncMethodQuality,
    SyncSelectionResult,
)

__all__ = [
    "SYNC_METHODS",
    "SyncMethodQuality",
    "SyncModel",
    "SyncSelectionResult",
    "apply_linear_time_transform",
    "apply_sync_model",
    "main",
    "synchronize_recording_all_methods",
    "synchronize_recording_chosen_method",
]
