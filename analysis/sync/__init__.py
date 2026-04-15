"""Synchronization package for the Multi-IMU analysis pipeline.

Tier hierarchy (strongest to weakest):
  1. multi_anchor        — shared anchor extraction + anchor-fit drift
  2. one_anchor_adaptive — first anchor + causal signal refinement
  3. one_anchor_prior    — first anchor + fixed drift prior
  4. signal_only         — SDA/LIDA without anchors
"""

from .model import (
    SyncModel,
    apply_linear_time_transform,
    apply_sync_model,
    make_sync_model,
)
from .orchestrate import (
    SYNC_METHODS,
    SyncMethodQuality,
    SyncSelectionResult,
)
from .pipeline import (
    main,
    synchronize_from_calibration,
    synchronize_recording_all_methods,
    synchronize_recording_chosen_method,
)

__all__ = [
    "SYNC_METHODS",
    "SyncMethodQuality",
    "SyncModel",
    "SyncSelectionResult",
    "apply_linear_time_transform",
    "apply_sync_model",
    "main",
    "make_sync_model",
    "synchronize_from_calibration",
    "synchronize_recording_all_methods",
    "synchronize_recording_chosen_method",
]
