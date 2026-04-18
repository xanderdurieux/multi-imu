"""Recording-level synchronization package.

Tier hierarchy (strongest to weakest):
  1. multi_anchor        — shared anchor extraction + anchor-fit drift
  2. one_anchor_adaptive — first anchor + causal signal refinement
  3. one_anchor_prior    — first anchor + fixed drift prior
  4. signal_only         — coarse signal alignment + windowed drift refinement
"""

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
