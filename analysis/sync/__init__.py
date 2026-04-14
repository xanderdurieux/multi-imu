"""Synchronization package for the Multi-IMU analysis pipeline.

Tier hierarchy (strongest to weakest):
  1. multi_anchor      — opening + closing protocol anchors
  2. one_anchor_adaptive — opening anchor + causal windowed drift
  3. one_anchor_prior   — opening anchor + pre-characterised drift
  4. signal_only        — SDA offset + LIDA-style windowed drift
"""

from .model import (
    SyncModel,
    apply_linear_time_transform,
    apply_sync_model,
    load_sync_model,
    make_sync_model,
    save_sync_model,
)
from .orchestrate import (
    METHOD_LABELS,
    METHOD_STAGES,
    SYNC_METHODS,
    SyncMethodQuality,
    SyncSelectionResult,
    compare_sync_models,
    method_label,
    method_stage,
    print_comparison,
    print_selection_result,
    select_best_sync_method,
)
from .pipeline import (
    ALL_METHODS,
    MethodResult,
    RecordingResult,
    apply_selection,
    main,
    prune_method_stage_directories,
    synchronize_from_calibration,
    synchronize_recording_all_methods,
    synchronize_recording_chosen_method,
    synchronize_session,
)

__all__ = [
    "ALL_METHODS",
    "METHOD_LABELS",
    "METHOD_STAGES",
    "SYNC_METHODS",
    "MethodResult",
    "RecordingResult",
    "SyncMethodQuality",
    "SyncModel",
    "SyncSelectionResult",
    "apply_linear_time_transform",
    "apply_selection",
    "apply_sync_model",
    "compare_sync_models",
    "load_sync_model",
    "main",
    "make_sync_model",
    "method_label",
    "method_stage",
    "print_comparison",
    "print_selection_result",
    "prune_method_stage_directories",
    "save_sync_model",
    "select_best_sync_method",
    "synchronize_from_calibration",
    "synchronize_recording_all_methods",
    "synchronize_recording_chosen_method",
    "synchronize_session",
]
