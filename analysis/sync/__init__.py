"""Synchronization package: four methods + automatic selection.

From ``analysis/``::

    python -m sync 2026-02-26_r5
    python -m sync 2026-02-26 --all

Runs every method on ``parsed/``, selects the best, writes flat ``synced/`` with plots.
See :mod:`sync.run`.
"""

from .core import (
    AlignmentSeries,
    OffsetEstimate,
    SyncModel,
    add_vector_norms,
    apply_linear_time_transform,
    apply_sync_model,
    build_activity_signal,
    build_alignment_series,
    estimate_lag,
    estimate_offset,
    estimate_sync_model,
    fit_sync_from_paths,
    load_stream,
    load_sync_model,
    remove_dropouts,
    resample_aligned_stream,
    resample_stream,
    resample_to_reference_timestamps,
    save_sync_model,
)
from .sda_sync import synchronize_recording_sda
from .lida_sync import synchronize, synchronize_recording
from .calibration_sync import (
    CalibrationWindowResult,
    estimate_sync_from_calibration,
    synchronize_from_calibration,
    synchronize_recording_from_calibration,
)
from .online_sync import (
    estimate_sync_from_opening_anchor,
    load_characterised_drift,
    synchronize_recording_online,
)
from .plots import (
    plot_method_scores,
    plot_methods_norm_grid,
    plot_synced_norm_overlay,
)
from .selection import (
    SyncMethodQuality,
    SyncSelectionResult,
    apply_selection,
    compare_sync_models,
    prune_method_stage_directories,
    print_comparison,
    print_selection_result,
    select_best_sync_method,
)
from .pipeline import (
    MethodResult,
    RecordingResult,
    synchronize_recording_all_methods,
    synchronize_session,
)

__all__ = [
    "AlignmentSeries",
    "OffsetEstimate",
    "SyncModel",
    "build_activity_signal",
    "build_alignment_series",
    "estimate_lag",
    "estimate_offset",
    "estimate_sync_model",
    "fit_sync_from_paths",
    "add_vector_norms",
    "apply_linear_time_transform",
    "apply_sync_model",
    "load_stream",
    "load_sync_model",
    "remove_dropouts",
    "resample_aligned_stream",
    "resample_stream",
    "resample_to_reference_timestamps",
    "save_sync_model",
    "synchronize_recording_sda",
    "synchronize",
    "synchronize_recording",
    "CalibrationWindowResult",
    "estimate_sync_from_calibration",
    "synchronize_from_calibration",
    "synchronize_recording_from_calibration",
    "estimate_sync_from_opening_anchor",
    "load_characterised_drift",
    "synchronize_recording_online",
    "SyncMethodQuality",
    "SyncSelectionResult",
    "apply_selection",
    "compare_sync_models",
    "plot_method_scores",
    "plot_methods_norm_grid",
    "plot_synced_norm_overlay",
    "prune_method_stage_directories",
    "print_comparison",
    "print_selection_result",
    "select_best_sync_method",
    "MethodResult",
    "RecordingResult",
    "synchronize_recording_all_methods",
    "synchronize_session",
]
