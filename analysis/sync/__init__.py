"""Synchronization package implementing SDA + LIDA stream alignment.

Also provides a calibration-sequence based sync variant via
:mod:`sync.calibration_sync`, and a comparison utility for evaluating
both approaches side-by-side via :mod:`sync.compare_sync`.
"""

from .align_df import (
    AlignmentSeries,
    OffsetEstimate,
    build_activity_signal,
    build_alignment_series,
    estimate_lag,
    estimate_offset,
)
from .common import (
    add_vector_norms,
    apply_linear_time_transform,
    load_stream,
    resample_stream,
    resample_to_reference_timestamps,
)
from .drift_estimator import (
    SyncModel,
    apply_sync_model,
    estimate_sync_model,
    fit_sync_from_paths,
    load_sync_model,
    resample_aligned_stream,
    save_sync_model,
)
from .calibration_sync import (
    CalibrationWindowResult,
    estimate_sync_from_calibration,
    synchronize_from_calibration,
    synchronize_recording_from_calibration,
)
from .compare_sync import (
    compare_sync_models,
    compare_all_recordings,
    plot_sync_comparison,
)

__all__ = [
    "CalibrationWindowResult",
    "estimate_sync_from_calibration",
    "synchronize_from_calibration",
    "synchronize_recording_from_calibration",
    "compare_sync_models",
    "compare_all_recordings",
    "plot_sync_comparison",
    "AlignmentSeries",
    "OffsetEstimate",
    "SyncModel",
    "add_vector_norms",
    "apply_linear_time_transform",
    "apply_sync_model",
    "build_activity_signal",
    "build_alignment_series",
    "estimate_lag",
    "estimate_offset",
    "estimate_sync_model",
    "fit_sync_from_paths",
    "load_stream",
    "load_sync_model",
    "resample_aligned_stream",
    "resample_stream",
    "resample_to_reference_timestamps",
    "save_sync_model",
]
