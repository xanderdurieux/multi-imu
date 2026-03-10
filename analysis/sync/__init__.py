"""Synchronization package for aligning dual-IMU streams.

Four sync methods are provided, in order from simplest to most sophisticated:

1. **SDA** (:mod:`sync.sda_sync`)
   Offset-only alignment via full-recording cross-correlation.
   Writes to ``synced_sda/``.

2. **LIDA** (:mod:`sync.lida_sync`)
   Offset + drift alignment (SDA coarse pass + windowed refinement).
   Writes to ``synced_lida/``.

3. **Calibration** (:mod:`sync.calibration_sync`)
   Offset + drift alignment anchored to known calibration tap-burst events.
   Writes to ``synced_cal/``.

4. **Online** (:mod:`sync.online_sync`)
   Causal single-anchor alignment using the opening calibration only.
   Drift is loaded from a pre-characterised historical estimate.
   Writes to ``synced_online/``.

After all methods have been run, :mod:`sync.selection` compares them and
copies the best result to ``synced/``.
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
    remove_dropouts,
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
from .selection import (
    SyncMethodQuality,
    SyncSelectionResult,
    apply_selection,
    compare_sync_models,
    compare_all_recordings,
    plot_sync_comparison,
    print_comparison,
    select_best_sync_method,
)
from .session import (
    MethodResult,
    RecordingResult,
    synchronize_recording_all_methods,
    synchronize_session,
)

__all__ = [
    # Core algorithms
    "AlignmentSeries",
    "OffsetEstimate",
    "SyncModel",
    "build_activity_signal",
    "build_alignment_series",
    "estimate_lag",
    "estimate_offset",
    "estimate_sync_model",
    "fit_sync_from_paths",
    # Common utilities
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
    # Method 1: SDA only
    "synchronize_recording_sda",
    # Method 2: SDA + LIDA
    "synchronize",
    "synchronize_recording",
    # Method 3: Calibration
    "CalibrationWindowResult",
    "estimate_sync_from_calibration",
    "synchronize_from_calibration",
    "synchronize_recording_from_calibration",
    # Method 4: Online
    "estimate_sync_from_opening_anchor",
    "load_characterised_drift",
    "synchronize_recording_online",
    # Comparison and selection
    "SyncMethodQuality",
    "SyncSelectionResult",
    "apply_selection",
    "compare_all_recordings",
    "compare_sync_models",
    "plot_sync_comparison",
    "print_comparison",
    "select_best_sync_method",
    # Session orchestration
    "MethodResult",
    "RecordingResult",
    "synchronize_recording_all_methods",
    "synchronize_session",
]
