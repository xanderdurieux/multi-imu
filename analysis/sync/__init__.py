"""Synchronization package: SDA, LIDA, calibration, and online methods + selection.

Run all methods and pick the best for a recording::

    python -m sync <recording> --apply --plot

See :mod:`sync.run` for the full CLI (including ``sync sda_sync …``, short aliases, and ``--select-only``).
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
from .selection import (
    SyncMethodQuality,
    SyncSelectionResult,
    apply_selection,
    compare_all_recordings,
    compare_sync_models,
    plot_sync_comparison,
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
    "compare_all_recordings",
    "compare_sync_models",
    "plot_sync_comparison",
    "print_comparison",
    "print_selection_result",
    "select_best_sync_method",
    "MethodResult",
    "RecordingResult",
    "synchronize_recording_all_methods",
    "synchronize_session",
]
