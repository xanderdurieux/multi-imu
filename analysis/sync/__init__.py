"""Synchronisation package with method modules, plotting, and the pipeline."""

from .align_df import AlignmentSeries, OffsetEstimate, build_activity_signal, build_alignment_series, estimate_lag, estimate_offset
from .drift_estimator import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_MIN_FIT_R2,
    DEFAULT_MIN_WINDOW_SCORE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    SyncModel,
    apply_sync_model,
    estimate_sync_model,
    fit_sync_from_paths,
    load_sync_model,
    resample_aligned_stream,
    save_sync_model,
)
from .helpers import METHOD_LABELS, METHOD_ORDER, METHOD_STAGES, SelectionResult, SyncArtifacts
from .pipeline import apply_best_method, compare_methods, run_pipeline, run_session, select_best_method
from .plotting import generate_sync_plots, plot_alignment, plot_method_comparison
from .sync_cal import CalibrationAnchor, estimate_sync_from_calibration
from .sync_cal import synchronize_recording as synchronize_recording_cal
from .sync_cal import synchronize_streams as synchronize_streams_cal
from .sync_lida import synchronize_recording as synchronize_recording_lida
from .sync_lida import synchronize_streams as synchronize_streams_lida
from .sync_online import estimate_sync_from_opening_anchor, load_characterised_drift_ppm
from .sync_online import synchronize_recording as synchronize_recording_online
from .sync_online import synchronize_streams as synchronize_streams_online
from .sync_sda import estimate_sync_model_sda
from .sync_sda import synchronize_recording as synchronize_recording_sda
from .sync_sda import synchronize_streams as synchronize_streams_sda

__all__ = [
    'AlignmentSeries',
    'OffsetEstimate',
    'SyncModel',
    'SyncArtifacts',
    'SelectionResult',
    'CalibrationAnchor',
    'METHOD_LABELS',
    'METHOD_ORDER',
    'METHOD_STAGES',
    'build_activity_signal',
    'build_alignment_series',
    'estimate_lag',
    'estimate_offset',
    'estimate_sync_model',
    'estimate_sync_model_sda',
    'estimate_sync_from_calibration',
    'estimate_sync_from_opening_anchor',
    'fit_sync_from_paths',
    'load_sync_model',
    'load_characterised_drift_ppm',
    'apply_sync_model',
    'save_sync_model',
    'resample_aligned_stream',
    'synchronize_recording_sda',
    'synchronize_recording_lida',
    'synchronize_recording_cal',
    'synchronize_recording_online',
    'synchronize_streams_sda',
    'synchronize_streams_lida',
    'synchronize_streams_cal',
    'synchronize_streams_online',
    'compare_methods',
    'select_best_method',
    'apply_best_method',
    'run_pipeline',
    'run_session',
    'plot_method_comparison',
    'plot_alignment',
    'generate_sync_plots',
    'DEFAULT_LOCAL_SEARCH_SECONDS',
    'DEFAULT_MIN_FIT_R2',
    'DEFAULT_MIN_WINDOW_SCORE',
    'DEFAULT_WINDOW_SECONDS',
    'DEFAULT_WINDOW_STEP_SECONDS',
]
