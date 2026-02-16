"""Synchronization utilities implementing SDA + LIDA algorithms for IMU stream alignment."""

from .align_df import OffsetEstimate, build_alignment_signal, estimate_lag, estimate_offset
from .common import apply_linear_time_transform, load_stream, resample_stream
from .drift_estimator import (
    SyncModel,
    apply_sync_model,
    estimate_sync_model,
    fit_sync_from_paths,
    load_sync_model,
    resample_aligned_stream,
    save_sync_model,
)
from .sync_streams import synchronize

__all__ = [
    "OffsetEstimate",
    "SyncModel",
    "apply_linear_time_transform",
    "load_stream",
    "resample_stream",
    "build_alignment_signal",
    "estimate_lag",
    "estimate_offset",
    "apply_sync_model",
    "estimate_sync_model",
    "fit_sync_from_paths",
    "load_sync_model",
    "resample_aligned_stream",
    "save_sync_model",
    "synchronize",
]
