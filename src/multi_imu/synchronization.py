"""Synchronization helpers for aligning IMU streams in time."""
from __future__ import annotations
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from scipy import signal

from .data_models import IMUSensorData, SyncedIMUData
from .preprocessing import normalize_axes


DEFAULT_AXES = ("ax", "ay", "az")


def _windowed_norm(df: pd.DataFrame, axes: Sequence[str]) -> np.ndarray:
    available = [axis for axis in axes if axis in df.columns]
    if not available:
        raise ValueError("No matching axes found for synchronization")
    vectors = df[available].to_numpy()
    return np.linalg.norm(vectors, axis=1)


def estimate_time_offset(reference: IMUSensorData, target: IMUSensorData, axes: Iterable[str] = DEFAULT_AXES) -> float:
    """Estimate the time offset between two IMU streams using cross-correlation."""
    ref_norm = _windowed_norm(normalize_axes(reference, axes).data, axes)
    tgt_norm = _windowed_norm(normalize_axes(target, axes).data, axes)

    corr = signal.correlate(ref_norm, tgt_norm, mode="full")
    lags = signal.correlation_lags(len(ref_norm), len(tgt_norm), mode="full")
    best_lag = lags[np.argmax(corr)]

    return best_lag / reference.sample_rate_hz


def synchronize_streams(
    reference: IMUSensorData,
    target: IMUSensorData,
    axes: Iterable[str] = DEFAULT_AXES,
    trim: bool = True,
) -> SyncedIMUData:
    """Return streams aligned in time by shifting the target stream."""
    offset = estimate_time_offset(reference, target, axes)

    shifted = target.data.copy()
    shifted["timestamp"] = shifted["timestamp"] + offset

    if trim:
        start = max(reference.data["timestamp"].min(), shifted["timestamp"].min())
        end = min(reference.data["timestamp"].max(), shifted["timestamp"].max())
        ref_trimmed = reference.data.query("@start <= timestamp <= @end").reset_index(drop=True)
        tgt_trimmed = shifted.query("@start <= timestamp <= @end").reset_index(drop=True)
    else:
        ref_trimmed = reference.data
        tgt_trimmed = shifted

    return SyncedIMUData(
        reference=IMUSensorData(reference.name, ref_trimmed, reference.sample_rate_hz),
        target=IMUSensorData(target.name, tgt_trimmed, target.sample_rate_hz),
        offset_seconds=offset,
    )


__all__ = ["estimate_time_offset", "synchronize_streams"]
