"""Shared 1-D signal helpers (vector norms, smoothing, NaN-aware filtering)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

log = logging.getLogger(__name__)


_DEFAULT_IMU_AXES: Mapping[str, Sequence[str]] = {
    "acc": ("ax", "ay", "az"),
    "gyro": ("gx", "gy", "gz"),
    "mag": ("mx", "my", "mz"),
}


def vector_norm(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Euclidean norm of the given columns per row, preserving NaN.

    Rows with any NaN component yield NaN; other rows get the standard
    Euclidean norm. Raises KeyError if any listed column is missing.
    """
    data = df[list(columns)].to_numpy(dtype=float)
    nan_mask = np.isnan(data).any(axis=1)
    norms = np.sqrt(np.nansum(data * data, axis=1))
    norms[nan_mask] = np.nan
    return norms


def norm_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norm of three component arrays (NaN propagates)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.sqrt(x * x + y * y + z * z)


def add_vector_norms(
    df: pd.DataFrame,
    vector_axes: Mapping[str, Iterable[str]] | None = None,
) -> pd.DataFrame:
    """Return a copy of *df* with ``{name}_norm`` columns for each (name, axes).

    When ``vector_axes`` is None, defaults to the standard IMU layout
    ({'acc', 'gyro', 'mag'}). Columns whose axes are absent become NaN.
    """
    axes_map = _DEFAULT_IMU_AXES if vector_axes is None else vector_axes
    out = df.copy()
    for name, axes in axes_map.items():
        axes = list(axes)
        col = f"{name}_norm"
        if all(c in out.columns for c in axes):
            out[col] = vector_norm(out, axes)
        else:
            out[col] = np.nan
    return out


def add_imu_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``acc_norm``, ``gyro_norm``, ``mag_norm`` columns."""
    return add_vector_norms(df)




def zscore_finite(signal: np.ndarray) -> np.ndarray:
    """Z-score finite samples and fill non-finite output with 0.0."""
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x, dtype=float)

    mu = float(np.nanmean(x[finite]))
    sigma = float(np.nanstd(x[finite]))
    if sigma < 1e-9:
        out = np.zeros_like(x, dtype=float)
        out[~finite] = 0.0
        return out

    out = (x - mu) / sigma
    out[~finite] = 0.0
    return out


def first_difference(signal: np.ndarray) -> np.ndarray:
    """One-step discrete difference preserving the input length."""
    x = np.asarray(signal, dtype=float)
    if x.size <= 1:
        return x.copy()
    return np.diff(x, prepend=x[0])

def smooth_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Centered box-car moving average, same length as input."""
    if window <= 1:
        return np.asarray(signal, dtype=float).copy()
    return (
        pd.Series(signal)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


_MIN_SAMPLES_BUTTER_ORDER4 = 3 * (4 + 1)


def butter_lowpass(
    data: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter that preserves NaN gaps.

    Contiguous finite segments are filtered independently so that NaN spans
    do not contaminate the rest of the trace. Segments shorter than
    ``3 * (order + 1)`` samples are left untouched.
    """
    min_samples = 3 * (order + 1)
    if len(data) <= min_samples:
        log.debug(
            "Too few samples (%d) for Butterworth filter (need > %d); returning original signal.",
            len(data),
            min_samples,
        )
        return data.copy()

    finite = np.isfinite(data)
    if not finite.any():
        return data.copy()

    nyq = 0.5 * sample_rate_hz
    normal_cutoff = min(cutoff_hz / nyq, 0.9999)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    out = data.copy()
    finite_i = finite.astype(np.int8)
    starts = np.flatnonzero(np.diff(np.r_[0, finite_i]) == 1)
    stops = np.flatnonzero(np.diff(np.r_[finite_i, 0]) == -1) + 1

    for start, stop in zip(starts, stops, strict=False):
        segment = data[start:stop]
        if len(segment) <= min_samples:
            continue
        out[start:stop] = filtfilt(b, a, segment)

    return out
