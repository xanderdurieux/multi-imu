"""Alignment-series construction for cross-correlation synchronization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stream_io import VECTOR_AXES, resample_stream, lowpass_filter
from .signals import build_activity_signal as _build_raw_signal

SIGNAL_MODE_ACC_NORM_DIFF = "acc_norm_diff"


@dataclass(frozen=True)
class AlignmentSeries:
    """Uniformly sampled 1-D activity signal with its time axis."""

    timestamps_seconds: np.ndarray
    signal: np.ndarray
    sample_rate_hz: float
    signal_mode: str = SIGNAL_MODE_ACC_NORM_DIFF


def build_alignment_series(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float,
    signal_mode: str = SIGNAL_MODE_ACC_NORM_DIFF,
    lowpass_cutoff_hz: float | None = None,
) -> AlignmentSeries:
    """Resample a stream and derive its 1-D activity signal."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    resampled = resample_stream(df, sample_rate_hz=sample_rate_hz)
    if resampled.empty:
        empty = np.asarray([], dtype=float)
        return AlignmentSeries(
            timestamps_seconds=empty,
            signal=empty,
            sample_rate_hz=float(sample_rate_hz),
            signal_mode=signal_mode,
        )

    if lowpass_cutoff_hz is not None:
        resampled = lowpass_filter(resampled, lowpass_cutoff_hz, sample_rate_hz)

    signal, resolved_mode = _build_raw_signal(
        resampled, vector_axes=VECTOR_AXES, signal_mode=signal_mode
    )
    ts_sec = (
        pd.to_numeric(resampled["timestamp"], errors="coerce").to_numpy(dtype=float)
        / 1000.0
    )
    return AlignmentSeries(
        timestamps_seconds=ts_sec,
        signal=signal,
        sample_rate_hz=float(sample_rate_hz),
        signal_mode=resolved_mode,
    )
