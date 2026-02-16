"""
SDA-style (Simple Data Alignment) coarse time offset estimation for IMU streams.

This module implements the SDA algorithm approach for initial time synchronization
between a reference stream and a target stream. SDA uses discrete sample shifts
(cross-correlation) to estimate the time offset, achieving alignment precision of
approximately one sample period (within one half sample period at best).

The alignment is performed using orientation-invariant vector magnitudes from
accelerometer, gyroscope, and/or magnetometer data, making it robust to sensor
orientation differences between devices.

Reference: Wang et al. (2023). Comparison between Two Time Synchronization and
Data Alignment Methods for Multi-Channel Wearable Biosensor Systems Using BLE
Protocol. Sensors, 23(5), 2465.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import add_vector_norms, resample_stream


@dataclass(frozen=True)
class OffsetEstimate:
    """
    SDA-style coarse time offset estimate between reference and target streams.

    Attributes:
        lag_samples: Discrete lag in samples (integer shift from cross-correlation).
        lag_seconds: Discrete lag converted to seconds.
        offset_seconds: Total time offset (includes stream start time difference + lag).
        score: Correlation score for the estimated lag (higher is better).
        sample_rate_hz: Sampling rate used for alignment signal computation.
    """

    lag_samples: int
    lag_seconds: float
    offset_seconds: float
    score: float
    sample_rate_hz: float


def _zscore(signal: np.ndarray) -> np.ndarray:
    """
    Normalize signal to zero mean and unit variance (z-score normalization).

    Used to make alignment signals amplitude-invariant, focusing correlation
    on temporal patterns rather than signal magnitude differences.
    """
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)

    mu = float(np.nanmean(x[finite]))
    sigma = float(np.nanstd(x[finite]))
    if sigma < 1e-9:
        out = np.zeros_like(x)
        out[~finite] = 0.0
        return out

    out = (x - mu) / sigma
    out[~finite] = 0.0
    return out


def build_alignment_signal(
    df: pd.DataFrame,
    *,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    differentiate: bool = True,
) -> np.ndarray:
    """
    Build a 1D alignment signal from IMU data for SDA-style cross-correlation.

    The signal is constructed from vector magnitudes (|acc|, |gyro|, |mag|) to be
    orientation-invariant, allowing alignment even when sensors have different
    orientations. Multiple sensor types are combined and normalized to focus
    correlation on temporal patterns rather than amplitude differences.

    Args:
        df: IMU dataframe with acceleration, gyroscope, and/or magnetometer columns.
        use_acc: Include accelerometer magnitude in alignment signal.
        use_gyro: Include gyroscope magnitude in alignment signal.
        use_mag: Include magnetometer magnitude in alignment signal.
        differentiate: Apply first-order difference to emphasize temporal changes.

    Returns:
        Normalized 1D alignment signal suitable for cross-correlation.

    Raises:
        ValueError: If no sensor channels are selected.
    """
    # Compute orientation-invariant vector magnitudes
    base = add_vector_norms(df)

    components: list[np.ndarray] = []
    if use_acc and "acc_norm" in base.columns:
        components.append(base["acc_norm"].to_numpy(dtype=float))
    if use_gyro and "gyro_norm" in base.columns:
        components.append(base["gyro_norm"].to_numpy(dtype=float))
    if use_mag and "mag_norm" in base.columns:
        components.append(base["mag_norm"].to_numpy(dtype=float))

    if not components:
        raise ValueError("no alignment channels selected")

    # Combine and normalize sensor components
    stacked = np.vstack([_zscore(c) for c in components])
    signal = np.nanmean(stacked, axis=0)

    # Emphasize temporal changes via differentiation
    if differentiate and len(signal) > 1:
        signal = np.diff(signal, prepend=signal[0])

    return _zscore(signal)


def _corr_score_for_lag(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    lag: int,
) -> float:
    """
    Compute cross-correlation score for a specific discrete lag (sample shift).

    The target signal is shifted by `lag` samples relative to the reference.
    Positive lag means target is delayed (shifted forward in time index).

    Args:
        reference_signal: Reference alignment signal.
        target_signal: Target alignment signal.
        lag: Discrete sample shift to test (can be negative).

    Returns:
        Normalized correlation score (higher is better), or -inf if insufficient overlap.
    """
    n_ref = len(reference_signal)
    n_tgt = len(target_signal)

    # Compute overlapping region after applying lag shift
    i0 = max(0, lag)
    i1 = min(n_ref, n_tgt + lag)
    if i1 - i0 < 10:
        return -np.inf

    ref_seg = reference_signal[i0:i1]
    tgt_seg = target_signal[i0 - lag : i1 - lag]
    return float(np.dot(ref_seg, tgt_seg) / (i1 - i0))


def estimate_lag(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    max_lag_samples: int,
) -> tuple[int, float]:
    """
    Estimate discrete lag (sample shift) between reference and target signals.

    Performs SDA-style discrete alignment by searching for the integer sample
    shift that maximizes cross-correlation. This provides coarse alignment with
    precision limited to one sample period.

    Args:
        reference_signal: Reference alignment signal (1D array).
        target_signal: Target alignment signal (1D array).
        max_lag_samples: Maximum absolute lag to search (symmetric search window).

    Returns:
        Tuple of (best_lag_samples, best_score) where:
        - best_lag_samples: Integer sample shift that maximizes correlation.
        - best_score: Correlation score for the best lag.

    Raises:
        ValueError: If max_lag_samples is negative.
    """
    if max_lag_samples < 0:
        raise ValueError("max_lag_samples must be >= 0")

    best_lag = 0
    best_score = -np.inf

    # Search symmetric lag window for maximum correlation
    for lag in range(-max_lag_samples, max_lag_samples + 1):
        score = _corr_score_for_lag(
            reference_signal,
            target_signal,
            lag,
        )
        if score > best_score:
            best_score = score
            best_lag = lag

    return best_lag, float(best_score)


def estimate_offset(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 20.0,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
) -> OffsetEstimate:
    """
    Estimate SDA-style coarse time offset between reference and target IMU streams.

    This implements the Simple Data Alignment (SDA) algorithm approach for initial
    time synchronization. The method:
    1. Resamples both streams to a common rate
    2. Builds orientation-invariant alignment signals from vector magnitudes
    3. Uses cross-correlation to find the discrete sample lag
    4. Computes total time offset including stream start time difference

    The alignment precision is limited to approximately one sample period (within
    one half sample period at best) due to discrete sample shifts. For sub-sample
    precision, use LIDA-style refinement (see drift_estimator module).

    Args:
        reference_df: Reference IMU stream dataframe.
        target_df: Target IMU stream dataframe to align to reference.
        sample_rate_hz: Common sampling rate for alignment signal computation.
        max_lag_seconds: Maximum absolute lag to search (symmetric window).
        use_acc: Include accelerometer in alignment signal.
        use_gyro: Include gyroscope in alignment signal.
        use_mag: Include magnetometer in alignment signal.

    Returns:
        OffsetEstimate containing discrete lag, total offset, correlation score,
        and sampling rate used.

    Reference:
        Wang et al. (2023). Comparison between Two Time Synchronization and Data
        Alignment Methods for Multi-Channel Wearable Biosensor Systems Using BLE
        Protocol. Sensors, 23(5), 2465.
    """
    # Resample both streams to common rate for alignment
    ref_rs = resample_stream(reference_df, sample_rate_hz)
    tgt_rs = resample_stream(target_df, sample_rate_hz)

    # Build orientation-invariant alignment signals
    ref_signal = build_alignment_signal(
        ref_rs,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )
    tgt_signal = build_alignment_signal(
        tgt_rs,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )

    # SDA: discrete lag estimation via cross-correlation
    max_lag_samples = max(1, int(round(max_lag_seconds * sample_rate_hz)))
    lag_samples, score = estimate_lag(ref_signal, tgt_signal, max_lag_samples=max_lag_samples)

    lag_seconds = lag_samples / sample_rate_hz

    # Compute total time offset (stream start difference + discrete lag)
    ref_start_sec = float(ref_rs["timestamp"].iloc[0]) / 1000.0
    tgt_start_sec = float(tgt_rs["timestamp"].iloc[0]) / 1000.0

    # Total offset: ref_time ≈ target_time + offset_seconds
    offset_seconds = (ref_start_sec - tgt_start_sec) + lag_seconds

    return OffsetEstimate(
        lag_samples=lag_samples,
        lag_seconds=lag_seconds,
        offset_seconds=offset_seconds,
        score=score,
        sample_rate_hz=sample_rate_hz,
    )
