"""SDA-style coarse offset estimation for two IMU streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import add_vector_norms, resample_stream


@dataclass(frozen=True)
class AlignmentSeries:
    """Uniformly sampled activity signal and its timestamps."""

    timestamps_seconds: np.ndarray
    signal: np.ndarray
    sample_rate_hz: float


@dataclass(frozen=True)
class OffsetEstimate:
    """SDA coarse lag/offset estimate."""

    lag_samples: int
    lag_seconds: float
    offset_seconds: float
    score: float
    sample_rate_hz: float


def _zscore(signal: np.ndarray) -> np.ndarray:
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


def build_activity_signal(
    df: pd.DataFrame,
    *,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    differentiate: bool = True,
) -> np.ndarray:
    """
    Build a single orientation-invariant activity signal from IMU data.

    The signal is the z-scored average of selected vector norms (acc/gyro/mag).
    """
    base = add_vector_norms(df)
    components: list[np.ndarray] = []

    def _append_if_selected(flag: bool, name: str) -> None:
        if not flag:
            return
        col = f"{name}_norm"
        if col not in base.columns:
            return
        values = base[col].to_numpy(dtype=float)
        if np.isfinite(values).any():
            components.append(values)

    _append_if_selected(use_acc, "acc")
    _append_if_selected(use_gyro, "gyro")
    _append_if_selected(use_mag, "mag")

    if not components:
        raise ValueError("No valid activity channels selected for alignment.")

    stacked = np.vstack([_zscore(c) for c in components])
    signal = np.nanmean(stacked, axis=0)

    if differentiate and signal.size > 1:
        signal = np.diff(signal, prepend=signal[0])
    return _zscore(signal)


def build_alignment_series(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    differentiate: bool = True,
) -> AlignmentSeries:
    """Resample one stream and derive its 1D activity-over-time signal."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    resampled = resample_stream(df, sample_rate_hz=sample_rate_hz, timestamp_col="timestamp")
    if resampled.empty:
        return AlignmentSeries(
            timestamps_seconds=np.asarray([], dtype=float),
            signal=np.asarray([], dtype=float),
            sample_rate_hz=float(sample_rate_hz),
        )

    signal = build_activity_signal(
        resampled,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )
    ts_sec = pd.to_numeric(resampled["timestamp"], errors="coerce").to_numpy(dtype=float) / 1000.0
    return AlignmentSeries(
        timestamps_seconds=ts_sec,
        signal=signal,
        sample_rate_hz=float(sample_rate_hz),
    )


def _fft_correlate_full(reference_signal: np.ndarray, target_signal: np.ndarray) -> np.ndarray:
    """Full cross-correlation with FFT (equivalent to np.correlate(..., mode='full'))."""
    ref = np.asarray(reference_signal, dtype=float)
    tgt = np.asarray(target_signal, dtype=float)
    n = ref.size + tgt.size - 1
    if n <= 0:
        return np.asarray([], dtype=float)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(ref, nfft) * np.fft.rfft(tgt[::-1], nfft), nfft)
    return corr[:n]


def estimate_lag(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    max_lag_samples: int | None = None,
    min_overlap_samples: int = 10,
) -> tuple[int, float]:
    """
    Estimate integer lag maximizing correlation score.

    Positive lag means target is delayed with respect to reference.
    """
    ref = np.asarray(reference_signal, dtype=float)
    tgt = np.asarray(target_signal, dtype=float)
    n_ref = ref.size
    n_tgt = tgt.size
    if n_ref == 0 or n_tgt == 0:
        return 0, float("-inf")

    corr = _fft_correlate_full(ref, tgt)
    lags = np.arange(-(n_tgt - 1), n_ref, dtype=int)

    overlap = np.minimum(n_ref, n_tgt + lags) - np.maximum(0, lags)
    valid = overlap >= max(1, int(min_overlap_samples))
    if max_lag_samples is not None:
        valid &= np.abs(lags) <= int(max_lag_samples)
    if not valid.any():
        return 0, float("-inf")

    norm_score = np.full(corr.shape, -np.inf, dtype=float)
    norm_score[valid] = corr[valid] / overlap[valid]
    idx = int(np.argmax(norm_score))
    return int(lags[idx]), float(norm_score[idx])


def estimate_offset_from_series(
    reference_series: AlignmentSeries,
    target_series: AlignmentSeries,
    *,
    max_lag_seconds: float = 30.0,
) -> OffsetEstimate:
    """Estimate SDA coarse offset from two prepared alignment series."""
    if reference_series.signal.size == 0 or target_series.signal.size == 0:
        raise ValueError("Alignment signals must be non-empty.")
    sample_rate_hz = float(reference_series.sample_rate_hz)
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")

    max_lag_samples = int(round(float(max_lag_seconds) * sample_rate_hz))
    lag_samples, score = estimate_lag(
        reference_series.signal,
        target_series.signal,
        max_lag_samples=max_lag_samples,
    )
    lag_seconds = float(lag_samples) / sample_rate_hz
    ref_start = float(reference_series.timestamps_seconds[0])
    tgt_start = float(target_series.timestamps_seconds[0])
    offset_seconds = (ref_start - tgt_start) + lag_seconds
    return OffsetEstimate(
        lag_samples=int(lag_samples),
        lag_seconds=float(lag_seconds),
        offset_seconds=float(offset_seconds),
        score=float(score),
        sample_rate_hz=sample_rate_hz,
    )


def estimate_offset(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 30.0,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    differentiate: bool = True,
) -> OffsetEstimate:
    """Estimate SDA coarse offset directly from input dataframes."""
    ref_series = build_alignment_series(
        reference_df,
        sample_rate_hz=sample_rate_hz,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )
    tgt_series = build_alignment_series(
        target_df,
        sample_rate_hz=sample_rate_hz,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )
    return estimate_offset_from_series(
        reference_series=ref_series,
        target_series=tgt_series,
        max_lag_seconds=max_lag_seconds,
    )
