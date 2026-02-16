"""Offset estimation utilities based on correlated IMU features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import add_vector_norms, resample_stream


@dataclass(frozen=True)
class OffsetEstimate:
    lag_samples: int
    lag_seconds: float
    offset_seconds: float
    score: float
    sample_rate_hz: float


def _zscore(signal: np.ndarray) -> np.ndarray:
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
    use_gyro: bool = True,
    use_mag: bool = False,
    differentiate: bool = True,
) -> np.ndarray:
    """Build a 1D alignment signal from IMU vectors, robust to orientation."""
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

    stacked = np.vstack([_zscore(c) for c in components])
    signal = np.nanmean(stacked, axis=0)

    if differentiate and len(signal) > 1:
        signal = np.diff(signal, prepend=signal[0])

    return _zscore(signal)


def _corr_score_for_lag(ref: np.ndarray, tgt: np.ndarray, lag: int) -> float:
    """Correlation score for a lag where target is shifted forward by ``lag``."""
    n_ref = len(ref)
    n_tgt = len(tgt)

    i0 = max(0, lag)
    i1 = min(n_ref, n_tgt + lag)
    if i1 - i0 < 10:
        return -np.inf

    ref_seg = ref[i0:i1]
    tgt_seg = tgt[i0 - lag : i1 - lag]
    return float(np.dot(ref_seg, tgt_seg) / (i1 - i0))


def estimate_lag(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    max_lag_samples: int,
) -> tuple[int, float]:
    """Return the best lag in samples and its score."""
    if max_lag_samples < 0:
        raise ValueError("max_lag_samples must be >= 0")

    best_lag = 0
    best_score = -np.inf

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        score = _corr_score_for_lag(reference_signal, target_signal, lag)
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
    """Estimate target->reference time offset with correlation."""
    ref_rs = resample_stream(reference_df, sample_rate_hz)
    tgt_rs = resample_stream(target_df, sample_rate_hz)

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

    max_lag_samples = max(1, int(round(max_lag_seconds * sample_rate_hz)))
    lag_samples, score = estimate_lag(ref_signal, tgt_signal, max_lag_samples=max_lag_samples)

    lag_seconds = lag_samples / sample_rate_hz

    ref_start_sec = float(ref_rs["timestamp"].iloc[0]) / 1000.0
    tgt_start_sec = float(tgt_rs["timestamp"].iloc[0]) / 1000.0

    # ref_time ~= target_time + offset
    offset_seconds = (ref_start_sec - tgt_start_sec) + lag_seconds

    return OffsetEstimate(
        lag_samples=lag_samples,
        lag_seconds=lag_seconds,
        offset_seconds=offset_seconds,
        score=score,
        sample_rate_hz=sample_rate_hz,
    )
