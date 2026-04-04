"""Shared stream I/O, SDA alignment signals, LIDA drift model, and sync metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from common.paths import read_csv
from .segments import (
    CALIBRATION_USAGE_FULL_STREAM,
    CALIBRATION_USAGE_ONLY,
    SegmentSelection,
    build_segment_selection,
    calibration_segments_to_windows_seconds,
    detect_calibration_segments_for_stream,
)
from .signals import build_activity_signal as build_sync_activity_signal

# ---------------------------------------------------------------------------
# Stream utilities
# ---------------------------------------------------------------------------

VECTOR_AXES: dict[str, list[str]] = {
    "acc": ["ax", "ay", "az"],
    "gyro": ["gx", "gy", "gz"],
    "mag": ["mx", "my", "mz"],
}


def load_stream(csv_path: Path | str) -> pd.DataFrame:
    """Load an IMU CSV, coerce numeric schema, and sort by timestamp."""
    path = Path(csv_path)
    df = read_csv(path).copy()
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_vector_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Add |acc|, |gyro|, and |mag| magnitude columns."""
    out = df.copy()
    for name, axes in VECTOR_AXES.items():
        if all(col in out.columns for col in axes):
            arr = out[axes].to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                out[f"{name}_norm"] = np.sqrt(np.nansum(arr * arr, axis=1))
        else:
            out[f"{name}_norm"] = np.nan
    return out


def infer_numeric_columns(df: pd.DataFrame, skip: Iterable[str] = ("timestamp",)) -> list[str]:
    """Infer numeric columns excluding the given names."""
    skip_set = set(skip)
    cols: list[str] = []
    for col in df.columns:
        if col in skip_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _sorted_numeric_timestamp(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_numeric(out[timestamp_col], errors="coerce")
    out = out.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    return out


def resample_stream(
    df: pd.DataFrame,
    sample_rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
    columns: list[str] | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> pd.DataFrame:
    """Resample stream to a uniform time grid with linear interpolation."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")

    base = _sorted_numeric_timestamp(df, timestamp_col)
    if base.empty:
        return pd.DataFrame(columns=[timestamp_col])

    ts = base[timestamp_col].to_numpy(dtype=float)
    step_ms = 1000.0 / float(sample_rate_hz)
    lo = float(ts[0] if start_ms is None else start_ms)
    hi = float(ts[-1] if end_ms is None else end_ms)

    if hi <= lo:
        out = pd.DataFrame({timestamp_col: np.asarray([lo], dtype=float)})
    else:
        grid = np.arange(lo, hi + 0.5 * step_ms, step_ms, dtype=float)
        out = pd.DataFrame({timestamp_col: grid})

    if columns is None:
        columns = infer_numeric_columns(base, skip=[timestamp_col])

    for col in columns:
        values = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(ts) & np.isfinite(values)
        if valid.sum() >= 2:
            out[col] = np.interp(out[timestamp_col].to_numpy(dtype=float), ts[valid], values[valid])
        elif valid.sum() == 1:
            out[col] = values[valid][0]
        else:
            out[col] = np.nan

    return out


def resample_to_reference_timestamps(
    target_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Resample target values at reference timestamps for sample-by-sample comparison."""
    if timestamp_col not in target_df.columns or timestamp_col not in reference_df.columns:
        raise ValueError(f"both dataframes must contain '{timestamp_col}'")

    ref = _sorted_numeric_timestamp(reference_df, timestamp_col)
    tgt = _sorted_numeric_timestamp(target_df, timestamp_col)
    ref_ts = ref[timestamp_col].to_numpy(dtype=float)
    tgt_ts = tgt[timestamp_col].to_numpy(dtype=float)

    out = pd.DataFrame({timestamp_col: ref_ts})
    if ref_ts.size == 0 or tgt_ts.size == 0:
        return out

    cols = infer_numeric_columns(tgt, skip=[timestamp_col])
    for col in cols:
        values = pd.to_numeric(tgt[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(tgt_ts) & np.isfinite(values)
        if valid.sum() < 2:
            out[col] = np.nan
            continue

        ts_valid = tgt_ts[valid]
        val_valid = values[valid]
        lo = float(ts_valid.min())
        hi = float(ts_valid.max())

        series = np.full(ref_ts.shape, np.nan, dtype=float)
        inside = (ref_ts >= lo) & (ref_ts <= hi)
        if inside.any():
            series[inside] = np.interp(ref_ts[inside], ts_valid, val_valid)
        out[col] = series

    return out


_IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def lowpass_filter(
    df: pd.DataFrame,
    cutoff_hz: float,
    sample_rate_hz: float,
    *,
    columns: list[str] | None = None,
    order: int = 4,
) -> pd.DataFrame:
    """Apply a zero-phase Butterworth low-pass filter to IMU columns."""
    nyq = 0.5 * float(sample_rate_hz)
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz must be in (0, {nyq:.1f}) for sample_rate_hz={sample_rate_hz:.1f} Hz; "
            f"got {cutoff_hz}."
        )

    b, a = butter(order, cutoff_hz / nyq, btype="low", analog=False)

    cols = columns if columns is not None else [c for c in _IMU_COLUMNS if c in df.columns]

    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float).copy()
        finite = np.isfinite(values)
        if finite.sum() < max(10, 3 * (order + 1)):
            continue
        values[~finite] = 0.0
        out[col] = filtfilt(b, a, values)

    return out


def remove_dropouts(df: pd.DataFrame, *, epsilon_fraction: float = 0.1) -> pd.DataFrame:
    """Remove rows where the acceleration norm is near-zero (sensor dropout packets)."""
    out = add_vector_norms(df)
    g_approx = float(out["acc_norm"].median())
    if g_approx <= 0:
        return df
    threshold = epsilon_fraction * g_approx
    valid = out["acc_norm"] >= threshold
    return df.loc[valid.values].reset_index(drop=True)


def apply_linear_time_transform(
    timestamp_ms: pd.Series | np.ndarray,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Map target timestamps to reference time using offset + linear drift."""
    ts_sec = np.asarray(timestamp_ms, dtype=float) / 1000.0
    aligned_sec = (
        ts_sec
        + float(offset_seconds)
        + float(drift_seconds_per_second) * (ts_sec - float(target_origin_seconds))
    )
    return aligned_sec * 1000.0


# ---------------------------------------------------------------------------
# SDA alignment (activity signal + lag)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentSeries:
    """Uniformly sampled activity signal and its timestamps."""

    timestamps_seconds: np.ndarray
    signal: np.ndarray
    sample_rate_hz: float
    valid_mask: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    segment_mask: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    segment_count: int = 0
    signal_mode: str = "legacy"


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
    sigma = float(np.nanstd(x))
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
    signal_mode: str | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    differentiate: bool = True,
) -> np.ndarray:
    """Build a single orientation-invariant activity signal from IMU data."""
    signal, _mode = build_sync_activity_signal(
        df,
        vector_axes=VECTOR_AXES,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )
    return signal


def build_alignment_series(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float,
    signal_mode: str | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    differentiate: bool = True,
    lowpass_cutoff_hz: float | None = None,
    calibration_usage_strategy: str = CALIBRATION_USAGE_FULL_STREAM,
    segment_aware: bool = False,
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
            valid_mask=np.asarray([], dtype=bool),
            segment_mask=np.asarray([], dtype=bool),
            segment_count=0,
            signal_mode=signal_mode or "legacy",
        )

    if lowpass_cutoff_hz is not None:
        resampled = lowpass_filter(resampled, lowpass_cutoff_hz, sample_rate_hz)

    signal, resolved_signal_mode = build_sync_activity_signal(
        resampled,
        vector_axes=VECTOR_AXES,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )
    ts_sec = pd.to_numeric(resampled["timestamp"], errors="coerce").to_numpy(dtype=float) / 1000.0
    effective_segment_aware = bool(segment_aware or calibration_usage_strategy == CALIBRATION_USAGE_ONLY)
    detected_segments = (
        detect_calibration_segments_for_stream(df, sample_rate_hz=sample_rate_hz)
        if effective_segment_aware
        else []
    )
    windows_seconds = calibration_segments_to_windows_seconds(df, detected_segments)
    selection: SegmentSelection = build_segment_selection(
        ts_sec,
        sample_rate_hz=sample_rate_hz,
        segment_windows_seconds=windows_seconds,
        calibration_usage_strategy=calibration_usage_strategy,
        segment_aware=effective_segment_aware,
    )
    return AlignmentSeries(
        timestamps_seconds=ts_sec,
        signal=signal,
        sample_rate_hz=float(sample_rate_hz),
        valid_mask=selection.valid_mask,
        segment_mask=selection.segment_mask,
        segment_count=selection.segment_count,
        signal_mode=resolved_signal_mode,
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


def _bool_mask(mask: np.ndarray | None, size: int) -> np.ndarray:
    if mask is None:
        return np.ones(size, dtype=bool)
    out = np.asarray(mask, dtype=bool)
    if out.size != size:
        raise ValueError(f"Mask length {out.size} does not match signal length {size}.")
    return out


def estimate_lag(
    reference_signal: np.ndarray,
    target_signal: np.ndarray,
    *,
    max_lag_samples: int | None = None,
    min_overlap_samples: int = 10,
    reference_valid_mask: np.ndarray | None = None,
    target_valid_mask: np.ndarray | None = None,
) -> tuple[int, float]:
    """Estimate integer lag maximizing correlation score."""
    ref = np.asarray(reference_signal, dtype=float)
    tgt = np.asarray(target_signal, dtype=float)
    n_ref = ref.size
    n_tgt = tgt.size
    if n_ref == 0 or n_tgt == 0:
        return 0, float("-inf")

    ref_mask = _bool_mask(reference_valid_mask, n_ref)
    tgt_mask = _bool_mask(target_valid_mask, n_tgt)
    ref_masked = np.where(ref_mask & np.isfinite(ref), ref, 0.0)
    tgt_masked = np.where(tgt_mask & np.isfinite(tgt), tgt, 0.0)

    corr = _fft_correlate_full(ref_masked, tgt_masked)
    lags = np.arange(-(n_tgt - 1), n_ref, dtype=int)
    overlap = _fft_correlate_full(ref_mask.astype(float), tgt_mask.astype(float))
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
    calibration_usage_strategy: str = CALIBRATION_USAGE_FULL_STREAM,
    return_diagnostics: bool = False,
) -> OffsetEstimate | tuple[OffsetEstimate, dict[str, Any]]:
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
        reference_valid_mask=reference_series.valid_mask,
        target_valid_mask=target_series.valid_mask,
    )
    lag_seconds = float(lag_samples) / sample_rate_hz
    ref_start = float(reference_series.timestamps_seconds[0])
    tgt_start = float(target_series.timestamps_seconds[0])
    offset_seconds = (ref_start - tgt_start) + lag_seconds
    estimate = OffsetEstimate(
        lag_samples=int(lag_samples),
        lag_seconds=float(lag_seconds),
        offset_seconds=float(offset_seconds),
        score=float(score),
        sample_rate_hz=sample_rate_hz,
    )
    if not return_diagnostics:
        return estimate
    diagnostics = {
        **_segment_selection_summary(
            reference_series,
            target_series,
            calibration_usage_strategy=calibration_usage_strategy,
        ),
        "signal_mode": reference_series.signal_mode,
        "accepted_windows": 0,
        "rejected_windows": 0,
        "anchor_count": 1,
        "local_corr_mean": None,
        "local_corr_median": None,
        "drift_r2": None,
        "residual_summary": None,
        "fallback_used": False,
    }
    return estimate, diagnostics


def estimate_offset(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 30.0,
    signal_mode: str | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    differentiate: bool = True,
    lowpass_cutoff_hz: float | None = None,
    calibration_usage_strategy: str = CALIBRATION_USAGE_FULL_STREAM,
    segment_aware: bool = False,
    return_diagnostics: bool = False,
) -> OffsetEstimate | tuple[OffsetEstimate, dict[str, Any]]:
    """Estimate SDA coarse offset directly from input dataframes."""
    ref_series = build_alignment_series(
        reference_df,
        sample_rate_hz=sample_rate_hz,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        calibration_usage_strategy=calibration_usage_strategy,
        segment_aware=segment_aware,
    )
    tgt_series = build_alignment_series(
        target_df,
        sample_rate_hz=sample_rate_hz,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        calibration_usage_strategy=calibration_usage_strategy,
        segment_aware=segment_aware,
    )
    return estimate_offset_from_series(
        reference_series=ref_series,
        target_series=tgt_series,
        max_lag_seconds=max_lag_seconds,
        calibration_usage_strategy=calibration_usage_strategy,
        return_diagnostics=return_diagnostics,
    )


# ---------------------------------------------------------------------------
# LIDA drift model
# ---------------------------------------------------------------------------

DEFAULT_WINDOW_SECONDS = 20.0
DEFAULT_WINDOW_STEP_SECONDS = 10.0
DEFAULT_LOCAL_SEARCH_SECONDS = 0.5
DEFAULT_MIN_WINDOW_SCORE = 0.10
DEFAULT_MIN_FIT_R2 = 0.10


@dataclass(frozen=True)
class SyncModel:
    """Global offset + drift model with minimal metadata."""

    reference_csv: str
    target_csv: str
    target_time_origin_seconds: float
    offset_seconds: float
    drift_seconds_per_second: float
    sample_rate_hz: float
    max_lag_seconds: float
    created_at_utc: str


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation (Pearson r) between two equal-length arrays."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm_a = float(np.linalg.norm(a_c))
    norm_b = float(np.linalg.norm(b_c))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a_c, b_c) / (norm_a * norm_b))


def _masked_ncc(
    a: np.ndarray,
    b: np.ndarray,
    *,
    a_valid: np.ndarray | None = None,
    b_valid: np.ndarray | None = None,
    min_valid_fraction: float = 0.0,
) -> tuple[float, float]:
    """Return (score, valid_fraction) for two equal-length windows."""
    if a.shape != b.shape:
        raise ValueError("Masked NCC requires equal-length arrays.")
    a_mask = _bool_mask(a_valid, a.size).reshape(a.shape)
    b_mask = _bool_mask(b_valid, b.size).reshape(b.shape)
    valid = a_mask & b_mask & np.isfinite(a) & np.isfinite(b)
    valid_fraction = float(valid.mean()) if valid.size else 0.0
    if valid.sum() < 3 or valid_fraction < float(min_valid_fraction):
        return float("-inf"), valid_fraction
    return _ncc(a[valid], b[valid]), valid_fraction


def _windowed_lag_refinement(
    reference_series: AlignmentSeries,
    target_series: AlignmentSeries,
    *,
    coarse_lag_samples: int,
    window_seconds: float,
    window_step_seconds: float,
    local_search_seconds: float,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ref_signal = reference_series.signal
    tgt_signal = target_series.signal
    ref_ts = reference_series.timestamps_seconds
    tgt_ts = target_series.timestamps_seconds
    sample_rate_hz = float(reference_series.sample_rate_hz)

    n_ref = ref_signal.size
    n_tgt = tgt_signal.size
    if n_ref == 0 or n_tgt == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, {"accepted_windows": 0, "rejected_windows": 0}

    window_n = max(20, int(round(float(window_seconds) * sample_rate_hz)))
    step_n = max(5, int(round(float(window_step_seconds) * sample_rate_hz)))
    search_n = max(1, int(round(float(local_search_seconds) * sample_rate_hz)))
    half = window_n // 2
    if n_ref < window_n:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty, {"accepted_windows": 0, "rejected_windows": 0}

    # NOTE: These times must be expressed in the *target* clock domain because the
    # subsequent drift fit models offset(t_target) as a function of target time.
    target_times: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []
    lags: list[float] = []
    rejected_windows = 0
    ref_valid = _bool_mask(reference_series.valid_mask, n_ref)
    tgt_valid = _bool_mask(target_series.valid_mask, n_tgt)

    for center in range(half, n_ref - half, step_n):
        left = center - half
        right = center + half
        ref_win = ref_signal[left:right]
        if ref_win.size < 10:
            continue

        best_lag: int | None = None
        best_score = float("-inf")

        for delta in range(-search_n, search_n + 1):
            lag = int(coarse_lag_samples + delta)
            t_left = left - lag
            t_right = right - lag
            if t_left < 0 or t_right > n_tgt:
                continue

            tgt_win = tgt_signal[t_left:t_right]
            if tgt_win.size != ref_win.size:
                continue

            score, valid_fraction = _masked_ncc(
                ref_win,
                tgt_win,
                a_valid=ref_valid[left:right],
                b_valid=tgt_valid[t_left:t_right],
                min_valid_fraction=min_valid_fraction,
            )
            if not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None or best_score < min_window_score:
            rejected_windows += 1
            continue

        tgt_idx = center - best_lag
        if tgt_idx < 0 or tgt_idx >= tgt_ts.size:
            continue

        ref_time = float(ref_ts[center])
        tgt_time = float(tgt_ts[tgt_idx])
        target_times.append(tgt_time)
        offsets.append(ref_time - tgt_time)
        scores.append(best_score)
        lags.append(float(best_lag))

    return (
        np.asarray(target_times, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(scores, dtype=float),
        np.asarray(lags, dtype=float),
        {
            "accepted_windows": len(target_times),
            "rejected_windows": rejected_windows,
        },
    )


def _adaptive_windowed_refinement(
    reference_series: AlignmentSeries,
    target_series: AlignmentSeries,
    *,
    initial_offset_seconds: float,
    initial_drift_seconds_per_second: float = 0.0,
    target_origin_seconds: float,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_points_for_drift: int = 3,
    min_valid_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Causal windowed lag refinement that updates the running model after each accepted window.

    Unlike :func:`_windowed_lag_refinement`, which uses the same global coarse lag for
    all window searches, this function feeds the model fitted on *past* windows back as
    the search centre for each *future* window — so no future data is ever used.

    Parameters
    ----------
    initial_offset_seconds, initial_drift_seconds_per_second:
        Seed model, typically obtained from the opening calibration anchor.
    target_origin_seconds:
        First target sample time in seconds; used as the drift-fit anchor.
    min_points_for_drift:
        Minimum accepted windows before the drift estimate is updated.
        Below this count the offset is adjusted but the seed drift is preserved.

    Returns
    -------
    ``(target_times_sec, offsets_sec, scores)`` – one entry per accepted window.
    """
    ref_signal = reference_series.signal
    tgt_signal = target_series.signal
    ref_ts = reference_series.timestamps_seconds
    tgt_ts = target_series.timestamps_seconds
    sample_rate_hz = float(reference_series.sample_rate_hz)

    n_ref = ref_signal.size
    n_tgt = tgt_signal.size
    if n_ref == 0 or n_tgt == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, {"accepted_windows": 0, "rejected_windows": 0}

    window_n = max(20, int(round(float(window_seconds) * sample_rate_hz)))
    step_n = max(5, int(round(float(window_step_seconds) * sample_rate_hz)))
    search_n = max(1, int(round(float(local_search_seconds) * sample_rate_hz)))
    half = window_n // 2

    if n_ref < window_n:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, {"accepted_windows": 0, "rejected_windows": 0}

    # Running model — seeded from opening calibration, updated as windows accumulate.
    current_offset = float(initial_offset_seconds)
    current_drift = float(initial_drift_seconds_per_second)

    target_times: list[float] = []
    offsets: list[float] = []
    scores: list[float] = []
    rejected_windows = 0
    ref_valid = _bool_mask(reference_series.valid_mask, n_ref)
    tgt_valid = _bool_mask(target_series.valid_mask, n_tgt)

    for center in range(half, n_ref - half, step_n):
        t_ref_center = float(ref_ts[center])

        # First-order inversion of t_ref = t_tgt + offset(t_tgt):
        #   t_tgt ≈ t_ref − offset − drift*(t_ref − offset − t_origin)
        t_tgt_predicted = (
            t_ref_center
            - current_offset
            - current_drift * (t_ref_center - current_offset - target_origin_seconds)
        )
        predicted_tgt_idx = int(np.clip(
            np.searchsorted(tgt_ts, t_tgt_predicted), 0, n_tgt - 1
        ))
        predicted_lag = center - predicted_tgt_idx

        left = center - half
        right = center + half
        ref_win = ref_signal[left:right]
        if ref_win.size < 10:
            continue

        best_lag: int | None = None
        best_score = float("-inf")

        for delta in range(-search_n, search_n + 1):
            lag = predicted_lag + delta
            t_left = left - lag
            t_right = right - lag
            if t_left < 0 or t_right > n_tgt:
                continue
            tgt_win = tgt_signal[t_left:t_right]
            if tgt_win.size != ref_win.size:
                continue
            score, valid_fraction = _masked_ncc(
                ref_win,
                tgt_win,
                a_valid=ref_valid[left:right],
                b_valid=tgt_valid[t_left:t_right],
                min_valid_fraction=min_valid_fraction,
            )
            if not np.isfinite(score):
                continue
            if score > best_score:
                best_score = score
                best_lag = lag

        if best_lag is None or best_score < min_window_score:
            rejected_windows += 1
            continue

        tgt_idx = center - best_lag
        if tgt_idx < 0 or tgt_idx >= tgt_ts.size:
            continue

        target_times.append(float(tgt_ts[tgt_idx]))
        offsets.append(float(ref_ts[center]) - float(tgt_ts[tgt_idx]))
        scores.append(best_score)

        # Update running model with all accepted observations so far.
        n_pts = len(target_times)
        if n_pts >= min_points_for_drift:
            t_arr = np.asarray(target_times, dtype=float)
            o_arr = np.asarray(offsets, dtype=float)
            w_arr = np.clip(np.asarray(scores, dtype=float), 0.0, None)
            intercept, slope, _ = _fit_offset_drift(
                t_arr, o_arr,
                target_origin_seconds=target_origin_seconds,
                weights=w_arr,
            )
            current_offset = intercept
            current_drift = slope
        elif n_pts >= 1:
            # Adjust offset by weighted mean; preserve seed drift.
            current_offset = float(np.average(offsets, weights=np.clip(scores, 0.0, None)))

    return (
        np.asarray(target_times, dtype=float),
        np.asarray(offsets, dtype=float),
        np.asarray(scores, dtype=float),
        {
            "accepted_windows": len(target_times),
            "rejected_windows": rejected_windows,
        },
    )


def _fit_offset_drift(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    *,
    target_origin_seconds: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if target_times_sec.size == 0:
        return 0.0, 0.0, 0.0
    if target_times_sec.size == 1:
        return float(offsets_sec[0]), 0.0, 0.0

    x = np.asarray(target_times_sec, dtype=float) - float(target_origin_seconds)
    y = np.asarray(offsets_sec, dtype=float)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = np.clip(w, a_min=0.0, a_max=None)
        if np.any(w > 0):
            slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(w))
        else:
            slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = np.polyfit(x, y, 1)

    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(intercept), float(slope), float(r2)


def _residual_summary(
    target_times_sec: np.ndarray,
    offsets_sec: np.ndarray,
    *,
    intercept: float,
    slope: float,
    target_origin_seconds: float,
) -> dict[str, float] | None:
    if target_times_sec.size == 0 or offsets_sec.size == 0:
        return None
    x = np.asarray(target_times_sec, dtype=float) - float(target_origin_seconds)
    y = np.asarray(offsets_sec, dtype=float)
    residuals = y - (float(intercept) + float(slope) * x)
    abs_res = np.abs(residuals)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "mae": float(np.mean(abs_res)),
        "median_abs": float(np.median(abs_res)),
        "max_abs": float(np.max(abs_res)),
    }


def _segment_selection_summary(
    reference_series: AlignmentSeries,
    target_series: AlignmentSeries,
    *,
    calibration_usage_strategy: str,
) -> dict[str, Any]:
    ref_valid = _bool_mask(reference_series.valid_mask, reference_series.signal.size)
    tgt_valid = _bool_mask(target_series.valid_mask, target_series.signal.size)
    sample_rate_hz = float(reference_series.sample_rate_hz) if reference_series.sample_rate_hz > 0 else 0.0
    usable_duration = float(min(ref_valid.sum(), tgt_valid.sum()) / sample_rate_hz) if sample_rate_hz > 0 else 0.0
    ref_segment_count = int(reference_series.segment_count)
    tgt_segment_count = int(target_series.segment_count)
    segment_aware_used = bool(
        calibration_usage_strategy != CALIBRATION_USAGE_FULL_STREAM
        and (reference_series.segment_mask.any() or target_series.segment_mask.any())
    )
    return {
        "calibration_usage_strategy": calibration_usage_strategy,
        "segment_aware_used": segment_aware_used,
        "segments_detected": {
            "reference": ref_segment_count,
            "target": tgt_segment_count,
        },
        "percentage_removed": {
            "reference": float(100.0 * (1.0 - ref_valid.mean())) if ref_valid.size else 0.0,
            "target": float(100.0 * (1.0 - tgt_valid.mean())) if tgt_valid.size else 0.0,
        },
        "percentage_used": {
            "reference": float(100.0 * ref_valid.mean()) if ref_valid.size else 0.0,
            "target": float(100.0 * tgt_valid.mean()) if tgt_valid.size else 0.0,
        },
        "usable_duration_seconds": usable_duration,
    }


def estimate_sync_model(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    reference_name: str,
    target_name: str,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 30.0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    min_valid_fraction: float = 0.5,
    signal_mode: str | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
    calibration_usage_strategy: str = CALIBRATION_USAGE_FULL_STREAM,
    segment_aware: bool = False,
    return_diagnostics: bool = False,
) -> SyncModel | tuple[SyncModel, dict[str, Any]]:
    """Estimate complete synchronization model: coarse SDA lag + LIDA-style linear drift fit."""
    ref_series = build_alignment_series(
        reference_df,
        sample_rate_hz=sample_rate_hz,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        calibration_usage_strategy=calibration_usage_strategy,
        segment_aware=segment_aware,
    )
    tgt_series = build_alignment_series(
        target_df,
        sample_rate_hz=sample_rate_hz,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        calibration_usage_strategy=calibration_usage_strategy,
        segment_aware=segment_aware,
    )
    if ref_series.signal.size == 0 or tgt_series.signal.size == 0:
        raise ValueError("Cannot estimate sync model from empty streams.")

    coarse_result = estimate_offset_from_series(
        reference_series=ref_series,
        target_series=tgt_series,
        max_lag_seconds=max_lag_seconds,
        calibration_usage_strategy=calibration_usage_strategy,
        return_diagnostics=True,
    )
    coarse, coarse_diag = coarse_result
    target_origin_seconds = float(tgt_series.timestamps_seconds[0])

    target_times, offsets, scores, _lags, window_stats = _windowed_lag_refinement(
        reference_series=ref_series,
        target_series=tgt_series,
        coarse_lag_samples=coarse.lag_samples,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_valid_fraction=min_valid_fraction,
    )

    fit_r2 = 0.0
    if offsets.size == 0:
        offset_seconds = float(coarse.offset_seconds)
        drift_seconds_per_second = 0.0
    else:
        weights = np.clip(scores, a_min=0.0, a_max=None)
        offset_seconds, drift_seconds_per_second, fit_r2 = _fit_offset_drift(
            target_times,
            offsets,
            target_origin_seconds=target_origin_seconds,
            weights=weights,
        )
        if fit_r2 < min_fit_r2:
            drift_seconds_per_second = 0.0

    model = SyncModel(
        reference_csv=str(reference_name),
        target_csv=str(target_name),
        target_time_origin_seconds=target_origin_seconds,
        offset_seconds=float(offset_seconds),
        drift_seconds_per_second=float(drift_seconds_per_second),
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(max_lag_seconds),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    if not return_diagnostics:
        return model
    diagnostics = {
        **coarse_diag,
        "signal_mode": ref_series.signal_mode,
        "accepted_windows": int(window_stats["accepted_windows"]),
        "rejected_windows": int(window_stats["rejected_windows"]),
        "kept_windows": int(window_stats["accepted_windows"]),
        "anchor_count": int(offsets.size),
        "local_corr_mean": float(np.mean(scores)) if scores.size else None,
        "local_corr_median": float(np.median(scores)) if scores.size else None,
        "drift_r2": float(fit_r2) if offsets.size else None,
        "residual_summary": _residual_summary(
            target_times,
            offsets,
            intercept=float(offset_seconds),
            slope=float(drift_seconds_per_second),
            target_origin_seconds=target_origin_seconds,
        ),
        "fallback_used": False,
    }
    return model, diagnostics


def apply_sync_model(
    target_df: pd.DataFrame,
    model: SyncModel,
    *,
    replace_timestamp: bool = True,
) -> pd.DataFrame:
    """Apply offset+drift model to target timestamps."""
    out = target_df.copy()
    out["timestamp_orig"] = pd.to_numeric(out["timestamp"], errors="coerce")
    aligned = apply_linear_time_transform(
        out["timestamp_orig"],
        offset_seconds=model.offset_seconds,
        drift_seconds_per_second=model.drift_seconds_per_second,
        target_origin_seconds=model.target_time_origin_seconds,
    )
    out["timestamp_aligned"] = aligned
    if replace_timestamp:
        out["timestamp"] = out["timestamp_aligned"]
    return out


def resample_aligned_stream(
    aligned_df: pd.DataFrame,
    *,
    resample_rate_hz: float,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Uniformly resample an aligned stream."""
    return resample_stream(
        aligned_df,
        sample_rate_hz=resample_rate_hz,
        timestamp_col=timestamp_col,
    )


def save_sync_model(model: SyncModel, path: Path | str) -> Path:
    """Serialize sync model JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(model), indent=2, sort_keys=False), encoding="utf-8")
    return out


def load_sync_model(path: Path | str) -> SyncModel:
    """Load sync model from JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_keys = {f.name for f in fields(SyncModel)}
    filtered = {k: v for k, v in data.items() if k in allowed_keys}
    return SyncModel(**filtered)


def fit_sync_from_paths(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 30.0,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    min_valid_fraction: float = 0.5,
    signal_mode: str | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
    calibration_usage_strategy: str = CALIBRATION_USAGE_FULL_STREAM,
    segment_aware: bool = False,
) -> SyncModel:
    """Load two CSV paths and estimate a sync model."""
    reference_path = Path(reference_csv)
    target_path = Path(target_csv)
    reference_df = load_stream(reference_path)
    target_df = load_stream(target_path)
    return estimate_sync_model(
        reference_df,
        target_df,
        reference_name=str(reference_path),
        target_name=str(target_path),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_fit_r2=min_fit_r2,
        min_valid_fraction=min_valid_fraction,
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        calibration_usage_strategy=calibration_usage_strategy,
        segment_aware=segment_aware,
    )


# ---------------------------------------------------------------------------
# Quality metrics (sync_info.json correlation block)
# ---------------------------------------------------------------------------


def acc_norm_correlation(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    sample_rate_hz: float,
) -> float | None:
    """Pearson r of ``acc_norm`` between two streams over their overlap."""
    ref_r = add_vector_norms(resample_stream(ref_df, sample_rate_hz))
    tgt_r = add_vector_norms(resample_stream(tgt_df, sample_rate_hz))

    ref_ts = ref_r["timestamp"].to_numpy(dtype=float)
    tgt_ts = tgt_r["timestamp"].to_numpy(dtype=float)
    if ref_ts.size == 0 or tgt_ts.size == 0:
        return None

    lo = max(float(ref_ts[0]), float(tgt_ts[0]))
    hi = min(float(ref_ts[-1]), float(tgt_ts[-1]))
    if lo >= hi:
        return None

    ref_acc = ref_r.loc[(ref_ts >= lo) & (ref_ts <= hi), "acc_norm"].to_numpy(dtype=float)
    tgt_acc = tgt_r.loc[(tgt_ts >= lo) & (tgt_ts <= hi), "acc_norm"].to_numpy(dtype=float)

    n = min(len(ref_acc), len(tgt_acc))
    if n < 10:
        return None

    x, y = ref_acc[:n], tgt_acc[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None

    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def compute_sync_correlations(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    model: SyncModel,
    *,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """Correlation of ``acc_norm`` before (offset only) and after (offset + drift) sync."""
    offset_only_model = replace(model, drift_seconds_per_second=0.0)
    offset_only_df = apply_sync_model(tgt_df, offset_only_model, replace_timestamp=True)

    full_model_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

    corr_offset = acc_norm_correlation(ref_df, offset_only_df, sample_rate_hz=sample_rate_hz)
    corr_full = acc_norm_correlation(ref_df, full_model_df, sample_rate_hz=sample_rate_hz)

    return {
        "offset_only": round(corr_offset, 4) if corr_offset is not None else None,
        "offset_and_drift": round(corr_full, 4) if corr_full is not None else None,
        "signal": "acc_norm",
        "sample_rate_hz": sample_rate_hz,
    }
