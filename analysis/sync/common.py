"""Shared utilities for IMU stream synchronization and resampling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from common import load_dataframe

VECTOR_AXES: dict[str, list[str]] = {
    "acc": ["ax", "ay", "az"],
    "gyro": ["gx", "gy", "gz"],
    "mag": ["mx", "my", "mz"],
}


def load_stream(csv_path: Path | str) -> pd.DataFrame:
    """Load an IMU CSV, coerce numeric schema, and sort by timestamp."""
    path = Path(csv_path)
    df = load_dataframe(path).copy()
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
    """Apply a zero-phase Butterworth low-pass filter to IMU columns.

    The stream must already be on a uniform time grid (e.g. after
    :func:`resample_stream`).  Columns that are absent in *df* are silently
    skipped.

    Parameters
    ----------
    df:
        Uniformly-sampled IMU DataFrame.
    cutoff_hz:
        Filter cutoff frequency in Hz (must be < sample_rate_hz / 2).
    sample_rate_hz:
        Sampling rate of *df* in Hz.
    columns:
        Columns to filter.  Defaults to all standard IMU axes present in *df*
        (``ax, ay, az, gx, gy, gz``).
    order:
        Butterworth filter order (default 4).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the selected columns low-pass filtered.
    """
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
    """Remove rows where the acceleration norm is near-zero (sensor dropout packets).

    Some sensors (e.g. Arduino) intermittently emit zero-valued packets whose
    ``acc_norm`` ≈ 0.  These corrupt cross-correlation and calibration detection
    and should be excluded before processing.

    Parameters
    ----------
    df:
        Input IMU DataFrame with ``ax``, ``ay``, ``az`` columns.
    epsilon_fraction:
        Rows with ``acc_norm < epsilon_fraction × median(acc_norm)`` are removed.
        Default 0.1 (10 % of the median gravity value).
    """
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
