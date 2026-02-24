"""Shared utilities for IMU stream synchronization and resampling."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common import load_dataframe

VECTOR_AXES = {
    "acc": ["ax", "ay", "az"],
    "gyro": ["gx", "gy", "gz"],
    "mag": ["mx", "my", "mz"],
}


def load_stream(csv_path: Path | str) -> pd.DataFrame:
    """Load a processed IMU stream CSV and return rows sorted by timestamp."""
    path = Path(csv_path)
    df = load_dataframe(path).copy()
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_vector_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Add orientation-invariant vector magnitude columns (|acc|, |gyro|, |mag|)."""
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
    """Infer numeric columns excluding ``skip``."""
    skip_set = set(skip)
    cols: list[str] = []
    for col in df.columns:
        if col in skip_set:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def resample_stream(
    df: pd.DataFrame,
    sample_rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
    columns: list[str] | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> pd.DataFrame:
    """Resample IMU stream to uniform sampling rate using linear interpolation (LIDA-style)."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")

    base = df.copy()
    base[timestamp_col] = pd.to_numeric(base[timestamp_col], errors="coerce")
    base = base.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if base.empty:
        return base

    ts = base[timestamp_col].to_numpy(dtype=float)
    step_ms = 1000.0 / sample_rate_hz

    lo = float(ts[0] if start_ms is None else start_ms)
    hi = float(ts[-1] if end_ms is None else end_ms)
    if hi <= lo:
        one = base.iloc[[0]].copy()
        one[timestamp_col] = lo
        return one

    grid = np.arange(lo, hi + 0.5 * step_ms, step_ms, dtype=float)
    out = pd.DataFrame({timestamp_col: grid})

    if columns is None:
        columns = infer_numeric_columns(base, skip=[timestamp_col])

    for col in columns:
        values = pd.to_numeric(base[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(ts) & np.isfinite(values)
        if valid.sum() >= 2:
            out[col] = np.interp(grid, ts[valid], values[valid])
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
    """
    Resample a target stream onto the timestamp series of a reference stream.

    Uses linear interpolation of the target values at the reference timestamps.
    Values outside the temporal support of the target become NaN so that only
    the overlapping region carries valid data.
    """
    if timestamp_col not in target_df.columns or timestamp_col not in reference_df.columns:
        raise ValueError(f"both dataframes must contain '{timestamp_col}'")

    ref = reference_df.copy()
    ref_ts = pd.to_numeric(ref[timestamp_col], errors="coerce").to_numpy(dtype=float)
    tgt = target_df.copy()
    tgt_ts = pd.to_numeric(tgt[timestamp_col], errors="coerce").to_numpy(dtype=float)

    if ref_ts.size == 0 or tgt_ts.size == 0:
        return pd.DataFrame({timestamp_col: ref_ts})

    out = pd.DataFrame({timestamp_col: ref_ts})

    columns = infer_numeric_columns(tgt, skip=[timestamp_col])
    for col in columns:
        values = pd.to_numeric(tgt[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(tgt_ts) & np.isfinite(values)
        if valid.sum() >= 2:
            ts_valid = tgt_ts[valid]
            vals_valid = values[valid]
            lo = float(ts_valid.min())
            hi = float(ts_valid.max())
            series = np.full(ref_ts.shape, np.nan, dtype=float)
            inside = (ref_ts >= lo) & (ref_ts <= hi)
            if inside.any():
                series[inside] = np.interp(ref_ts[inside], ts_valid, vals_valid)
            out[col] = series
        elif valid.sum() == 1:
            # Single valid point: propagate it only where reference is close to that time.
            ts0 = float(tgt_ts[valid][0])
            vals0 = float(values[valid][0])
            series = np.full(ref_ts.shape, np.nan, dtype=float)
            series[np.isclose(ref_ts, ts0)] = vals0
            out[col] = series
        else:
            out[col] = np.nan

    return out


def apply_linear_time_transform(
    timestamp_ms: pd.Series | np.ndarray,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Apply linear time transformation (offset + drift) to map target timestamps to reference clock."""
    ts_sec = np.asarray(timestamp_ms, dtype=float) / 1000.0
    aligned_sec = (
        ts_sec
        + offset_seconds
        + drift_seconds_per_second * (ts_sec - target_origin_seconds)
    )
    return aligned_sec * 1000.0
