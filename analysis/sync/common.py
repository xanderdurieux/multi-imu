"""Shared helpers for IMU stream synchronization and resampling."""

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
    """Load a processed IMU stream and return rows sorted by timestamp."""
    path = Path(csv_path)
    df = load_dataframe(path).copy()
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_vector_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with orientation-invariant vector norms."""
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
    rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
    columns: list[str] | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> pd.DataFrame:
    """Resample ``df`` to a uniform sampling rate using linear interpolation."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")

    base = df.copy()
    base[timestamp_col] = pd.to_numeric(base[timestamp_col], errors="coerce")
    base = base.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if base.empty:
        return base

    ts = base[timestamp_col].to_numpy(dtype=float)
    step_ms = 1000.0 / rate_hz

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


def apply_linear_time_transform(
    timestamp_ms: pd.Series | np.ndarray,
    *,
    offset_seconds: float,
    drift_seconds_per_second: float,
    target_origin_seconds: float,
) -> np.ndarray:
    """Map target timestamps to reference clock with a linear offset+drift model."""
    ts_sec = np.asarray(timestamp_ms, dtype=float) / 1000.0
    aligned_sec = (
        ts_sec
        + offset_seconds
        + drift_seconds_per_second * (ts_sec - target_origin_seconds)
    )
    return aligned_sec * 1000.0
