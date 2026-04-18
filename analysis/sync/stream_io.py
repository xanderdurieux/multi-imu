"""Stream loading, resampling, filtering, and dropout removal."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from common.paths import read_csv
from common.signals import add_imu_norms

VECTOR_AXES: dict[str, list[str]] = {
    "acc": ["ax", "ay", "az"],
    "gyro": ["gx", "gy", "gz"],
    "mag": ["mx", "my", "mz"],
}

_IMU_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]


def load_stream(csv_path: Path | str) -> pd.DataFrame:
    """Load an IMU CSV, coerce types, sort by timestamp."""
    df = read_csv(Path(csv_path)).copy()
    df = df.dropna(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def infer_numeric_columns(
    df: pd.DataFrame, skip: Iterable[str] = ("timestamp",)
) -> list[str]:
    skip_set = set(skip)
    return [
        c for c in df.columns
        if c not in skip_set and pd.api.types.is_numeric_dtype(df[c])
    ]


def resample_stream(
    df: pd.DataFrame,
    sample_rate_hz: float,
    *,
    timestamp_col: str = "timestamp",
    columns: list[str] | None = None,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> pd.DataFrame:
    """Resample stream to a uniform time grid via linear interpolation."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be > 0")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")

    base = df.copy()
    base[timestamp_col] = pd.to_numeric(base[timestamp_col], errors="coerce")
    base = base.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
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
            out[col] = np.interp(
                out[timestamp_col].to_numpy(dtype=float), ts[valid], values[valid]
            )
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
    """Resample target values at reference timestamps."""
    if timestamp_col not in target_df.columns or timestamp_col not in reference_df.columns:
        raise ValueError(f"both DataFrames must contain '{timestamp_col}'")

    ref = reference_df.copy()
    ref[timestamp_col] = pd.to_numeric(ref[timestamp_col], errors="coerce")
    ref = ref.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    tgt = target_df.copy()
    tgt[timestamp_col] = pd.to_numeric(tgt[timestamp_col], errors="coerce")
    tgt = tgt.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    ref_ts = ref[timestamp_col].to_numpy(dtype=float)
    tgt_ts = tgt[timestamp_col].to_numpy(dtype=float)
    out = pd.DataFrame({timestamp_col: ref_ts})
    if ref_ts.size == 0 or tgt_ts.size == 0:
        return out

    for col in infer_numeric_columns(tgt, skip=[timestamp_col]):
        values = pd.to_numeric(tgt[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(tgt_ts) & np.isfinite(values)
        if valid.sum() < 2:
            out[col] = np.nan
            continue
        ts_v, val_v = tgt_ts[valid], values[valid]
        series = np.full(ref_ts.shape, np.nan, dtype=float)
        inside = (ref_ts >= ts_v.min()) & (ref_ts <= ts_v.max())
        if inside.any():
            series[inside] = np.interp(ref_ts[inside], ts_v, val_v)
        out[col] = series
    return out


def lowpass_filter(
    df: pd.DataFrame,
    cutoff_hz: float,
    sample_rate_hz: float,
    *,
    columns: list[str] | None = None,
    order: int = 4,
) -> pd.DataFrame:
    """Apply a zero-phase Butterworth low-pass filter."""
    nyq = 0.5 * float(sample_rate_hz)
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz must be in (0, {nyq:.1f}); got {cutoff_hz}.")
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


def remove_dropouts(
    df: pd.DataFrame, *, epsilon_fraction: float = 0.1
) -> pd.DataFrame:
    """Remove rows where acc_norm is near-zero (BLE dropout packets)."""
    out = add_imu_norms(df)
    g_approx = float(out["acc_norm"].median())
    if g_approx <= 0:
        return df
    threshold = epsilon_fraction * g_approx
    valid = out["acc_norm"] >= threshold
    return df.loc[valid.values].reset_index(drop=True)
