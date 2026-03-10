"""Shared low-level utilities for visualization modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mask_dropout_packets(df: pd.DataFrame, epsilon_fraction: float = 0.1) -> pd.DataFrame:
    """Set sensor columns to NaN for rows where acc_norm is near-zero.

    The timestamp column is preserved so dropout periods appear as visible
    gaps rather than zero-value spikes on the time axis.
    """
    acc_cols = ["ax", "ay", "az"]
    if not all(c in df.columns for c in acc_cols):
        return df
    acc_norm = np.sqrt((df[acc_cols].to_numpy(dtype=float) ** 2).sum(axis=1))
    g_approx = float(np.nanmedian(acc_norm))
    if g_approx <= 0:
        return df
    dropout = acc_norm < epsilon_fraction * g_approx
    if not dropout.any():
        return df
    out = df.copy()
    sensor_cols = [c for c in df.columns if c != "timestamp"]
    out.loc[dropout, sensor_cols] = np.nan
    return out


def time_axis_seconds(timestamps_ms: pd.Series) -> pd.Series:
    """Convert a millisecond timestamp series to seconds starting at zero."""
    ts = timestamps_ms.astype(float)
    return (ts - float(ts.iloc[0])) / 1000.0


def acc_norm_series(df: pd.DataFrame) -> np.ndarray:
    """Compute accelerometer vector norm for each row."""
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    return np.sqrt((acc ** 2).sum(axis=1))
