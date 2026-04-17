"""Shared 1-D signal helpers (acceleration norms, smoothing, peak picking)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def vector_norm(df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Euclidean norm of the given columns per row."""
    data = df[list(columns)].to_numpy(dtype=float)
    nan_mask = np.isnan(data).any(axis=1)
    norms = np.sqrt(np.nansum(data**2, axis=1))
    norms[nan_mask] = np.nan
    return norms


def add_imu_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` with `acc_norm`, `gyro_norm`, and `mag_norm` columns added."""
    out = df.copy()
    out["acc_norm"] = vector_norm(out, ["ax", "ay", "az"])
    out["gyro_norm"] = vector_norm(out, ["gx", "gy", "gz"])
    out["mag_norm"] = vector_norm(out, ["mx", "my", "mz"])
    return out


def smooth_moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Centered box-car moving average, same length as input."""
    if window <= 1:
        return np.asarray(signal, dtype=float).copy()
    return (
        pd.Series(signal)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    ) 