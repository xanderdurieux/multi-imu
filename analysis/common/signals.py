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


def rolling_pearson(a: np.ndarray, b: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson *r* using a centered window. Returns NaN at edges."""
    n = len(a)
    out = np.full(n, np.nan)
    half = window // 2
    for i in range(half, n - half):
        x = a[i - half : i + half]
        y = b[i - half : i + half]
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            continue
        out[i] = np.corrcoef(x, y)[0, 1]
    return out


def find_peaks_simple(
    signal: np.ndarray,
    *,
    height: float = 0.0,
    distance: int = 1,
) -> np.ndarray:
    """Indices of local maxima above *height* with minimum index *distance*."""
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 3:
        return np.empty(0, dtype=int)

    candidates = [
        i
        for i in range(1, n - 1)
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] >= height
    ]
    if not candidates:
        return np.empty(0, dtype=int)

    if distance <= 1:
        return np.array(candidates, dtype=int)

    cands = np.array(candidates, dtype=int)
    order = np.argsort(x[cands])[::-1]
    sorted_cands = cands[order]

    keep = np.ones(len(sorted_cands), dtype=bool)
    for i in range(len(sorted_cands)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(sorted_cands)):
            if keep[j] and abs(int(sorted_cands[i]) - int(sorted_cands[j])) < distance:
                keep[j] = False

    return np.sort(sorted_cands[keep])
