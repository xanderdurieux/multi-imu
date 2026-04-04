"""Shared 1-D signal helpers (acceleration norms, smoothing, peak picking)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def acc_norm_from_imu_df(df: pd.DataFrame) -> np.ndarray:
    """Euclidean norm of ax, ay, az per row (NaNs treated via nansum)."""
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    return np.sqrt(np.nansum(acc**2, axis=1))


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
