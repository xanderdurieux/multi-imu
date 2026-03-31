"""Internal visualization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

def _as_float_vector(values: np.ndarray) -> np.ndarray:
    """Return a flattened float vector for plotting helpers."""
    return np.ravel(np.asarray(values, dtype=float))


def mask_valid_plot_x(x: np.ndarray) -> np.ndarray:
    """Return boolean mask of finite, monotonically non-decreasing values."""
    x = _as_float_vector(x)
    if x.size == 0:
        return np.array([], dtype=bool)
    finite = np.isfinite(x)
    # Also mask negative jumps (time going backwards).
    diffs = np.diff(x, prepend=x[0])
    monotone = diffs >= 0
    return finite & monotone


def mask_valid_plot_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return boolean mask of points safe to plot for one series."""
    x = _as_float_vector(x)
    y = _as_float_vector(y)
    if x.shape != y.shape:
        raise ValueError(f"Plot arrays must have the same shape, got {x.shape} and {y.shape}")
    return mask_valid_plot_x(x) & np.isfinite(y)


def filter_valid_plot_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays already filtered to values safe to plot."""
    x = _as_float_vector(x)
    y = _as_float_vector(y)
    valid = mask_valid_plot_xy(x, y)
    return x[valid], y[valid]


def strict_vector_norm(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Compute vector norm, requiring all components to be finite."""
    if not cols:
        return np.array([], dtype=float)
    arr = df[cols].to_numpy(dtype=float)
    valid = np.all(np.isfinite(arr), axis=1)
    norm = np.full(arr.shape[0], np.nan, dtype=float)
    norm[valid] = np.sqrt(np.sum(arr[valid] ** 2, axis=1))
    return norm


def mask_dropout_packets(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where acc_norm is near-zero (sensor dropout packets)."""
    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    if not acc_cols:
        return df
    arr = df[acc_cols].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(arr ** 2, axis=1))
    g_approx = float(np.nanmedian(norm))
    if g_approx <= 0:
        return df
    threshold = 0.1 * g_approx
    valid = norm >= threshold
    return df.loc[valid].reset_index(drop=True)
