"""Internal visualization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

def _as_float_vector(values: np.ndarray) -> np.ndarray:
    """Return a flattened float vector for plotting helpers."""
    return np.ravel(np.asarray(values, dtype=float))


def _mask_valid_plot_x(x: np.ndarray) -> np.ndarray:
    """Return boolean mask of finite, monotonically non-decreasing values."""
    x = _as_float_vector(x)
    if x.size == 0:
        return np.array([], dtype=bool)
    finite = np.isfinite(x)
    # Also mask negative jumps (time going backwards).
    diffs = np.diff(x, prepend=x[0])
    monotone = diffs >= 0
    return finite & monotone


def filter_valid_plot_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays already filtered to values safe to plot."""
    x = _as_float_vector(x)
    y = _as_float_vector(y)
    if x.shape != y.shape:
        raise ValueError(f"Plot arrays must have the same shape, got {x.shape} and {y.shape}")
    valid = _mask_valid_plot_x(x) & np.isfinite(y)
    return x[valid], y[valid]


def strict_vector_norm(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Return row-wise vector norm while preserving NaN for incomplete rows."""
    if not cols:
        return np.array([], dtype=float)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for vector norm: {missing}")

    arr = np.column_stack([pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in cols])
    out = np.full(arr.shape[0], np.nan, dtype=float)
    valid = np.isfinite(arr).all(axis=1)
    out[valid] = np.linalg.norm(arr[valid], axis=1)
    return out


def timestamps_to_relative_seconds(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert timestamp-like values in ms to relative seconds."""
    ts = _as_float_vector(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    if ts.size == 0:
        return ts
    finite = np.isfinite(ts)
    if not finite.any():
        return np.full_like(ts, np.nan, dtype=float)
    t0 = ts[finite][0]
    out = (ts - t0) / 1000.0
    out[~finite] = np.nan
    return out
