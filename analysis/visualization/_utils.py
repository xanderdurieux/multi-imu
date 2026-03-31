"""Internal visualization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.csv_schema import load_dataframe


def mask_valid_plot_x(x: np.ndarray) -> np.ndarray:
    """Return boolean mask of finite, monotonically non-decreasing values."""
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    # Also mask negative jumps (time going backwards).
    diffs = np.diff(x, prepend=x[0])
    monotone = diffs >= 0
    return finite & monotone


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
