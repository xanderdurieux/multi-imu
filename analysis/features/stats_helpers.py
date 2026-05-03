"""Window-level helpers shared across feature families."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.statistics import (
    safe_energy,
    safe_iqr,
    safe_kurtosis,
    safe_max,
    safe_mean,
    safe_min,
    safe_skew,
    safe_std,
    zero_crossings,
)


def get_col(df: pd.DataFrame | None, col: str) -> np.ndarray:
    """Return column as float array, or empty array if df is None/empty/missing col."""
    if df is None or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float)


def signal_stats(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    """Return basic stats dict for signal array under ``prefix``."""
    return {
        f"{prefix}_mean": safe_mean(arr),
        f"{prefix}_std": safe_std(arr),
        f"{prefix}_min": safe_min(arr),
        f"{prefix}_max": safe_max(arr),
        f"{prefix}_iqr": safe_iqr(arr),
        f"{prefix}_skew": safe_skew(arr),
        f"{prefix}_kurtosis": safe_kurtosis(arr),
        f"{prefix}_energy": safe_energy(arr),
        f"{prefix}_zero_crossings": zero_crossings(arr),
    }


def window_valid_ratio_imu(
    window_signals: pd.DataFrame | None,
    window_calibrated: pd.DataFrame | None,
    *,
    signal_col: str = "acc_norm",
) -> float:
    """Fraction of samples with finite acceleration norm in the window.

    Uses derived ``signal_col`` when present; if that slice is empty or
    entirely non-finite, falls back to ``acc_norm`` from the calibrated stream.

    Without this fallback, timestamps misaligned between calibrated and derived
    CSVs (or a missing derived slice) incorrectly yield *zero* validity even
    when the calibrated stream has hundreds of good samples.
    """
    arr = get_col(window_signals, signal_col)
    if len(arr) > 0:
        r = float(np.sum(np.isfinite(arr)) / len(arr))
        if r > 0.0:
            return float(np.clip(r, 0.0, 1.0))

    cal_arr = get_col(window_calibrated, "acc_norm")
    if cal_arr.size == 0:
        return 0.0
    return float(np.clip(np.isfinite(cal_arr).mean(), 0.0, 1.0))


def cross_window_valid_ratio(window_cross: pd.DataFrame | None) -> float:
    """Mean finite-data rate across core cross-sensor columns (0-1).

    When both IMUs contribute to fusion, cross features should have usable
    samples in the same window; pervasive NaNs indicate failed alignment or
    missing overlap for that interval.
    """
    if window_cross is None or window_cross.empty:
        return 0.0
    candidates = ["acc_correlation", "gyro_diff_norm", "acc_diff_norm", "acc_ratio"]
    if "disagree_score" in window_cross.columns:
        candidates.append("disagree_score")
    elif "disagree_combined_heuristic" in window_cross.columns:
        candidates.append("disagree_combined_heuristic")
    cols = [c for c in candidates if c in window_cross.columns]
    if not cols:
        return 1.0
    arr = window_cross[cols].to_numpy(dtype=float)
    return float(np.mean(np.isfinite(arr)))
