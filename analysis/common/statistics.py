"""NaN-safe scalar statistics over 1-D arrays.

All helpers return ``float('nan')`` when the input has no finite values, and
ignore NaNs otherwise. Kept purely scalar for use inside feature-extraction
loops; for vectorised aggregates use ``numpy``'s ``nanmean`` / ``nanstd``
directly.
"""

from __future__ import annotations

import numpy as np


_EPSILON = 1e-9


def safe_mean(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    v = float(np.nanstd(arr))
    return v if np.isfinite(v) else float("nan")


def safe_min(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmin(arr))


def safe_max(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmax(arr))


def safe_iqr(arr: np.ndarray) -> float:
    clean = np.asarray(arr, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return float("nan")
    q75, q25 = np.percentile(clean, [75, 25])
    return float(q75 - q25)


def safe_skew(arr: np.ndarray) -> float:
    clean = np.asarray(arr, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 3:
        return float("nan")
    mu = np.mean(clean)
    sigma = np.std(clean)
    if sigma < _EPSILON:
        return 0.0
    return float(np.mean(((clean - mu) / sigma) ** 3))


def safe_kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition, normal → 0)."""
    clean = np.asarray(arr, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 4:
        return float("nan")
    mu = np.mean(clean)
    sigma = np.std(clean)
    if sigma < _EPSILON:
        return 0.0
    return float(np.mean(((clean - mu) / sigma) ** 4) - 3.0)


def safe_energy(arr: np.ndarray) -> float:
    """Mean of squared finite values."""
    clean = np.asarray(arr, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan")
    return float(np.mean(clean * clean))


def zero_crossings(arr: np.ndarray) -> int:
    """Number of sign changes among finite values (zeros are skipped)."""
    clean = np.asarray(arr, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return 0
    signs = np.sign(clean)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(np.diff(nonzero) != 0))
