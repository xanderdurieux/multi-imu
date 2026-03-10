"""Low-level windowed feature extraction for IMU streams.

The core entry point is :func:`compute_time_series_features`, which expects a
DataFrame with at least::

    timestamp, ax, ay, az, gx, gy, gz

and computes a compact set of descriptive statistics suitable for incident
analysis:

- Duration and sample statistics.
- Acceleration and angular-rate magnitude statistics.
- Simple jerk (finite-difference of acceleration) statistics.

These helpers are used by :mod:`features.section_features` to build
per-recording and per-section feature tables.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _safe_stats(arr: np.ndarray) -> dict[str, float]:
    """Return a standard set of robust statistics for a 1D array."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
        }

    finite = np.isfinite(arr)
    if not np.any(finite):
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
        }

    vals = arr[finite]
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p25": float(np.percentile(vals, 25.0)),
        "p50": float(np.percentile(vals, 50.0)),
        "p75": float(np.percentile(vals, 75.0)),
    }


def compute_time_series_features(
    df: pd.DataFrame,
    *,
    sensor: str,
    recording_name: str,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute basic IMU features for one contiguous time series.

    Parameters
    ----------
    df:
        DataFrame with at least ``timestamp``, ``ax``, ``ay``, ``az``,
        ``gx``, ``gy``, ``gz``. Additional columns are ignored here.
    sensor:
        Logical sensor name (e.g. ``"sporsa"`` or ``"arduino"``).
    recording_name:
        Recording identifier, attached to the feature row.
    context:
        Optional extra metadata (e.g. section id, stage) to attach to the
        feature dictionary.

    Returns
    -------
    dict
        A flat feature dictionary; all keys are JSON-serialisable.
    """
    if df.empty:
        features: dict[str, Any] = {
            "recording": recording_name,
            "sensor": sensor,
            "n_samples": 0,
            "duration_s": float("nan"),
        }
        if context:
            features.update(context)
        return features

    ts = df["timestamp"].to_numpy(dtype=float)
    t0 = float(ts[0])
    t1 = float(ts[-1])
    duration_s = (t1 - t0) / 1000.0 if t1 >= t0 else float("nan")

    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyro = df[["gx", "gy", "gz"]].to_numpy(dtype=float)

    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)

    # Approximate jerk (time-derivative of acceleration magnitude).
    if len(acc_norm) >= 2 and np.all(np.isfinite(ts)):
        dt = np.diff(ts) / 1000.0
        dt[dt <= 0] = np.nan
        jerk = np.diff(acc_norm) / dt
        # Pad to original length for convenience.
        jerk = np.concatenate([[np.nan], jerk])
    else:
        jerk = np.full_like(acc_norm, np.nan, dtype=float)

    acc_stats = _safe_stats(acc_norm)
    gyro_stats = _safe_stats(gyro_norm)
    jerk_stats = _safe_stats(jerk)

    features = {
        "recording": recording_name,
        "sensor": sensor,
        "n_samples": int(len(df)),
        "duration_s": float(duration_s),
        "acc_min": acc_stats["min"],
        "acc_max": acc_stats["max"],
        "acc_mean": acc_stats["mean"],
        "acc_std": acc_stats["std"],
        "acc_p25": acc_stats["p25"],
        "acc_p50": acc_stats["p50"],
        "acc_p75": acc_stats["p75"],
        "gyro_min": gyro_stats["min"],
        "gyro_max": gyro_stats["max"],
        "gyro_mean": gyro_stats["mean"],
        "gyro_std": gyro_stats["std"],
        "gyro_p25": gyro_stats["p25"],
        "gyro_p50": gyro_stats["p50"],
        "gyro_p75": gyro_stats["p75"],
        "jerk_min": jerk_stats["min"],
        "jerk_max": jerk_stats["max"],
        "jerk_mean": jerk_stats["mean"],
        "jerk_std": jerk_stats["std"],
        "jerk_p25": jerk_stats["p25"],
        "jerk_p50": jerk_stats["p50"],
        "jerk_p75": jerk_stats["p75"],
    }

    if context:
        features.update(context)

    return features

