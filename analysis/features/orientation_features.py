"""Orientation-based features (pitch/roll statistics)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.statistics import safe_mean, safe_std


def orientation_features(
    sensor_prefix: str,
    window_orient: pd.DataFrame | None,
) -> dict[str, Any]:
    """Extract pitch/roll mean/std and their sample-to-sample rate std."""
    base = {
        f"{sensor_prefix}_pitch_mean": float("nan"),
        f"{sensor_prefix}_pitch_std": float("nan"),
        f"{sensor_prefix}_roll_mean": float("nan"),
        f"{sensor_prefix}_roll_std": float("nan"),
        f"{sensor_prefix}_pitch_rate_std": float("nan"),
        f"{sensor_prefix}_roll_rate_std": float("nan"),
    }
    if window_orient is None or window_orient.empty:
        return base

    if "pitch_deg" in window_orient.columns:
        pitch = window_orient["pitch_deg"].to_numpy(dtype=float)
        base[f"{sensor_prefix}_pitch_mean"] = safe_mean(pitch)
        base[f"{sensor_prefix}_pitch_std"] = safe_std(pitch)
        if len(pitch) >= 2:
            pitch_rate = np.diff(pitch[np.isfinite(pitch)])
            base[f"{sensor_prefix}_pitch_rate_std"] = safe_std(pitch_rate)

    if "roll_deg" in window_orient.columns:
        roll = window_orient["roll_deg"].to_numpy(dtype=float)
        base[f"{sensor_prefix}_roll_mean"] = safe_mean(roll)
        base[f"{sensor_prefix}_roll_std"] = safe_std(roll)
        if len(roll) >= 2:
            roll_rate = np.diff(roll[np.isfinite(roll)])
            base[f"{sensor_prefix}_roll_rate_std"] = safe_std(roll_rate)

    return base
