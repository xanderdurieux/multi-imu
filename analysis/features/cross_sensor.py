"""Cross-sensor features (agreement, disagreement, energy ratio)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from common.statistics import safe_max, safe_mean
from .stats_helpers import get_col


def cross_sensor_features(window_cross: pd.DataFrame | None) -> dict[str, Any]:
    """Extract cross-sensor features from the cross_sensor_signals slice."""
    feats: dict[str, Any] = {
        "cross_acc_correlation_mean": float("nan"),
        "cross_gyro_diff_mean": float("nan"),
        "cross_acc_diff_mean": float("nan"),
        "cross_disagree_score_mean": float("nan"),
        "cross_disagree_score_max": float("nan"),
        "cross_acc_energy_ratio": float("nan"),
    }
    if window_cross is None or window_cross.empty:
        return feats

    if "acc_correlation" in window_cross.columns:
        feats["cross_acc_correlation_mean"] = safe_mean(get_col(window_cross, "acc_correlation"))
    if "gyro_diff_norm" in window_cross.columns:
        feats["cross_gyro_diff_mean"] = safe_mean(get_col(window_cross, "gyro_diff_norm"))
    if "acc_diff_norm" in window_cross.columns:
        feats["cross_acc_diff_mean"] = safe_mean(get_col(window_cross, "acc_diff_norm"))
    if "disagree_score" in window_cross.columns:
        ds = get_col(window_cross, "disagree_score")
        feats["cross_disagree_score_mean"] = safe_mean(ds)
        feats["cross_disagree_score_max"] = safe_max(ds)
    if "acc_ratio" in window_cross.columns:
        feats["cross_acc_energy_ratio"] = safe_mean(get_col(window_cross, "acc_ratio"))

    return feats
