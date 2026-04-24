"""Cross-sensor features (agreement, disagreement, energy ratio)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.statistics import safe_max, safe_mean, safe_min, safe_std
from .stats_helpers import get_col

# Mirrors EventConfig.disagree_threshold — fraction-above feature uses same boundary.
_DISAGREE_THRESH = 4.0


def _linear_slope(arr: np.ndarray) -> float:
    """Least-squares linear trend. Returns nan when fewer than 2 finite values."""
    finite_idx = np.where(np.isfinite(arr))[0]
    if len(finite_idx) < 2:
        return float("nan")
    slope = np.polyfit(finite_idx.astype(float), arr[finite_idx], 1)[0]
    return float(slope) if np.isfinite(slope) else float("nan")


def cross_sensor_features(window_cross: pd.DataFrame | None) -> dict[str, Any]:
    """Extract cross-sensor features from the cross_sensor_signals slice."""
    feats: dict[str, Any] = {
        # --- agreement level (original) ---
        "cross_acc_correlation_mean": float("nan"),
        "cross_gyro_diff_mean": float("nan"),
        "cross_acc_diff_mean": float("nan"),
        "cross_disagree_score_mean": float("nan"),
        "cross_disagree_score_max": float("nan"),
        "cross_acc_energy_ratio": float("nan"),
        # --- agreement variability & dynamics (new) ---
        "cross_acc_correlation_std": float("nan"),
        "cross_acc_correlation_min": float("nan"),
        "cross_gyro_diff_std": float("nan"),
        "cross_acc_diff_std": float("nan"),
        "cross_disagree_score_std": float("nan"),
        "cross_disagree_score_trend": float("nan"),
        "cross_disagree_fraction": float("nan"),
    }
    if window_cross is None or window_cross.empty:
        return feats

    if "acc_correlation" in window_cross.columns:
        corr = get_col(window_cross, "acc_correlation")
        feats["cross_acc_correlation_mean"] = safe_mean(corr)
        feats["cross_acc_correlation_std"] = safe_std(corr)
        feats["cross_acc_correlation_min"] = safe_min(corr)

    if "gyro_diff_norm" in window_cross.columns:
        gd = get_col(window_cross, "gyro_diff_norm")
        feats["cross_gyro_diff_mean"] = safe_mean(gd)
        feats["cross_gyro_diff_std"] = safe_std(gd)

    if "acc_diff_norm" in window_cross.columns:
        ad = get_col(window_cross, "acc_diff_norm")
        feats["cross_acc_diff_mean"] = safe_mean(ad)
        feats["cross_acc_diff_std"] = safe_std(ad)

    if "disagree_score" in window_cross.columns:
        ds = get_col(window_cross, "disagree_score")
        feats["cross_disagree_score_mean"] = safe_mean(ds)
        feats["cross_disagree_score_max"] = safe_max(ds)
        feats["cross_disagree_score_std"] = safe_std(ds)
        feats["cross_disagree_score_trend"] = _linear_slope(ds)
        finite = ds[np.isfinite(ds)]
        feats["cross_disagree_fraction"] = (
            float((finite > _DISAGREE_THRESH).mean()) if finite.size > 0 else float("nan")
        )

    if "acc_ratio" in window_cross.columns:
        feats["cross_acc_energy_ratio"] = safe_mean(get_col(window_cross, "acc_ratio"))

    return feats
