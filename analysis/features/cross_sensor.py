"""Cross-sensor features (agreement, disagreement, energy ratio)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.statistics import safe_max, safe_mean, safe_min, safe_std
from .stats_helpers import get_col

# Boundary used by cross_disagree_fraction; values above this count as "disagreeing".
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
        # --- HF coupling (helmet-on-head indicator) ---
        # Rolling acc_hf/gyro_hf correlation is the direct "do both sensors
        # see the same road vibration" signal.  High coupling → helmet is
        # mechanically attached to the rider on the bike.  Low coupling →
        # helmet is decoupled (off head, being handled, bike stationary).
        "cross_acc_hf_correlation_mean": float("nan"),
        "cross_acc_hf_correlation_std": float("nan"),
        "cross_acc_hf_correlation_min": float("nan"),
        "cross_acc_hf_correlation_max": float("nan"),
        "cross_gyro_hf_correlation_mean": float("nan"),
        "cross_gyro_hf_correlation_std": float("nan"),
        # Ratio of HF energies — near 1 when both are shaken by the same road,
        # >>1 when the helmet sensor is being shaken independently.
        "cross_acc_hf_ratio_mean": float("nan"),
        "cross_acc_hf_ratio_std": float("nan"),
        "cross_acc_hf_ratio_log_mean": float("nan"),
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

    if "acc_hf_correlation" in window_cross.columns:
        hf_corr = get_col(window_cross, "acc_hf_correlation")
        feats["cross_acc_hf_correlation_mean"] = safe_mean(hf_corr)
        feats["cross_acc_hf_correlation_std"] = safe_std(hf_corr)
        feats["cross_acc_hf_correlation_min"] = safe_min(hf_corr)
        feats["cross_acc_hf_correlation_max"] = safe_max(hf_corr)

    if "gyro_hf_correlation" in window_cross.columns:
        ghf_corr = get_col(window_cross, "gyro_hf_correlation")
        feats["cross_gyro_hf_correlation_mean"] = safe_mean(ghf_corr)
        feats["cross_gyro_hf_correlation_std"] = safe_std(ghf_corr)

    if "acc_hf_ratio" in window_cross.columns:
        hf_ratio = get_col(window_cross, "acc_hf_ratio")
        feats["cross_acc_hf_ratio_mean"] = safe_mean(hf_ratio)
        feats["cross_acc_hf_ratio_std"] = safe_std(hf_ratio)
        # Log-ratio is symmetric around 0 and compresses the heavy right tail
        # (helmet-off-head can produce ratios in the hundreds); more useful for
        # downstream learners than the raw mean.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(np.where(hf_ratio > 0, hf_ratio, np.nan))
        feats["cross_acc_hf_ratio_log_mean"] = safe_mean(log_ratio)

    return feats
