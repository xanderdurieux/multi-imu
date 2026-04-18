"""Window-level quality scoring.

Combines per-sensor validity, alignment confidence, calibration quality,
and cross-sensor finite-rate into a single numeric score + tier label.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Calibration quality -> numeric score. Three distinct levels so 'marginal'
# is meaningfully penalised relative to 'good'.
CAL_QUALITY_SCORES: dict[str, float] = {
    "good": 1.0,
    "marginal": 0.65,
    "poor": 0.0,
}


def quality_features(
    valid_ratio_sporsa: float,
    valid_ratio_arduino: float,
    calibration_quality: str,
    yaw_conf_sporsa: float,
    yaw_conf_arduino: float,
    cross_valid_ratio: float,
) -> dict[str, Any]:
    """Multi-IMU quality score with a dedicated cross-sensor usability term.

    Dual-stream (both IMUs have samples in the window): weighted sum of bike
    validity, rider validity, alignment, calibration, and cross-row finite rate.

    Single-stream: cross term omitted; ``quality_cross`` stored as 1.0 (N/A).
    """
    q_bike = float(np.clip(valid_ratio_sporsa, 0.0, 1.0)) if np.isfinite(valid_ratio_sporsa) else 0.0
    q_rider = float(np.clip(valid_ratio_arduino, 0.0, 1.0)) if np.isfinite(valid_ratio_arduino) else 0.0
    q_cal = CAL_QUALITY_SCORES.get(calibration_quality, 0.5)

    yc_s = float(np.clip(yaw_conf_sporsa, 0.0, 1.0)) if np.isfinite(yaw_conf_sporsa) else 0.5
    yc_a = float(np.clip(yaw_conf_arduino, 0.0, 1.0)) if np.isfinite(yaw_conf_arduino) else 0.5

    q_align = min(yc_s, yc_a) if q_rider > 0.0 else yc_s

    dual_stream = q_bike > 1e-6 and q_rider > 1e-6
    q_cross = float(np.clip(cross_valid_ratio, 0.0, 1.0)) if dual_stream else 1.0

    if dual_stream:
        score = (
            0.22 * q_bike
            + 0.22 * q_rider
            + 0.18 * q_align
            + 0.18 * q_cal
            + 0.20 * q_cross
        )
        q_cross_out = q_cross
    elif q_bike > 1e-6:
        score = 0.45 * q_bike + 0.30 * q_align + 0.25 * q_cal
        q_cross_out = 1.0
    elif q_rider > 1e-6:
        score = 0.45 * q_rider + 0.30 * q_align + 0.25 * q_cal
        q_cross_out = 1.0
    else:
        score = 0.0
        q_cross_out = 1.0

    if score >= 0.72:
        label, tier = "good", "A"
    elif score >= 0.42:
        label, tier = "marginal", "B"
    else:
        label, tier = "poor", "C"

    return {
        "quality_bike": round(q_bike, 4),
        "quality_rider": round(q_rider, 4),
        "quality_alignment": round(q_align, 4),
        "quality_calibration": round(q_cal, 4),
        "quality_cross": round(q_cross_out, 4),
        "overall_quality_score": round(score, 4),
        "overall_quality_label": label,
        "quality_tier": tier,
    }
