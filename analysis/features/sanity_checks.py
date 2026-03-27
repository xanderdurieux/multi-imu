"""Unit-like sanity checks for degenerate feature windows."""

from __future__ import annotations

import numpy as np

from .extract import _window_sanity_flags
from .families import _lag_seconds, extract_grouped_features
from .signal_stats import safe_ratio


def run_sanity_checks() -> None:
    # Constant signals -> low variance flags.
    acc = np.ones((10, 3), dtype=float)
    gyro = np.ones((10, 3), dtype=float)
    flags = _window_sanity_flags(acc, gyro)
    assert "acc_low_variance" in flags
    assert "gyro_low_variance" in flags

    # Very short windows -> short_window flag.
    flags_short = _window_sanity_flags(np.ones((4, 3)), np.ones((4, 3)))
    assert "short_window" in flags_short

    # Near-zero denominator protected.
    assert np.isnan(safe_ratio(1.0, 0.0))
    assert np.isnan(_lag_seconds(np.zeros(20), np.zeros(20), 0.02))

    # Group feature extraction on degenerate signals should not raise.
    out = extract_grouped_features(
        bike_acc=np.zeros((40, 3)),
        rider_acc=np.zeros((40, 3)),
        bike_acc_norm=np.zeros(40),
        rider_acc_norm=np.zeros(40),
        bike_gyro_norm=np.zeros(40),
        rider_gyro_norm=np.zeros(40),
        bike_pitch=np.zeros(40),
        rider_pitch=np.zeros(40),
        bike_roll=np.zeros(40),
        rider_roll=np.zeros(40),
        dt_s=0.02,
        roll_dt_s=0.02,
        fs_hz=50.0,
        vec_disagreement=0.0,
        axis_energy_ratios=(np.nan, np.nan, np.nan),
    )
    assert "bump_shock_attenuation_ratio" in out
    assert np.isnan(out["bump_shock_attenuation_ratio"])


if __name__ == "__main__":
    run_sanity_checks()
    print("sanity checks passed")
