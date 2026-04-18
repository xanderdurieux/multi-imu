"""Windowed feature extraction dispatcher for dual-IMU calibrated data.

All features are extracted for a single time window. Every public function
takes pre-sliced DataFrames and returns a flat dict of scalar feature values.

Feature families (split into sibling modules):

- ``stats_helpers`` - basic window stats (mean, std, iqr, skew, ...)
- ``spectral``      - FFT-based centroid / band energies / dominant freq
- ``orientation_features`` - pitch/roll mean/std + rate std
- ``cross_sensor``  - inter-sensor agreement, disagreement, energy ratio
- ``event_features`` - per-window event overlap counts and confidences
- ``labels``        - fine/coarse/binary label taxonomy + assignment
- ``quality``       - multi-IMU quality scoring

Naming conventions for feature columns:

- ``bike_``  prefix -> sporsa sensor (frame / bike)
- ``rider_`` prefix -> arduino sensor (rider / body)
- ``cross_`` prefix -> cross-sensor comparison features
- ``events_`` prefix -> event-overlap features
"""

from __future__ import annotations

import logging

import pandas as pd

from .cross_sensor import cross_sensor_features
from .event_features import event_features
from .labels import label_feature, to_binary_label, to_coarse_label
from .orientation_features import orientation_features
from .quality import quality_features
from .spectral import spectral_features
from .stats_helpers import (
    cross_window_valid_ratio,
    get_col,
    signal_stats,
    window_valid_ratio_imu,
)

log = logging.getLogger(__name__)

_SENSOR_SIGNALS = ("acc_norm", "gyro_norm", "acc_vertical", "acc_hf", "jerk_norm")
_SPECTRAL_SIGNALS = ("acc_norm", "gyro_norm")
_SENSOR_PREFIX = {"sporsa": "bike", "arduino": "rider"}


def extract_window_features(
    window_sporsa: pd.DataFrame,
    window_arduino: pd.DataFrame,
    window_sporsa_signals: pd.DataFrame,
    window_arduino_signals: pd.DataFrame,
    window_cross: pd.DataFrame,
    window_orientation_sporsa: pd.DataFrame | None,
    window_orientation_arduino: pd.DataFrame | None,
    *,
    section_id: str,
    window_start_ms: float,
    window_end_ms: float,
    window_idx: int,
    sample_rate_hz: float = 100.0,
    calibration_quality: str = "good",
    sync_confidence: float = 1.0,
    yaw_conf_sporsa: float = 1.0,
    yaw_conf_arduino: float = 1.0,
    events_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
) -> dict[str, float | str | int]:
    """Extract all features for one window and return a flat dict.

    See submodule docstrings for the feature definitions themselves. This
    function wires window slices to each family and flattens the output.
    """
    feats: dict[str, float | str | int] = {}

    # ------------------------------------------------------------------
    # 1. Window metadata
    # ------------------------------------------------------------------
    duration_s = (window_end_ms - window_start_ms) / 1000.0
    feats["section_id"] = section_id
    feats["window_idx"] = window_idx
    feats["window_start_ms"] = float(window_start_ms)
    feats["window_end_ms"] = float(window_end_ms)
    feats["window_duration_s"] = float(duration_s)

    n_sporsa = len(window_sporsa) if window_sporsa is not None else 0
    n_arduino = len(window_arduino) if window_arduino is not None else 0
    feats["window_n_samples_sporsa"] = n_sporsa
    feats["window_n_samples_arduino"] = n_arduino

    cross_valid_ratio = cross_window_valid_ratio(window_cross)

    valid_ratio_sporsa = window_valid_ratio_imu(window_sporsa_signals, window_sporsa)
    valid_ratio_arduino = window_valid_ratio_imu(window_arduino_signals, window_arduino)
    feats["window_valid_ratio_sporsa"] = valid_ratio_sporsa
    feats["window_valid_ratio_arduino"] = valid_ratio_arduino

    # ------------------------------------------------------------------
    # 2. Quality metadata
    # ------------------------------------------------------------------
    feats["calibration_quality"] = str(calibration_quality)
    feats["sync_confidence"] = float(sync_confidence)
    feats.update(
        quality_features(
            valid_ratio_sporsa,
            valid_ratio_arduino,
            calibration_quality,
            yaw_conf_sporsa,
            yaw_conf_arduino,
            cross_valid_ratio,
        )
    )

    # ------------------------------------------------------------------
    # 3. Basic stats - per sensor, per signal
    # ------------------------------------------------------------------
    signal_dfs = {
        "sporsa": window_sporsa_signals,
        "arduino": window_arduino_signals,
    }
    for sensor, sig_df in signal_dfs.items():
        prefix_root = _SENSOR_PREFIX[sensor]
        for signal in _SENSOR_SIGNALS:
            arr = get_col(sig_df, signal)
            feats.update(signal_stats(f"{prefix_root}_{signal}", arr))

    # ------------------------------------------------------------------
    # 4. Spectral features
    # ------------------------------------------------------------------
    for sensor, sig_df in signal_dfs.items():
        prefix_root = _SENSOR_PREFIX[sensor]
        for signal in _SPECTRAL_SIGNALS:
            arr = get_col(sig_df, signal)
            feats.update(spectral_features(f"{prefix_root}_{signal}", arr, sample_rate_hz))

    # ------------------------------------------------------------------
    # 5. Orientation features
    # ------------------------------------------------------------------
    feats.update(orientation_features("bike", window_orientation_sporsa))
    feats.update(orientation_features("rider", window_orientation_arduino))

    # ------------------------------------------------------------------
    # 6. Cross-sensor features
    # ------------------------------------------------------------------
    feats.update(cross_sensor_features(window_cross))

    # ------------------------------------------------------------------
    # 7. Event features
    # ------------------------------------------------------------------
    feats.update(event_features(events_df, window_start_ms, window_end_ms))

    # ------------------------------------------------------------------
    # 8. Labels - three parallel representations of the same annotation.
    #
    #   scenario_label        fine-grained (17 classes) - for inspection / audit
    #   scenario_label_coarse 4-class taxonomy           - recommended main target
    #   scenario_label_binary incident / normal          - binary safety target
    # ------------------------------------------------------------------
    fine = label_feature(labels_df, window_start_ms, window_end_ms)
    feats["scenario_label"] = fine
    feats["scenario_label_coarse"] = to_coarse_label(fine)
    feats["scenario_label_binary"] = to_binary_label(fine)

    return feats
