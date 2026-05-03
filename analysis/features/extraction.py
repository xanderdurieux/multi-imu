"""Extraction helpers for extract labelled sliding-window features from section signals."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, welch

from .cross_sensor import cross_sensor_features
from .label_config import LabelConfig, default_label_config
from .labels import (
    label_feature,
    label_feature_set,
    to_activity_label,
    to_binary_label,
    to_coarse_label,
    to_set_based_label,
)
from .orientation_features import orientation_cross_features, orientation_features
from .quality import quality_features
from .spectral import spectral_features
from .stats_helpers import (
    cross_window_valid_ratio,
    get_col,
    signal_stats,
    window_valid_ratio_imu,
)

log = logging.getLogger(__name__)

_SENSOR_SIGNALS = ("acc_norm", "gyro_norm", "acc_vertical", "acc_hf", "jerk_norm", "alpha_norm", "energy_acc")
_SPECTRAL_SIGNALS = ("acc_norm", "gyro_norm", "acc_hf", "gyro_hf")
_SENSOR_PREFIX = {"sporsa": "bike", "arduino": "rider"}
_PEAK_SIGNALS = ("acc_deviation", "acc_hf", "jerk_norm", "gyro_norm", "alpha_norm", "energy_acc")
_BANDS = ((0.5, 2.0), (2.0, 8.0), (8.0, 20.0))

try:
    _np_trapz = np.trapezoid  # NumPy 2.0+
except AttributeError:
    _np_trapz = np.trapz  # type: ignore[attr-defined]


def _peak_features(prefix: str, arr: np.ndarray) -> dict[str, float]:
    """Return peak features."""
    out = {
        f"{prefix}_peak_count": 0.0,
        f"{prefix}_peak_max": float("nan"),
        f"{prefix}_peak_prominence_mean": float("nan"),
        f"{prefix}_peak_prominence_max": float("nan"),
        f"{prefix}_peak_width_mean": float("nan"),
    }
    x = arr[np.isfinite(arr)]
    if len(x) < 4:
        return out
    pidx, props = find_peaks(x, prominence=max(np.std(x) * 0.25, 1e-6))
    out[f"{prefix}_peak_count"] = float(len(pidx))
    if len(pidx) > 0:
        out[f"{prefix}_peak_max"] = float(np.max(x[pidx]))
        prom = props.get("prominences", np.array([], dtype=float))
        widths = peak_widths(x, pidx, rel_height=0.5)[0]
        out[f"{prefix}_peak_prominence_mean"] = float(np.mean(prom)) if prom.size else float("nan")
        out[f"{prefix}_peak_prominence_max"] = float(np.max(prom)) if prom.size else float("nan")
        out[f"{prefix}_peak_width_mean"] = float(np.mean(widths)) if widths.size else float("nan")
    return out


def _bandpower_features(prefix: str, arr: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    """Return bandpower features."""
    out: dict[str, float] = {}
    x = arr[np.isfinite(arr)]
    for lo, hi in _BANDS:
        out[f"{prefix}_bandpower_{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}"] = float("nan")
    if len(x) < 8:
        return out
    freqs, pxx = welch(x, fs=sample_rate_hz, nperseg=min(256, len(x)))
    for lo, hi in _BANDS:
        m = (freqs >= lo) & (freqs < hi)
        key = f"{prefix}_bandpower_{str(lo).replace('.', 'p')}_{str(hi).replace('.', 'p')}"
        out[key] = float(_np_trapz(pxx[m], freqs[m])) if np.any(m) else 0.0
    return out


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
    labels_df: pd.DataFrame | None = None,
    label_config: LabelConfig | None = None,
    quality_metadata: dict[str, float | str | int | bool] | None = None,
) -> dict[str, float | str | int]:
    """Return extract window features."""
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
    if quality_metadata:
        feats.update(quality_metadata)

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
        for signal in _PEAK_SIGNALS:
            feats.update(_peak_features(f"{prefix_root}_{signal}", get_col(sig_df, signal)))

        # Dedicated bandpower aliases for ablation-friendly naming.
        for signal, sprefix in (("acc_deviation", f"{prefix_root}_acc"), ("gyro_norm", f"{prefix_root}_gyro")):
            feats.update(_bandpower_features(sprefix, get_col(sig_df, signal), sample_rate_hz))

    # ------------------------------------------------------------------
    # 5. Orientation features
    # ------------------------------------------------------------------
    feats.update(orientation_features("bike", window_orientation_sporsa))
    feats.update(orientation_features("rider", window_orientation_arduino))
    feats.update(orientation_cross_features(
        window_orientation_sporsa, window_orientation_arduino,
    ))

    # ------------------------------------------------------------------
    # 6. Cross-sensor features
    # ------------------------------------------------------------------
    feats.update(cross_sensor_features(window_cross))

    # ------------------------------------------------------------------
    # 7. Labels - parallel representations of the same annotation.
    #
    #   scenario_labels      pipe-delimited set of all overlapping labels on
    #                        this window (multi-label source-of-truth;
    #                        per-objective resolvers select from it)
    #   scenario_label       priority-collapsed dominant fine label
    #                        (legacy single-label inspection target)
    #   scenario_label_activity / _coarse / _binary
    #                        single-label derived schemes resolved from the
    #                        priority-dominant token
    #   set-based binary schemes (configured under set_based_binary_schemes
    #                        in labels.default.json — riding, cornering,
    #                        head_motion, ...) resolved from the FULL
    #                        overlap set, ignoring priority.
    # ------------------------------------------------------------------
    cfg = label_config or default_label_config()
    label_set = label_feature_set(labels_df, window_start_ms, window_end_ms)
    fine = label_feature(labels_df, window_start_ms, window_end_ms, config=cfg)
    feats["scenario_labels"] = "|".join(label_set) if label_set else "unlabeled"
    feats["scenario_label"] = fine
    feats["scenario_label_activity"] = to_activity_label(fine, config=cfg)
    feats["scenario_label_coarse"] = to_coarse_label(fine, config=cfg)
    feats["scenario_label_binary"] = to_binary_label(fine, config=cfg)
    for scheme in cfg.set_based_scheme_names:
        feats[scheme] = to_set_based_label(scheme, label_set, config=cfg)

    return feats
