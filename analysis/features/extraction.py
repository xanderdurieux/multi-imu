"""Windowed feature extraction from dual-IMU calibrated data.

All features are extracted for a single time window. This module is
stateless: every public function takes pre-sliced DataFrames and returns
a flat dict of scalar feature values.

Naming conventions
------------------
- ``bike_``  prefix  → sporsa sensor (frame / bike)
- ``rider_`` prefix  → arduino sensor (rider / body)
- ``cross_`` prefix  → cross-sensor comparison features
- ``events_`` prefix → event-overlap features
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EPSILON = 1e-9


def _safe_mean(arr: np.ndarray) -> float:
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return float("nan")
    return float(np.mean(clean))


def _safe_std(arr: np.ndarray) -> float:
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return float("nan")
    v = np.std(clean)
    return float(v) if np.isfinite(v) else float("nan")


def _safe_min(arr: np.ndarray) -> float:
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmin(arr))


def _safe_max(arr: np.ndarray) -> float:
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _safe_iqr(arr: np.ndarray) -> float:
    clean = arr[np.isfinite(arr)]
    if len(clean) < 2:
        return float("nan")
    q75, q25 = np.percentile(clean, [75, 25])
    return float(q75 - q25)


def _safe_skew(arr: np.ndarray) -> float:
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n < 3:
        return float("nan")
    mu = np.mean(clean)
    sigma = np.std(clean)
    if sigma < _EPSILON:
        return 0.0
    return float(np.mean(((clean - mu) / sigma) ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition, normal → 0)."""
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n < 4:
        return float("nan")
    mu = np.mean(clean)
    sigma = np.std(clean)
    if sigma < _EPSILON:
        return 0.0
    return float(np.mean(((clean - mu) / sigma) ** 4) - 3.0)


def _safe_energy(arr: np.ndarray) -> float:
    """Mean of squared values."""
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return float("nan")
    return float(np.mean(clean ** 2))


def _zero_crossings(arr: np.ndarray) -> int:
    clean = arr[np.isfinite(arr)]
    if len(clean) < 2:
        return 0
    signs = np.sign(clean)
    # Only count actual sign flips (exclude zeros).
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return 0
    return int(np.sum(np.diff(nonzero) != 0))


def _get_col(df: pd.DataFrame, col: str) -> np.ndarray:
    """Return column as float array, or empty array if column missing."""
    if df is None or df.empty or col not in df.columns:
        return np.array([], dtype=float)
    return df[col].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Basic statistical features for one signal array
# ---------------------------------------------------------------------------

def _signal_stats(prefix: str, arr: np.ndarray) -> dict[str, Any]:
    """Return basic stats dict for signal array under ``prefix``."""
    return {
        f"{prefix}_mean": _safe_mean(arr),
        f"{prefix}_std": _safe_std(arr),
        f"{prefix}_min": _safe_min(arr),
        f"{prefix}_max": _safe_max(arr),
        f"{prefix}_iqr": _safe_iqr(arr),
        f"{prefix}_skew": _safe_skew(arr),
        f"{prefix}_kurtosis": _safe_kurtosis(arr),
        f"{prefix}_energy": _safe_energy(arr),
        f"{prefix}_zero_crossings": _zero_crossings(arr),
    }


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

def _spectral_features(
    prefix: str,
    arr: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, Any]:
    """Compute spectral features via rfft with Hanning window."""
    feats: dict[str, Any] = {
        f"{prefix}_spectral_centroid": float("nan"),
        f"{prefix}_spectral_energy_low": float("nan"),
        f"{prefix}_spectral_energy_mid": float("nan"),
        f"{prefix}_spectral_energy_high": float("nan"),
        f"{prefix}_dominant_freq": float("nan"),
    }
    clean = arr[np.isfinite(arr)]
    n = len(clean)
    if n < 4:
        return feats

    window = np.hanning(n)
    windowed = clean * window
    spectrum = np.fft.rfft(windowed)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)

    total_power = power.sum()
    if total_power < _EPSILON:
        return feats

    # Spectral centroid.
    centroid = float(np.sum(freqs * power) / total_power)
    feats[f"{prefix}_spectral_centroid"] = centroid

    # Band energies (sums — not normalised).
    low_mask = freqs < 2.0
    mid_mask = (freqs >= 2.0) & (freqs < 8.0)
    high_mask = freqs >= 8.0

    feats[f"{prefix}_spectral_energy_low"] = float(power[low_mask].sum())
    feats[f"{prefix}_spectral_energy_mid"] = float(power[mid_mask].sum())
    feats[f"{prefix}_spectral_energy_high"] = float(power[high_mask].sum())

    # Dominant frequency.
    dom_idx = int(np.argmax(power))
    feats[f"{prefix}_dominant_freq"] = float(freqs[dom_idx])

    return feats


# ---------------------------------------------------------------------------
# Orientation features
# ---------------------------------------------------------------------------

def _orientation_features(
    sensor_prefix: str,
    window_orient: pd.DataFrame | None,
) -> dict[str, Any]:
    """Extract orientation statistics from Madgwick orientation slice."""
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
        base[f"{sensor_prefix}_pitch_mean"] = _safe_mean(pitch)
        base[f"{sensor_prefix}_pitch_std"] = _safe_std(pitch)
        if len(pitch) >= 2:
            pitch_rate = np.diff(pitch[np.isfinite(pitch)])
            base[f"{sensor_prefix}_pitch_rate_std"] = _safe_std(pitch_rate)

    if "roll_deg" in window_orient.columns:
        roll = window_orient["roll_deg"].to_numpy(dtype=float)
        base[f"{sensor_prefix}_roll_mean"] = _safe_mean(roll)
        base[f"{sensor_prefix}_roll_std"] = _safe_std(roll)
        if len(roll) >= 2:
            roll_rate = np.diff(roll[np.isfinite(roll)])
            base[f"{sensor_prefix}_roll_rate_std"] = _safe_std(roll_rate)

    return base


# ---------------------------------------------------------------------------
# Cross-sensor features
# ---------------------------------------------------------------------------

def _cross_sensor_features(window_cross: pd.DataFrame) -> dict[str, Any]:
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
        feats["cross_acc_correlation_mean"] = _safe_mean(
            _get_col(window_cross, "acc_correlation")
        )
    if "gyro_diff_norm" in window_cross.columns:
        feats["cross_gyro_diff_mean"] = _safe_mean(
            _get_col(window_cross, "gyro_diff_norm")
        )
    if "acc_diff_norm" in window_cross.columns:
        feats["cross_acc_diff_mean"] = _safe_mean(
            _get_col(window_cross, "acc_diff_norm")
        )
    if "disagree_score" in window_cross.columns:
        ds = _get_col(window_cross, "disagree_score")
        feats["cross_disagree_score_mean"] = _safe_mean(ds)
        feats["cross_disagree_score_max"] = _safe_max(ds)
    if "acc_ratio" in window_cross.columns:
        feats["cross_acc_energy_ratio"] = _safe_mean(
            _get_col(window_cross, "acc_ratio")
        )

    return feats


# ---------------------------------------------------------------------------
# Event features
# ---------------------------------------------------------------------------

def _event_features(
    events_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
) -> dict[str, Any]:
    """Count events that overlap the window."""
    feats: dict[str, Any] = {
        "events_bump_count": 0,
        "events_brake_count": 0,
        "events_swerve_count": 0,
        "events_disagree_count": 0,
        "events_max_bump_confidence": 0.0,
        "events_max_swerve_confidence": 0.0,
        "events_any": 0,
    }
    if events_df is None or events_df.empty:
        return feats

    # Filter to events that overlap [window_start_ms, window_end_ms].
    required = {"event_type", "start_ms", "end_ms"}
    if not required.issubset(events_df.columns):
        return feats

    # Overlap condition: event starts before window ends AND event ends after window starts.
    mask = (
        (events_df["start_ms"] < window_end_ms)
        & (events_df["end_ms"] > window_start_ms)
    )
    overlapping = events_df[mask]

    if overlapping.empty:
        return feats

    types = overlapping["event_type"].str.lower() if "event_type" in overlapping.columns else pd.Series(dtype=str)

    feats["events_bump_count"] = int((types == "bump").sum())
    feats["events_brake_count"] = int((types == "brake").sum())
    feats["events_swerve_count"] = int((types == "swerve").sum())
    feats["events_disagree_count"] = int((types == "disagree").sum())
    feats["events_any"] = 1 if len(overlapping) > 0 else 0

    if "confidence" in overlapping.columns:
        bump_mask = types == "bump"
        swerve_mask = types == "swerve"
        if bump_mask.any():
            feats["events_max_bump_confidence"] = float(
                overlapping.loc[bump_mask, "confidence"].max()
            )
        if swerve_mask.any():
            feats["events_max_swerve_confidence"] = float(
                overlapping.loc[swerve_mask, "confidence"].max()
            )

    return feats


# ---------------------------------------------------------------------------
# Label features
# ---------------------------------------------------------------------------

# Priority order for resolving simultaneous overlapping labels.
# Higher index = higher priority (last entry wins when multiple labels overlap).
_LABEL_PRIORITY: list[str] = [
    "calibration_sequence",
    "helmet_move",
    "grounded",
    "riding",
    "riding_standing",
    "forest",
    "uneven_road",
    "head_movement",
    "shoulder_check",
    "accelerating",
    "cornering",
    "braking",
    "sprinting",
    "sprint_standing",
    "hard_braking",
    "swerving",
    "fall",
]

_LABEL_PRIORITY_RANK: dict[str, int] = {lbl: i for i, lbl in enumerate(_LABEL_PRIORITY)}


def _label_feature(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float = 0.5,
) -> str:
    """Return the single highest-priority scenario_label for the window, or 'unlabeled'.

    Labeling strategy:
    1. Compute the containment ratio (overlap / window duration) for every label
       that touches the window.
    2. Keep labels whose containment ratio >= *containment_threshold* (the label
       covers at least half the window). Using window duration ensures long
       scenario labels are always captured.
    3. If no label meets the threshold, fall back to all overlapping labels.
    4. From the candidate labels, return the single highest-priority label
       according to ``_LABEL_PRIORITY``. Unknown labels fall back to the one
       with the greatest absolute overlap.
    """
    if labels_df is None or labels_df.empty:
        return "unlabeled"

    required = {"start_ms", "end_ms"}
    if not required.issubset(labels_df.columns):
        return "unlabeled"

    mask = (
        (labels_df["start_ms"] < window_end_ms)
        & (labels_df["end_ms"] > window_start_ms)
    )
    overlapping = labels_df[mask].copy()
    if overlapping.empty:
        return "unlabeled"

    overlap_ms = (
        overlapping["end_ms"].clip(upper=window_end_ms)
        - overlapping["start_ms"].clip(lower=window_start_ms)
    )
    window_dur_ms = max(window_end_ms - window_start_ms, 1.0)
    containment = overlap_ms / window_dur_ms

    well_contained = overlapping[containment >= containment_threshold]
    if well_contained.empty:
        well_contained = overlapping
        overlap_ms = overlap_ms.loc[well_contained.index]

    def _row_label(row: pd.Series) -> str:
        if "scenario_label" in row.index and pd.notna(row["scenario_label"]) and str(row["scenario_label"]).strip():
            return str(row["scenario_label"])
        if "label" in row.index and pd.notna(row["label"]) and str(row["label"]).strip():
            return str(row["label"])
        return ""

    candidates: list[tuple[str, float]] = []
    for (idx, row), ov in zip(well_contained.iterrows(), overlap_ms.loc[well_contained.index]):
        lbl = _row_label(row)
        if lbl:
            candidates.append((lbl, float(ov)))

    if not candidates:
        return "unlabeled"

    # Pick the highest-priority known label; for unknowns use largest overlap.
    known = [(lbl, ov) for lbl, ov in candidates if lbl in _LABEL_PRIORITY_RANK]
    if known:
        return max(known, key=lambda t: _LABEL_PRIORITY_RANK[t[0]])[0]
    return max(candidates, key=lambda t: t[1])[0]


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def _quality_features(
    valid_ratio_sporsa: float,
    calibration_quality: str,
) -> dict[str, Any]:
    cal_ok = 1.0 if calibration_quality != "poor" else 0.0
    vr = valid_ratio_sporsa if np.isfinite(valid_ratio_sporsa) else 0.0
    score = 0.8 * (1.0 if vr > 0.9 else vr) + 0.2 * cal_ok

    if score >= 0.8:
        label = "good"
        tier = "A"
    elif score >= 0.5:
        label = "marginal"
        tier = "B"
    else:
        label = "poor"
        tier = "C"

    return {
        "overall_quality_score": round(float(score), 4),
        "overall_quality_label": label,
        "quality_tier": tier,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    events_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
) -> dict[str, float | str | int]:
    """Extract all features for one window.

    Parameters
    ----------
    window_sporsa:
        Calibrated sporsa slice for the window.
    window_arduino:
        Calibrated arduino slice for the window.
    window_sporsa_signals:
        Derived signals slice for sporsa.
    window_arduino_signals:
        Derived signals slice for arduino.
    window_cross:
        Cross-sensor derived signals slice.
    window_orientation_sporsa:
        Orientation CSV slice for sporsa (Madgwick), or None.
    window_orientation_arduino:
        Orientation CSV slice for arduino (Madgwick), or None.
    section_id:
        Section folder name used as identifier.
    window_start_ms:
        Window start time in milliseconds.
    window_end_ms:
        Window end time in milliseconds.
    window_idx:
        Zero-based index of this window within the section.
    sample_rate_hz:
        Nominal sampling rate for spectral feature computation.
    calibration_quality:
        Overall calibration quality string from calibration.json.
    sync_confidence:
        Synchronisation confidence (0–1) from calibration.json.
    events_df:
        Full events DataFrame (will be filtered to window).
    labels_df:
        Full labels DataFrame (will be filtered to window).

    Returns
    -------
    dict
        Flat dict of all features for this window.
    """
    feats: dict[str, Any] = {}

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

    # Valid ratio for sporsa: fraction of non-NaN acc_norm in signals.
    sporsa_acc_norm = _get_col(window_sporsa_signals, "acc_norm")
    if len(sporsa_acc_norm) > 0:
        valid_ratio_sporsa = float(np.sum(np.isfinite(sporsa_acc_norm)) / len(sporsa_acc_norm))
    else:
        valid_ratio_sporsa = 0.0
    feats["window_valid_ratio_sporsa"] = valid_ratio_sporsa

    # ------------------------------------------------------------------
    # 2. Quality metadata
    # ------------------------------------------------------------------
    feats["calibration_quality"] = str(calibration_quality)
    feats["sync_confidence"] = float(sync_confidence)
    feats.update(_quality_features(valid_ratio_sporsa, calibration_quality))

    # ------------------------------------------------------------------
    # 3. Basic stats — per sensor, per signal
    # ------------------------------------------------------------------
    signal_dfs = {
        "sporsa": window_sporsa_signals,
        "arduino": window_arduino_signals,
    }
    for sensor, sig_df in signal_dfs.items():
        prefix_root = _SENSOR_PREFIX[sensor]
        for signal in _SENSOR_SIGNALS:
            arr = _get_col(sig_df, signal)
            feats.update(_signal_stats(f"{prefix_root}_{signal}", arr))

    # ------------------------------------------------------------------
    # 4. Spectral features
    # ------------------------------------------------------------------
    for sensor, sig_df in signal_dfs.items():
        prefix_root = _SENSOR_PREFIX[sensor]
        for signal in _SPECTRAL_SIGNALS:
            arr = _get_col(sig_df, signal)
            feats.update(_spectral_features(f"{prefix_root}_{signal}", arr, sample_rate_hz))

    # ------------------------------------------------------------------
    # 5. Orientation features
    # ------------------------------------------------------------------
    feats.update(_orientation_features("bike", window_orientation_sporsa))
    feats.update(_orientation_features("rider", window_orientation_arduino))

    # ------------------------------------------------------------------
    # 6. Cross-sensor features
    # ------------------------------------------------------------------
    feats.update(_cross_sensor_features(window_cross))

    # ------------------------------------------------------------------
    # 7. Event features
    # ------------------------------------------------------------------
    feats.update(_event_features(events_df, window_start_ms, window_end_ms))

    # ------------------------------------------------------------------
    # 8. Label
    # ------------------------------------------------------------------
    feats["scenario_label"] = _label_feature(labels_df, window_start_ms, window_end_ms)

    return feats
