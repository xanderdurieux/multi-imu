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

from common.signals import acc_norm_from_imu_df

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


def _window_valid_ratio_imu(
    window_signals: pd.DataFrame | None,
    window_calibrated: pd.DataFrame | None,
    *,
    signal_col: str = "acc_norm",
) -> float:
    """Fraction of samples with finite acceleration norm in the window.

    Uses derived ``signal_col`` when present; if that slice is empty or
    entirely non-finite, falls back to ``|acc|`` from calibrated ax/ay/az.

    Without this fallback, timestamps misaligned between calibrated and derived
    CSVs (or a missing derived slice) incorrectly yield *zero* validity even
    when the calibrated stream has hundreds of good samples — which was
    collapsing almost all windows to ``overall_quality_label='poor'``.
    """
    arr = _get_col(window_signals, signal_col)
    if len(arr) > 0:
        r = float(np.sum(np.isfinite(arr)) / len(arr))
        if r > 0.0:
            return float(np.clip(r, 0.0, 1.0))

    if (
        window_calibrated is not None
        and not window_calibrated.empty
        and all(c in window_calibrated.columns for c in ("ax", "ay", "az"))
    ):
        try:
            norms = acc_norm_from_imu_df(window_calibrated)
            if len(norms) == 0:
                return 0.0
            return float(np.clip(np.sum(np.isfinite(norms)) / len(norms), 0.0, 1.0))
        except Exception:
            return 0.0
    return 0.0


def _cross_window_valid_ratio(window_cross: pd.DataFrame | None) -> float:
    """Mean finite-data rate across core cross-sensor columns (0–1).

    When both IMUs contribute to fusion, cross features should have usable
    samples in the same window; pervasive NaNs indicate failed alignment or
    missing overlap for that interval.
    """
    if window_cross is None or window_cross.empty:
        return 0.0
    cols = [
        c
        for c in (
            "acc_correlation",
            "gyro_diff_norm",
            "acc_diff_norm",
            "disagree_score",
            "acc_ratio",
        )
        if c in window_cross.columns
    ]
    if not cols:
        return 1.0
    arr = window_cross[cols].to_numpy(dtype=float)
    return float(np.mean(np.isfinite(arr)))


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

# Priority order for resolving simultaneous overlapping fine-grained labels.
# Higher index = higher priority. This is used *only* to break ties when two
# annotations cover the same window equally; it does NOT define the class
# hierarchy used in training (see _COARSE_MAP for that).
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

# ---------------------------------------------------------------------------
# Coarse 4-class taxonomy for the main thesis experiment.
#
# Trade-offs vs. using raw fine-grained labels:
#   Fine-grained (17 classes)
#     + Maximum semantic granularity.
#     - Many classes have very few windows (fall, sprint_standing), making
#       per-class F1 unreliable and confusion matrices noisy.
#     - Pipe-separated annotations ("riding|cornering") produce composite
#       strings that are not in any class schema.
#   Coarse (4 classes)  ← default for main experiment
#     + Sufficient samples per class for stable cross-validated metrics.
#     + Semantically interpretable (non_riding / steady / active / incident).
#     - Loses intra-class distinctions (cornering vs. braking both → active).
#   Binary (incident / normal)
#     + Maximum class balance and clearest safety interpretation.
#     + Directly answers "can the system detect safety-critical events?".
#     - "non_riding" windows must still be filtered at the dataset level.
#     - Loses distinctions between riding modes.
# ---------------------------------------------------------------------------
_COARSE_MAP: dict[str, str] = {
    # Windows with no meaningful riding signal; typically filtered before training.
    "calibration_sequence": "non_riding",
    "helmet_move": "non_riding",
    "grounded": "non_riding",
    # Low-demand, steady-state riding.
    "riding": "steady_riding",
    "riding_standing": "steady_riding",
    "forest": "steady_riding",
    "uneven_road": "steady_riding",
    # Moderate-demand intentional maneuvers.
    "head_movement": "active_riding",
    "shoulder_check": "active_riding",
    "accelerating": "active_riding",
    "cornering": "active_riding",
    "braking": "active_riding",
    "sprinting": "active_riding",
    "sprint_standing": "active_riding",
    # Safety-critical / high-consequence events.
    "hard_braking": "incident",
    "swerving": "incident",
    "fall": "incident",
}

_INCIDENT_LABELS: frozenset[str] = frozenset({"hard_braking", "swerving", "fall"})
_NON_RIDING_LABELS: frozenset[str] = frozenset({"calibration_sequence", "helmet_move", "grounded"})


def _label_feature(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float = 0.5,
) -> str:
    """Return a single fine-grained label for the window (no multi-label strings).

    Labeling strategy (thesis default)
    ----------------------------------
    1. Find all annotation rows overlapping [window_start_ms, window_end_ms].
    2. Compute containment = overlap_ms / window_duration_ms; keep rows with
       containment >= *containment_threshold*, else fall back to all overlaps.
    3. **Pipe-split** compound strings (``"riding|cornering"``) into tokens;
       each token inherits its source row's overlap_ms.
    4. Choose the token with **largest overlap_ms**; break ties with
       ``_LABEL_PRIORITY_RANK`` (higher index wins).  Unknown tokens use rank -1.

    **Why overlap-first:** pure priority ranking can assign a rare short tag
    over a dominant long ``riding`` span. Overlap-first matches the window's
    primary semantic content; priority resolves genuine ties (e.g. equal
    partial overlaps). ``scenario_label_coarse`` / ``scenario_label_binary``
    remain available for aggregated experiments.
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

    def _row_label(row: pd.Series) -> str:
        if "scenario_label" in row.index and pd.notna(row["scenario_label"]) and str(row["scenario_label"]).strip():
            return str(row["scenario_label"])
        if "label" in row.index and pd.notna(row["label"]) and str(row["label"]).strip():
            return str(row["label"])
        return ""

    # Split pipe-separated compound annotations into individual tokens.
    # Each token inherits the full overlap_ms of its source row so that
    # priority ranking is not distorted by partial credit.
    candidates: list[tuple[str, float]] = []
    for idx, row in well_contained.iterrows():
        ov = float(overlap_ms.loc[idx])
        raw = _row_label(row)
        if not raw:
            continue
        for token in raw.split("|"):
            token = token.strip()
            if token:
                candidates.append((token, ov))

    if not candidates:
        return "unlabeled"

    def _rank(lbl: str) -> int:
        return _LABEL_PRIORITY_RANK.get(lbl, -1)

    return max(candidates, key=lambda t: (t[1], _rank(t[0])))[0]


def _to_coarse_label(fine_label: str) -> str:
    """Map a fine-grained label to one of four coarse thesis classes.

    Returns ``'unlabeled'`` for unlabeled windows and ``'unknown'`` for any
    label not in ``_COARSE_MAP`` (e.g. future annotations).
    """
    if fine_label == "unlabeled":
        return "unlabeled"
    return _COARSE_MAP.get(fine_label, "unknown")


def _to_binary_label(fine_label: str) -> str:
    """Map a fine-grained label to a binary incident / normal target.

    ``non_riding`` windows are kept as a distinct value so that the caller
    (or the dataset filter in ``run_evaluation``) can exclude them explicitly
    rather than silently collapsing them into 'normal'.

    Classes
    -------
    incident    hard_braking, swerving, fall
    normal      all riding labels that are not safety-critical
    non_riding  calibration_sequence, helmet_move, grounded
    unlabeled   no annotation covers this window
    """
    if fine_label == "unlabeled":
        return "unlabeled"
    if fine_label in _NON_RIDING_LABELS:
        return "non_riding"
    if fine_label in _INCIDENT_LABELS:
        return "incident"
    return "normal"


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

# Calibration quality → numeric score.
# Three distinct levels — "marginal" is meaningfully penalised relative to
# "good" rather than being collapsed into the same value (old binary scheme).
_CAL_QUALITY_SCORES: dict[str, float] = {
    "good": 1.0,
    "marginal": 0.65,
    "poor": 0.0,
}

def _quality_features(
    valid_ratio_sporsa: float,
    valid_ratio_arduino: float,
    calibration_quality: str,
    yaw_conf_sporsa: float,
    yaw_conf_arduino: float,
    cross_valid_ratio: float,
) -> dict[str, Any]:
    """Multi-IMU quality score with a dedicated cross-sensor data-usability term.

    **Dual-stream** (bike and rider both have samples in the window): weighted
    sum of bike validity, rider validity, alignment, calibration, and cross-row
    finite rate (see module docstring in ``extract_window_features``).

    **Single-stream**: cross term omitted; ``quality_cross`` stored as 1.0 (N/A).
    """
    q_bike = float(np.clip(valid_ratio_sporsa, 0.0, 1.0)) if np.isfinite(valid_ratio_sporsa) else 0.0
    q_rider = float(np.clip(valid_ratio_arduino, 0.0, 1.0)) if np.isfinite(valid_ratio_arduino) else 0.0
    q_cal = _CAL_QUALITY_SCORES.get(calibration_quality, 0.5)

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
    yaw_conf_sporsa: float = 1.0,
    yaw_conf_arduino: float = 1.0,
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
        Sporsa yaw confidence (0–1) stored for backward compatibility.
    yaw_conf_sporsa:
        Sporsa orientation alignment confidence from
        ``calibration.json → alignment.sporsa.yaw_confidence``.
        Defaults to 1.0 (maximum) so existing call sites without this
        argument are not penalised.
    yaw_conf_arduino:
        Arduino orientation alignment confidence from
        ``calibration.json → alignment.arduino.yaw_confidence``.
        Defaults to 1.0.  Together with ``yaw_conf_sporsa`` this enters the
        ``q_align`` component of the quality score as
        ``min(yaw_conf_sporsa, yaw_conf_arduino)`` when the rider sensor is
        present.
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

    cross_valid_ratio = _cross_window_valid_ratio(window_cross)

    valid_ratio_sporsa = _window_valid_ratio_imu(
        window_sporsa_signals,
        window_sporsa,
    )
    feats["window_valid_ratio_sporsa"] = valid_ratio_sporsa

    valid_ratio_arduino = _window_valid_ratio_imu(
        window_arduino_signals,
        window_arduino,
    )
    feats["window_valid_ratio_arduino"] = valid_ratio_arduino

    # ------------------------------------------------------------------
    # 2. Quality metadata
    # ------------------------------------------------------------------
    feats["calibration_quality"] = str(calibration_quality)
    feats["sync_confidence"] = float(sync_confidence)
    feats.update(
        _quality_features(
            valid_ratio_sporsa,
            valid_ratio_arduino,
            calibration_quality,
            yaw_conf_sporsa,
            yaw_conf_arduino,
            cross_valid_ratio,
        )
    )

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
    # 8. Labels — three parallel representations of the same annotation.
    #
    #   scenario_label        fine-grained (17 classes) — for inspection / audit
    #   scenario_label_coarse 4-class taxonomy          — recommended main target
    #   scenario_label_binary incident / normal          — binary safety target
    #
    # Use label_col in run_evaluation to select which column to train on.
    # ------------------------------------------------------------------
    fine = _label_feature(labels_df, window_start_ms, window_end_ms)
    feats["scenario_label"] = fine
    feats["scenario_label_coarse"] = _to_coarse_label(fine)
    feats["scenario_label_binary"] = _to_binary_label(fine)

    return feats
