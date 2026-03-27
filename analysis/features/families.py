"""Physically interpreted feature families for dual-IMU cycling windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal_stats import (
    mean_coherence_band,
    normalized_band_energy,
    safe_ratio,
)


@dataclass(frozen=True)
class FeatureDef:
    """Human-readable feature metadata entry."""

    name: str
    group: str
    hypothesis: str
    description: str


GROUP_EXPLANATIONS: dict[str, str] = {
    "bumps_disturbances": (
        "Road bumps inject short, high-amplitude vertical shocks on the bicycle. "
        "Rider-mounted response should be attenuated and slightly delayed if damping works."
    ),
    "braking_deceleration": (
        "Braking produces strong negative longitudinal acceleration and bike-rider forward pitch coupling."
    ),
    "cornering_swerving": (
        "Cornering and swerving create lateral acceleration, roll activity, and coordinated bike-rider lean."
    ),
    "sprinting_exertion": (
        "Sprinting is periodic and energetic: cadence-like oscillations and elevated angular motion energy."
    ),
    "rider_bicycle_disagreement": (
        "Destabilization appears as disagreement between sensors in timing, shape, and directional energy."
    ),
}


FEATURE_DEFS: list[FeatureDef] = [
    FeatureDef(
        "bump_vertical_peak_ms2",
        "bumps_disturbances",
        "Bump severity",
        "Peak absolute vertical acceleration on bicycle IMU within window.",
    ),
    FeatureDef(
        "bump_shock_attenuation_ratio",
        "bumps_disturbances",
        "Shock attenuation",
        "Bike/rider peak vertical acceleration ratio; >1 suggests attenuation toward rider.",
    ),
    FeatureDef(
        "bump_response_lag_s",
        "bumps_disturbances",
        "Shock transmission lag",
        "Time lag at peak cross-correlation of bike vs rider vertical acceleration.",
    ),
    FeatureDef(
        "brake_longitudinal_decel_peak_ms2",
        "braking_deceleration",
        "Hard braking",
        "Most negative longitudinal acceleration on bicycle IMU.",
    ),
    FeatureDef(
        "brake_pitch_change_deg",
        "braking_deceleration",
        "Forward pitch excursion",
        "Pitch change (end-start) on bicycle orientation track within window.",
    ),
    FeatureDef(
        "brake_pitch_coupling_corr",
        "braking_deceleration",
        "Bike-rider coupling",
        "Correlation between bike and rider pitch traces in the window.",
    ),
    FeatureDef(
        "corner_lateral_energy_ms2_sq",
        "cornering_swerving",
        "Cornering intensity",
        "Sum of squared bicycle lateral acceleration in the window.",
    ),
    FeatureDef(
        "corner_roll_rate_rms_deg_s",
        "cornering_swerving",
        "Lean dynamics",
        "RMS roll rate on bicycle orientation signal.",
    ),
    FeatureDef(
        "corner_roll_coupling_corr",
        "cornering_swerving",
        "Coordinated leaning",
        "Correlation of bike vs rider roll traces.",
    ),
    FeatureDef(
        "sprint_cadence_band_fraction",
        "sprinting_exertion",
        "Cadence-like periodicity",
        "Fraction of acceleration energy in cadence-like 1.2–3.5 Hz band (bicycle IMU).",
    ),
    FeatureDef(
        "sprint_dom_freq_hz",
        "sprinting_exertion",
        "Dominant periodic motion",
        "Dominant frequency of bicycle acceleration norm.",
    ),
    FeatureDef(
        "sprint_gyro_energy_sum",
        "sprinting_exertion",
        "Whole-body exertion",
        "Combined bike+rider gyro norm energy.",
    ),
    FeatureDef(
        "disagree_vec_diff_mean_ms2",
        "rider_bicycle_disagreement",
        "Motion mismatch",
        "Mean vector acceleration disagreement between bike and rider.",
    ),
    FeatureDef(
        "disagree_vertical_coherence",
        "rider_bicycle_disagreement",
        "Frequency-domain mismatch",
        "Vertical-axis coherence in 0.5–10 Hz band between sensors.",
    ),
    FeatureDef(
        "disagree_energy_axis_ratio_var",
        "rider_bicycle_disagreement",
        "Directional mismatch",
        "Variance of axis-wise bike/rider energy ratios (x,y,z).",
    ),
]


def feature_def_by_name() -> dict[str, FeatureDef]:
    return {f.name: f for f in FEATURE_DEFS}


def grouped_feature_definitions() -> dict[str, list[FeatureDef]]:
    out: dict[str, list[FeatureDef]] = {}
    for feat in FEATURE_DEFS:
        out.setdefault(feat.group, []).append(feat)
    return out


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return float("nan")
    if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _lag_seconds(a: np.ndarray, b: np.ndarray, dt_s: float, max_lag_s: float = 0.5) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0)
    b = np.nan_to_num(np.asarray(b, dtype=float), nan=0.0)
    c = np.correlate(a - np.mean(a), b - np.mean(b), mode="full")
    mid = len(c) // 2
    max_lag = int(max_lag_s / max(dt_s, 1e-6))
    lo = max(0, mid - max_lag)
    hi = min(len(c), mid + max_lag + 1)
    i = int(np.argmax(c[lo:hi])) + lo
    return float((i - mid) * dt_s)


def extract_grouped_features(
    *,
    bike_acc: np.ndarray,
    rider_acc: np.ndarray,
    bike_acc_norm: np.ndarray,
    rider_acc_norm: np.ndarray,
    bike_gyro_norm: np.ndarray,
    rider_gyro_norm: np.ndarray,
    bike_pitch: np.ndarray | None,
    rider_pitch: np.ndarray | None,
    bike_roll: np.ndarray | None,
    rider_roll: np.ndarray | None,
    dt_s: float,
    fs_hz: float,
    vec_disagreement: float,
    axis_energy_ratios: tuple[float, float, float],
) -> dict[str, float]:
    """Compute compact feature families tied to physical hypotheses."""
    out: dict[str, float] = {}

    bike_z = bike_acc[:, 2] if bike_acc.shape[1] >= 3 else np.array([], dtype=float)
    rider_z = rider_acc[:, 2] if rider_acc.shape[1] >= 3 else np.array([], dtype=float)
    bike_x = bike_acc[:, 0] if bike_acc.shape[1] >= 1 else np.array([], dtype=float)
    bike_y = bike_acc[:, 1] if bike_acc.shape[1] >= 2 else np.array([], dtype=float)

    # 1) Bumps / disturbances
    z_peak_bike = float(np.nanmax(np.abs(bike_z))) if len(bike_z) else float("nan")
    z_peak_rider = float(np.nanmax(np.abs(rider_z))) if len(rider_z) else float("nan")
    out["bump_vertical_peak_ms2"] = z_peak_bike
    out["bump_shock_attenuation_ratio"] = safe_ratio(z_peak_bike, z_peak_rider)
    out["bump_response_lag_s"] = _lag_seconds(bike_z, rider_z, dt_s)

    # 2) Braking / deceleration
    out["brake_longitudinal_decel_peak_ms2"] = (
        float(np.nanmin(bike_x)) if len(bike_x) else float("nan")
    )
    if bike_pitch is not None and len(bike_pitch) >= 2:
        out["brake_pitch_change_deg"] = float(bike_pitch[-1] - bike_pitch[0])
    else:
        out["brake_pitch_change_deg"] = float("nan")
    out["brake_pitch_coupling_corr"] = (
        _corr_safe(bike_pitch, rider_pitch)
        if bike_pitch is not None and rider_pitch is not None
        else float("nan")
    )

    # 3) Cornering / swerving
    out["corner_lateral_energy_ms2_sq"] = float(np.nansum(bike_y * bike_y)) if len(bike_y) else float("nan")
    if bike_roll is not None and len(bike_roll) >= 3:
        roll_rate = np.diff(bike_roll) / max(dt_s, 1e-6)
        out["corner_roll_rate_rms_deg_s"] = float(np.sqrt(np.nanmean(roll_rate * roll_rate)))
    else:
        out["corner_roll_rate_rms_deg_s"] = float("nan")
    out["corner_roll_coupling_corr"] = (
        _corr_safe(bike_roll, rider_roll)
        if bike_roll is not None and rider_roll is not None
        else float("nan")
    )

    # 4) Sprinting / exertion
    out["sprint_cadence_band_fraction"] = normalized_band_energy(
        bike_acc_norm, fs_hz, (1.2, 3.5)
    )
    out["sprint_dom_freq_hz"] = float("nan")
    if len(bike_acc_norm) >= 16:
        from .signal_stats import dominant_frequency_hz

        out["sprint_dom_freq_hz"] = dominant_frequency_hz(bike_acc_norm, fs_hz)
    out["sprint_gyro_energy_sum"] = float(
        np.nansum(bike_gyro_norm * bike_gyro_norm) + np.nansum(rider_gyro_norm * rider_gyro_norm)
    )

    # 5) Rider-bike disagreement
    out["disagree_vec_diff_mean_ms2"] = vec_disagreement
    out["disagree_vertical_coherence"] = mean_coherence_band(bike_z, rider_z, fs_hz=fs_hz, f_max_hz=10.0)
    ratios = np.array(axis_energy_ratios, dtype=float)
    out["disagree_energy_axis_ratio_var"] = float(np.nanvar(ratios)) if np.any(np.isfinite(ratios)) else float("nan")

    return out
