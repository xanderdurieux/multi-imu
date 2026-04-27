"""Orientation-based features (pitch/roll/yaw statistics, cross-sensor divergence)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common.statistics import safe_iqr, safe_max, safe_mean, safe_skew, safe_std

_PITCH_ROLL_KEYS = ("pitch_mean", "pitch_std", "roll_mean", "roll_std",
                    "pitch_rate_std", "roll_rate_std",
                    "pitch_range", "roll_range",
                    "yaw_rate_std", "yaw_rate_max_abs",
                    # Extra yaw-rate stats targeted at brief, asymmetric head
                    # turns (shoulder check, head_movement).  std + max alone
                    # under-represent these because they are dwarfed by the
                    # within-window noise floor unless the turn is sustained.
                    "yaw_rate_mean_abs", "yaw_rate_p95_abs",
                    "yaw_rate_range", "yaw_rate_skew", "yaw_rate_iqr")


def _finite(arr: np.ndarray) -> np.ndarray:
    """Return the finite subset of *arr*."""
    return arr[np.isfinite(arr)]


def _angular_diff_deg(angles_deg: np.ndarray) -> np.ndarray:
    """Discrete angular difference wrapped to [-180, 180]°.

    Prevents yaw wraparound from producing spurious ~360° spikes.
    """
    diff = np.diff(angles_deg)
    return (diff + 180.0) % 360.0 - 180.0


def _peak_to_peak(arr: np.ndarray) -> float:
    """Peak-to-peak range across finite samples (nan if <2 finite)."""
    finite = _finite(arr)
    if finite.size < 2:
        return float("nan")
    return float(np.max(finite) - np.min(finite))


def orientation_features(
    sensor_prefix: str,
    window_orient: pd.DataFrame | None,
) -> dict[str, Any]:
    """Extract per-sensor orientation stats.

    Includes pitch/roll mean/std, pitch/roll rate std, pitch/roll peak-to-peak
    range, and yaw rate stats (std and max magnitude).  Yaw itself has an
    arbitrary frame origin so mean/std aren't meaningful; yaw *rate* is.
    """
    base = {f"{sensor_prefix}_{k}": float("nan") for k in _PITCH_ROLL_KEYS}
    if window_orient is None or window_orient.empty:
        return base

    if "pitch_deg" in window_orient.columns:
        pitch = window_orient["pitch_deg"].to_numpy(dtype=float)
        base[f"{sensor_prefix}_pitch_mean"] = safe_mean(pitch)
        base[f"{sensor_prefix}_pitch_std"] = safe_std(pitch)
        base[f"{sensor_prefix}_pitch_range"] = _peak_to_peak(pitch)
        finite = _finite(pitch)
        if finite.size >= 2:
            base[f"{sensor_prefix}_pitch_rate_std"] = safe_std(np.diff(finite))

    if "roll_deg" in window_orient.columns:
        roll = window_orient["roll_deg"].to_numpy(dtype=float)
        base[f"{sensor_prefix}_roll_mean"] = safe_mean(roll)
        base[f"{sensor_prefix}_roll_std"] = safe_std(roll)
        base[f"{sensor_prefix}_roll_range"] = _peak_to_peak(roll)
        finite = _finite(roll)
        if finite.size >= 2:
            base[f"{sensor_prefix}_roll_rate_std"] = safe_std(np.diff(finite))

    if "yaw_deg" in window_orient.columns:
        yaw = window_orient["yaw_deg"].to_numpy(dtype=float)
        finite = _finite(yaw)
        if finite.size >= 2:
            yaw_rate = _angular_diff_deg(finite)
            abs_rate = np.abs(yaw_rate)
            base[f"{sensor_prefix}_yaw_rate_std"] = safe_std(yaw_rate)
            base[f"{sensor_prefix}_yaw_rate_max_abs"] = safe_max(abs_rate)
            base[f"{sensor_prefix}_yaw_rate_mean_abs"] = safe_mean(abs_rate)
            # p95 catches brief peaks (shoulder-check duration < window) that
            # mean/max blur or saturate.
            base[f"{sensor_prefix}_yaw_rate_p95_abs"] = float(np.percentile(abs_rate, 95))
            base[f"{sensor_prefix}_yaw_rate_range"] = float(np.max(yaw_rate) - np.min(yaw_rate))
            # Head turns are typically one-sided within a window (look left OR
            # right), so signed-rate skew is informative; a steady cornering
            # rate is symmetric around its mean and produces low skew.
            base[f"{sensor_prefix}_yaw_rate_skew"] = safe_skew(yaw_rate)
            base[f"{sensor_prefix}_yaw_rate_iqr"] = safe_iqr(yaw_rate)

    return base


# ---------------------------------------------------------------------------
# Cross-sensor orientation divergence
# ---------------------------------------------------------------------------

_CROSS_ORIENT_KEYS = (
    "cross_pitch_diff_mean",
    "cross_pitch_diff_std",
    "cross_pitch_corr",
    "cross_roll_diff_mean",
    "cross_roll_diff_std",
    "cross_roll_corr",
    "cross_yaw_rate_diff_std",
    "cross_yaw_rate_diff_max_abs",
    # Head-turn-specific yaw decoupling.  When the rider checks over their
    # shoulder, the helmet yaws while the bike heading is steady, so
    # |rider_rate - bike_rate| spikes briefly.  These complement the
    # std/max stats with a robust mean and a normalised decoupling ratio.
    "cross_yaw_rate_diff_mean_abs",
    "cross_yaw_rate_diff_p95_abs",
    "cross_yaw_rate_decoupling_ratio",
)


def _interp_onto(
    t_target: np.ndarray, t_src: np.ndarray, v_src: np.ndarray
) -> np.ndarray:
    """Linear interpolation of *v_src* (at *t_src*) onto *t_target*.

    Samples outside the source range or where the source is non-finite
    become NaN; the source is sorted into ascending time and its finite
    subset is used for np.interp.
    """
    order = np.argsort(t_src)
    t_src = t_src[order]
    v_src = v_src[order]
    finite = np.isfinite(t_src) & np.isfinite(v_src)
    if finite.sum() < 2:
        return np.full_like(t_target, np.nan, dtype=float)
    return np.interp(t_target, t_src[finite], v_src[finite],
                     left=np.nan, right=np.nan)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation on the finite, non-constant subset."""
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a_f, b_f = a[mask], b[mask]
    if np.std(a_f) == 0 or np.std(b_f) == 0:
        return float("nan")
    return float(np.corrcoef(a_f, b_f)[0, 1])


def orientation_cross_features(
    window_bike: pd.DataFrame | None,
    window_rider: pd.DataFrame | None,
) -> dict[str, Any]:
    """Cross-sensor orientation divergence features.

    Captures helmet-vs-bike behaviour that unimodal features cannot:

    * ``cross_pitch_diff_*``     rider forward/back lean independent of bike
    * ``cross_roll_diff_*``      rider body lean independent of bike roll
    * ``cross_pitch_corr``       lock-step seated vs. standing pedalling
    * ``cross_roll_corr``        matched-lean cornering vs. swerve/countersteer
    * ``cross_yaw_rate_diff_*``  helmet turning while frame holds straight
                                 (shoulder check, head scanning)

    Rider orientation is resampled onto bike timestamps within the window so
    per-sample differences/correlations are aligned.
    """
    base = {k: float("nan") for k in _CROSS_ORIENT_KEYS}
    if (window_bike is None or window_bike.empty
            or window_rider is None or window_rider.empty):
        return base
    if "timestamp" not in window_bike.columns or "timestamp" not in window_rider.columns:
        return base

    t_bike = window_bike["timestamp"].to_numpy(dtype=float)
    t_rider = window_rider["timestamp"].to_numpy(dtype=float)
    if t_bike.size < 2 or t_rider.size < 2:
        return base

    for angle in ("pitch", "roll"):
        col = f"{angle}_deg"
        if col not in window_bike.columns or col not in window_rider.columns:
            continue
        b = window_bike[col].to_numpy(dtype=float)
        r = _interp_onto(t_bike, t_rider, window_rider[col].to_numpy(dtype=float))
        diff = b - r
        base[f"cross_{angle}_diff_mean"] = safe_mean(diff)
        base[f"cross_{angle}_diff_std"] = safe_std(diff)
        base[f"cross_{angle}_corr"] = _corr(b, r)

    if "yaw_deg" in window_bike.columns and "yaw_deg" in window_rider.columns:
        b_yaw = window_bike["yaw_deg"].to_numpy(dtype=float)
        r_yaw = _interp_onto(t_bike, t_rider,
                             window_rider["yaw_deg"].to_numpy(dtype=float))
        b_rate = _angular_diff_deg(b_yaw)
        r_rate = _angular_diff_deg(r_yaw)
        diff_rate = b_rate - r_rate
        base["cross_yaw_rate_diff_std"] = safe_std(diff_rate)
        finite = _finite(diff_rate)
        if finite.size > 0:
            abs_diff = np.abs(finite)
            base["cross_yaw_rate_diff_max_abs"] = float(np.max(abs_diff))
            base["cross_yaw_rate_diff_mean_abs"] = float(np.mean(abs_diff))
            base["cross_yaw_rate_diff_p95_abs"] = float(np.percentile(abs_diff, 95))

        # Decoupling ratio: how much rider yaw is *not* explained by bike yaw.
        # During cornering both rotate together (low ratio); during a shoulder
        # check the bike is steady so the diff is dominated by rider motion,
        # pushing the ratio close to 1.
        b_rate_finite = _finite(b_rate)
        r_rate_finite = _finite(r_rate)
        if (
            finite.size > 0
            and b_rate_finite.size > 0
            and r_rate_finite.size > 0
        ):
            diff_std = safe_std(diff_rate)
            r_std = safe_std(r_rate)
            if r_std > 1e-6:
                base["cross_yaw_rate_decoupling_ratio"] = float(diff_std / r_std)

    return base
