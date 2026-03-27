"""Per-section feature extraction from calibrated and orientation data."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from common import load_dataframe

from .schema import write_feature_schema_json
from .families import extract_grouped_features
from .signal_stats import (
    band_energy_ratio,
    crest_factor,
    dominant_frequency_hz,
    mean_coherence_band,
    peak_time_difference_s,
    safe_ratio,
    signal_entropy,
    vec_disagreement_ms2,
    zero_crossing_rate,
)

log = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.81
ACC_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]

PER_SENSOR_FEATURES = [
    "acc_norm_mean",
    "acc_norm_max",
    "acc_norm_energy",
    "jerk_norm_max",
    "gyro_norm_max",
    "gyro_energy",
    "vertical_acc_mean",
    "vertical_acc_std",
]

PER_SENSOR_EXTRA = [
    "acc_norm_rms",
    "acc_norm_std",
    "acc_norm_ptp",
    "gyro_norm_rms",
    "gyro_norm_std",
    "gyro_norm_ptp",
    "dom_freq_acc_norm_hz",
    "acc_band_low_energy",
    "acc_band_high_energy",
    "acc_band_high_fraction",
    "crest_factor_acc_norm",
    "entropy_acc_norm",
    "zcr_acc_norm",
    "longitudinal_acc_std",
    "lateral_acc_std",
]

ORIENT_FEATURE_SUFFIXES = [
    "pitch_mean_deg",
    "pitch_std_deg",
    "roll_mean_deg",
    "roll_std_deg",
    "pitch_rate_mean_deg_s",
    "roll_rate_mean_deg_s",
]

CROSS_SENSOR_FEATURES = [
    "acc_norm_corr",
    "acc_norm_lag_ms",
    "acc_energy_ratio",
    "gyro_energy_ratio",
    "pitch_corr",
    "pitch_divergence_std",
]

CROSS_SENSOR_EXTRA = [
    "acc_norm_coherence_mean",
    "gyro_norm_coherence_mean",
    "peak_time_diff_acc_norm_s",
    "shock_peak_ratio_bike_to_rider",
    "shock_peak_ratio_rider_to_bike",
    "vec_disagreement_mean_ms2",
    "roll_corr",
    "roll_divergence_std",
    "energy_ratio_longitudinal",
    "energy_ratio_lateral",
    "energy_ratio_vertical",
]

GROUPED_FEATURES = [
    "bump_vertical_peak_ms2",
    "bump_shock_attenuation_ratio",
    "bump_response_lag_s",
    "brake_longitudinal_decel_peak_ms2",
    "brake_pitch_change_deg",
    "brake_pitch_coupling_corr",
    "corner_lateral_energy_ms2_sq",
    "corner_roll_rate_rms_deg_s",
    "corner_roll_coupling_corr",
    "sprint_cadence_band_fraction",
    "sprint_dom_freq_hz",
    "sprint_gyro_energy_sum",
    "disagree_vec_diff_mean_ms2",
    "disagree_vertical_coherence",
    "disagree_energy_axis_ratio_var",
]


def _window_sanity_flags(
    acc: np.ndarray,
    gyro: np.ndarray,
    *,
    min_len: int = 8,
    min_std: float = 1e-6,
) -> str:
    """Return semi-colon-separated sanity flags for degenerate windows."""
    flags: list[str] = []
    if len(acc) < min_len or len(gyro) < min_len:
        flags.append("short_window")
    if np.nanstd(acc) < min_std:
        flags.append("acc_low_variance")
    if np.nanstd(gyro) < min_std:
        flags.append("gyro_low_variance")
    if np.all(~np.isfinite(acc)) or np.all(~np.isfinite(gyro)):
        flags.append("all_nan")
    return ";".join(flags) if flags else "ok"


def _acc_norm(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(arr * arr, axis=1))


def _gyro_norm(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(arr * arr, axis=1))


def _extract_per_sensor_features(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float,
    fs_hz: float,
) -> dict[str, float]:
    """Extract per-sensor features for one window (legacy + extended)."""
    acc_n = _acc_norm(acc)
    gyro_n = _gyro_norm(gyro)
    vertical_acc = acc[:, 2] if acc.shape[1] > 2 else np.full(len(acc), np.nan)

    jerk = np.abs(np.diff(acc_n, prepend=acc_n[0])) / max(dt, 1e-9)

    e_lo, e_hi, e_tot = band_energy_ratio(acc_n, fs_hz)
    high_frac = e_hi / e_tot if np.isfinite(e_tot) and e_tot > 1e-18 else np.nan

    out: dict[str, float] = {
        "acc_norm_mean": float(np.nanmean(acc_n)),
        "acc_norm_max": float(np.nanmax(acc_n)),
        "acc_norm_energy": float(np.nansum(acc_n * acc_n)),
        "jerk_norm_max": float(np.nanmax(jerk)),
        "gyro_norm_max": float(np.nanmax(gyro_n)),
        "gyro_energy": float(np.nansum(gyro_n * gyro_n)),
        "vertical_acc_mean": float(np.nanmean(vertical_acc)),
        "vertical_acc_std": float(np.nanstd(vertical_acc)) if len(vertical_acc) > 1 else 0.0,
        "acc_norm_rms": float(np.sqrt(np.nanmean(acc_n * acc_n))),
        "acc_norm_std": float(np.nanstd(acc_n)) if len(acc_n) > 1 else 0.0,
        "acc_norm_ptp": float(np.nanmax(acc_n) - np.nanmin(acc_n)),
        "gyro_norm_rms": float(np.sqrt(np.nanmean(gyro_n * gyro_n))),
        "gyro_norm_std": float(np.nanstd(gyro_n)) if len(gyro_n) > 1 else 0.0,
        "gyro_norm_ptp": float(np.nanmax(gyro_n) - np.nanmin(gyro_n)),
        "dom_freq_acc_norm_hz": dominant_frequency_hz(acc_n, fs_hz),
        "acc_band_low_energy": e_lo,
        "acc_band_high_energy": e_hi,
        "acc_band_high_fraction": float(high_frac) if np.isfinite(high_frac) else np.nan,
        "crest_factor_acc_norm": crest_factor(acc_n),
        "entropy_acc_norm": signal_entropy(acc_n),
        "zcr_acc_norm": zero_crossing_rate(acc_n),
        "longitudinal_acc_std": float(np.nanstd(acc[:, 0])) if acc.shape[1] > 0 else np.nan,
        "lateral_acc_std": float(np.nanstd(acc[:, 1])) if acc.shape[1] > 1 else np.nan,
    }
    return out


def _orient_window_features(
    odf: pd.DataFrame,
    t0: float,
    window_start_s: float,
    window_end_s: float,
) -> dict[str, float]:
    """Pitch/roll stats for samples inside [window_start_s, window_end_s] (section time)."""
    ts = (odf["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
    mask = (ts >= window_start_s) & (ts <= window_end_s)
    if np.sum(mask) < 3:
        return {k: np.nan for k in ORIENT_FEATURE_SUFFIXES}
    pitch = odf.loc[mask, "pitch_deg"].to_numpy(dtype=float)
    roll = odf.loc[mask, "roll_deg"].to_numpy(dtype=float)
    tsub = ts[mask]
    if len(tsub) < 2:
        return {k: np.nan for k in ORIENT_FEATURE_SUFFIXES}
    dp = np.diff(pitch)
    dr = np.diff(roll)
    dts = np.diff(tsub)
    dts = np.where(np.abs(dts) > 1e-9, dts, np.nan)
    pr = np.abs(dp / dts)
    rr = np.abs(dr / dts)
    return {
        "pitch_mean_deg": float(np.nanmean(pitch)),
        "pitch_std_deg": float(np.nanstd(pitch)),
        "roll_mean_deg": float(np.nanmean(roll)),
        "roll_std_deg": float(np.nanstd(roll)),
        "pitch_rate_mean_deg_s": float(np.nanmean(pr)),
        "roll_rate_mean_deg_s": float(np.nanmean(rr)),
    }


def _cross_corr_lag_ms(
    a: np.ndarray,
    b: np.ndarray,
    dt_ms: float,
    max_lag_ms: float = 500.0,
) -> float:
    """Lag in ms at max cross-correlation (-max_lag to +max_lag)."""
    if len(a) < 2 or len(b) < 2 or not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return np.nan
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    corr = np.correlate(a, b, mode="full")
    mid = len(corr) // 2
    dt_ms = max(dt_ms, 0.1)
    max_lag_samp = int(max_lag_ms / dt_ms)
    lo = max(0, mid - max_lag_samp)
    hi = min(len(corr), mid + max_lag_samp + 1)
    region = corr[lo:hi]
    idx = np.argmax(region)
    lag_samp = idx - (mid - lo)
    return float(lag_samp * dt_ms)


def _align_to_common_time(
    ts_a: np.ndarray,
    vals_a: np.ndarray,
    ts_b: np.ndarray,
    vals_b: np.ndarray,
    t_center: float,
    half_window_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract aligned segments for cross-sensor features. Returns (a, b) or None."""
    t_lo = t_center - half_window_s
    t_hi = t_center + half_window_s
    mask_a = (ts_a >= t_lo) & (ts_a <= t_hi)
    mask_b = (ts_b >= t_lo) & (ts_b <= t_hi)
    if np.sum(mask_a) < 5 or np.sum(mask_b) < 5:
        return None
    grid = np.linspace(t_lo, t_hi, max(20, min(np.sum(mask_a), np.sum(mask_b))))
    a_interp = np.interp(grid, ts_a[mask_a], vals_a[mask_a])
    b_interp = np.interp(grid, ts_b[mask_b], vals_b[mask_b])
    return a_interp, b_interp


def _align_acc_vectors(
    ts_a: np.ndarray,
    acc_a: np.ndarray,
    ts_b: np.ndarray,
    acc_b: np.ndarray,
    t_center: float,
    half_window_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate 3-axis acc onto common time grid."""
    t_lo = t_center - half_window_s
    t_hi = t_center + half_window_s
    mask_a = (ts_a >= t_lo) & (ts_a <= t_hi)
    mask_b = (ts_b >= t_lo) & (ts_b <= t_hi)
    if np.sum(mask_a) < 5 or np.sum(mask_b) < 5:
        return None
    n = max(20, min(np.sum(mask_a), np.sum(mask_b)))
    grid = np.linspace(t_lo, t_hi, n)
    out_a = np.column_stack([np.interp(grid, ts_a[mask_a], acc_a[mask_a, i]) for i in range(3)])
    out_b = np.column_stack([np.interp(grid, ts_b[mask_b], acc_b[mask_b, i]) for i in range(3)])
    return out_a, out_b


def _worst_calibration_quality(cal_json: dict[str, Any] | None) -> str:
    if not cal_json:
        return "unknown"
    tiers = {"poor": 0, "marginal": 1, "good": 2, "unknown": -1}
    worst = "good"
    worst_v = 3
    for _k, block in cal_json.items():
        if not isinstance(block, dict):
            continue
        q = str(block.get("calibration_quality", "unknown"))
        if tiers.get(q, -1) < worst_v:
            worst_v = tiers.get(q, -1)
            worst = q
    return worst


def extract_section(
    section_path: Path,
    section_name: str,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    write_plots: bool = True,
    orientation_variant: str = "complementary_orientation",
    label_index: Any | None = None,
    sync_method: str = "",
    recording_id: str | None = None,
    section_id: str | None = None,
    event_windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract features for a section. Returns DataFrame with one row per window."""
    section_path = Path(section_path)
    calibrated_dir = section_path / "calibrated"
    orient_dir = section_path / "orientation"
    features_dir = section_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    if recording_id is None or section_id is None:
        sec = Path(section_name.replace("\\", "/")).name
        section_id = section_id or sec
        if recording_id is None:
            try:
                from common.paths import parse_section_folder_name

                rec, _idx = parse_section_folder_name(section_id)
                recording_id = rec
            except Exception:
                parts = section_name.replace("\\", "/").split("/")
                recording_id = parts[0] if parts else ""

    cal_json: dict[str, Any] | None = None
    cal_path = calibrated_dir / "calibration.json"
    if cal_path.exists():
        cal_json = json.loads(cal_path.read_text(encoding="utf-8"))
    calib_q = _worst_calibration_quality(cal_json)

    dfs: dict[str, pd.DataFrame] = {}
    orient_dfs: dict[str, pd.DataFrame] = {}
    for sensor in ("sporsa", "arduino"):
        p = calibrated_dir / f"{sensor}.csv"
        if p.exists():
            df = load_dataframe(p)
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            dfs[sensor] = df
        p_orient = orient_dir / f"{sensor}__{orientation_variant}.csv"
        if p_orient.exists():
            orient_dfs[sensor] = pd.read_csv(p_orient)

    has_orient = bool(orient_dfs)

    if not dfs:
        raise FileNotFoundError(f"No calibrated data in {calibrated_dir}")

    t0 = float(dfs[list(dfs.keys())[0]]["timestamp"].iloc[0])
    t_end = float(dfs[list(dfs.keys())[0]]["timestamp"].iloc[-1])
    duration_s = (t_end - t0) / 1000.0

    half_window_s = window_s / 2.0
    rows = []

    if event_windows is not None and len(event_windows):
        centers = pd.to_numeric(event_windows["window_center_s"], errors="coerce").dropna().to_numpy(dtype=float)
        centers = np.array([c for c in centers if c - half_window_s >= 0 and c + half_window_s <= duration_s], dtype=float)
    else:
        centers = np.arange(half_window_s, duration_s - half_window_s + 1e-12, hop_s, dtype=float)

    for t_center_s in centers:
        t_center_ms = t0 + t_center_s * 1000.0
        window_start_s = (t_center_ms - t0) / 1000.0 - half_window_s
        window_end_s = window_start_s + window_s

        row: dict[str, Any] = {
            "section": section_name,
            "recording_id": recording_id,
            "section_id": section_id,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "window_center_s": t_center_s,
            "sync_method": sync_method,
            "orientation_method": orientation_variant,
            "calibration_quality": calib_q,
            "window_source": "event_centered" if event_windows is not None else "sliding",
            "event_type": "",
            "event_confidence": np.nan,
            "event_timestamp": np.nan,
        }
        if event_windows is not None and len(event_windows):
            # Nearest event center metadata (exact in common case).
            idx_evt = int(np.nanargmin(np.abs(pd.to_numeric(event_windows["window_center_s"], errors="coerce").to_numpy(dtype=float) - t_center_s)))
            evt = event_windows.iloc[idx_evt]
            row["event_type"] = str(evt.get("event_type", "") or "")
            row["event_confidence"] = float(evt.get("confidence")) if pd.notna(evt.get("confidence")) else np.nan
            row["event_timestamp"] = float(evt.get("event_timestamp")) if pd.notna(evt.get("event_timestamp")) else np.nan

        for sensor, df in dfs.items():
            ts_raw = df["timestamp"].to_numpy(dtype=float)
            ts = (ts_raw - t0) / 1000.0
            mask = (ts >= window_start_s) & (ts <= window_end_s)
            if np.sum(mask) < 5:
                for f in PER_SENSOR_FEATURES + PER_SENSOR_EXTRA:
                    row[f"{sensor}__{f}"] = np.nan
            else:
                acc = df.loc[mask, ACC_COLS].to_numpy(dtype=float)
                gyro = df.loc[mask, GYRO_COLS].to_numpy(dtype=float)
                dt = float(np.nanmean(np.diff(ts_raw[mask]))) / 1000.0 if np.sum(mask) > 1 else 0.01
                fs_hz = 1.0 / max(dt, 1e-6)
                feats = _extract_per_sensor_features(acc, gyro, dt, fs_hz)
                for k, v in feats.items():
                    row[f"{sensor}__{k}"] = v

            if sensor in orient_dfs:
                of = orient_dfs[sensor]
                ofe = _orient_window_features(of, t0, window_start_s, window_end_s)
                for k, v in ofe.items():
                    row[f"{sensor}__{k}"] = v
            elif has_orient:
                for k in ORIENT_FEATURE_SUFFIXES:
                    row[f"{sensor}__{k}"] = np.nan

        if "sporsa" in dfs and "arduino" in dfs:
            sporsa_df = dfs["sporsa"]
            arduino_df = dfs["arduino"]
            ts_s = (sporsa_df["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
            ts_a = (arduino_df["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
            acc_n_s = _acc_norm(sporsa_df[ACC_COLS].to_numpy(dtype=float))
            acc_n_a = _acc_norm(arduino_df[ACC_COLS].to_numpy(dtype=float))
            gyro_n_s = _gyro_norm(sporsa_df[GYRO_COLS].to_numpy(dtype=float))
            gyro_n_a = _gyro_norm(arduino_df[GYRO_COLS].to_numpy(dtype=float))
            dt_ms = (
                1000.0 * float(np.nanmean(np.diff(sporsa_df["timestamp"])))
                if len(sporsa_df) > 1
                else 10.0
            )
            fs_hz = 1000.0 / max(dt_ms, 0.1)

            aligned = _align_to_common_time(
                ts_s, acc_n_s, ts_a, acc_n_a, t_center_s, half_window_s
            )
            if aligned is not None:
                a_s, a_a = aligned
                if len(a_s) > 2 and (np.ptp(a_s) > 1e-12 or np.ptp(a_a) > 1e-12):
                    try:
                        corr = stats.pearsonr(a_s, a_a)[0]
                    except Exception:
                        corr = np.nan
                else:
                    corr = np.nan
                row["acc_norm_corr"] = corr if np.isfinite(corr) else np.nan
                dt_interp_ms = 1000.0 * window_s / max(len(a_s), 1)
                row["acc_norm_lag_ms"] = _cross_corr_lag_ms(a_s, a_a, dt_interp_ms)
                row["acc_norm_coherence_mean"] = mean_coherence_band(
                    a_s, a_a, fs_hz=1.0 / max(window_s / max(len(a_s), 1), 1e-6)
                )
            else:
                row["acc_norm_corr"] = np.nan
                row["acc_norm_lag_ms"] = np.nan
                row["acc_norm_coherence_mean"] = np.nan

            aligned_g = _align_to_common_time(
                ts_s, gyro_n_s, ts_a, gyro_n_a, t_center_s, half_window_s
            )
            if aligned_g is not None:
                g_s, g_a = aligned_g
                dt_g = window_s / max(len(g_s), 1)
                row["gyro_norm_coherence_mean"] = mean_coherence_band(g_s, g_a, fs_hz=1.0 / max(dt_g, 1e-6))
            else:
                row["gyro_norm_coherence_mean"] = np.nan

            e_s = row.get("sporsa__acc_norm_energy", np.nan)
            e_a = row.get("arduino__acc_norm_energy", np.nan)
            row["acc_energy_ratio"] = e_s / e_a if np.isfinite(e_a) and e_a != 0 else np.nan

            g_s_e = row.get("sporsa__gyro_energy", np.nan)
            g_a_e = row.get("arduino__gyro_energy", np.nan)
            row["gyro_energy_ratio"] = g_s_e / g_a_e if np.isfinite(g_a_e) and g_a_e != 0 else np.nan

            acc_pair = _align_acc_vectors(
                ts_s,
                sporsa_df[ACC_COLS].to_numpy(dtype=float),
                ts_a,
                arduino_df[ACC_COLS].to_numpy(dtype=float),
                t_center_s,
                half_window_s,
            )
            if acc_pair is not None:
                va, vb = acc_pair
                row["vec_disagreement_mean_ms2"] = vec_disagreement_ms2(va, vb)
                exs = np.nansum(va[:, 0] ** 2)
                exa = np.nansum(vb[:, 0] ** 2)
                eys = np.nansum(va[:, 1] ** 2)
                eya = np.nansum(vb[:, 1] ** 2)
                ezs = np.nansum(va[:, 2] ** 2)
                eza = np.nansum(vb[:, 2] ** 2)
                row["energy_ratio_longitudinal"] = safe_ratio(exs, exa)
                row["energy_ratio_lateral"] = safe_ratio(eys, eya)
                row["energy_ratio_vertical"] = safe_ratio(ezs, eza)
                mxs = float(np.nanmax(acc_n_s[(ts_s >= t_center_s - half_window_s) & (ts_s <= t_center_s + half_window_s)]))
                mxa = float(np.nanmax(acc_n_a[(ts_a >= t_center_s - half_window_s) & (ts_a <= t_center_s + half_window_s)]))
                row["shock_peak_ratio_bike_to_rider"] = safe_ratio(mxs, mxa, eps=1e-9)
                row["shock_peak_ratio_rider_to_bike"] = safe_ratio(mxa, mxs, eps=1e-9)
                m_s = (ts_s >= t_center_s - half_window_s) & (ts_s <= t_center_s + half_window_s)
                m_a = (ts_a >= t_center_s - half_window_s) & (ts_a <= t_center_s + half_window_s)
                row["peak_time_diff_acc_norm_s"] = peak_time_difference_s(
                    ts_s[m_s],
                    acc_n_s[m_s],
                    ts_a[m_a],
                    acc_n_a[m_a],
                )
            else:
                row["vec_disagreement_mean_ms2"] = np.nan
                row["energy_ratio_longitudinal"] = np.nan
                row["energy_ratio_lateral"] = np.nan
                row["energy_ratio_vertical"] = np.nan
                row["shock_peak_ratio_bike_to_rider"] = np.nan
                row["shock_peak_ratio_rider_to_bike"] = np.nan
                row["peak_time_diff_acc_norm_s"] = np.nan

            # Physically interpreted compact families (aligned window).
            if acc_pair is not None and aligned is not None:
                bike_pitch = rider_pitch = bike_roll = rider_roll = None
                roll_dt_s: float | None = None
                if "sporsa" in orient_dfs and "arduino" in orient_dfs:
                    od_s = orient_dfs["sporsa"]
                    od_a = orient_dfs["arduino"]
                    ts_os = (od_s["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
                    ts_oa = (od_a["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
                    pitch_pair = _align_to_common_time(
                        ts_os,
                        od_s["pitch_deg"].to_numpy(dtype=float),
                        ts_oa,
                        od_a["pitch_deg"].to_numpy(dtype=float),
                        t_center_s,
                        half_window_s,
                    )
                    roll_pair = _align_to_common_time(
                        ts_os,
                        od_s["roll_deg"].to_numpy(dtype=float),
                        ts_oa,
                        od_a["roll_deg"].to_numpy(dtype=float),
                        t_center_s,
                        half_window_s,
                    )
                    if pitch_pair is not None:
                        bike_pitch, rider_pitch = pitch_pair
                    if roll_pair is not None:
                        bike_roll, rider_roll = roll_pair
                        roll_dt_s = window_s / max(len(bike_roll), 1)

                va, vb = acc_pair
                bike_gyro_interp = _align_to_common_time(ts_s, gyro_n_s, ts_a, gyro_n_a, t_center_s, half_window_s)
                if bike_gyro_interp is not None:
                    gsa, gaa = bike_gyro_interp
                else:
                    gsa = np.array([], dtype=float)
                    gaa = np.array([], dtype=float)
                axis_ratios = (
                    row.get("energy_ratio_longitudinal", np.nan),
                    row.get("energy_ratio_lateral", np.nan),
                    row.get("energy_ratio_vertical", np.nan),
                )
                grouped = extract_grouped_features(
                    bike_acc=va,
                    rider_acc=vb,
                    bike_acc_norm=aligned[0],
                    rider_acc_norm=aligned[1],
                    bike_gyro_norm=gsa,
                    rider_gyro_norm=gaa,
                    bike_pitch=bike_pitch,
                    rider_pitch=rider_pitch,
                    bike_roll=bike_roll,
                    rider_roll=rider_roll,
                    dt_s=window_s / max(len(va), 1),
                    roll_dt_s=roll_dt_s,
                    fs_hz=1.0 / max(window_s / max(len(va), 1), 1e-6),
                    vec_disagreement=row.get("vec_disagreement_mean_ms2", np.nan),
                    axis_energy_ratios=axis_ratios,
                )
                row.update(grouped)
            else:
                for f in GROUPED_FEATURES:
                    row[f] = np.nan

            if "sporsa" in orient_dfs and "arduino" in orient_dfs:
                od_s = orient_dfs["sporsa"]
                od_a = orient_dfs["arduino"]
                ts_os = (od_s["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
                ts_oa = (od_a["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
                pitch_s = od_s["pitch_deg"].to_numpy(dtype=float)
                pitch_a = od_a["pitch_deg"].to_numpy(dtype=float)
                roll_s = od_s["roll_deg"].to_numpy(dtype=float)
                roll_a = od_a["roll_deg"].to_numpy(dtype=float)
                al = _align_to_common_time(ts_os, pitch_s, ts_oa, pitch_a, t_center_s, half_window_s)
                if al is not None:
                    ps, pa = al
                    if len(ps) > 2 and (np.ptp(ps) > 1e-12 or np.ptp(pa) > 1e-12):
                        try:
                            row["pitch_corr"] = stats.pearsonr(ps, pa)[0]
                        except Exception:
                            row["pitch_corr"] = np.nan
                    else:
                        row["pitch_corr"] = np.nan
                    row["pitch_divergence_std"] = float(np.nanstd(ps - pa))
                else:
                    row["pitch_corr"] = np.nan
                    row["pitch_divergence_std"] = np.nan
                alr = _align_to_common_time(ts_os, roll_s, ts_oa, roll_a, t_center_s, half_window_s)
                if alr is not None:
                    rs, ra = alr
                    if len(rs) > 2 and (np.ptp(rs) > 1e-12 or np.ptp(ra) > 1e-12):
                        try:
                            row["roll_corr"] = stats.pearsonr(rs, ra)[0]
                        except Exception:
                            row["roll_corr"] = np.nan
                    else:
                        row["roll_corr"] = np.nan
                    row["roll_divergence_std"] = float(np.nanstd(rs - ra))
                else:
                    row["roll_corr"] = np.nan
                    row["roll_divergence_std"] = np.nan
            else:
                row["pitch_corr"] = np.nan
                row["pitch_divergence_std"] = np.nan
                row["roll_corr"] = np.nan
                row["roll_divergence_std"] = np.nan
        else:
            for f in CROSS_SENSOR_FEATURES + CROSS_SENSOR_EXTRA:
                row[f] = np.nan
            for f in GROUPED_FEATURES:
                row[f] = np.nan

        scen = ""
        src = "none"
        if label_index is not None:
            scen, src = label_index.resolve(
                recording_id,
                section_id,
                window_start_s,
                window_end_s,
            )
        row["scenario_label"] = scen
        row["label_source"] = src
        # Window sanity checks from per-sensor windows if available.
        if "sporsa" in dfs:
            sdf = dfs["sporsa"]
            st = (sdf["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
            smask = (st >= window_start_s) & (st <= window_end_s)
            sacc = sdf.loc[smask, ACC_COLS].to_numpy(dtype=float)
            sgyro = sdf.loc[smask, GYRO_COLS].to_numpy(dtype=float)
            row["sporsa__window_sanity"] = _window_sanity_flags(sacc, sgyro)
        if "arduino" in dfs:
            adf = dfs["arduino"]
            at = (adf["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
            amask = (at >= window_start_s) & (at <= window_end_s)
            aacc = adf.loc[amask, ACC_COLS].to_numpy(dtype=float)
            agyro = adf.loc[amask, GYRO_COLS].to_numpy(dtype=float)
            row["arduino__window_sanity"] = _window_sanity_flags(aacc, agyro)

        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = features_dir / "features.csv"
    out_df.to_csv(out_path, index=False)

    stats_dict: dict[str, dict[str, float]] = {}
    for col in out_df.select_dtypes(include=[np.number]).columns:
        vals = out_df[col].dropna()
        stats_dict[col] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
            "min": float(vals.min()) if len(vals) else float("nan"),
            "max": float(vals.max()) if len(vals) else float("nan"),
            "nan_fraction": float(out_df[col].isna().mean()),
        }
    with (features_dir / "features_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    write_feature_schema_json(features_dir / "feature_schema.json")

    if write_plots and len(out_df) > 0:
        from .plots import plot_features_timeline, plot_scenario_feature_summary

        plot_features_timeline(features_dir, out_df)
        plot_scenario_feature_summary(features_dir, out_df, GROUPED_FEATURES)

    return out_df


def _resolve_target(name: str) -> tuple[str, Path | None, bool]:
    """Resolve CLI argument into (recording_or_session, section_dir_or_none, is_session).

    Accepted:
    - section directory path (e.g. data/sections/2026-02-26_r2s1)
    - section folder name (e.g. 2026-02-26_r2s1)
    - recording name (e.g. 2026-02-26_r2) with --all-sections
    - session name (e.g. 2026-02-26) with --all
    """
    s = name.strip().rstrip("/").replace("\\", "/")
    if not s:
        raise ValueError("name must be a non-empty section path, section folder, recording name, or session name")

    p = Path(s)
    if p.is_dir():
        return "", p.resolve(), False

    from common.paths import parse_section_folder_name, sections_root

    try:
        rec, _idx = parse_section_folder_name(s)
    except Exception:
        rec = ""
    else:
        sec_dir = sections_root() / s
        if sec_dir.is_dir():
            return rec, sec_dir.resolve(), False

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s, None, True
    if re.fullmatch(r".+_r\d+", s):
        return s, None, False

    raise ValueError(
        f"Unrecognized name {name!r}. Expected a section folder like '2026-02-26_r2s1', "
        f"a section path like 'data/sections/2026-02-26_r2s1', a recording like '2026-02-26_r2', "
        f"or a session date like '2026-02-26'."
    )


def extract_from_args(
    name: str,
    *,
    all_sections: bool = False,
    all_recordings: bool = False,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    orientation_variant: str = "complementary_orientation",
    labels_path: Path | str | None = None,
    event_centered: bool = False,
    event_types: set[str] | None = None,
    min_event_confidence: float = 0.0,
) -> list[Path]:
    """Run feature extraction from CLI args."""
    from common import recording_dir, recordings_root

    label_index = None
    if labels_path is not None:
        from labels.parser import load_labels_from_path

        label_index = load_labels_from_path(Path(labels_path))

    rec, section_dir, is_session = _resolve_target(name)

    if all_recordings:
        if not is_session:
            raise ValueError("--all expects a session name like '2026-02-26'")
        rec_dirs = sorted(
            d
            for d in (recordings_root()).iterdir()
            if d.is_dir() and d.name.startswith(rec + "_r")
        )
    else:
        if section_dir is not None:
            rec_dirs = []
        else:
            rec_dir = recording_dir(rec)
            if not rec_dir.exists():
                raise FileNotFoundError(f"Recording not found: {rec_dir}")
            rec_dirs = [rec_dir]

    done = []
    if section_dir is not None:
        section_id = section_dir.name
        try:
            from common.paths import parse_section_folder_name

            rec_name, _idx = parse_section_folder_name(section_id)
        except Exception:
            rec_name = rec or ""
        sync_method = _read_sync_method(rec_name) if rec_name else ""
        section_name = section_id
        cal_dir = section_dir / "calibrated"
        if not (cal_dir / "calibration.json").exists():
            log.warning("Skipping %s (no calibration)", section_name)
            return []
        event_windows = None
        if event_centered:
            from events.extract import load_event_windows

            event_windows = load_event_windows(
                section_dir,
                event_types=event_types,
                min_confidence=min_event_confidence,
            )
        extract_section(
            section_dir,
            section_name,
            window_s=window_s,
            hop_s=hop_s,
            orientation_variant=orientation_variant,
            label_index=label_index,
            sync_method=sync_method,
            recording_id=rec_name or None,
            section_id=section_id,
            event_windows=event_windows,
        )
        return [section_dir]

    for rdir in rec_dirs:
        rec_name = rdir.name
        sync_method = _read_sync_method(rec_name)
        from common.paths import iter_sections_for_recording

        if not (all_sections or all_recordings):
            raise ValueError(
                "Pass a section folder/path, or use --all-sections with a recording, or --all with a session."
            )

        section_dirs = iter_sections_for_recording(rec_name)

        for sdir in section_dirs:
            section_id = sdir.name
            section_name = section_id
            cal_dir = sdir / "calibrated"
            if not (cal_dir / "calibration.json").exists():
                log.warning("Skipping %s (no calibration)", section_name)
                continue
            event_windows = None
            if event_centered:
                from events.extract import load_event_windows

                event_windows = load_event_windows(
                    sdir,
                    event_types=event_types,
                    min_confidence=min_event_confidence,
                )
            extract_section(
                sdir,
                section_name,
                window_s=window_s,
                hop_s=hop_s,
                orientation_variant=orientation_variant,
                label_index=label_index,
                sync_method=sync_method,
                recording_id=rec_name,
                section_id=section_id,
                event_windows=event_windows,
            )
            done.append(sdir)

    return done


def _read_sync_method(recording_name: str) -> str:
    from common.paths import recording_dir

    p = recording_dir(recording_name) / "synced" / "all_methods.json"
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return str(data.get("selected_method", "") or "")
    except Exception:
        return ""


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(prog="python -m features.extract")
    parser.add_argument("name", help="Section path, recording, or session")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument("--all", action="store_true", dest="all_recordings")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--hop", type=float, default=0.5)
    parser.add_argument(
        "--orientation",
        default="complementary_orientation",
        help="Orientation CSV suffix (default complementary_orientation)",
    )
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV or JSON")
    parser.add_argument(
        "--event-centered",
        action="store_true",
        help="Use event-centered windows from section events/event_candidates.csv instead of sliding windows.",
    )
    parser.add_argument(
        "--event-types",
        type=str,
        default="",
        help="Comma-separated event types to keep when --event-centered is used.",
    )
    parser.add_argument(
        "--min-event-confidence",
        type=float,
        default=0.0,
        help="Minimum event confidence to keep for event-centered extraction.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    extract_from_args(
        args.name,
        all_sections=args.all_sections,
        all_recordings=args.all_recordings,
        window_s=args.window,
        hop_s=args.hop,
        orientation_variant=args.orientation,
        labels_path=args.labels,
        event_centered=args.event_centered,
        event_types={x.strip() for x in args.event_types.split(",") if x.strip()} or None,
        min_event_confidence=args.min_event_confidence,
    )
