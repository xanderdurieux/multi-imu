"""Compute thesis-friendly derived motion signals after calibration/orientation.

This stage transforms calibrated/oriented data into physically interpretable time-series:
- gravity-compensated linear acceleration,
- longitudinal/lateral/vertical acceleration,
- angular rates around interpretable axes,
- tilt derivatives,
- rider-minus-bicycle residuals,
- shock transmission signals,
- optional robust-normalized variants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_dataframe

GRAVITY_M_S2 = 9.81
SENSORS = ("sporsa", "arduino")
FULL_ALIGNMENT_MODES = {"gravity_plus_forward", "section_horizontal_frame"}

def _safe_gradient(values: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return np.full(len(values), np.nan)
    out = np.gradient(values, t_s)
    out[~np.isfinite(out)] = np.nan
    return out


def _robust_zscore(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median and MAD (scaled by 1.4826)."""
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        return np.full_like(x, np.nan, dtype=float)
    return (x - med) / scale


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.abs(x)
    s = pd.Series(np.asarray(x, dtype=float))
    return np.sqrt(s.pow(2).rolling(window=window, min_periods=max(3, window // 3), center=True).mean()).to_numpy()


def _dt_seconds(df: pd.DataFrame) -> float:
    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
    if len(ts) < 2:
        return 0.01
    dt = float(np.nanmedian(np.diff(ts)) / 1000.0)
    return max(dt, 1e-3)


def _load_orientation(section_path: Path, sensor: str, variant: str) -> pd.DataFrame | None:
    p = section_path / "orientation" / f"{sensor}__{variant}.csv"
    if not p.exists():
        return None
    odf = pd.read_csv(p)
    if "timestamp" not in odf.columns:
        return None
    return odf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def _interp_column(src_t: np.ndarray, src_v: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    mask = np.isfinite(src_t) & np.isfinite(src_v)
    if np.sum(mask) < 3:
        return np.full_like(dst_t, np.nan, dtype=float)
    return np.interp(dst_t, src_t[mask], src_v[mask], left=np.nan, right=np.nan)


def derive_section_signals(
    section_path: Path,
    *,
    orientation_variant: str = "complementary_orientation",
    include_normalized: bool = True,
) -> dict[str, Any]:
    """Compute and export derived signals for one section.

    Writes:
    - ``derived/<sensor>_signals.csv`` for bike/rider signals.
    - ``derived/cross_sensor_signals.csv`` for residual/transmission signals.
    - ``derived/derived_signals_meta.json`` quality/dependency metadata.
    """
    section_path = Path(section_path)
    cal_dir = section_path / "calibrated"
    out_dir = section_path / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_meta_path = cal_dir / "calibration.json"
    if not cal_meta_path.exists():
        raise FileNotFoundError(f"Missing calibration metadata: {cal_meta_path}")
    cal_meta = json.loads(cal_meta_path.read_text(encoding="utf-8"))

    frame_alignment = str(cal_meta.get("sporsa", {}).get("frame_alignment", "gravity_only"))
    full_alignment = frame_alignment in FULL_ALIGNMENT_MODES

    sensor_outputs: dict[str, pd.DataFrame] = {}
    quality: dict[str, Any] = {
        "frame_alignment": frame_alignment,
        "full_horizontal_alignment_available": full_alignment,
        "signals": {},
    }

    for sensor in SENSORS:
        p = cal_dir / f"{sensor}.csv"
        if not p.exists():
            continue
        df = load_dataframe(p)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            continue

        dt = _dt_seconds(df)
        t_s = (df["timestamp"].to_numpy(dtype=float) - float(df["timestamp"].iloc[0])) / 1000.0

        ax = pd.to_numeric(df["ax"], errors="coerce").to_numpy(dtype=float)
        ay = pd.to_numeric(df["ay"], errors="coerce").to_numpy(dtype=float)
        az = pd.to_numeric(df["az"], errors="coerce").to_numpy(dtype=float)
        gx = pd.to_numeric(df["gx"], errors="coerce").to_numpy(dtype=float)
        gy = pd.to_numeric(df["gy"], errors="coerce").to_numpy(dtype=float)
        gz = pd.to_numeric(df["gz"], errors="coerce").to_numpy(dtype=float)

        lin_x = ax
        lin_y = ay
        lin_z = az - GRAVITY_M_S2

        odf = _load_orientation(section_path, sensor, orientation_variant)
        roll_rate = np.full(len(df), np.nan)
        pitch_rate = np.full(len(df), np.nan)
        if odf is not None and {"roll_deg", "pitch_deg"}.issubset(odf.columns):
            ot = odf["timestamp"].to_numpy(dtype=float)
            roll = np.deg2rad(odf["roll_deg"].to_numpy(dtype=float))
            pitch = np.deg2rad(odf["pitch_deg"].to_numpy(dtype=float))
            droll = _safe_gradient(roll, (ot - ot[0]) / 1000.0)
            dpitch = _safe_gradient(pitch, (ot - ot[0]) / 1000.0)
            roll_rate = _interp_column(ot, droll, df["timestamp"].to_numpy(dtype=float))
            pitch_rate = _interp_column(ot, dpitch, df["timestamp"].to_numpy(dtype=float))

        out = pd.DataFrame({
            "timestamp": df["timestamp"],
            "time_s": t_s,
            "lin_acc_world_x_m_s2": lin_x,
            "lin_acc_world_y_m_s2": lin_y,
            "lin_acc_world_z_m_s2": lin_z,
            "acc_longitudinal_m_s2": lin_x if full_alignment else np.nan,
            "acc_lateral_m_s2": lin_y if full_alignment else np.nan,
            "acc_vertical_m_s2": lin_z,
            "omega_roll_axis_rad_s": gx if full_alignment else np.nan,
            "omega_pitch_axis_rad_s": gy if full_alignment else np.nan,
            "omega_yaw_axis_rad_s": gz if full_alignment else np.nan,
            "tilt_roll_rate_rad_s": roll_rate,
            "tilt_pitch_rate_rad_s": pitch_rate,
            "quality_full_alignment_ok": int(full_alignment),
            "quality_orientation_ok": int(odf is not None),
            "quality_calibration": str(cal_meta.get(sensor, {}).get("calibration_quality", "unknown")),
        })

        if include_normalized:
            for c in (
                "lin_acc_world_x_m_s2",
                "lin_acc_world_y_m_s2",
                "lin_acc_world_z_m_s2",
                "acc_vertical_m_s2",
                "tilt_roll_rate_rad_s",
                "tilt_pitch_rate_rad_s",
            ):
                out[f"{c}__robust_z"] = _robust_zscore(out[c].to_numpy(dtype=float))

        out.to_csv(out_dir / f"{sensor}_signals.csv", index=False)
        sensor_outputs[sensor] = out

    cross_meta: dict[str, Any] = {
        "residual_requires_full_alignment": True,
        "shock_vertical_valid_without_full_alignment": True,
    }

    if "sporsa" in sensor_outputs and "arduino" in sensor_outputs:
        bike = sensor_outputs["sporsa"]
        rider = sensor_outputs["arduino"]
        t_ref = bike["timestamp"].to_numpy(dtype=float)

        def interp(sig: str) -> tuple[np.ndarray, np.ndarray]:
            b = bike[sig].to_numpy(dtype=float)
            r = _interp_column(rider["timestamp"].to_numpy(dtype=float), rider[sig].to_numpy(dtype=float), t_ref)
            return b, r

        bike_vert, rider_vert = interp("acc_vertical_m_s2")
        bike_long, rider_long = interp("acc_longitudinal_m_s2")
        bike_lat, rider_lat = interp("acc_lateral_m_s2")

        residual_vert = rider_vert - bike_vert
        residual_long = rider_long - bike_long if full_alignment else np.full(len(t_ref), np.nan)
        residual_lat = rider_lat - bike_lat if full_alignment else np.full(len(t_ref), np.nan)

        dt = _dt_seconds(bike)
        window = max(3, int(round(0.25 / dt)))
        bike_shock_rms = _rolling_rms(bike_vert, window)
        rider_shock_rms = _rolling_rms(rider_vert, window)
        shock_gain = rider_shock_rms / np.maximum(bike_shock_rms, 1e-3)

        cross = pd.DataFrame({
            "timestamp": t_ref,
            "time_s": bike["time_s"],
            "residual_vertical_m_s2": residual_vert,
            "residual_longitudinal_m_s2": residual_long,
            "residual_lateral_m_s2": residual_lat,
            "shock_bike_vertical_rms_m_s2": bike_shock_rms,
            "shock_rider_vertical_rms_m_s2": rider_shock_rms,
            "shock_transmission_gain": shock_gain,
            "quality_residual_full_alignment_ok": int(full_alignment),
            "quality_shock_valid": 1,
        })
        if include_normalized:
            for c in (
                "residual_vertical_m_s2",
                "residual_longitudinal_m_s2",
                "residual_lateral_m_s2",
                "shock_transmission_gain",
            ):
                cross[f"{c}__robust_z"] = _robust_zscore(cross[c].to_numpy(dtype=float))
        cross.to_csv(out_dir / "cross_sensor_signals.csv", index=False)

    quality["signals"] = {
        "gravity_compensated_linear_acceleration": {
            "depends_on": "gravity alignment",
            "trustworthy": True,
        },
        "longitudinal_lateral_axes": {
            "depends_on": "full horizontal alignment",
            "trustworthy": bool(full_alignment),
        },
        "angular_velocity_interpretable_axes": {
            "depends_on": "full horizontal alignment",
            "trustworthy": bool(full_alignment),
        },
        "tilt_rates": {
            "depends_on": f"orientation variant={orientation_variant}",
            "trustworthy": all(df["quality_orientation_ok"].iloc[0] == 1 for df in sensor_outputs.values()) if sensor_outputs else False,
        },
        "rider_minus_bike_residual_long_lateral": {
            "depends_on": "full horizontal alignment",
            "trustworthy": bool(full_alignment),
        },
        "shock_transmission_vertical": {
            "depends_on": "gravity alignment (vertical axis)",
            "trustworthy": True,
        },
    }
    quality["cross_sensor_notes"] = cross_meta

    meta_path = out_dir / "derived_signals_meta.json"
    meta_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    return quality
