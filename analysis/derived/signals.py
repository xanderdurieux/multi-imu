"""Derived signal computation from calibrated IMU data.

Functions operate on a pandas DataFrame with IMU columns:
    timestamp, ax, ay, az, gx, gy, gz, [ax_world, ay_world, az_world, gx_world, gy_world, gz_world]

Timestamps in milliseconds, accelerometer in m/s², gyro in degrees/s.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from common.signals import butter_lowpass, norm_xyz

log = logging.getLogger(__name__)

_GRAVITY = 9.81
_QUAT_COLS = ("qw", "qx", "qy", "qz")
_DOM_EPS = 1e-6


def _rolling_rms(series: pd.Series, window: int) -> pd.Series:
    """Rolling root-mean-square with min_periods=1."""
    return series.pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)


def _rolling_lagged_xcorr(
    x: np.ndarray, y: np.ndarray, *, window: int, max_lag_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling lag-aware normalized cross-correlation max and lag (seconds not applied)."""
    n = len(x)
    out_max = np.full(n, np.nan, dtype=float)
    out_lag = np.full(n, np.nan, dtype=float)
    if n == 0 or window <= 1:
        return out_max, out_lag

    x_s = pd.Series(x, dtype=float)
    y_s = pd.Series(y, dtype=float)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=int)
    corr_mat = np.full((len(lags), n), np.nan, dtype=float)

    min_periods = max(4, min(window, 20))
    for i, lag in enumerate(lags):
        corr_mat[i, :] = x_s.rolling(window=window, min_periods=min_periods).corr(y_s.shift(lag)).to_numpy(dtype=float)

    abs_corr = np.abs(corr_mat)
    has_valid = np.any(np.isfinite(abs_corr), axis=0)
    if not np.any(has_valid):
        return out_max, out_lag

    best_idx = np.nanargmax(abs_corr[:, has_valid], axis=0)
    valid_cols = np.where(has_valid)[0]
    out_max[valid_cols] = corr_mat[best_idx, valid_cols]
    out_lag[valid_cols] = lags[best_idx]
    return out_max, out_lag


def _compute_linear_acc(
    df_cal: pd.DataFrame,
    df_orient: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (ax_lin, ay_lin, az_lin, norm) arrays aligned to df_cal rows.

    Uses the time-varying orientation quaternion to rotate calibrated
    accelerometer readings into the world frame, then removes gravity.
    The input acceleration is taken from ax_world/ay_world/az_world when
    available (matching the input used by the orientation filter), otherwise
    falls back to ax/ay/az.

    Parameters
    ----------
    df_cal:
        Calibrated sensor DataFrame (must contain timestamp + acc columns).
    df_orient:
        Orientation DataFrame with timestamp, qw, qx, qy, qz columns.

    Returns
    -------
    Tuple of four arrays (all aligned to df_cal rows), or None if orientation
    data is unavailable or missing required quaternion columns.
    """
    if df_orient is None or df_orient.empty:
        return None
    if not all(c in df_orient.columns for c in _QUAT_COLS):
        log.debug("Orientation DataFrame missing quaternion columns — skipping linear acc.")
        return None

    has_world = all(c in df_cal.columns for c in ("ax_world", "ay_world", "az_world"))
    ax = df_cal["ax_world" if has_world else "ax"].to_numpy(dtype=float)
    ay = df_cal["ay_world" if has_world else "ay"].to_numpy(dtype=float)
    az = df_cal["az_world" if has_world else "az"].to_numpy(dtype=float)

    cal_ts = df_cal["timestamp"].to_numpy(dtype=float)
    orient_ts = df_orient["timestamp"].to_numpy(dtype=float)

    # Map each calibrated timestamp to its nearest orientation row.
    sort_idx = np.argsort(orient_ts)
    orient_ts_s = orient_ts[sort_idx]
    ins = np.searchsorted(orient_ts_s, cal_ts).clip(0, len(orient_ts_s) - 1)
    left = (ins - 1).clip(0, len(orient_ts_s) - 1)
    nearest_sorted = np.where(
        np.abs(cal_ts - orient_ts_s[left]) <= np.abs(cal_ts - orient_ts_s[ins]),
        left,
        ins,
    )
    near = sort_idx[nearest_sorted]

    qw = df_orient["qw"].to_numpy(dtype=float)[near]
    qx = df_orient["qx"].to_numpy(dtype=float)[near]
    qy = df_orient["qy"].to_numpy(dtype=float)[near]
    qz = df_orient["qz"].to_numpy(dtype=float)[near]

    # Vectorised body→world rotation: v_world = R(q) @ v_body.
    # R(q) elements for a unit quaternion [w, x, y, z]:
    ax_w = (1 - 2*(qy**2 + qz**2))*ax + 2*(qx*qy - qw*qz)*ay + 2*(qx*qz + qw*qy)*az
    ay_w = 2*(qx*qy + qw*qz)*ax + (1 - 2*(qx**2 + qz**2))*ay + 2*(qy*qz - qw*qx)*az
    az_w = 2*(qx*qz - qw*qy)*ax + 2*(qy*qz + qw*qx)*ay + (1 - 2*(qx**2 + qy**2))*az

    # Remove gravity (world +Z = 9.81 m/s²).
    az_w -= _GRAVITY

    norm = norm_xyz(ax_w, ay_w, az_w)
    return ax_w, ay_w, az_w, norm


def compute_sensor_signals(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    df_orient: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute derived signals for one sensor.

    Parameters
    ----------
    df:
        DataFrame with at minimum columns: timestamp, ax, ay, az, gx, gy, gz.
        Optionally: ax_world, ay_world, az_world.
    sample_rate_hz:
        Nominal sampling rate used for filter design and jerk scaling.
    df_orient:
        Optional orientation DataFrame (timestamp, qw, qx, qy, qz) produced
        by the orientation stage.  When provided, orientation-aware linear
        acceleration columns are appended (see below).

    Returns
    -------
    DataFrame with timestamp plus derived signal columns:

    acc_norm          - sqrt(ax²+ay²+az²)
    gyro_norm         - sqrt(gx²+gy²+gz²)
    acc_vertical      - az_world if available, else az
    acc_horizontal    - sqrt(ax_world²+ay_world²) or sqrt(ax²+ay²)
    acc_deviation     - |acc_norm - 9.81|
    acc_lf            - 4th-order Butterworth lowpass at 2 Hz of acc_norm
    acc_hf            - acc_norm - acc_lf
    jerk_norm         - norm of numerical derivative of [ax,ay,az] * sample_rate_hz
    gyro_lf           - lowpass at 1 Hz of gyro_norm
    gyro_hf           - gyro_norm - gyro_lf
    energy_acc        - rolling RMS of acc_deviation over 1s window
    energy_gyro       - rolling RMS of gyro_norm over 1s window

    When df_orient is provided:
    acc_linear_x      - dynamic (gravity-removed) world-frame acceleration, X
    acc_linear_y      - dynamic (gravity-removed) world-frame acceleration, Y
    acc_linear_z      - dynamic (gravity-removed) world-frame acceleration, Z
    acc_linear_norm   - sqrt(acc_linear_x²+acc_linear_y²+acc_linear_z²)
    """
    out = pd.DataFrame()
    out["timestamp"] = df["timestamp"].values

    ax = df["ax"].to_numpy(dtype=float)
    ay = df["ay"].to_numpy(dtype=float)
    az = df["az"].to_numpy(dtype=float)
    gx = df["gx"].to_numpy(dtype=float)
    gy = df["gy"].to_numpy(dtype=float)
    gz = df["gz"].to_numpy(dtype=float)

    # Norms.
    acc_norm = norm_xyz(ax, ay, az)
    gyro_norm = norm_xyz(gx, gy, gz)
    out["acc_norm"] = acc_norm
    out["gyro_norm"] = gyro_norm

    # World-frame columns (optional).
    has_world = all(c in df.columns for c in ("ax_world", "ay_world", "az_world"))
    if has_world:
        ax_w = df["ax_world"].to_numpy(dtype=float)
        ay_w = df["ay_world"].to_numpy(dtype=float)
        az_w = df["az_world"].to_numpy(dtype=float)
        out["acc_vertical"] = az_w
        out["acc_horizontal"] = np.hypot(ax_w, ay_w)
    else:
        out["acc_vertical"] = az
        out["acc_horizontal"] = np.hypot(ax, ay)

    # Deviation from gravity.
    acc_deviation = np.abs(acc_norm - _GRAVITY)
    out["acc_deviation"] = acc_deviation

    # Lowpass / highpass splits.
    acc_lf = butter_lowpass(acc_norm, cutoff_hz=2.0, sample_rate_hz=sample_rate_hz)
    out["acc_lf"] = acc_lf
    out["acc_hf"] = acc_norm - acc_lf

    # Jerk: gradient of [ax, ay, az] scaled by sample_rate_hz.
    dax = np.gradient(ax) * sample_rate_hz
    day = np.gradient(ay) * sample_rate_hz
    daz = np.gradient(az) * sample_rate_hz
    out["jerk_norm"] = norm_xyz(dax, day, daz)

    # Gyro lowpass / highpass.
    gyro_lf = butter_lowpass(gyro_norm, cutoff_hz=1.0, sample_rate_hz=sample_rate_hz)
    out["gyro_lf"] = gyro_lf
    out["gyro_hf"] = gyro_norm - gyro_lf

    # Angular acceleration and energy.
    agx = np.gradient(gx) * sample_rate_hz
    agy = np.gradient(gy) * sample_rate_hz
    agz = np.gradient(gz) * sample_rate_hz
    alpha_norm = norm_xyz(agx, agy, agz)
    out["alpha_norm"] = alpha_norm

    # Rolling energy (RMS) over 1-second window.
    window = int(1.0 * sample_rate_hz)
    window = max(window, 1)
    out["energy_acc"] = _rolling_rms(pd.Series(acc_deviation), window).values
    out["energy_gyro"] = _rolling_rms(pd.Series(gyro_norm), window).values
    out["alpha_energy"] = _rolling_rms(pd.Series(alpha_norm), window).values

    # Orientation-aware linear acceleration (gravity removed via quaternion).
    if df_orient is not None:
        lin = _compute_linear_acc(df, df_orient)
        if lin is not None:
            out["acc_linear_x"] = lin[0]
            out["acc_linear_y"] = lin[1]
            out["acc_linear_z"] = lin[2]
            out["acc_linear_norm"] = lin[3]
            log.debug("Linear acceleration computed (%d samples).", len(df))
        else:
            log.debug("Linear acceleration skipped (insufficient orientation data).")

    return out


def compute_cross_sensor_signals(
    sporsa_df: pd.DataFrame,
    arduino_df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
) -> pd.DataFrame:
    """Compute cross-sensor disagreement/coupling signals on a common time grid.

    Both DataFrames are interpolated onto a common uniform timestamp grid
    covering the overlap of their time ranges.

    Parameters
    ----------
    sporsa_df:
        Derived signals DataFrame for the sporsa sensor (output of
        ``compute_sensor_signals``).  Must contain timestamp and derived
        signal columns.
    arduino_df:
        Derived signals DataFrame for the arduino sensor (same schema).
    sample_rate_hz:
        Nominal rate used for rolling-window sizing.

    Returns
    -------
    DataFrame with timestamp plus:

    acc_diff_norm        - |acc_norm_sporsa - acc_norm_arduino|
    gyro_diff_norm       - |gyro_norm_sporsa - gyro_norm_arduino|
    acc_correlation      - rolling Pearson r of acc_norm over 2s window
    acc_ratio            - acc_norm_arduino / (acc_norm_sporsa + 1e-6)
    vertical_diff        - acc_vertical_arduino - acc_vertical_sporsa
    disagree_score       - weighted combination: 0.6*norm(acc_diff_norm) + 0.4*norm(gyro_diff_norm)
    """
    # Build common time grid over overlap.
    t_start = max(sporsa_df["timestamp"].iloc[0], arduino_df["timestamp"].iloc[0])
    t_end = min(sporsa_df["timestamp"].iloc[-1], arduino_df["timestamp"].iloc[-1])

    if t_start >= t_end:
        log.warning("No temporal overlap between sporsa and arduino; returning empty cross-sensor DataFrame.")
        return pd.DataFrame(columns=[
            "timestamp",
            "acc_diff_norm",
            "gyro_diff_norm",
            "acc_correlation",
            "acc_ratio",
            "vertical_diff",
            "disagree_score",
            "disagree_acc_norm",
            "disagree_gyro_norm",
            "disagree_combined_heuristic",
        ])

    dt_ms = 1000.0 / sample_rate_hz
    common_ts = np.arange(t_start, t_end + dt_ms * 0.5, dt_ms)

    def _interp_series(src_df: pd.DataFrame, col: str, target_ts: np.ndarray) -> np.ndarray:
        return np.interp(target_ts, src_df["timestamp"].to_numpy(dtype=float), src_df[col].to_numpy(dtype=float))

    sporsa_acc = _interp_series(sporsa_df, "acc_norm", common_ts)
    arduino_acc = _interp_series(arduino_df, "acc_norm", common_ts)
    sporsa_gyro = _interp_series(sporsa_df, "gyro_norm", common_ts)
    arduino_gyro = _interp_series(arduino_df, "gyro_norm", common_ts)
    s_acc_dev = _interp_series(sporsa_df, "acc_deviation", common_ts)
    a_acc_dev = _interp_series(arduino_df, "acc_deviation", common_ts)
    s_gyro_hf = _interp_series(sporsa_df, "gyro_hf", common_ts) if "gyro_hf" in sporsa_df.columns else sporsa_gyro
    a_gyro_hf = _interp_series(arduino_df, "gyro_hf", common_ts) if "gyro_hf" in arduino_df.columns else arduino_gyro
    s_gyro = sporsa_gyro
    a_gyro = arduino_gyro
    sporsa_vert = _interp_series(sporsa_df, "acc_vertical", common_ts)
    arduino_vert = _interp_series(arduino_df, "acc_vertical", common_ts)

    out = pd.DataFrame()
    out["timestamp"] = common_ts

    acc_diff = np.abs(sporsa_acc - arduino_acc)
    gyro_diff = np.abs(sporsa_gyro - arduino_gyro)
    out["acc_diff_norm"] = acc_diff
    out["gyro_diff_norm"] = gyro_diff

    # Rolling Pearson correlation over 2-second window (min 20 samples).
    corr_window = max(int(2.0 * sample_rate_hz), 20)
    sporsa_acc_series = pd.Series(sporsa_acc)
    arduino_acc_series = pd.Series(arduino_acc)
    out["acc_correlation"] = sporsa_acc_series.rolling(window=corr_window, min_periods=20).corr(arduino_acc_series).values

    out["acc_ratio"] = arduino_acc / (sporsa_acc + 1e-6)
    out["vertical_diff"] = arduino_vert - sporsa_vert

    # Disagree score: normalize each diff to [0,1] range then combine.
    # Use nanmax because interpolation over sensor gaps propagates NaNs;
    # a plain .max() would poison the whole column.
    acc_diff_max = np.nanmax(acc_diff) if np.any(np.isfinite(acc_diff)) else np.nan
    gyro_diff_max = np.nanmax(gyro_diff) if np.any(np.isfinite(gyro_diff)) else np.nan
    acc_diff_norm = acc_diff / (acc_diff_max + 1e-9)
    gyro_diff_norm_scaled = gyro_diff / (gyro_diff_max + 1e-9)
    out["disagree_acc_norm"] = acc_diff_norm
    out["disagree_gyro_norm"] = gyro_diff_norm_scaled
    disagree_score = 0.6 * acc_diff_norm + 0.4 * gyro_diff_norm_scaled
    out["disagree_score"] = disagree_score
    out["disagree_combined_heuristic"] = disagree_score

    # Bounded dominance scores (+ rider/arduino dominant, - bike/sporsa dominant).
    out["acc_dominance"] = (a_acc_dev - s_acc_dev) / (a_acc_dev + s_acc_dev + _DOM_EPS)
    out["gyro_dominance"] = (a_gyro - s_gyro) / (a_gyro + s_gyro + _DOM_EPS)

    # Rolling lag-aware xcorr on transient-centric channels.
    xcorr_window = max(int(2.0 * sample_rate_hz), 4)
    max_lag_samples = max(int(0.5 * sample_rate_hz), 1)
    acc_xcorr, acc_lag = _rolling_lagged_xcorr(
        s_acc_dev, a_acc_dev, window=xcorr_window, max_lag_samples=max_lag_samples,
    )
    out["xcorr_acc_max"] = acc_xcorr
    out["xcorr_acc_lag_s"] = acc_lag / sample_rate_hz

    gyro_xcorr, gyro_lag = _rolling_lagged_xcorr(
        s_gyro_hf, a_gyro_hf, window=xcorr_window, max_lag_samples=max_lag_samples,
    )
    out["xcorr_gyro_max"] = gyro_xcorr
    out["xcorr_gyro_lag_s"] = gyro_lag / sample_rate_hz

    return out
