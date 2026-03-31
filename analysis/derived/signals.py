"""Derived signal computation from calibrated IMU data.

Functions operate on a pandas DataFrame with IMU columns:
    timestamp, ax, ay, az, gx, gy, gz, [ax_world, ay_world, az_world, gx_world, gy_world, gz_world]

Timestamps in milliseconds, accelerometer in m/s², gyro in degrees/s.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

log = logging.getLogger(__name__)

_GRAVITY = 9.81
_MIN_SAMPLES_BUTTER = 3 * (4 + 1)  # 3 * (order + 1) for 4th-order filter


def _butter_lowpass(data: np.ndarray, cutoff_hz: float, sample_rate_hz: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth lowpass filter. Falls back to identity if too few samples."""
    if len(data) < _MIN_SAMPLES_BUTTER:
        log.debug(
            "Too few samples (%d) for Butterworth filter (need >= %d); returning original signal.",
            len(data),
            _MIN_SAMPLES_BUTTER,
        )
        return data.copy()
    nyq = 0.5 * sample_rate_hz
    normal_cutoff = cutoff_hz / nyq
    # Clamp to avoid ValueError from scipy when cutoff >= nyquist.
    normal_cutoff = min(normal_cutoff, 0.9999)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def _rolling_rms(series: pd.Series, window: int) -> pd.Series:
    """Rolling root-mean-square with min_periods=1."""
    return series.pow(2).rolling(window=window, min_periods=1).mean().pow(0.5)


def compute_sensor_signals(df: pd.DataFrame, *, sample_rate_hz: float = 100.0) -> pd.DataFrame:
    """Compute derived signals for one sensor.

    Parameters
    ----------
    df:
        DataFrame with at minimum columns: timestamp, ax, ay, az, gx, gy, gz.
        Optionally: ax_world, ay_world, az_world.
    sample_rate_hz:
        Nominal sampling rate used for filter design and jerk scaling.

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
    acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)
    out["acc_norm"] = acc_norm
    out["gyro_norm"] = gyro_norm

    # World-frame columns (optional).
    has_world = all(c in df.columns for c in ("ax_world", "ay_world", "az_world"))
    if has_world:
        ax_w = df["ax_world"].to_numpy(dtype=float)
        ay_w = df["ay_world"].to_numpy(dtype=float)
        az_w = df["az_world"].to_numpy(dtype=float)
        out["acc_vertical"] = az_w
        out["acc_horizontal"] = np.sqrt(ax_w**2 + ay_w**2)
    else:
        out["acc_vertical"] = az
        out["acc_horizontal"] = np.sqrt(ax**2 + ay**2)

    # Deviation from gravity.
    acc_deviation = np.abs(acc_norm - _GRAVITY)
    out["acc_deviation"] = acc_deviation

    # Lowpass / highpass splits.
    acc_lf = _butter_lowpass(acc_norm, cutoff_hz=2.0, sample_rate_hz=sample_rate_hz)
    out["acc_lf"] = acc_lf
    out["acc_hf"] = acc_norm - acc_lf

    # Jerk: gradient of [ax, ay, az] scaled by sample_rate_hz.
    dax = np.gradient(ax) * sample_rate_hz
    day = np.gradient(ay) * sample_rate_hz
    daz = np.gradient(az) * sample_rate_hz
    out["jerk_norm"] = np.sqrt(dax**2 + day**2 + daz**2)

    # Gyro lowpass / highpass.
    gyro_lf = _butter_lowpass(gyro_norm, cutoff_hz=1.0, sample_rate_hz=sample_rate_hz)
    out["gyro_lf"] = gyro_lf
    out["gyro_hf"] = gyro_norm - gyro_lf

    # Rolling energy (RMS) over 1-second window.
    window = int(1.0 * sample_rate_hz)
    window = max(window, 1)
    out["energy_acc"] = _rolling_rms(pd.Series(acc_deviation), window).values
    out["energy_gyro"] = _rolling_rms(pd.Series(gyro_norm), window).values

    return out


def compute_cross_sensor_signals(
    sporsa_df: pd.DataFrame,
    arduino_df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
) -> pd.DataFrame:
    """Compute cross-sensor disagreement signals on a common time grid.

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
        ])

    dt_ms = 1000.0 / sample_rate_hz
    common_ts = np.arange(t_start, t_end + dt_ms * 0.5, dt_ms)

    def _interp_series(src_df: pd.DataFrame, col: str, target_ts: np.ndarray) -> np.ndarray:
        return np.interp(target_ts, src_df["timestamp"].to_numpy(dtype=float), src_df[col].to_numpy(dtype=float))

    sporsa_acc = _interp_series(sporsa_df, "acc_norm", common_ts)
    arduino_acc = _interp_series(arduino_df, "acc_norm", common_ts)
    sporsa_gyro = _interp_series(sporsa_df, "gyro_norm", common_ts)
    arduino_gyro = _interp_series(arduino_df, "gyro_norm", common_ts)
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
    acc_diff_max = acc_diff.max()
    gyro_diff_max = gyro_diff.max()
    acc_diff_norm = acc_diff / (acc_diff_max + 1e-9)
    gyro_diff_norm_scaled = gyro_diff / (gyro_diff_max + 1e-9)
    out["disagree_score"] = 0.6 * acc_diff_norm + 0.4 * gyro_diff_norm_scaled

    return out
