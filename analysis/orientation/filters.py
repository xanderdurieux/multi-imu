"""Filters helpers for estimate sensor orientation for calibrated section data."""

from __future__ import annotations

import logging

import numpy as np

from common.quaternion import quat_normalize, tilt_quat_from_acc

log = logging.getLogger(__name__)

try:
    import ahrs as _ahrs
    _HAS_AHRS = True
    log.debug("ahrs %s available", getattr(_ahrs, "__version__", "unknown"))
except ImportError:
    _HAS_AHRS = False

_G = 9.81
_DEFAULT_KP = 2.0
_DEFAULT_KI = 0.005


def run_mahony(
    acc: np.ndarray,
    gyro_rad: np.ndarray,
    dt_arr: np.ndarray,
    *,
    q0: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    Kp: float = _DEFAULT_KP,
    Ki: float = _DEFAULT_KI,
) -> np.ndarray:
    """Run mahony."""
    N = len(acc)
    if N == 0:
        return np.zeros((0, 4))

    acc, gyro_rad = _ffill_imu(acc, gyro_rad)

    if q0 is None:
        q0 = tilt_quat_from_acc(acc[0])
    q0 = quat_normalize(np.asarray(q0, dtype=float))

    mag_filled: np.ndarray | None = None
    if mag is not None:
        mag_filled = _ffill_mag(np.asarray(mag, dtype=float))
        if not np.any(np.all(np.isfinite(mag_filled), axis=1)):
            mag_filled = None

    if _HAS_AHRS:
        try:
            return _run_ahrs(acc, gyro_rad, dt_arr, q0, mag_filled, Kp, Ki)
        except Exception as exc:
            log.warning("ahrs Mahony failed (%s) — falling back to numpy", exc)

    return _run_numpy(acc, gyro_rad, dt_arr, q0, mag_filled, Kp, Ki)


def _run_ahrs(
    acc: np.ndarray,
    gyro_rad: np.ndarray,
    dt_arr: np.ndarray,
    q0: np.ndarray,
    mag: np.ndarray | None,
    Kp: float,
    Ki: float,
) -> np.ndarray:
    """Run ahrs."""
    freq = 1.0 / float(np.median(dt_arr))
    kwargs: dict = dict(gyr=gyro_rad, acc=acc, frequency=freq, q0=q0, k_P=Kp, k_I=Ki)
    if mag is not None:
        kwargs["mag"] = mag
    Q = np.asarray(_ahrs.filters.Mahony(**kwargs).Q, dtype=float)
    if not np.all(np.isfinite(Q)):
        raise ValueError("ahrs Mahony produced non-finite quaternions")
    return Q


def _run_numpy(
    acc: np.ndarray,
    gyro_rad: np.ndarray,
    dt_arr: np.ndarray,
    q0: np.ndarray,
    mag: np.ndarray | None,
    Kp: float,
    Ki: float,
) -> np.ndarray:
    """Run numpy."""
    from .mahony import MahonyFilter
    sr_hz = 1.0 / float(np.median(dt_arr))
    filt = MahonyFilter(Kp=Kp, Ki=Ki, sample_rate_hz=sr_hz)
    filt.reset(q0)

    N = len(acc)
    Q = np.zeros((N, 4))
    pending_dt = 0.0

    for i in range(N):
        dt_i = float(dt_arr[i])
        g = gyro_rad[i]
        if not np.all(np.isfinite(g)):
            Q[i] = filt.current_quaternion()
            pending_dt += dt_i
            continue
        pending_dt += dt_i
        m = mag[i] if mag is not None else None
        mag_i = m if (m is not None and np.all(np.isfinite(m))) else None
        Q[i] = filt.update(acc[i], g, mag=mag_i, dt=pending_dt)
        pending_dt = 0.0

    return Q


def _ffill_mag(mag: np.ndarray) -> np.ndarray:
    """Return ffill mag."""
    out = mag.copy()
    first_valid: np.ndarray | None = None
    last: np.ndarray | None = None
    for i in range(len(out)):
        if np.all(np.isfinite(out[i])):
            if first_valid is None:
                first_valid = out[i].copy()
            last = out[i].copy()
        elif last is not None:
            out[i] = last
    if first_valid is not None:
        for i in range(len(out)):
            if not np.all(np.isfinite(out[i])):
                out[i] = first_valid
            else:
                break
    return out


def _ffill_imu(acc: np.ndarray, gyro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill NaN rows in acc and gyro (interleaved sensor streams)."""
    acc_out = acc.copy()
    gyro_out = gyro.copy()
    last_acc: np.ndarray | None = None
    last_gyro: np.ndarray | None = None
    for i in range(len(acc_out)):
        if np.all(np.isfinite(acc_out[i])):
            last_acc = acc_out[i].copy()
        elif last_acc is not None:
            acc_out[i] = last_acc
        if np.all(np.isfinite(gyro_out[i])):
            last_gyro = gyro_out[i].copy()
        elif last_gyro is not None:
            gyro_out[i] = last_gyro
    return acc_out, gyro_out
