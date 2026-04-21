"""Batch AHRS orientation filter interface.

All four filters (Madgwick, Mahony, Complementary, EKF) share a single
entry point ``run_filter()`` that returns a (N, 4) quaternion array.

The ahrs library is used for the batch Madgwick and Mahony paths when
available; all four fall back to the per-sample numpy implementations in
their respective modules (madgwick.py, mahony.py, complementary.py, ekf.py).

Conventions
-----------
- Quaternion: [w, x, y, z], body→world rotation
- Accelerometer: m/s², at rest measures [0, 0, ~9.81] (Z-up / ENU)
- Gyroscope: rad/s
- Magnetometer: any consistent unit (normalised internally)
- Magnetometer NaN values are forward-filled before processing to handle
  variable-rate sensors (e.g. Arduino 10 Hz mag at 100 Hz IMU rate).
"""

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

METHODS: tuple[str, ...] = ("madgwick", "mahony", "complementary", "ekf")

DEFAULT_PARAMS: dict[str, dict] = {
    "madgwick":      {"beta": 0.1},
    "mahony":        {"Kp": 2.0, "Ki": 0.005},
    "complementary": {"alpha": 0.98},
    "ekf":           {"sigma_gyro": 0.01, "sigma_acc": 0.1, "sigma_mag": 0.1},
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _remove_imu_outliers(
    acc: np.ndarray,
    gyro: np.ndarray,
    method: str = "iqr",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect and forward-fill outliers in IMU data.

    Parameters
    ----------
    acc:
        Accelerometer array (N, 3) in m/s².
    gyro:
        Gyroscope array (N, 3) in rad/s.
    method:
        ``"iqr"``: Interquartile range method (fast, robust).

    Returns
    -------
    (acc_clean, gyro_clean, outlier_mask)
        Clean arrays with outliers forward-filled.
        outlier_mask[i] = True where sample i was an outlier.
    """
    N = len(acc)
    outlier_mask = np.zeros(N, dtype=bool)

    # Acceleration norm analysis
    acc_norms = np.linalg.norm(acc, axis=1)
    acc_valid = np.isfinite(acc_norms)

    if acc_valid.sum() > 0:
        valid_norms = acc_norms[acc_valid]
        Q1, Q3 = np.percentile(valid_norms, [25, 75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        acc_outliers = (acc_norms < lower) | (acc_norms > upper) | ~acc_valid
        outlier_mask |= acc_outliers

    # Gyro norm analysis
    gyro_norms = np.linalg.norm(gyro, axis=1)
    gyro_valid = np.isfinite(gyro_norms)

    if gyro_valid.sum() > 0:
        valid_norms = gyro_norms[gyro_valid]
        if len(valid_norms) > 4:
            Q1, Q3 = np.percentile(valid_norms, [25, 75])
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            gyro_outliers = (gyro_norms < lower) | (gyro_norms > upper) | ~gyro_valid
            outlier_mask |= gyro_outliers

    # Forward-fill outliers
    acc_clean = acc.copy()
    gyro_clean = gyro.copy()

    if outlier_mask.any():
        for i in np.where(outlier_mask)[0]:
            # Find last valid sample
            valid_before = np.where(~outlier_mask[:i])[0]
            if len(valid_before) > 0:
                last_valid = valid_before[-1]
                acc_clean[i] = acc_clean[last_valid]
                gyro_clean[i] = gyro_clean[last_valid]

    return acc_clean, gyro_clean, outlier_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_filter(
    method: str,
    acc: np.ndarray,
    gyro_rad: np.ndarray,
    dt_arr: np.ndarray,
    *,
    q0: np.ndarray | None = None,
    mag: np.ndarray | None = None,
    remove_outliers: bool = True,
    **params,
) -> np.ndarray:
    """Run an AHRS orientation filter over a calibrated IMU section.

    Parameters
    ----------
    method:
        One of ``"madgwick"``, ``"mahony"``, ``"complementary"``, ``"ekf"``.
    acc:
        Accelerometer array (N, 3) in m/s².
    gyro_rad:
        Gyroscope array (N, 3) in rad/s.
    dt_arr:
        Per-sample time steps (N,) in seconds.
    q0:
        Initial quaternion [w, x, y, z].  When ``None``, estimated from the
        first accelerometer sample (tilt-only, yaw = 0).
    mag:
        Magnetometer array (N, 3) or ``None``.  NaN rows are forward-filled
        so that variable-rate sensors (e.g. 10 Hz) work at 100 Hz IMU rate.
        Ignored by the complementary filter.
    remove_outliers:
        If ``True`` (default), detect and forward-fill outliers before filtering.
    **params:
        Filter-specific parameter overrides (merged with ``DEFAULT_PARAMS``).

    Returns
    -------
    np.ndarray of shape (N, 4)
        Per-sample quaternions [w, x, y, z].
    """
    N = len(acc)
    if N == 0:
        return np.zeros((0, 4))

    # Forward-fill NaN rows from interleaved sensor streams before any processing.
    acc, gyro_rad = _ffill_imu(acc, gyro_rad)

    # Preprocess: remove outliers
    if remove_outliers and N > 4:
        acc, gyro_rad, _ = _remove_imu_outliers(acc, gyro_rad)

    if q0 is None:
        q0 = tilt_quat_from_acc(acc[0])
    q0 = quat_normalize(np.asarray(q0, dtype=float))

    merged = {**DEFAULT_PARAMS.get(method, {}), **params}

    # Forward-fill NaN rows in magnetometer for variable-rate sensors.
    mag_filled: np.ndarray | None = None
    if mag is not None:
        mag_filled = _ffill_mag(np.asarray(mag, dtype=float))
        if not np.any(np.all(np.isfinite(mag_filled), axis=1)):
            mag_filled = None  # no valid readings at all

    # ahrs batch interface for Madgwick and Mahony (faster, validated).
    if _HAS_AHRS and method in ("madgwick", "mahony"):
        try:
            return _run_ahrs_batch(method, acc, gyro_rad, dt_arr, q0, mag_filled, merged)
        except Exception as exc:
            log.warning("ahrs %s failed (%s) — falling back to numpy", method, exc)

    return _run_numpy(method, acc, gyro_rad, dt_arr, q0, mag_filled, merged)


# ---------------------------------------------------------------------------
# ahrs batch runner
# ---------------------------------------------------------------------------


def _run_ahrs_batch(
    method: str,
    acc: np.ndarray,
    gyro_rad: np.ndarray,
    dt_arr: np.ndarray,
    q0: np.ndarray,
    mag: np.ndarray | None,
    params: dict,
) -> np.ndarray:
    freq = 1.0 / float(np.median(dt_arr))

    if method == "madgwick":
        kwargs: dict = dict(
            gyr=gyro_rad, acc=acc, frequency=freq,
            q0=q0, beta=params.get("beta", 0.1),
        )
        if mag is not None:
            kwargs["mag"] = mag
        Q = _ahrs.filters.Madgwick(**kwargs).Q

    elif method == "mahony":
        kwargs = dict(
            gyr=gyro_rad, acc=acc, frequency=freq,
            q0=q0, k_P=params.get("Kp", 2.0), k_I=params.get("Ki", 0.005),
        )
        if mag is not None:
            kwargs["mag"] = mag
        Q = _ahrs.filters.Mahony(**kwargs).Q

    else:
        raise ValueError(f"ahrs batch not supported for method '{method}'")

    # ahrs returns (N, 4) with [w, x, y, z] convention — matches ours.
    return np.asarray(Q, dtype=float)


# ---------------------------------------------------------------------------
# Numpy sample-by-sample runner
# ---------------------------------------------------------------------------


def _run_numpy(
    method: str,
    acc: np.ndarray,
    gyro_rad: np.ndarray,
    dt_arr: np.ndarray,
    q0: np.ndarray,
    mag: np.ndarray | None,
    params: dict,
) -> np.ndarray:
    sr_hz = 1.0 / float(np.median(dt_arr))
    filt = _build_filter(method, params, sr_hz)
    filt.reset(q0)

    N = len(acc)
    Q = np.zeros((N, 4))
    pending_dt = 0.0

    for i in range(N):
        dt_i = float(dt_arr[i])
        a = acc[i]
        g = gyro_rad[i]
        m = mag[i] if mag is not None else None

        if not np.all(np.isfinite(g)):
            Q[i] = filt.current_quaternion()
            pending_dt += dt_i
            continue

        pending_dt += dt_i
        mag_i = m if (m is not None and np.all(np.isfinite(m))) else None
        Q[i] = filt.update(a, g, mag=mag_i, dt=pending_dt)
        pending_dt = 0.0

    return Q


def _build_filter(method: str, params: dict, sample_rate_hz: float):
    if method == "madgwick":
        from .madgwick import MadgwickFilter
        return MadgwickFilter(beta=params.get("beta", 0.1), sample_rate_hz=sample_rate_hz)
    if method == "mahony":
        from .mahony import MahonyFilter
        return MahonyFilter(Kp=params.get("Kp", 2.0), Ki=params.get("Ki", 0.005), sample_rate_hz=sample_rate_hz)
    if method == "complementary":
        from .complementary import ComplementaryFilter
        return ComplementaryFilter(alpha=params.get("alpha", 0.98), sample_rate_hz=sample_rate_hz)
    if method == "ekf":
        from .ekf import EKFFilter
        return EKFFilter(
            sigma_gyro=params.get("sigma_gyro", 0.01),
            sigma_acc=params.get("sigma_acc", 0.1),
            sigma_mag=params.get("sigma_mag", 0.1),
            sample_rate_hz=sample_rate_hz,
        )
    raise ValueError(f"Unknown orientation method: '{method}'")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ffill_mag(mag: np.ndarray) -> np.ndarray:
    """Forward-fill NaN rows (variable-rate magnetometer → uniform array)."""
    out = mag.copy()
    last: np.ndarray | None = None
    for i in range(len(out)):
        if np.all(np.isfinite(out[i])):
            last = out[i].copy()
        elif last is not None:
            out[i] = last
    return out


def _ffill_imu(
    acc: np.ndarray,
    gyro: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-fill NaN rows in acc and gyro independently.

    Handles sensors where acc and gyro are sampled at different rates and
    interleaved, leaving NaN in the non-updating channel each row.
    """
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
