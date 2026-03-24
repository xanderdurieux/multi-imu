"""Madgwick orientation filter — from-scratch implementation.

Implements the gradient-descent orientation filter from:
  Madgwick, S.O.H. (2010). An efficient orientation filter for inertial and
  inertial/magnetic sensor arrays. Technical report, University of Bristol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from common.quaternion import (
    quat_conjugate,
    quat_from_rotation_matrix,
    quat_multiply,
    quat_normalize,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

GRAVITY_REF = np.array([0.0, 0.0, 1.0])


def _as_dt_array(dt: float | np.ndarray, n: int) -> np.ndarray:
    if np.isscalar(dt):
        return np.full(n, float(dt), dtype=float)
    arr = np.asarray(dt, dtype=float).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"dt length mismatch: expected {n}, got {len(arr)}")
    return arr


def _f_g(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    ax, ay, az = a[0], a[1], a[2]
    return np.array([
        2.0 * (x * z - w * y) - ax,
        2.0 * (w * x + y * z) - ay,
        2.0 * (0.5 - x * x - y * y) - az,
    ])


def _J_g(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [-2 * y, 2 * z, -2 * w, 2 * x],
        [2 * x, 2 * w, 2 * z, 2 * y],
        [0, -4 * x, -4 * y, 0],
    ])


def _gradient_f(q: np.ndarray, a: np.ndarray) -> np.ndarray:
    return _J_g(q).T @ _f_g(q, a)


def _init_q_from_acc(acc: np.ndarray) -> np.ndarray:
    for a in acc:
        n = np.linalg.norm(a)
        if n > 1e-6 and np.all(np.isfinite(a)):
            rot, _ = Rotation.align_vectors([[0, 0, 1]], [a / n])
            q_bw = quat_from_rotation_matrix(rot.as_matrix())
            return quat_conjugate(q_bw)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def madgwick_acc_only(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float | np.ndarray,
    beta: float = 0.033,
) -> np.ndarray:
    n = len(acc)
    dt_arr = _as_dt_array(dt, n)
    q = _init_q_from_acc(acc)
    out = np.empty((n, 4))
    out_raw = q.copy()

    for i in range(n):
        a = np.asarray(acc[i], dtype=float)
        g = np.asarray(gyro[i], dtype=float)
        dti = float(dt_arr[i]) if np.isfinite(dt_arr[i]) else 0.0

        # Gyro integration only when gyro is valid and dt positive
        if np.all(np.isfinite(g)) and dti > 0.0:
            omega_quat = np.array([0.0, g[0], g[1], g[2]], dtype=float)
            q_dot_omega = 0.5 * quat_multiply(q, omega_quat)
        else:
            q_dot_omega = np.zeros(4, dtype=float)
            dti = 0.0

        # Acc correction only when acc is valid
        if np.all(np.isfinite(a)) and np.linalg.norm(a) > 1e-6:
            a_norm = a / np.linalg.norm(a)
            grad = _gradient_f(q, a_norm)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                q_dot_err = grad / grad_norm
                q = q + (q_dot_omega - beta * q_dot_err) * dti
            else:
                q = q + q_dot_omega * dti
        else:
            q = q + q_dot_omega * dti

        q = quat_normalize(q)
        if i > 0 and np.dot(q, out_raw) < 0:
            q = -q
        out_raw = q.copy()
        out[i] = quat_conjugate(q)

    return out


def madgwick_9dof(
    acc: np.ndarray,
    gyro: np.ndarray,
    mag: np.ndarray,
    dt: float | np.ndarray,
    beta: float = 0.033,
) -> np.ndarray:
    # Current implementation keeps same correction term as acc-only for stability.
    n = len(acc)
    dt_arr = _as_dt_array(dt, n)
    q = _init_q_from_acc(acc)
    out = np.empty((n, 4))
    out_raw = q.copy()

    for i in range(n):
        a = np.asarray(acc[i], dtype=float)
        g = np.asarray(gyro[i], dtype=float)
        dti = float(dt_arr[i]) if np.isfinite(dt_arr[i]) else 0.0

        if np.all(np.isfinite(g)) and dti > 0.0:
            omega_quat = np.array([0.0, g[0], g[1], g[2]], dtype=float)
            q_dot_omega = 0.5 * quat_multiply(q, omega_quat)
        else:
            q_dot_omega = np.zeros(4, dtype=float)
            dti = 0.0

        if np.all(np.isfinite(a)) and np.linalg.norm(a) > 1e-6:
            a_norm = a / np.linalg.norm(a)
            grad = _gradient_f(q, a_norm)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                q_dot_err = grad / grad_norm
                q = q + (q_dot_omega - beta * q_dot_err) * dti
            else:
                q = q + q_dot_omega * dti
        else:
            q = q + q_dot_omega * dti

        q = quat_normalize(q)
        if i > 0 and np.dot(q, out_raw) < 0:
            q = -q
        out_raw = q.copy()
        out[i] = quat_conjugate(q)

    return out
