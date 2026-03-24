"""Complementary filter for orientation estimation.

Dropout handling:
- Gyro NaN rows do not update orientation.
- When gyro resumes, integration uses elapsed time since the previous valid gyro sample
  (provided via per-sample dt array from estimate.py).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from common.quaternion import (
    quat_from_gyro,
    quat_multiply,
    quat_normalize,
    quat_from_rotation_matrix,
)


def _as_dt_array(dt: float | np.ndarray, n: int) -> np.ndarray:
    if np.isscalar(dt):
        return np.full(n, float(dt), dtype=float)
    arr = np.asarray(dt, dtype=float).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"dt length mismatch: expected {n}, got {len(arr)}")
    return arr


def _tilt_quat_from_acc(acc: np.ndarray) -> np.ndarray:
    a = np.asarray(acc, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    a = a / n
    rot, _ = Rotation.align_vectors([[0, 0, 1]], [a])
    return quat_from_rotation_matrix(rot.as_matrix())


def complementary_orientation(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float | np.ndarray,
    alpha: float = 0.98,
) -> np.ndarray:
    n = len(acc)
    dt_arr = _as_dt_array(dt, n)
    out = np.empty((n, 4))

    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    for i in range(n):
        if np.all(np.isfinite(acc[i])) and np.linalg.norm(acc[i]) > 1e-6:
            q = _tilt_quat_from_acc(acc[i])
            break

    for i in range(n):
        g = np.asarray(gyro[i], dtype=float)
        a = np.asarray(acc[i], dtype=float)
        dti = float(dt_arr[i]) if np.isfinite(dt_arr[i]) else 0.0

        if np.all(np.isfinite(g)) and dti > 0.0:
            dq = quat_from_gyro(g, dti)
            q_gyro = quat_multiply(q, dq)
        else:
            q_gyro = q.copy()

        if np.all(np.isfinite(a)) and np.linalg.norm(a) > 1e-6:
            q_acc = _tilt_quat_from_acc(a)
            if np.dot(q_gyro, q_acc) < 0:
                q_acc = -q_acc
            q_blend = alpha * q_gyro + (1.0 - alpha) * q_acc
        else:
            q_blend = q_gyro

        q = quat_normalize(q_blend)
        out[i] = q

    return out
