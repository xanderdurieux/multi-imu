"""Quaternion EKF orientation estimator (gyro prediction + accelerometer update).

This is a lightweight 6-DoF EKF:
- State: body->world quaternion [w, x, y, z]
- Prediction: gyro integration
- Measurement: normalized accelerometer direction
"""

from __future__ import annotations

import numpy as np

from common.quaternion import (
    quat_conjugate,
    quat_from_gyro,
    quat_multiply,
    quat_normalize,
    quat_rotate,
)


def _as_dt_array(dt: float | np.ndarray, n: int) -> np.ndarray:
    if np.isscalar(dt):
        return np.full(n, float(dt), dtype=float)
    arr = np.asarray(dt, dtype=float).reshape(-1)
    if len(arr) != n:
        raise ValueError(f"dt length mismatch: expected {n}, got {len(arr)}")
    return arr


def _init_q_from_acc(acc: np.ndarray) -> np.ndarray:
    for a in acc:
        if not np.all(np.isfinite(a)):
            continue
        n = np.linalg.norm(a)
        if n <= 1e-6:
            continue
        a_n = a / n
        # Align world +Z to measured gravity direction in body frame.
        z_world_in_body = a_n
        z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        v = np.cross(z_ref, z_world_in_body)
        c = float(np.dot(z_ref, z_world_in_body))
        s = np.linalg.norm(v)
        if s < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if c > 0 else np.array(
                [0.0, 1.0, 0.0, 0.0],
                dtype=float,
            )
        axis = v / s
        angle = np.arctan2(s, c)
        half = 0.5 * angle
        return quat_normalize(np.array([np.cos(half), *(axis * np.sin(half))], dtype=float))
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _h_acc(q_bw: np.ndarray) -> np.ndarray:
    """Expected normalized body-frame gravity direction from quaternion."""
    g_world = np.array([0.0, 0.0, 1.0], dtype=float)
    return quat_rotate(quat_conjugate(q_bw), g_world)


def _jacobian_numeric(q_bw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian dh/dq for h(q)=expected normalized gravity in body frame."""
    H = np.zeros((3, 4), dtype=float)
    h0 = _h_acc(q_bw)
    for j in range(4):
        dq = np.zeros(4, dtype=float)
        dq[j] = eps
        qp = quat_normalize(q_bw + dq)
        hm = h0
        hp = _h_acc(qp)
        H[:, j] = (hp - hm) / eps
    return H


def ekf_orientation(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float | np.ndarray,
    *,
    process_noise: float = 2e-3,
    measurement_noise: float = 2e-2,
) -> np.ndarray:
    """Estimate body->world quaternions using a quaternion-state EKF."""
    n = len(acc)
    dt_arr = _as_dt_array(dt, n)
    out = np.empty((n, 4), dtype=float)

    q = _init_q_from_acc(acc)
    P = np.eye(4, dtype=float) * 1e-2
    Q = np.eye(4, dtype=float) * float(process_noise)
    R = np.eye(3, dtype=float) * float(measurement_noise)

    for i in range(n):
        g = np.asarray(gyro[i], dtype=float)
        a = np.asarray(acc[i], dtype=float)
        dti = float(dt_arr[i]) if np.isfinite(dt_arr[i]) else 0.0

        # Prediction (gyro integration).
        if np.all(np.isfinite(g)) and dti > 0.0:
            dq = quat_from_gyro(g, dti)
            q = quat_normalize(quat_multiply(q, dq))

        P = P + Q * max(dti, 1e-3)

        # Update (accelerometer direction).
        if np.all(np.isfinite(a)) and np.linalg.norm(a) > 1e-6:
            z = a / np.linalg.norm(a)
            h = _h_acc(q)
            H = _jacobian_numeric(q)
            y = z - h
            S = H @ P @ H.T + R
            try:
                K = P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P @ H.T @ np.linalg.pinv(S)
            dq_state = K @ y
            q = quat_normalize(q + dq_state)
            I = np.eye(4, dtype=float)
            P = (I - K @ H) @ P

        # Keep quaternion sign continuous for plotting/readability.
        if i > 0 and np.dot(q, out[i - 1]) < 0:
            q = -q
        out[i] = q

    return out
