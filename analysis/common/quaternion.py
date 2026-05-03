"""Quaternion math helpers for orientation estimation."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np

log = logging.getLogger(__name__)


def quat_identity() -> np.ndarray:
    """Return the identity quaternion (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit quaternion."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < eps:
        log.debug("quat_normalize: degenerate quaternion (|q|=%.3e); resetting to identity.", n)
        return quat_identity()
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of quaternion ``q``."""
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([w, -x, -y, -z], dtype=float)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Return the Hamilton product of two quaternions."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z], dtype=float)


def quat_from_axis_angle(axis: Iterable[float], angle_rad: float) -> np.ndarray:
    """Create quaternion from rotation axis and angle (in radians)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0.0 or angle_rad == 0.0:
        return quat_identity()

    axis = axis / norm
    half = 0.5 * angle_rad
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def quat_from_gyro(omega_body_rad: Iterable[float], dt: float) -> np.ndarray:
    """Return a small rotation quaternion from gyro data."""
    omega = np.asarray(omega_body_rad, dtype=float)
    angle = np.linalg.norm(omega) * dt
    if angle == 0.0:
        return quat_identity()
    axis = omega / np.linalg.norm(omega)
    return quat_from_axis_angle(axis, angle)


def quat_rotate(q_bw: np.ndarray, v_body: Iterable[float]) -> np.ndarray:
    """Rotate a 3D vector from body frame into world frame."""
    q = quat_normalize(q_bw)
    v = np.asarray(v_body, dtype=float)
    v_quat = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    q_conj = quat_conjugate(q)
    rotated = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return rotated[1:]


def quat_rotate_batch(Q: np.ndarray, V_body: np.ndarray) -> np.ndarray:
    """Rotate body-frame vectors into the world frame."""
    Q = np.asarray(Q, dtype=float)
    V = np.asarray(V_body, dtype=float)
    if Q.ndim != 2 or Q.shape[1] != 4:
        raise ValueError(f"Q must have shape (N, 4), got {Q.shape}")
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"V_body must have shape (N, 3), got {V.shape}")
    if len(Q) != len(V):
        raise ValueError(f"length mismatch: Q={len(Q)} V={len(V)}")

    qw, qx, qy, qz = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    x, y, z = V[:, 0], V[:, 1], V[:, 2]

    # R(q) @ v written out per element; identical to quat_rotate but vectorised.
    out = np.empty_like(V)
    out[:, 0] = (1 - 2*(qy**2 + qz**2)) * x + 2*(qx*qy - qw*qz) * y + 2*(qx*qz + qw*qy) * z
    out[:, 1] = 2*(qx*qy + qw*qz) * x + (1 - 2*(qx**2 + qz**2)) * y + 2*(qy*qz - qw*qx) * z
    out[:, 2] = 2*(qx*qz - qw*qy) * x + 2*(qy*qz + qw*qx) * y + (1 - 2*(qx**2 + qy**2)) * z
    return out


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two quaternions."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))

    # Ensure shortest path.
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        # Nearly colinear - use linear interpolation.
        q = q0 + t * (q1 - q0)
        return quat_normalize(q)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)

    theta = theta_0 * t
    sin_theta = np.sin(theta)

    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return q0 * s0 + q1 * s1


def euler_from_quat(q_bw: np.ndarray) -> Tuple[float, float, float]:
    """Return yaw, pitch, and roll from a quaternion."""
    q = quat_normalize(q_bw)
    w, x, y, z = q

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    return yaw, pitch, roll


def quat_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Create a body-to-world quaternion from yaw, pitch, and roll."""
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return quat_normalize(np.array([w, x, y, z], dtype=float))


def quat_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a quaternion."""
    R = np.asarray(R, dtype=float)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z], dtype=float))


def tilt_quat_from_acc(acc_body: Iterable[float], g: float = 9.81) -> np.ndarray:
    """Estimate tilt from accelerometer data."""
    ax, ay, az = np.asarray(acc_body, dtype=float)
    # Avoid division by zero; fall back to identity if completely invalid.
    if np.allclose([ax, ay, az], 0.0):
        return quat_identity()

    # Normalize to unit vector.
    norm = np.linalg.norm([ax, ay, az])
    if norm == 0.0:
        return quat_identity()
    ax /= norm
    ay /= norm
    az /= norm

    # Gravity expected to point along negative world Z.
    # Compute roll and pitch that align measured gravity to world -Z.
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
    yaw = 0.0
    return quat_from_euler(yaw, pitch, roll)

