"""Quaternion utilities for IMU orientation estimation.

Quaternions are represented as NumPy arrays of shape (4,) in ``[w, x, y, z]`` order.

Convention
----------
All quaternions in this module represent the rotation that maps vectors from the
sensor/body frame into the world frame:

    v_world = quat_rotate(q_bw, v_body)

This is convenient because it lets you take raw accelerometer readings in the
sensor frame and rotate them into a common world frame using the same
orientation estimate used everywhere else.
"""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import numpy as np

log = logging.getLogger(__name__)


def quat_identity() -> np.ndarray:
    """Return the identity quaternion (no rotation)."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def quat_normalize(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return ``q`` normalized to unit length.

    Falls back to identity (and logs at DEBUG) when the input norm is below
    ``eps`` — this happens with underflow in orientation filter loops and
    is worth surfacing when diagnosing divergence.
    """
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
    """Hamilton product ``q = q1 ⊗ q2``."""
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
    """Small rotation quaternion from body-frame angular velocity over ``dt``.

    Parameters
    ----------
    omega_body_rad:
        Angular velocity vector in body frame ``[wx, wy, wz]`` in rad/s.
    dt:
        Time step in seconds.
    """
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


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions.

    Parameters
    ----------
    q0, q1:
        End-point quaternions (will be normalized internally).
    t:
        Interpolation parameter in ``[0, 1]``.
    """
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
    """Return yaw, pitch, roll (Z-Y-X) from body→world quaternion.

    Angles are in radians, using the aerospace convention:
    - yaw   ψ: rotation around world Z
    - pitch θ: rotation around world Y
    - roll  φ: rotation around world X
    """
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
    """Create body→world quaternion from yaw, pitch, roll (Z-Y-X) in radians."""
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
    """Convert a 3×3 rotation matrix to a unit quaternion [w, x, y, z].

    Uses Shepperd's method for numerical stability.  The input matrix must be
    a proper rotation matrix (det ≈ +1, R.T @ R ≈ I).

    Parameters
    ----------
    R:
        3×3 rotation matrix (sensor→world or body→world convention).

    Returns
    -------
    np.ndarray
        Unit quaternion ``[w, x, y, z]`` representing the same rotation.
    """
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
    """Estimate roll and pitch from accelerometer, assuming it measures gravity.

    Heading (yaw) is set to zero in the world frame by construction.
    """
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

