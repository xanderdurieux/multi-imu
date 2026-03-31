"""Calibration estimation and application for a single IMU stream.

Conventions
-----------
- World frame: Z points up (against gravity), X points forward, Y points left.
- Gravity in world frame: [0, 0, -g] (pointing down in the real world, so
  accelerometer at rest measures [0, 0, +g] in world frame).
- All timestamps in milliseconds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from common.calibration_segments import CalibrationSegment, find_calibration_segments
from common.quaternion import (
    quat_from_rotation_matrix,
    quat_normalize,
    quat_rotate,
    tilt_quat_from_acc,
    euler_from_quat,
)

log = logging.getLogger(__name__)

_G = 9.81  # m/s²


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CalibrationParams:
    """Per-sensor calibration parameters."""

    acc_bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro_bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gravity_vector_body: list[float] = field(default_factory=lambda: [0.0, 0.0, 9.81])
    rotation_matrix: list[list[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    gravity_residual_ms2: float = 0.0
    forward_confidence: float = 0.0
    quality: str = "good"
    fallback_used: bool = False
    quality_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationParams":
        return cls(
            acc_bias=list(d.get("acc_bias", [0.0, 0.0, 0.0])),
            gyro_bias=list(d.get("gyro_bias", [0.0, 0.0, 0.0])),
            gravity_vector_body=list(d.get("gravity_vector_body", [0.0, 0.0, 9.81])),
            rotation_matrix=list(d.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            gravity_residual_ms2=float(d.get("gravity_residual_ms2", 0.0)),
            forward_confidence=float(d.get("forward_confidence", 0.0)),
            quality=str(d.get("quality", "good")),
            fallback_used=bool(d.get("fallback_used", False)),
            quality_tags=list(d.get("quality_tags", [])),
        )


@dataclass
class SectionCalibration:
    """Combined calibration for a section (both sensors)."""

    sporsa: CalibrationParams
    arduino: CalibrationParams
    frame_alignment: str = "gravity_only"
    calibration_quality: str = "good"
    quality_tags: list[str] = field(default_factory=list)
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "sporsa": self.sporsa.to_dict(),
            "arduino": self.arduino.to_dict(),
            "frame_alignment": self.frame_alignment,
            "calibration_quality": self.calibration_quality,
            "quality_tags": self.quality_tags,
            "created_at_utc": self.created_at_utc,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SectionCalibration":
        return cls(
            sporsa=CalibrationParams.from_dict(d.get("sporsa", {})),
            arduino=CalibrationParams.from_dict(d.get("arduino", {})),
            frame_alignment=str(d.get("frame_alignment", "gravity_only")),
            calibration_quality=str(d.get("calibration_quality", "good")),
            quality_tags=list(d.get("quality_tags", [])),
            created_at_utc=str(d.get("created_at_utc", "")),
        )


# ---------------------------------------------------------------------------
# Estimation helpers
# ---------------------------------------------------------------------------


def _static_mask(df: pd.DataFrame, *, sample_rate_hz: float = 100.0) -> np.ndarray:
    """Return a boolean mask of rows in static (low-motion) segments."""
    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    if not acc_cols:
        return np.zeros(len(df), dtype=bool)
    acc = df[acc_cols].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(acc ** 2, axis=1))
    deviation = np.abs(norm - _G)
    return deviation < 0.5  # within 0.5 m/s² of g


def _estimate_static_segments_from_calibration(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    static_min_s: float = 2.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) for detected static regions."""
    cals = find_calibration_segments(
        df,
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
    )
    if not cals:
        return []
    ranges: list[tuple[int, int]] = []
    for cal in cals:
        # Take the flat static flank at the start of each calibration.
        s = cal.start_idx
        if cal.peak_indices:
            e = cal.peak_indices[0]
        else:
            e = cal.end_idx
        ranges.append((s, e))
    return ranges


def _gravity_from_static(
    df: pd.DataFrame,
    static_ranges: list[tuple[int, int]],
) -> tuple[np.ndarray, float]:
    """Estimate gravity vector in body frame from static regions.

    Returns (gravity_vector, residual_ms2).
    """
    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    if not acc_cols or not static_ranges:
        g_body = np.array([0.0, 0.0, _G])
        return g_body, float("nan")

    all_acc: list[np.ndarray] = []
    for s, e in static_ranges:
        chunk = df.iloc[s:e][acc_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(chunk), axis=1)
        if finite.any():
            all_acc.append(chunk[finite])

    if not all_acc:
        return np.array([0.0, 0.0, _G]), float("nan")

    combined = np.vstack(all_acc)
    g_body = np.nanmean(combined, axis=0)
    g_norm = np.linalg.norm(g_body)
    residual = abs(g_norm - _G)

    # Normalize to expected gravity magnitude.
    if g_norm > 0.1:
        g_body = g_body * (_G / g_norm)

    return g_body, float(residual)


def _gyro_bias_from_static(
    df: pd.DataFrame,
    static_ranges: list[tuple[int, int]],
) -> np.ndarray:
    """Estimate gyroscope bias from static regions."""
    gyro_cols = [c for c in ["gx", "gy", "gz"] if c in df.columns]
    if not gyro_cols or not static_ranges:
        return np.zeros(3)

    all_gyro: list[np.ndarray] = []
    for s, e in static_ranges:
        chunk = df.iloc[s:e][gyro_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(chunk), axis=1)
        if finite.any():
            all_gyro.append(chunk[finite])

    if not all_gyro:
        return np.zeros(3)
    combined = np.vstack(all_gyro)
    return np.nanmean(combined, axis=0)


def _gravity_alignment_rotation(g_body: np.ndarray) -> np.ndarray:
    """Compute rotation matrix R such that R @ g_body ≈ [0, 0, g].

    The resulting rotation maps the sensor body frame to a gravity-aligned
    world frame where Z points up.
    """
    g_norm = np.linalg.norm(g_body)
    if g_norm < 1e-6:
        return np.eye(3)

    g_hat = g_body / g_norm  # measured gravity direction in body

    # Target: gravity points along +Z in world frame.
    target = np.array([0.0, 0.0, 1.0])

    # Axis = g_hat × target (rotation axis).
    axis = np.cross(g_hat, target)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(np.dot(g_hat, target))

    if sin_angle < 1e-8:
        # Already aligned or exactly anti-aligned.
        if cos_angle > 0:
            return np.eye(3)
        # 180° flip around X.
        return np.diag([1.0, -1.0, -1.0])

    axis = axis / sin_angle
    # Rodrigues' rotation formula.
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = (
        np.eye(3) * cos_angle
        + sin_angle * K
        + (1.0 - cos_angle) * np.outer(axis, axis)
    )
    return R


def _estimate_forward_direction(
    df: pd.DataFrame,
    R_gravity: np.ndarray,
    *,
    min_speed_ms2: float = 2.0,
    sample_rate_hz: float = 100.0,
) -> tuple[np.ndarray, float]:
    """Estimate forward direction from motion segments.

    Returns (forward_unit_vec_in_gravity_aligned_frame, confidence_in_0_1).
    """
    acc_cols = [c for c in ["ax", "ay", "az"] if c in df.columns]
    if not acc_cols:
        return np.array([1.0, 0.0, 0.0]), 0.0

    acc = df[acc_cols].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(acc ** 2, axis=1))

    # Motion = deviation from gravity level.
    deviation = np.abs(norm - _G)
    motion_mask = deviation > min_speed_ms2

    if motion_mask.sum() < 20:
        return np.array([1.0, 0.0, 0.0]), 0.0

    motion_acc = acc[motion_mask]
    # Rotate into gravity-aligned frame.
    motion_world = (R_gravity @ motion_acc.T).T

    # Horizontal components (X, Y in world frame) capture forward direction.
    horizontal = motion_world[:, :2]
    finite = np.all(np.isfinite(horizontal), axis=1)
    if finite.sum() < 10:
        return np.array([1.0, 0.0, 0.0]), 0.0

    # Use PCA on horizontal motion to find dominant forward direction.
    H = horizontal[finite]
    cov = np.cov(H.T)
    vals, vecs = np.linalg.eigh(cov)
    # Largest eigenvector = dominant horizontal direction.
    forward_2d = vecs[:, -1]
    # Explained variance ratio as confidence proxy.
    confidence = float(vals[-1] / (vals.sum() + 1e-9))
    forward_3d = np.array([forward_2d[0], forward_2d[1], 0.0])
    norm_fwd = np.linalg.norm(forward_3d)
    if norm_fwd > 1e-8:
        forward_3d /= norm_fwd
    return forward_3d, min(confidence, 1.0)


def _build_full_rotation(
    R_gravity: np.ndarray,
    forward_world: np.ndarray,
    confidence: float,
    *,
    min_confidence: float = 0.3,
) -> np.ndarray:
    """Extend gravity rotation with forward direction alignment."""
    if confidence < min_confidence:
        return R_gravity

    fwd = np.asarray(forward_world, dtype=float)
    fwd[2] = 0.0  # ensure it's horizontal
    norm = np.linalg.norm(fwd)
    if norm < 1e-8:
        return R_gravity

    fwd /= norm
    # Target forward is +X.
    angle = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return R_yaw @ R_gravity


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------


def estimate_calibration(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    frame_alignment: str = "gravity_only",
    static_min_s: float = 2.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
) -> CalibrationParams:
    """Estimate calibration parameters for one IMU sensor stream.

    Parameters
    ----------
    df:
        IMU DataFrame with columns ax/ay/az/gx/gy/gz and timestamp.
    sample_rate_hz:
        Approximate sample rate used for calibration-segment detection.
    frame_alignment:
        ``"gravity_only"`` or ``"gravity_plus_forward"``.
    """
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    quality_tags: list[str] = []
    fallback_used = False

    # Detect static segments from calibration sequences.
    static_ranges = _estimate_static_segments_from_calibration(
        df,
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
    )

    if not static_ranges:
        # Fallback: use first 5 s as static if df is long enough.
        ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        if ts.size > 0:
            t0 = ts[0]
            mask = ts <= t0 + 5000.0
            n_static = int(mask.sum())
            if n_static > 10:
                static_ranges = [(0, n_static)]
                fallback_used = True
                quality_tags.append("fallback_static")
            else:
                quality_tags.append("no_static_found")
        else:
            quality_tags.append("empty_stream")

    # Estimate biases.
    g_body, gravity_residual = _gravity_from_static(df, static_ranges)
    gyro_bias = _gyro_bias_from_static(df, static_ranges)

    # Accelerometer bias = (measured g) - (expected g_body direction at g magnitude).
    g_body_expected = g_body / (np.linalg.norm(g_body) + 1e-9) * _G
    acc_bias = (g_body - g_body_expected).tolist()

    # Build rotation to world frame.
    R_gravity = _gravity_alignment_rotation(g_body)

    forward_confidence = 0.0
    R_final = R_gravity

    if frame_alignment == "gravity_plus_forward":
        forward_world, forward_confidence = _estimate_forward_direction(
            df, R_gravity, sample_rate_hz=sample_rate_hz
        )
        R_final = _build_full_rotation(R_gravity, forward_world, forward_confidence)

    # Assess quality.
    if np.isnan(gravity_residual) or gravity_residual > 1.0:
        quality_tags.append("poor_gravity_alignment")
        quality = "poor" if gravity_residual > 2.0 else "marginal"
    else:
        quality = "good"

    return CalibrationParams(
        acc_bias=list(np.array(acc_bias).round(6).tolist()),
        gyro_bias=list(np.round(gyro_bias, 6).tolist()),
        gravity_vector_body=list(np.round(g_body, 6).tolist()),
        rotation_matrix=[list(row) for row in np.round(R_final, 8).tolist()],
        gravity_residual_ms2=round(float(gravity_residual) if not np.isnan(gravity_residual) else 99.0, 4),
        forward_confidence=round(float(forward_confidence), 4),
        quality=quality,
        fallback_used=fallback_used,
        quality_tags=quality_tags,
    )


# ---------------------------------------------------------------------------
# Application function
# ---------------------------------------------------------------------------


def apply_calibration(
    df: pd.DataFrame,
    params: CalibrationParams,
) -> pd.DataFrame:
    """Apply calibration to an IMU DataFrame.

    Steps:
    1. Subtract accelerometer bias.
    2. Subtract gyroscope bias.
    3. Rotate acc and gyro into world frame using rotation_matrix.

    Returns a new DataFrame with corrected columns and a ``_world`` suffix for
    rotated vectors.  Original columns are preserved.
    """
    out = df.copy()
    R = np.array(params.rotation_matrix, dtype=float)
    acc_bias = np.array(params.acc_bias, dtype=float)
    gyro_bias = np.array(params.gyro_bias, dtype=float)

    acc_cols = [c for c in ["ax", "ay", "az"] if c in out.columns]
    gyro_cols = [c for c in ["gx", "gy", "gz"] if c in out.columns]

    if len(acc_cols) == 3:
        acc = out[acc_cols].to_numpy(dtype=float)
        acc_corr = acc - acc_bias
        acc_world = (R @ acc_corr.T).T
        out["ax"] = acc_corr[:, 0]
        out["ay"] = acc_corr[:, 1]
        out["az"] = acc_corr[:, 2]
        out["ax_world"] = acc_world[:, 0]
        out["ay_world"] = acc_world[:, 1]
        out["az_world"] = acc_world[:, 2]

    if len(gyro_cols) == 3:
        gyro = out[gyro_cols].to_numpy(dtype=float)
        gyro_corr = gyro - gyro_bias
        gyro_world = (R @ gyro_corr.T).T
        out["gx"] = gyro_corr[:, 0]
        out["gy"] = gyro_corr[:, 1]
        out["gz"] = gyro_corr[:, 2]
        out["gx_world"] = gyro_world[:, 0]
        out["gy_world"] = gyro_world[:, 1]
        out["gz_world"] = gyro_world[:, 2]

    return out
