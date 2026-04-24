"""Calibration estimation and application for a single IMU stream.

Conventions
-----------
- World frame: Z points up, X forward, Y left.
- Gravity in world frame: [0, 0, -g]; accelerometer at rest measures [0, 0, +g].
- All timestamps in milliseconds.

Recording protocol
------------------
Each section begins with: ~5 s static → taps (sync anchors) → ~5 s static.

Calibration outputs per sensor
-------------------------------
- ``intrinsics``: gyro bias (always); acc bias/scale when a static hardware
  calibration reference is provided.
- ``alignment``: body-to-world rotation from the post-tap static window; only 
  aligns gravity (roll + pitch), yaw is undefined.

The rotation matrix is valid only at the alignment window and is used solely
to initialise the orientation filter.  Do not use per-axis world-frame columns
as time series without real-time orientation from the ``orientation`` stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from parser.calibration_segments import CalibrationSegment
from common.signals import add_imu_norms

log = logging.getLogger(__name__)

_G = 9.81


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OpeningSequence:
    tap_times_ms: list[float]
    pre_static_start_ms: float
    pre_static_end_ms: float
    post_static_start_ms: float
    post_static_end_ms: float
    n_taps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpeningSequence":
        return cls(
            tap_times_ms=[float(v) for v in d.get("tap_times_ms", [])],
            pre_static_start_ms=float(d.get("pre_static_start_ms", 0.0)),
            pre_static_end_ms=float(d.get("pre_static_end_ms", 0.0)),
            post_static_start_ms=float(d.get("post_static_start_ms", 0.0)),
            post_static_end_ms=float(d.get("post_static_end_ms", 0.0)),
            n_taps=int(d.get("n_taps", 0)),
        )


@dataclass
class SensorIntrinsics:
    gyro_bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acc_bias: list[float] | None = None
    acc_scale: list[float] | None = None
    static_residual_ms2: float = 0.0
    quality: str = "good"
    quality_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gyro_bias": self.gyro_bias,
            "acc_bias": self.acc_bias,
            "acc_scale": self.acc_scale,
            "static_residual_ms2": self.static_residual_ms2,
            "quality": self.quality,
            "quality_tags": self.quality_tags,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SensorIntrinsics":
        return cls(
            gyro_bias=list(d.get("gyro_bias", [0.0, 0.0, 0.0])),
            acc_bias=list(d["acc_bias"]) if d.get("acc_bias") is not None else None,
            acc_scale=list(d["acc_scale"]) if d.get("acc_scale") is not None else None,
            static_residual_ms2=float(d.get("static_residual_ms2", 0.0)),
            quality=str(d.get("quality", "good")),
            quality_tags=list(d.get("quality_tags", [])),
        )


@dataclass
class SensorAlignment:
    rotation_matrix: list[list[float]] = field(
        default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    gravity_estimate: list[float] = field(default_factory=lambda: [0.0, 0.0, _G])
    yaw_source: str = "gravity_only"
    yaw_confidence: float = 0.0
    alignment_window_start_ms: float = 0.0
    alignment_window_end_ms: float = 0.0
    gravity_residual_ms2: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SensorAlignment":
        return cls(
            rotation_matrix=list(d.get("rotation_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])),
            gravity_estimate=list(d.get("gravity_estimate", [0.0, 0.0, _G])),
            yaw_source=str(d.get("yaw_source", "gravity_only")),
            yaw_confidence=float(d.get("yaw_confidence", 0.0)),
            alignment_window_start_ms=float(d.get("alignment_window_start_ms", 0.0)),
            alignment_window_end_ms=float(d.get("alignment_window_end_ms", 0.0)),
            gravity_residual_ms2=float(d.get("gravity_residual_ms2", 0.0)),
        )


@dataclass
class SectionCalibration:
    protocol_detected: bool = False
    opening_sequence: dict[str, OpeningSequence] = field(default_factory=dict)
    closing_sequence: dict[str, OpeningSequence] = field(default_factory=dict)
    intrinsics: dict[str, SensorIntrinsics] = field(default_factory=dict)
    alignment: dict[str, SensorAlignment] = field(default_factory=dict)
    quality: dict[str, Any] = field(
        default_factory=lambda: {"overall": "good", "tags": []}
    )
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "protocol_detected": self.protocol_detected,
            "opening_sequence": {k: v.to_dict() for k, v in self.opening_sequence.items()},
            "closing_sequence": {k: v.to_dict() for k, v in self.closing_sequence.items()},
            "intrinsics": {k: v.to_dict() for k, v in self.intrinsics.items()},
            "alignment": {k: v.to_dict() for k, v in self.alignment.items()},
            "quality": self.quality,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SectionCalibration":
        raw_opening = d.get("opening_sequence") or {}
        opening = {s: OpeningSequence.from_dict(p) for s, p in raw_opening.items() if p}
        raw_closing = d.get("closing_sequence") or {}
        closing = {s: OpeningSequence.from_dict(p) for s, p in raw_closing.items() if p}
        return cls(
            protocol_detected=bool(d.get("protocol_detected", False)),
            opening_sequence=opening,
            closing_sequence=closing,
            intrinsics={k: SensorIntrinsics.from_dict(v) for k, v in d.get("intrinsics", {}).items()},
            alignment={k: SensorAlignment.from_dict(v) for k, v in d.get("alignment", {}).items()},
            quality=dict(d.get("quality", {"overall": "good", "tags": []})),
            provenance=dict(d.get("provenance", {})),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gravity_from_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """Mean body-frame gravity from static windows; returns (g_body, residual_ms2)."""
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not acc_cols or not ranges:
        return np.array([0.0, 0.0, _G]), float("nan")

    ts = df["timestamp"].to_numpy(dtype=float)
    chunks: list[np.ndarray] = []
    for s_ms, e_ms in ranges:
        mask = (ts >= s_ms) & (ts <= e_ms)
        chunk = df.loc[mask, acc_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(chunk), axis=1)
        if finite.any():
            chunks.append(chunk[finite])

    if not chunks:
        return np.array([0.0, 0.0, _G]), float("nan")

    g_body = np.nanmean(np.vstack(chunks), axis=0)
    g_norm = np.linalg.norm(g_body)
    residual = abs(g_norm - _G)
    if g_norm > 0.1:
        g_body = g_body * (_G / g_norm)
    return g_body, float(residual)


def _gyro_bias_from_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]],
) -> np.ndarray:
    """Mean gyroscope bias from static windows."""
    gyro_cols = [c for c in ("gx", "gy", "gz") if c in df.columns]
    if not gyro_cols or not ranges:
        return np.zeros(3)

    ts = df["timestamp"].to_numpy(dtype=float)
    chunks: list[np.ndarray] = []
    for s_ms, e_ms in ranges:
        mask = (ts >= s_ms) & (ts <= e_ms)
        chunk = df.loc[mask, gyro_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(chunk), axis=1)
        if finite.any():
            chunks.append(chunk[finite])

    if not chunks:
        return np.zeros(3)
    return np.nanmean(np.vstack(chunks), axis=0)


def _gravity_alignment_rotation(g_body: np.ndarray) -> np.ndarray:
    """R such that R @ g_body ≈ [0, 0, g] (Z-up world frame)."""
    g_norm = np.linalg.norm(g_body)
    if g_norm < 1e-6:
        return np.eye(3)

    g_hat = g_body / g_norm
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(g_hat, target)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(np.dot(g_hat, target))

    if sin_angle < 1e-8:
        return np.eye(3) if cos_angle > 0 else np.diag([1.0, -1.0, -1.0])

    axis = axis / sin_angle
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) * cos_angle + sin_angle * K + (1.0 - cos_angle) * np.outer(axis, axis)


def _segment_static_ranges(seg: CalibrationSegment) -> list[tuple[float, float]]:
    """Pre- and post-tap static windows from a calibration segment."""
    ranges: list[tuple[float, float]] = []
    pre_end = seg.start_ms + seg.static_pre_ms
    if seg.static_pre_ms > 0 and pre_end > seg.start_ms:
        ranges.append((seg.start_ms, pre_end))
    post_start = seg.end_ms - seg.static_post_ms
    if seg.static_post_ms > 0 and seg.end_ms > post_start:
        ranges.append((post_start, seg.end_ms))
    if not ranges:
        ranges.append((seg.start_ms, seg.end_ms))
    return ranges


def detect_protocol_landmarks(
    segments: list[CalibrationSegment],
) -> tuple[CalibrationSegment | None, list[tuple[float, float]], CalibrationSegment | None]:
    """Return (opening_segment, static_ranges, closing_segment) from detected segments."""
    if not segments:
        return None, [], None
    opening = segments[0]
    static_ranges = _segment_static_ranges(opening)
    closing = segments[-1] if len(segments) > 1 else None
    return opening, static_ranges, closing


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

_GYRO_BIAS_DRIFT_THRESHOLD_DEG_S = 1.0
_ACC_SANITY_THRESHOLD_MS2 = 0.1


def estimate_sensor_intrinsics(
    df: pd.DataFrame,
    static_ranges: list[tuple[float, float]],
    *,
    static_cal: dict[str, Any] | None = None,
) -> SensorIntrinsics:
    """Estimate gyro bias and optionally acc bias/scale from static windows.

    Dynamic gyro bias (from opening window) is always applied; it tracks
    session-level temperature drift that a static hardware cal cannot capture.
    Accelerometer bias/scale are carried from the static hardware calibration
    reference when provided — they are stable across sessions.
    """
    quality_tags: list[str] = []

    gyro_bias = _gyro_bias_from_ranges(df, static_ranges)
    g_body, residual = _gravity_from_ranges(df, static_ranges)

    if not static_ranges:
        quality = "poor"
        quality_tags.append("no_static_ranges")
    elif np.isnan(residual) or residual > 2.0:
        quality = "poor"
        quality_tags.append("poor_static_gravity")
    elif residual > 1.0:
        quality = "marginal"
        quality_tags.append("marginal_static_gravity")
    else:
        quality = "good"

    acc_bias: list[float] | None = None
    acc_scale: list[float] | None = None

    if static_cal is not None:
        acc_cal = static_cal.get("accelerometer", {})
        if acc_cal.get("bias"):
            b = acc_cal["bias"]
            acc_bias = [float(b.get("x", 0.0)), float(b.get("y", 0.0)), float(b.get("z", 0.0))]
        if acc_cal.get("scale"):
            sc = acc_cal["scale"]
            acc_scale = [float(sc.get("x", 1.0)), float(sc.get("y", 1.0)), float(sc.get("z", 1.0))]

        gyro_cal = static_cal.get("gyroscope", {})
        if gyro_cal.get("bias_deg_s"):
            gb = gyro_cal["bias_deg_s"]
            static_gyro_bias = np.array([
                float(gb.get("x", 0.0)), float(gb.get("y", 0.0)), float(gb.get("z", 0.0))
            ])
            drift = float(np.linalg.norm(gyro_bias - static_gyro_bias))
            if drift > _GYRO_BIAS_DRIFT_THRESHOLD_DEG_S:
                # Informational only: dynamic bias is used for correction regardless,
                # so drift vs. the hardware reference does not degrade correction quality.
                quality_tags.append(f"gyro_bias_drift_{drift:.2f}deg_s")

        if acc_bias is not None or acc_scale is not None:
            corrected = g_body.copy()
            if acc_bias is not None:
                corrected = corrected - np.array(acc_bias, dtype=float)
            if acc_scale is not None:
                corrected = corrected * np.array(acc_scale, dtype=float)
            corrected_norm = float(np.linalg.norm(corrected))
            if np.isfinite(corrected_norm):
                post_residual = abs(corrected_norm - _G)
                if post_residual > _ACC_SANITY_THRESHOLD_MS2:
                    quality_tags.append(f"acc_cal_stale_{post_residual:.3f}ms2")
                    if quality == "good":
                        quality = "marginal"

    return SensorIntrinsics(
        gyro_bias=list(np.round(gyro_bias, 6).tolist()),
        acc_bias=[round(v, 6) for v in acc_bias] if acc_bias is not None else None,
        acc_scale=[round(v, 6) for v in acc_scale] if acc_scale is not None else None,
        static_residual_ms2=round(float(residual) if not np.isnan(residual) else 99.0, 4),
        quality=quality,
        quality_tags=quality_tags,
    )


def estimate_sensor_alignment(
    df: pd.DataFrame,
    window_range: tuple[float, float],
) -> SensorAlignment:
    """Gravity-only body-to-world alignment from a stable window (roll + pitch; yaw undefined)."""
    win_start_ms, win_end_ms = float(window_range[0]), float(window_range[1])
    g_body, residual = _gravity_from_ranges(df, [(win_start_ms, win_end_ms)])
    R = _gravity_alignment_rotation(g_body)

    return SensorAlignment(
        rotation_matrix=[list(row) for row in np.round(R, 8).tolist()],
        gravity_estimate=list(np.round(g_body, 6).tolist()),
        yaw_source="gravity_only",
        yaw_confidence=0.0,
        alignment_window_start_ms=round(win_start_ms, 1),
        alignment_window_end_ms=round(win_end_ms, 1),
        gravity_residual_ms2=round(float(residual) if not np.isnan(residual) else 99.0, 4),
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def apply_calibration(
    df: pd.DataFrame,
    intrinsics: SensorIntrinsics,
    alignment: SensorAlignment,
) -> pd.DataFrame:
    """Apply bias correction and world-frame rotation to a raw IMU DataFrame.

    Steps: subtract gyro bias → subtract acc bias, apply acc scale → rotate
    both to world frame. World-frame columns (_world suffix) are valid only at
    the alignment window; use the orientation stage for time-varying rotation.
    """
    out = df.copy()
    R = np.array(alignment.rotation_matrix, dtype=float)
    gyro_bias = np.array(intrinsics.gyro_bias, dtype=float)

    acc_cols = [c for c in ("ax", "ay", "az") if c in out.columns]
    gyro_cols = [c for c in ("gx", "gy", "gz") if c in out.columns]

    if len(acc_cols) == 3:
        acc = out[acc_cols].to_numpy(dtype=float)
        acc_corr = acc.copy()
        if intrinsics.acc_bias is not None:
            acc_corr -= np.array(intrinsics.acc_bias, dtype=float)
        if intrinsics.acc_scale is not None:
            acc_corr *= np.array(intrinsics.acc_scale, dtype=float)
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

    return add_imu_norms(out)
