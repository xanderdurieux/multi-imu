"""Protocol-aware calibration estimation and application for a single IMU stream.

Conventions
-----------
- World frame: Z points up (against gravity), X points forward, Y points left.
- Gravity in world frame: [0, 0, -g] (pointing down in the real world, so
  accelerometer at rest measures [0, 0, +g] in world frame).
- All timestamps in milliseconds.

Recording protocol
------------------
Each recording section begins with an opening routine:

  1. ~5 s static  (pre-tap static)  → gyro bias estimation
  2. ~5 acceleration taps           → sync anchors
  3. ~5 s static  (post-tap static) → additional gyro bias source
  4. Mount transition (Arduino only): sensor is strapped to the bike
  5. First sufficiently stable post-mount window → alignment anchor (Arduino)

Sporsa does not undergo a mount change; its opening static windows are used
for both intrinsic estimation and alignment.

Calibration is split into two independent outputs per sensor:
- ``intrinsics``: gyro bias (always); accelerometer bias/scale only when a
  static hardware calibration reference is provided.
- ``alignment``: body-to-world rotation derived from the first appropriate
  stable window after mount, plus yaw resolved by priority (mag → PCA → none).
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

_G = 9.81  # m/s²


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class OpeningSequence:
    """Protocol-detected opening routine in a section.

    All range fields are millisecond timestamps.
    """

    tap_times_ms: list[float]      # tap timestamps
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
    """Per-sensor intrinsic calibration parameters.

    ``gyro_bias`` is always estimated from opening static windows.
    ``acc_bias`` and ``acc_scale`` are populated only when a static hardware
    calibration reference is provided; otherwise they are ``None`` and no
    accelerometer correction is applied.
    """

    gyro_bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    acc_bias: list[float] | None = None
    acc_scale: list[float] | None = None
    static_residual_ms2: float = 0.0
    quality: str = "good"
    quality_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "gyro_bias": self.gyro_bias,
            "acc_bias": self.acc_bias,
            "acc_scale": self.acc_scale,
            "static_residual_ms2": self.static_residual_ms2,
            "quality": self.quality,
            "quality_tags": self.quality_tags,
        }
        return d

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
    """Per-sensor alignment: body-to-world rotation and yaw resolution.

    ``yaw_source`` records which method resolved the yaw degree of freedom:
    - ``"mag"``: magnetometer-based heading
    - ``"pca_forward"``: dominant motion direction via PCA
    - ``"gravity_only"``: only roll/pitch resolved; yaw is undefined
    """

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
    """Combined protocol-aware calibration for a section (both sensors)."""

    protocol_detected: bool = False
    opening_sequence: OpeningSequence | None = None
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
            "opening_sequence": self.opening_sequence.to_dict() if self.opening_sequence else None,
            "intrinsics": {k: v.to_dict() for k, v in self.intrinsics.items()},
            "alignment": {k: v.to_dict() for k, v in self.alignment.items()},
            "quality": self.quality,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SectionCalibration":
        raw_opening = d.get("opening_sequence")
        opening = OpeningSequence.from_dict(raw_opening) if raw_opening else None
        return cls(
            protocol_detected=bool(d.get("protocol_detected", False)),
            opening_sequence=opening,
            intrinsics={k: SensorIntrinsics.from_dict(v) for k, v in d.get("intrinsics", {}).items()},
            alignment={k: SensorAlignment.from_dict(v) for k, v in d.get("alignment", {}).items()},
            quality=dict(d.get("quality", {"overall": "good", "tags": []})),
            provenance=dict(d.get("provenance", {})),
        )


# ---------------------------------------------------------------------------
# Gravity / rotation helpers (internal)
# ---------------------------------------------------------------------------


def _gravity_from_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """Estimate body-frame gravity vector from static timestamp ranges (ms).

    Parameters
    ----------
    ranges:
        List of ``(start_ms, end_ms)`` timestamp pairs defining static windows.

    Returns (gravity_vector_body, residual_ms2).
    """
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

    combined = np.vstack(chunks)
    g_body = np.nanmean(combined, axis=0)
    g_norm = np.linalg.norm(g_body)
    residual = abs(g_norm - _G)

    if g_norm > 0.1:
        g_body = g_body * (_G / g_norm)

    return g_body, float(residual)


def _gyro_bias_from_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]],
) -> np.ndarray:
    """Estimate gyroscope bias from static timestamp ranges (ms)."""
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
    return (
        np.eye(3) * cos_angle
        + sin_angle * K
        + (1.0 - cos_angle) * np.outer(axis, axis)
    )


def _mag_from_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]],
) -> np.ndarray:
    """Estimate mean magnetometer vector over timestamp ranges (ms)."""
    mag_cols = [c for c in ("mx", "my", "mz") if c in df.columns]
    if not mag_cols or not ranges:
        return np.zeros(3)

    ts = df["timestamp"].to_numpy(dtype=float)
    chunks: list[np.ndarray] = []
    for s_ms, e_ms in ranges:
        mask = (ts >= s_ms) & (ts <= e_ms)
        chunk = df.loc[mask, mag_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(chunk), axis=1)
        if finite.any():
            chunks.append(chunk[finite])

    if not chunks:
        return np.zeros(3)
    return np.nanmean(np.vstack(chunks), axis=0)


def _assess_mag_reliability(
    df: pd.DataFrame,
    ranges: list[tuple[float, float]],
    *,
    field_min: float = 1.0,
    field_max: float = 200.0,
    max_cv: float = 0.15,
) -> tuple[bool, list[str]]:
    mag_cols = [c for c in ("mx", "my", "mz") if c in df.columns]
    if len(mag_cols) < 3:
        return False, ["mag_columns_missing"]

    ts = df["timestamp"].to_numpy(dtype=float)
    mags: list[np.ndarray] = []
    for s_ms, e_ms in ranges:
        mask = (ts >= s_ms) & (ts <= e_ms)
        chunk = df.loc[mask, mag_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(chunk), axis=1)
        if finite.any():
            mags.append(np.linalg.norm(chunk[finite], axis=1))

    if not mags:
        arr = df[mag_cols].to_numpy(dtype=float)
        finite = np.all(np.isfinite(arr), axis=1)
        if not finite.any():
            return False, ["mag_all_nan"]
        mags = [np.linalg.norm(arr[finite], axis=1)]

    all_norms = np.concatenate(mags)
    if all_norms.size == 0:
        return False, ["mag_all_nan"]

    mean_norm = float(np.mean(all_norms))
    if not (field_min <= mean_norm <= field_max):
        return False, [f"mag_field_out_of_range_{mean_norm:.1f}"]

    if all_norms.size > 1:
        cv = float(np.std(all_norms) / (mean_norm + 1e-9))
        if cv > max_cv:
            return False, [f"mag_unstable_cv_{cv:.2f}"]

    return True, []


def _yaw_from_mag(
    mag_body: np.ndarray,
    R_gravity: np.ndarray,
) -> tuple[float, float]:
    """Return (yaw_angle_rad, confidence) for mag-based heading."""
    mag_world = R_gravity @ mag_body
    hx, hy = float(mag_world[0]), float(mag_world[1])
    horiz = float(np.sqrt(hx ** 2 + hy ** 2))
    total = float(np.linalg.norm(mag_world))
    confidence = horiz / (total + 1e-9)
    yaw = float(np.arctan2(hy, hx))
    return yaw, confidence


def _forward_from_pca(
    df: pd.DataFrame,
    R_gravity: np.ndarray,
    *,
    min_speed_ms2: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Estimate dominant horizontal motion direction via PCA."""
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not acc_cols:
        return np.array([1.0, 0.0, 0.0]), 0.0

    acc = df[acc_cols].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(acc ** 2, axis=1))
    motion_mask = np.abs(norm - _G) > min_speed_ms2

    if motion_mask.sum() < 20:
        return np.array([1.0, 0.0, 0.0]), 0.0

    motion_world = (R_gravity @ acc[motion_mask].T).T
    horizontal = motion_world[:, :2]
    finite = np.all(np.isfinite(horizontal), axis=1)
    if finite.sum() < 10:
        return np.array([1.0, 0.0, 0.0]), 0.0

    H = horizontal[finite]
    cov = np.cov(H.T)
    vals, vecs = np.linalg.eigh(cov)
    forward_2d = vecs[:, -1]
    confidence = float(vals[-1] / (vals.sum() + 1e-9))
    forward_3d = np.array([forward_2d[0], forward_2d[1], 0.0])
    n = np.linalg.norm(forward_3d)
    if n > 1e-8:
        forward_3d /= n
    return forward_3d, min(confidence, 1.0)


def _apply_yaw(R_gravity: np.ndarray, yaw_angle_rad: float) -> np.ndarray:
    c, s = np.cos(-yaw_angle_rad), np.sin(-yaw_angle_rad)
    R_yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R_yaw @ R_gravity




# ---------------------------------------------------------------------------
# Protocol detection
# ---------------------------------------------------------------------------


def _find_first_stable_window(
    df: pd.DataFrame,
    after_ms: float,
    *,
    min_duration_ms: float = 3000.0,
    threshold_ms2: float = 1.5,
) -> tuple[float, float] | None:
    """Find the first stable (low-motion) window that starts at or after *after_ms*.

    A sample is stable when ``||acc_norm| - g| < threshold_ms2``.  Duration
    is measured in milliseconds from the timestamp column.

    Returns ``(start_ms, end_ms)`` or ``None`` when no window is found.
    """
    acc_cols = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not acc_cols:
        return None

    ts = df["timestamp"].to_numpy(dtype=float)
    mask_after = ts >= after_ms
    if not np.any(mask_after):
        return None

    ts_after = ts[mask_after]
    acc = df.loc[mask_after, acc_cols].to_numpy(dtype=float)
    norm = np.sqrt(np.nansum(acc ** 2, axis=1))
    is_stable = np.abs(norm - _G) < threshold_ms2

    run_start: int | None = None
    for i, flag in enumerate(is_stable):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if ts_after[i - 1] - ts_after[run_start] >= min_duration_ms:
                return float(ts_after[run_start]), float(ts_after[i - 1])
            run_start = None

    if run_start is not None:
        if ts_after[-1] - ts_after[run_start] >= min_duration_ms:
            return float(ts_after[run_start]), float(ts_after[-1])

    return None


def detect_protocol_landmarks(
    segments: list[CalibrationSegment],
) -> tuple[CalibrationSegment | None, list[tuple[float, float]]]:
    """Derive opening-routine static ranges from pre-detected calibration segments.

    The first detected segment is the opening routine.  Its peak timestamps
    split the segment into a pre-tap static window and a post-tap static window.

    Parameters
    ----------
    segments:
        Pre-detected calibration segments loaded from JSON (parser output).
        Pass an empty list if none were found.

    Returns
    -------
    (opening_segment, static_ranges)
        ``opening_segment``: the first :class:`CalibrationSegment`, or ``None``.
        ``static_ranges``: list of ``(start_ms, end_ms)`` timestamp pairs for
        the static flanks.  Empty when no segments were found.
    """
    if not segments:
        return None, []

    seg = segments[0]  # opening routine = first calibration sequence
    static_ranges: list[tuple[float, float]] = []

    if seg.peak_timestamps:
        pre_end_ms = seg.peak_timestamps[0]
        if pre_end_ms > seg.start_timestamp:
            static_ranges.append((seg.start_timestamp, pre_end_ms))

        post_start_ms = seg.peak_timestamps[-1]
        if seg.end_timestamp > post_start_ms:
            static_ranges.append((post_start_ms, seg.end_timestamp))
    else:
        static_ranges.append((seg.start_timestamp, seg.end_timestamp))

    return seg, static_ranges


# ---------------------------------------------------------------------------
# Intrinsics estimation
# ---------------------------------------------------------------------------


def estimate_sensor_intrinsics(
    df: pd.DataFrame,
    static_ranges: list[tuple[float, float]],
    *,
    static_cal: dict[str, Any] | None = None,
) -> SensorIntrinsics:
    """Estimate sensor intrinsics from static windows.

    Parameters
    ----------
    df:
        Raw IMU DataFrame.
    static_ranges:
        ``(start_ms, end_ms)`` timestamp pairs of static windows to use.
    static_cal:
        Optional static hardware calibration reference dict.  Must contain
        ``accelerometer.bias`` and/or ``accelerometer.scale`` and
        ``gyroscope.bias_deg_s`` sub-dicts to be used.  When provided, its
        gyro bias overrides the dynamically-estimated value.

    Returns
    -------
    SensorIntrinsics
    """
    quality_tags: list[str] = []

    # Gyro bias
    gyro_bias = _gyro_bias_from_ranges(df, static_ranges)

    # Gravity estimate for quality assessment
    g_body, residual = _gravity_from_ranges(df, static_ranges)

    # Assess quality
    if np.isnan(residual) or residual > 2.0:
        quality = "poor"
        quality_tags.append("poor_static_gravity")
    elif residual > 1.0:
        quality = "marginal"
        quality_tags.append("marginal_static_gravity")
    else:
        quality = "good"

    if not static_ranges:
        quality_tags.append("no_static_ranges")
        quality = "poor"

    # Accelerometer intrinsics: only from hardware static calibration
    acc_bias: list[float] | None = None
    acc_scale: list[float] | None = None

    if static_cal is not None:
        acc_cal = static_cal.get("accelerometer", {})
        if acc_cal.get("bias"):
            b = acc_cal["bias"]
            acc_bias = [float(b.get("x", 0)), float(b.get("y", 0)), float(b.get("z", 0))]
        if acc_cal.get("scale"):
            sc = acc_cal["scale"]
            acc_scale = [float(sc.get("x", 1)), float(sc.get("y", 1)), float(sc.get("z", 1))]
        # Override gyro bias with static reference if available
        gyro_cal = static_cal.get("gyroscope", {})
        if gyro_cal.get("bias_deg_s"):
            gb = gyro_cal["bias_deg_s"]
            gyro_bias = np.array([
                float(gb.get("x", 0)), float(gb.get("y", 0)), float(gb.get("z", 0))
            ])
            quality_tags.append("gyro_bias_from_static_cal")

    return SensorIntrinsics(
        gyro_bias=list(np.round(gyro_bias, 6).tolist()),
        acc_bias=[round(v, 6) for v in acc_bias] if acc_bias is not None else None,
        acc_scale=[round(v, 6) for v in acc_scale] if acc_scale is not None else None,
        static_residual_ms2=round(float(residual) if not np.isnan(residual) else 99.0, 4),
        quality=quality,
        quality_tags=quality_tags,
    )


# ---------------------------------------------------------------------------
# Alignment estimation
# ---------------------------------------------------------------------------


def estimate_sensor_alignment(
    df: pd.DataFrame,
    window_range: tuple[float, float],
    *,
    full_df: pd.DataFrame | None = None,
) -> SensorAlignment:
    """Estimate body-to-world alignment from a stable window.

    Yaw is resolved in priority order:
    1. Magnetometer (if reliable in the alignment window)
    2. PCA of dominant horizontal motion (over ``full_df`` if provided,
       otherwise ``df``)
    3. Gravity-only (roll + pitch only; yaw undefined)

    Parameters
    ----------
    df:
        Raw IMU DataFrame.
    window_range:
        ``(start_ms, end_ms)`` timestamp range of the stable window to use
        for gravity and magnetometer estimation.
    full_df:
        Full section DataFrame (used for PCA forward-direction estimation).
        Falls back to ``df`` when not provided.
    """
    quality_tags: list[str] = []
    win_start_ms, win_end_ms = float(window_range[0]), float(window_range[1])

    # Gravity estimate from alignment window
    g_body, residual = _gravity_from_ranges(df, [(win_start_ms, win_end_ms)])
    R_gravity = _gravity_alignment_rotation(g_body)

    # --- Yaw resolution ---
    R_final = R_gravity
    yaw_source = "gravity_only"
    yaw_confidence = 0.0

    # Priority 1: magnetometer
    mag_ranges = [(win_start_ms, win_end_ms)]
    mag_reliable, mag_tags = _assess_mag_reliability(df, mag_ranges)
    quality_tags.extend(mag_tags)

    if mag_reliable:
        mag_body = _mag_from_ranges(df, mag_ranges)
        yaw_angle, yaw_conf = _yaw_from_mag(mag_body, R_gravity)
        if yaw_conf >= 0.2:
            R_final = _apply_yaw(R_gravity, yaw_angle)
            yaw_source = "mag"
            yaw_confidence = round(float(yaw_conf), 4)
            quality_tags.append("yaw_from_mag")
        else:
            quality_tags.append("mag_low_horizontal_component")

    # Priority 2: PCA forward direction (uses full section for motion samples)
    if yaw_source == "gravity_only":
        motion_df = full_df if full_df is not None else df
        forward_world, fwd_conf = _forward_from_pca(motion_df, R_gravity)
        if fwd_conf >= 0.3:
            angle = float(np.arctan2(forward_world[1], forward_world[0]))
            R_final = _apply_yaw(R_gravity, angle)
            yaw_source = "pca_forward"
            yaw_confidence = round(float(fwd_conf), 4)
            quality_tags.append("yaw_from_pca")
        else:
            quality_tags.append("yaw_undetermined")

    if np.isnan(residual) or residual > 1.0:
        quality_tags.append("poor_alignment_gravity")

    return SensorAlignment(
        rotation_matrix=[list(row) for row in np.round(R_final, 8).tolist()],
        gravity_estimate=list(np.round(g_body, 6).tolist()),
        yaw_source=yaw_source,
        yaw_confidence=yaw_confidence,
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
    """Apply calibration to a raw IMU DataFrame.

    Steps:
    1. Subtract gyroscope bias (always).
    2. Subtract accelerometer bias and/or apply scale correction if available.
    3. Rotate acc and gyro into world frame using the alignment rotation matrix.
       World-frame columns are written with ``_world`` suffix.
       Original (bias-corrected) columns are updated in-place.

    Returns a new DataFrame with corrected columns.
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
            # Scale correction: divide each axis by its scale factor
            acc_corr /= np.array(intrinsics.acc_scale, dtype=float)
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

    # Axis values changed — recompute norms to stay consistent.
    return add_imu_norms(out)
