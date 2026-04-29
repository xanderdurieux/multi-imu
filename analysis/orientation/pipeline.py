"""Orientation estimation pipeline for calibrated IMU sections.

For each section:
1. Load calibrated/{sensor}.csv and calibration.json.
2. Initialise Mahony from the calibration body→world rotation matrix.
3. Apply magnetometer hard/soft-iron correction where available.
4. Run Mahony filter and score gravity residual in the known-static windows.
5. Write per-sensor quaternion CSVs, a stats JSON, and comparison plots.

Output layout
-------------
::

    orientation/
        sporsa.csv               # quaternions + Euler angles
        arduino.csv
        orientation_stats.json   # residual, quality, parameters
        sporsa_orientation.png
        arduino_orientation.png
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    read_json_file,
    project_relative_path,
    read_csv,
    section_stage_dir,
    static_calibration_json,
    write_csv,
    write_json_file,
)
from common.quaternion import (
    euler_from_quat,
    quat_conjugate,
    quat_from_rotation_matrix,
    quat_normalize,
    quat_rotate,
    quat_rotate_batch,
    tilt_quat_from_acc,
)

from .filters import run_mahony, _DEFAULT_KP, _DEFAULT_KI
from .mag_calibration import MagCalibration, fit_mag_ellipsoid

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_G = 9.81
_DEG = 180.0 / np.pi
_RAD = np.pi / 180.0


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _load_section_calibration(section_dir: Path) -> dict[str, Any]:
    cal_path = section_stage_dir(section_dir.name, "calibrated") / "calibration.json"
    return read_json_file(cal_path)


def _initial_quaternion(cal: dict[str, Any], sensor: str) -> np.ndarray | None:
    try:
        R = np.array(cal["alignment"][sensor]["rotation_matrix"], dtype=float)
        return quat_from_rotation_matrix(R)
    except (KeyError, TypeError, ValueError):
        return None


def _static_windows(cal: dict[str, Any], sensor: str) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    for seq_key in ("opening_sequence", "closing_sequence"):
        try:
            seq = cal[seq_key][sensor]
            pre_s = float(seq.get("pre_static_start_ms", 0))
            pre_e = float(seq.get("pre_static_end_ms", 0))
            post_s = float(seq.get("post_static_start_ms", 0))
            post_e = float(seq.get("post_static_end_ms", 0))
            if pre_e > pre_s:
                windows.append((pre_s, pre_e))
            if post_e > post_s:
                windows.append((post_s, post_e))
        except (KeyError, TypeError):
            pass
    return windows


def _load_static_mag_calibration(sensor: str) -> MagCalibration | None:
    """Load a stored ellipsoid (or hard-iron-only) mag calibration.

    Layered design:
    * If the static-calibration JSON has a full ellipsoid (``offset`` +
      ``transform``), use it directly.
    * If it has only a legacy ``hard_iron_bias`` block, lift it to a
      hard-iron-only :class:`MagCalibration` (identity transform).
    * Otherwise return None — caller will fit an ellipsoid in-section.
    """
    if sensor != "arduino":
        return None
    cal_path = static_calibration_json()
    if not cal_path.exists():
        return None
    try:
        cal = read_json_file(cal_path)
        mag_block = cal.get("magnetometer") or {}

        ellipsoid = mag_block.get("ellipsoid")
        if isinstance(ellipsoid, dict) and "offset" in ellipsoid and "transform" in ellipsoid:
            return MagCalibration(
                offset=np.array(ellipsoid["offset"], dtype=float),
                transform=np.array(ellipsoid["transform"], dtype=float),
                radius=float(ellipsoid.get("radius", 0.0)),
                residual=float(ellipsoid.get("residual", float("nan"))),
                n_samples=int(ellipsoid.get("n_samples", 0)),
            )

        hi = mag_block.get("hard_iron_bias")
        if hi is not None:
            return MagCalibration(
                offset=np.array([hi["x"], hi["y"], hi["z"]], dtype=float),
                transform=np.eye(3),
                radius=0.0,
                residual=float("nan"),
                n_samples=0,
            )
    except Exception as exc:
        log.warning("Could not load mag calibration: %s", exc)
    return None


def _resolve_mag_calibration(sensor: str, mag: np.ndarray | None) -> MagCalibration | None:
    """Pick the best mag calibration available for this section.

    Priority: stored static cal (loaded from disk) > in-section ellipsoid fit
    on the actual cycling data > None (filter runs gyro+acc only).  The fit
    requires enough rotation diversity, which a full cycling section
    typically provides whereas the six-face static recordings did not.

    Arduino benefits from a pre-recorded static ellipsoid; other sensors fall
    through directly to the in-section fit.
    """
    if mag is None:
        return None

    static_cal = _load_static_mag_calibration(sensor)
    if static_cal is not None and static_cal.n_samples > 0:
        return static_cal  # full ellipsoid from a previous run

    fitted = fit_mag_ellipsoid(mag)
    if fitted.n_samples >= 100:
        return fitted

    # Fall back to legacy hard-iron-only correction if loaded above.
    return static_cal


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _gravity_residual(
    timestamps: np.ndarray,
    acc: np.ndarray,
    Q: np.ndarray,
    static_windows: list[tuple[float, float]],
) -> float:
    """Mean gravity error in static windows (m/s²); inf if no static windows."""
    indices: list[int] = []
    for t0, t1 in static_windows:
        indices.extend(np.where((timestamps >= t0) & (timestamps <= t1))[0].tolist())
    if not indices:
        return float("inf")

    errors = []
    for i in indices:
        if i >= len(Q):
            continue
        acc_world = quat_rotate(Q[i], acc[i])
        err = float(np.linalg.norm(acc_world - np.array([0.0, 0.0, _G])))
        if np.isfinite(err):
            errors.append(err)
    return float(np.mean(errors)) if errors else float("inf")


def _quality_label(residual: float) -> str:
    if residual < 0.5:
        return "good"
    if residual < 1.5:
        return "marginal"
    return "poor"


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------


def _build_output_df(
    timestamps: np.ndarray,
    Q: np.ndarray,
    gyro_dps: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the per-sample orientation CSV frame.

    When ``gyro_dps`` is provided, the calibrated body-frame gyroscope is
    rotated into the world frame using the time-varying quaternion to give
    instantaneous angular rate around each world axis.  ``gyro_world_z_dps``
    is the directly-measured yaw rate — preferred over differencing
    ``yaw_deg`` because it has no Mahony lag, no ±180° wraparound, and uses
    the full sensor bandwidth (head turns of <0.5 s are visible).
    """
    eulers = np.array([euler_from_quat(q) for q in Q], dtype=float)  # (N, 3) yaw, pitch, roll
    out = pd.DataFrame({
        "timestamp": timestamps.astype(float),
        "qw": Q[:, 0].astype(float),
        "qx": Q[:, 1].astype(float),
        "qy": Q[:, 2].astype(float),
        "qz": Q[:, 3].astype(float),
        "roll_deg":  (eulers[:, 2] * _DEG).astype(float),
        "pitch_deg": (eulers[:, 1] * _DEG).astype(float),
        "yaw_deg":   (eulers[:, 0] * _DEG).astype(float),
    })

    if gyro_dps is not None and len(gyro_dps) == len(Q):
        gyro_world = quat_rotate_batch(Q, gyro_dps)
        out["gyro_world_x_dps"] = gyro_world[:, 0]
        out["gyro_world_y_dps"] = gyro_world[:, 1]
        out["gyro_world_z_dps"] = gyro_world[:, 2]

    return out


# ---------------------------------------------------------------------------
# Public section/recording pipeline
# ---------------------------------------------------------------------------


def process_section_orientation(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    Kp: float = _DEFAULT_KP,
    Ki: float = _DEFAULT_KI,
) -> dict[str, Any]:
    """Run Mahony orientation estimation for one section.

    Parameters
    ----------
    section_dir:
        Path to the section directory.
    sample_rate_hz:
        Nominal IMU sampling rate in Hz.
    force:
        Overwrite existing outputs.
    Kp, Ki:
        Mahony filter gains.

    Returns
    -------
    Stats dict written to ``orientation_stats.json``.
    """
    orient_dir = section_stage_dir(section_dir.name, "orientation")
    stats_json = orient_dir / "orientation_stats.json"

    if stats_json.exists() and not force:
        log.info("Orientation already exists for %s — skipping", project_relative_path(section_dir))
        return read_json_file(stats_json)

    orient_dir.mkdir(parents=True, exist_ok=True)

    cal = _load_section_calibration(section_dir)
    dt_nominal = 1.0 / sample_rate_hz

    stats: dict[str, Any] = {
        "method": "mahony",
        "Kp": Kp,
        "Ki": Ki,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sensors": {},
    }

    for sensor in _SENSORS:
        cal_csv = section_stage_dir(section_dir.name, "calibrated") / f"{sensor}.csv"
        if not cal_csv.exists():
            log.warning("Calibrated CSV missing for %s in %s", sensor, section_dir.name)
            continue
        df = read_csv(cal_csv)
        if df.empty:
            log.warning("Empty calibrated CSV for %s in %s", sensor, section_dir.name)
            continue

        timestamps = df["timestamp"].to_numpy(dtype=float)
        acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
        gyro_dps = df[["gx", "gy", "gz"]].to_numpy(dtype=float)
        gyro_rad = gyro_dps * _RAD

        dt_arr = np.diff(timestamps, prepend=timestamps[0] - dt_nominal * 1000.0) / 1000.0
        dt_arr = np.clip(dt_arr, dt_nominal * 0.1, dt_nominal * 10.0)

        mag_cols = [c for c in ("mx", "my", "mz") if c in df.columns]
        mag: np.ndarray | None = None
        mag_cal: MagCalibration | None = None
        if len(mag_cols) == 3:
            mag_raw = df[mag_cols].to_numpy(dtype=float)
            mag_cal = _resolve_mag_calibration(sensor, mag_raw)
            if mag_cal is not None:
                mag = mag_cal.apply(mag_raw)
            else:
                mag = mag_raw

        q0 = _initial_quaternion(cal, sensor)
        mag_available = mag is not None and np.any(np.all(np.isfinite(mag), axis=1))

        if q0 is None:
            first_valid = np.where(np.all(np.isfinite(acc), axis=1))[0]
            q0 = tilt_quat_from_acc(acc[first_valid[0]]) if len(first_valid) > 0 else quat_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
        elif not mag_available:
            # No mag: strip yaw from calibration q0 so filter and init are consistent.
            g_body = quat_rotate(quat_conjugate(q0), np.array([0.0, 0.0, _G]))
            q0 = tilt_quat_from_acc(g_body)

        Q = run_mahony(acc, gyro_rad, dt_arr, q0=q0, mag=mag, Kp=Kp, Ki=Ki)

        static_wins = _static_windows(cal, sensor)
        residual = _gravity_residual(timestamps, acc, Q, static_wins)

        out_df = _build_output_df(timestamps, Q, gyro_dps=gyro_dps)
        write_csv(out_df, orient_dir / f"{sensor}.csv")

        stats["sensors"][sensor] = {
            "gravity_residual_ms2": round(residual, 4) if np.isfinite(residual) else None,
            "quality": _quality_label(residual),
            "mag_calibration": mag_cal.to_dict() if mag_cal is not None else None,
        }
        log.info(
            "%s/%s: gravity_residual=%.4f m/s² (%s)",
            section_dir.name, sensor, residual, _quality_label(residual),
        )

        try:
            from visualization.plot_orientation import plot_orientation_stage
            plot_orientation_stage(section_dir)
        except Exception as exc:
            log.warning("Orientation plot failed for %s/%s: %s", section_dir.name, sensor, exc)

    write_json_file(stats_json, stats)
    log.info("Orientation done for %s → %s", section_dir.name, project_relative_path(orient_dir))
    return stats


def process_recording_orientation(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    Kp: float = _DEFAULT_KP,
    Ki: float = _DEFAULT_KI,
    **_ignored,
) -> list[dict[str, Any]]:
    """Run orientation estimation for all sections of a recording."""
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'", recording_name)
        return []

    results: list[dict[str, Any]] = []
    for sec_dir in section_dirs:
        log.info("Processing orientation for %s ...", sec_dir.name)
        try:
            stats = process_section_orientation(
                sec_dir, sample_rate_hz=sample_rate_hz, force=force, Kp=Kp, Ki=Ki,
            )
            results.append(stats)
        except Exception as exc:
            log.error("Orientation failed for %s: %s", sec_dir.name, exc)
    return results
