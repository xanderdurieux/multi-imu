"""Orientation estimation pipeline for calibrated IMU sections.

For each section:
1. Load ``calibrated/{sensor}.csv`` and ``calibration/calibration.json``.
2. Initialise filters from the calibration body→world rotation matrix so the
   estimate is physically grounded from the first sample.
3. Apply magnetometer hard-iron correction for sensors that have a static
   calibration with magnetometer data (Arduino).
4. Run the configured method(s), score each with a gravity-residual metric
   evaluated in the known-static opening windows, and select the best.
5. Write per-sensor quaternion CSVs (selected method), a per-sensor comparison
   CSV with Euler angles for all methods, a stats JSON, and comparison plots.

Output layout
-------------
::

    orientation/
        sporsa.csv               # selected method quaternions + Euler angles
        arduino.csv
        sporsa_all_methods.csv   # Euler angles from every method run
        arduino_all_methods.csv
        orientation_stats.json   # scores, selected method, parameters
        sporsa_orientation.png   # comparison plot
        arduino_orientation.png

The selected method's quaternion CSVs are the canonical input for the
``derived`` stage.  The all-methods CSVs and plots are retained for analysis.

Subsequent stages that need to be updated when orientation output improves
--------------------------------------------------------------------------
- ``derived/signals.py``: ``_compute_linear_acc`` already uses the quaternion
  to rotate body-frame acc to world and subtract gravity.  No change needed.
- ``features/``: features that reference body-frame axes (ax, ay, az) should
  be re-examined — they are currently meaningful only at the alignment window.
  A future edit should rotate these to world frame using orientation before
  computing features.
- ``events/``: impact and swerve detection based on body-frame acc will
  benefit from world-frame acc once orientation is propagated there.
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
    load_orientation_config_data,
    read_json_file,
    project_relative_path,
    read_csv,
    section_stage_dir,
    static_calibration_json,
    write_csv,
    write_json_file,
)
from common.quaternion import euler_from_quat, quat_conjugate, quat_from_rotation_matrix, quat_normalize, quat_rotate, tilt_quat_from_acc

from .filters import DEFAULT_PARAMS, METHODS, run_filter

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_G = 9.81
_DEG = 180.0 / np.pi
_RAD = np.pi / 180.0

# Score threshold: if best − second-best gravity residual exceeds this value
# (m/s²), the best method is considered "clearly superior" for the section.
_SCORE_CLEAR_MARGIN = 0.30


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _load_section_calibration(section_dir: Path) -> dict[str, Any]:
    cal_path = section_stage_dir(section_dir.name, "calibrated") / "calibration.json"
    return read_json_file(cal_path)

def _initial_quaternion(cal: dict[str, Any], sensor: str) -> np.ndarray | None:
    """Extract the body→world quaternion from the calibration alignment matrix."""
    try:
        R = np.array(cal["alignment"][sensor]["rotation_matrix"], dtype=float)
        return quat_from_rotation_matrix(R)
    except (KeyError, TypeError, ValueError):
        return None


def _static_windows(cal: dict[str, Any], sensor: str) -> list[tuple[float, float]]:
    """Return (start_ms, end_ms) pairs for all known-static windows (opening and closing)."""
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


def _load_mag_hard_iron(sensor: str) -> np.ndarray | None:
    """Load magnetometer hard-iron bias from the static calibration JSON.

    Only Arduino has static calibration recordings; Sporsa returns None.
    """
    if sensor != "arduino":
        return None
    cal_path = static_calibration_json()
    if not cal_path.exists():
        return None
    try:
        cal = read_json_file(cal_path)
        hi = cal.get("magnetometer", {}).get("hard_iron_bias")
        if hi is not None:
            return np.array([hi["x"], hi["y"], hi["z"]], dtype=float)
    except Exception as exc:
        log.warning("Could not load mag hard-iron from static calibration: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_gravity_residual(
    timestamps: np.ndarray,
    acc: np.ndarray,
    Q: np.ndarray,
    static_windows: list[tuple[float, float]],
) -> dict[str, float]:
    """Comprehensive gravity-based scoring.
    
    Returns dict with:
    - residual_static: mean error in static windows
    - residual_dynamic: variance metric outside static windows
    - score: combined metric
    """
    from common.quaternion import quat_rotate

    indices_static: list[int] = []
    indices_dynamic: list[int] = []
    
    for t0, t1 in static_windows:
        mask = np.where((timestamps >= t0) & (timestamps <= t1))[0]
        indices_static.extend(mask.tolist())
    
    all_indices = set(range(len(timestamps)))
    indices_dynamic = list(all_indices - set(indices_static))

    # Static window residual (primary metric)
    residual_static = float("inf")
    if indices_static:
        errors = []
        for i in indices_static:
            if i >= len(Q):
                continue
            acc_world = quat_rotate(Q[i], acc[i])
            err = float(np.linalg.norm(acc_world - np.array([0.0, 0.0, _G])))
            if np.isfinite(err):
                errors.append(err)
        if errors:
            residual_static = float(np.mean(errors))

    # Dynamic window metric (secondary: should be smooth)
    residual_dynamic = 0.0
    if indices_dynamic and len(indices_dynamic) > 10:
        az_world = np.array([
            float(quat_rotate(Q[i], acc[i])[2]) for i in indices_dynamic
            if i < len(Q) and np.all(np.isfinite(Q[i])) and np.all(np.isfinite(acc[i]))
        ])
        if len(az_world) > 1:
            residual_dynamic = float(np.std(az_world))

    # Combined score: prefer low static error, smooth dynamics
    if np.isfinite(residual_static):
        score = residual_static + 0.1 * residual_dynamic
    else:
        score = float("inf")

    return {
        "residual_static": residual_static,
        "residual_dynamic": residual_dynamic,
        "score": score,
    }



def _quality_label(residual: float) -> str:
    if residual < 0.5:
        return "good"
    if residual < 1.5:
        return "marginal"
    return "poor"


# ---------------------------------------------------------------------------
# Core per-sensor filter runner
# ---------------------------------------------------------------------------


def _run_sensor(
    df: pd.DataFrame,
    sensor: str,
    methods: list[str],
    method_params: dict[str, dict],
    q0: np.ndarray | None,
    mag_hard_iron: np.ndarray | None,
    static_windows: list[tuple[float, float]],
    sample_rate_hz: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], np.ndarray]:
    """Run all methods on one sensor; return (Q_dict, residual_dict, timestamps)."""
    timestamps = df["timestamp"].to_numpy(dtype=float)
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyro_deg = df[["gx", "gy", "gz"]].to_numpy(dtype=float)
    gyro_rad = gyro_deg * _RAD

    # Compute per-sample dt from timestamps (ms → s)
    dt_nominal = 1.0 / sample_rate_hz
    dt_arr = np.diff(timestamps, prepend=timestamps[0] - dt_nominal * 1000.0) / 1000.0
    dt_arr = np.clip(dt_arr, dt_nominal * 0.1, dt_nominal * 10.0)

    # Magnetometer (optional)
    mag_cols = [c for c in ("mx", "my", "mz") if c in df.columns]
    mag: np.ndarray | None = None
    if len(mag_cols) == 3:
        mag = df[mag_cols].to_numpy(dtype=float)
        if mag_hard_iron is not None:
            # Subtract hard-iron bias where readings are valid.
            valid = np.all(np.isfinite(mag), axis=1)
            mag[valid] -= mag_hard_iron

    # Resolve initial quaternion. When no magnetometer is available yaw is
    # unobservable, so filters drift towards yaw=0 regardless of how they
    # start. Strip yaw from the calibration-derived q0 in that case so the
    # initial orientation is consistent with where the filter will settle.
    mag_available = mag is not None and np.any(np.all(np.isfinite(mag), axis=1))
    if q0 is None:
        first_valid = np.where(np.all(np.isfinite(acc), axis=1))[0]
        if len(first_valid) == 0:
            q0 = quat_normalize(quat_from_rotation_matrix(np.eye(3)))
        else:
            q0 = tilt_quat_from_acc(acc[first_valid[0]])
    elif not mag_available:
        # Keep calibration roll/pitch but zero yaw: rotate world-z to body
        # frame via the calibration quaternion, then re-derive tilt-only q0.
        g_body = quat_rotate(quat_conjugate(q0), np.array([0.0, 0.0, _G]))
        q0 = tilt_quat_from_acc(g_body)

    Q_dict: dict[str, np.ndarray] = {}
    residual_dict: dict[str, float] = {}

    for method in methods:
        params = method_params.get(method, {})
        try:
            Q = run_filter(
                method, acc, gyro_rad, dt_arr,
                q0=q0.copy() if q0 is not None else None,
                mag=mag,
                **params,
            )
            score_dict = _score_gravity_residual(timestamps, acc, Q, static_windows)
            residual = score_dict.get("score", float("inf"))
        except Exception as exc:
            log.error("Filter %s failed for sensor %s: %s", method, sensor, exc)
            Q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(timestamps), 1))
            residual = float("inf")

        Q_dict[method] = Q
        residual_dict[method] = residual
        log.debug(
            "%s/%s: gravity residual = %.4f m/s² (%s)",
            sensor, method, residual, _quality_label(residual),
        )

    return Q_dict, residual_dict, timestamps


# ---------------------------------------------------------------------------
# Comparison CSV builder
# ---------------------------------------------------------------------------


def _build_comparison_df(
    timestamps: np.ndarray,
    Q_dict: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build a flat DataFrame with Euler angles from all methods."""
    df = pd.DataFrame({"timestamp": timestamps})
    for method, Q in Q_dict.items():
        rolls, pitches, yaws = [], [], []
        for q in Q:
            yaw_r, pitch_r, roll_r = euler_from_quat(q)
            rolls.append(float(roll_r * _DEG))
            pitches.append(float(pitch_r * _DEG))
            yaws.append(float(yaw_r * _DEG))
        df[f"{method}_roll_deg"] = rolls
        df[f"{method}_pitch_deg"] = pitches
        df[f"{method}_yaw_deg"] = yaws
    return df


def _build_output_df(
    timestamps: np.ndarray,
    Q: np.ndarray,
) -> pd.DataFrame:
    """Build the canonical orientation output DataFrame."""
    rows = []
    for i, q in enumerate(Q):
        yaw_r, pitch_r, roll_r = euler_from_quat(q)
        rows.append({
            "timestamp": float(timestamps[i]),
            "qw": float(q[0]), "qx": float(q[1]),
            "qy": float(q[2]), "qz": float(q[3]),
            "roll_deg":  float(roll_r  * _DEG),
            "pitch_deg": float(pitch_r * _DEG),
            "yaw_deg":   float(yaw_r   * _DEG),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------


def _select_method(
    residuals_by_sensor: dict[str, dict[str, float]],
    methods: list[str],
    preferred: str,
) -> str:
    """Select the best orientation method.

    Picks the method with lowest worst-case residual across sensors.
    If multiple methods are close (within margin), prefers the configured default.
    """
    if not methods:
        return "madgwick"

    combined: dict[str, float] = {}
    for method in methods:
        sensor_residuals = [
            residuals_by_sensor[s].get(method, float("inf"))
            for s in _SENSORS
            if s in residuals_by_sensor
        ]
        # Use worst-case (max) to ensure method works for both sensors
        combined[method] = max(sensor_residuals) if sensor_residuals else float("inf")

    sorted_methods = sorted(methods, key=lambda m: combined[m])
    best = sorted_methods[0]
    second_best = sorted_methods[1] if len(sorted_methods) > 1 else best

    score_best = combined[best]
    score_second = combined[second_best]

    # If preferred is competitive, use it
    if preferred in methods and combined[preferred] <= score_best + _SCORE_CLEAR_MARGIN:
        return preferred

    # If no clear winner, prefer configured default
    if (score_second - score_best) < _SCORE_CLEAR_MARGIN and preferred in methods:
        return preferred

    return best


# ---------------------------------------------------------------------------
# Public section/recording pipeline
# ---------------------------------------------------------------------------


def process_section_orientation(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    method: str = "auto",
    method_params: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Run orientation estimation for one section.

    Parameters
    ----------
    section_dir:
        Path to the section directory.
    sample_rate_hz:
        Nominal IMU sampling rate in Hz.
    force:
        Overwrite existing outputs.
    method:
        ``"auto"`` — run all configured methods and select the best.
        Any other value (``"madgwick"``, ``"mahony"``, etc.) — run only that
        method, skip comparison and selection.
    method_params:
        Per-method parameter overrides.  Falls back to ``DEFAULT_PARAMS``.

    Returns
    -------
    Stats dict written to ``orientation_stats.json``.
    """
    orient_dir = section_stage_dir(section_dir.name, "orientation")
    stats_json = orient_dir / "orientation_stats.json"

    if stats_json.exists() and not force:
        log.info(
            "Orientation already exists for %s — skipping (force=True to overwrite)",
            project_relative_path(section_dir),
        )
        return read_json_file(stats_json)

    orient_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which methods to run.
    cfg = load_orientation_config_data()
    if method == "auto":
        methods = list(cfg.get("methods", list(METHODS)))
        preferred = str(cfg.get("method", "auto"))
        if preferred == "auto":
            preferred = "madgwick"
    else:
        methods = [method]
        preferred = method

    if method_params is None:
        method_params = {m: dict(cfg.get(m, {})) for m in methods}

    cal = _load_section_calibration(section_dir)
    created_at = datetime.now(UTC).isoformat()

    sensor_Q: dict[str, dict[str, np.ndarray]] = {}
    sensor_residuals: dict[str, dict[str, float]] = {}
    sensor_timestamps: dict[str, np.ndarray] = {}

    for sensor in _SENSORS:
        cal_csv = section_stage_dir(section_dir.name, "calibrated") / f"{sensor}.csv"
        if not cal_csv.exists():
            log.warning("Calibrated CSV missing for %s in %s", sensor, section_dir.name)
            continue
        df = read_csv(cal_csv)
        if df.empty:
            log.warning("Empty calibrated CSV for %s in %s", sensor, section_dir.name)
            continue

        q0 = _initial_quaternion(cal, sensor)
        if q0 is not None:
            log.debug("%s/%s: initialising from calibration rotation matrix", section_dir.name, sensor)
        else:
            log.debug("%s/%s: no calibration rotation matrix — tilt init from acc", section_dir.name, sensor)

        mag_hi = _load_mag_hard_iron(sensor)
        static_wins = _static_windows(cal, sensor)

        Q_dict, residuals, timestamps = _run_sensor(
            df, sensor, methods, method_params,
            q0, mag_hi, static_wins, sample_rate_hz,
        )
        sensor_Q[sensor] = Q_dict
        sensor_residuals[sensor] = residuals
        sensor_timestamps[sensor] = timestamps

        # Comparison CSV (all methods, Euler angles only)
        cmp_df = _build_comparison_df(timestamps, Q_dict)
        write_csv(cmp_df, orient_dir / f"{sensor}_all_methods.csv")

    # Select method
    if len(methods) > 1:
        selected = _select_method(sensor_residuals, methods, preferred)
    else:
        selected = methods[0]

    log.info("%s: selected orientation method '%s'", section_dir.name, selected)

    # Write canonical outputs and plots
    for sensor in _SENSORS:
        if sensor not in sensor_Q:
            continue
        Q_dict = sensor_Q[sensor]
        timestamps = sensor_timestamps[sensor]
        static_wins = _static_windows(cal, sensor)

        out_df = _build_output_df(timestamps, Q_dict[selected])
        write_csv(out_df, orient_dir / f"{sensor}.csv")

        try:
            from visualization.plot_orientation import plot_orientation_methods_comparison
            plot_orientation_methods_comparison(orient_dir, sensor)
        except Exception as exc:
            log.warning("Orientation plot failed for %s/%s: %s", section_dir.name, sensor, exc)

    # Build stats JSON
    stats: dict[str, Any] = {
        "selected_method": selected,
        "method_params": {m: method_params.get(m, DEFAULT_PARAMS.get(m, {})) for m in methods},
        "created_at_utc": created_at,
        "sensors": {},
        "all_methods": {},
    }
    for sensor in _SENSORS:
        if sensor not in sensor_residuals:
            continue
        residuals = sensor_residuals[sensor]
        stats["sensors"][sensor] = {
            "selected_residual_ms2": round(residuals.get(selected, float("inf")), 4),
            "quality": _quality_label(residuals.get(selected, float("inf"))),
        }
        stats["all_methods"][sensor] = {
            m: {
                "gravity_residual_ms2": round(r, 4),
                "quality": _quality_label(r),
            }
            for m, r in residuals.items()
        }

    write_json_file(stats_json, stats)
    log.info(
        "Orientation done for %s → %s (method=%s)",
        section_dir.name, project_relative_path(orient_dir), selected,
    )
    return stats


def process_recording_orientation(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    method: str = "auto",
    method_params: dict[str, dict] | None = None,
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
                sec_dir,
                sample_rate_hz=sample_rate_hz,
                force=force,
                method=method,
                method_params=method_params,
            )
            results.append(stats)
        except Exception as exc:
            log.error("Orientation failed for %s: %s", sec_dir.name, exc)
    return results
