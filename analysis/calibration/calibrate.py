"""Ride-level world-frame calibration: static correction and gravity alignment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from common import (
    load_dataframe,
    recording_dir,
    recordings_root,
    write_dataframe,
)

from ._paths import static_calibration_json_path

log = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.81
DEG_TO_RAD = np.pi / 180.0
AXES = ("x", "y", "z")
ACC_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]


def _acc_norm(df: pd.DataFrame) -> np.ndarray:
    """Compute accelerometer norm per row."""
    acc = df[ACC_COLS].to_numpy(dtype=float)
    return np.sqrt(np.nansum(acc * acc, axis=1))


def _find_static_window(
    df: pd.DataFrame,
    *,
    static_window_seconds: float = 3.0,
    variance_threshold: float = 0.5,
) -> tuple[int, int] | None:
    """Find first N-second window where acc_norm variance is below threshold.

    Returns (start_idx, end_idx) or None if not found.
    """
    if df.empty or "timestamp" not in df.columns:
        return None
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    ts = ts.dropna()
    if ts.empty or len(ts) < 2:
        return None
    t0 = float(ts.iloc[0])
    dt_ms = (ts.iloc[-1] - t0) / max(1, len(ts) - 1)
    # Samples per window based on time span; require at least 10 samples
    samples_per_window = max(10, int(static_window_seconds * 1000.0 / max(1.0, dt_ms)))
    samples_per_window = min(samples_per_window, len(df) - 1)
    if samples_per_window < 10:
        samples_per_window = min(10, len(df))
    if len(df) < samples_per_window:
        samples_per_window = len(df)
    acc_norm = _acc_norm(df)
    valid = np.isfinite(acc_norm)
    for start in range(0, len(df) - samples_per_window + 1):
        end = start + samples_per_window
        window_norm = acc_norm[start:end]
        window_valid = valid[start:end]
        if np.sum(window_valid) < samples_per_window // 2:
            continue
        window_norm = window_norm[window_valid]
        var = np.var(window_norm)
        if var < variance_threshold**2:  # threshold is std-like
            return (start, end)
    return None


def _compute_rotation_matrix(g_hat: np.ndarray) -> np.ndarray:
    """Compute rotation that aligns g_hat to [0, 0, 9.81] (world +Z)."""
    g = np.asarray(g_hat, dtype=float)
    if np.linalg.norm(g) < 1e-6:
        return np.eye(3)
    target = np.array([[0.0, 0.0, GRAVITY_M_S2]])
    src = g.reshape(1, 3)
    rot, _ = Rotation.align_vectors(target, src)
    # align_vectors returns R such that R @ src = target, so R @ g_hat = [0,0,9.81]
    # We want to rotate sensor frame INTO world frame: v_world = R @ v_sensor
    # So R maps g_hat (sensor) -> target (world). The Rotation we get does that.
    return rot.as_matrix()


def _apply_rotation_to_columns(
    df: pd.DataFrame,
    R: np.ndarray,
    acc_cols: list[str],
    gyro_cols: list[str],
) -> pd.DataFrame:
    """Apply rotation matrix to acc and gyro columns in-place style; return new df."""
    out = df.copy()
    for cols in (acc_cols, gyro_cols):
        if not all(c in out.columns for c in cols):
            continue
        arr = out[cols].to_numpy(dtype=float)
        rotated = (R @ arr.T).T
        for i, c in enumerate(cols):
            out[c] = rotated[:, i]
    return out


def calibrate_section(
    section_path: Path,
    *,
    static_window_seconds: float = 3.0,
    variance_threshold: float = 0.5,
    static_calib_path: Path | None = None,
    write_plots: bool = True,
    frame_alignment: str = "gravity_only",
    forward_min_motion_ms2: float = 0.35,
) -> dict[str, Any]:
    """Calibrate a section: apply static calibration (Arduino), align to world frame.

    Reads sporsa.csv and arduino.csv from section_path.
    Writes calibrated/*.csv, calibration.json, and plots to section_path/calibrated/.

    Parameters
    ----------
    frame_alignment:
        ``gravity_only`` (default), ``gravity_plus_forward`` (per-sensor mean horizontal
        specific force), or ``section_horizontal_frame`` (recommended; section-level yaw
        estimated from reference bike sensor horizontal dynamics and transferred to both sensors).
    forward_min_motion_ms2:
        Motion threshold (m/s²) for forward-axis estimation samples.

    Returns calibration metadata dict.
    """
    section_path = Path(section_path)
    if not section_path.exists():
        raise FileNotFoundError(f"Section directory not found: {section_path}")

    # Resolve section path: may be absolute or relative to recordings root
    if not section_path.is_absolute():
        section_path = (recordings_root() / section_path).resolve()
    if not section_path.exists():
        raise FileNotFoundError(f"Section directory not found: {section_path}")

    calibrated_dir = section_path / "calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)

    # Load static calibration for Arduino
    cal_path = static_calib_path or static_calibration_json_path()
    arduino_cal: dict[str, Any] | None = None
    if cal_path.exists():
        with cal_path.open("r", encoding="utf-8") as f:
            arduino_cal = json.load(f)
    else:
        log.warning("Static calibration not found at %s; Arduino will use raw values", cal_path)

    result: dict[str, Any] = {}
    prepared: dict[str, dict[str, Any]] = {}
    sensor_order = ("sporsa", "arduino")
    alignment_mode = frame_alignment.strip().lower()
    if alignment_mode not in {"gravity_only", "gravity_plus_forward", "section_horizontal_frame"}:
        log.warning("Unknown frame_alignment %r — using gravity_only", frame_alignment)
        alignment_mode = "gravity_only"

    for sensor in sensor_order:
        csv_path = section_path / f"{sensor}.csv"
        if not csv_path.exists():
            log.warning("Missing %s — skipping", csv_path)
            continue

        df = load_dataframe(csv_path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            log.warning("%s is empty — skipping", csv_path)
            continue

        # Apply static calibration for Arduino only
        if sensor == "arduino" and arduino_cal is not None:
            from static_calibration.imu_static import apply_calibration_to_dataframe
            df = apply_calibration_to_dataframe(df, arduino_cal)

        # Convert gyro deg/s -> rad/s
        for c in GYRO_COLS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce") * DEG_TO_RAD

        # Find static window
        window = _find_static_window(
            df,
            static_window_seconds=static_window_seconds,
            variance_threshold=variance_threshold,
        )
        if window is None:
            log.warning(
                "No static window found for %s (window=%.1fs, threshold=%.2f) — using first 3s",
                sensor, static_window_seconds, variance_threshold,
            )
            n_samples = min(int(3.0 * 100), len(df))  # ~100 Hz estimate
            window = (0, min(n_samples, len(df)))

        start_idx, end_idx = window
        static_acc = df.iloc[start_idx:end_idx][ACC_COLS].to_numpy(dtype=float)
        g_hat = np.nanmean(static_acc, axis=0)
        g_hat = np.asarray(g_hat, dtype=float)
        if np.linalg.norm(g_hat) < 1e-6:
            g_hat = np.array([0.0, 0.0, -GRAVITY_M_S2])  # fallback
        else:
            g_hat = g_hat / np.linalg.norm(g_hat) * GRAVITY_M_S2

        acc_norm_static = _acc_norm(df.iloc[start_idx:end_idx])
        gravity_residual = float(np.nanstd(np.abs(acc_norm_static - GRAVITY_M_S2)))
        n_static_samples = end_idx - start_idx

        R_grav = _compute_rotation_matrix(g_hat)
        prepared[sensor] = {
            "df": df,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "g_hat": g_hat,
            "gravity_residual": gravity_residual,
            "n_static_samples": n_static_samples,
            "R_grav": R_grav,
        }

    if not prepared:
        return result

    section_frame_meta: dict[str, Any] = {
        "frame_alignment": alignment_mode,
        "reference_sensor": "sporsa",
        "fallback": False,
        "fallback_reason": "",
    }
    R_section_yaw = np.eye(3)
    if alignment_mode == "section_horizontal_frame":
        from calibration.frame_alignment import estimate_section_horizontal_frame

        ref_sensor = "sporsa" if "sporsa" in prepared else next(iter(prepared.keys()))
        ref = prepared[ref_sensor]
        acc_ref = ref["df"][ACC_COLS].to_numpy(dtype=float)
        acc_ref_world = (ref["R_grav"] @ acc_ref.T).T
        mag_ref_world: np.ndarray | None = None
        if all(c in ref["df"].columns for c in ("mx", "my", "mz")):
            mag_ref = ref["df"][["mx", "my", "mz"]].to_numpy(dtype=float)
            mag_ref_world = (ref["R_grav"] @ mag_ref.T).T
        R_section_yaw, section_frame_meta = estimate_section_horizontal_frame(
            acc_ref_world,
            mag_world_ref=mag_ref_world,
            static_indices=slice(ref["start_idx"], ref["end_idx"]),
            min_motion_ms2=forward_min_motion_ms2,
        )
        section_frame_meta["frame_alignment"] = "section_horizontal_frame"
        section_frame_meta["reference_sensor"] = ref_sensor

    for sensor in sensor_order:
        if sensor not in prepared:
            continue
        block = prepared[sensor]
        df = block["df"]
        R = block["R_grav"]
        forward_meta: dict[str, Any] = {}
        if alignment_mode == "gravity_plus_forward":
            from calibration.frame_alignment import estimate_yaw_align_forward

            acc_sensor = df[ACC_COLS].to_numpy(dtype=float)
            acc_world = (R @ acc_sensor.T).T
            static_slice = slice(block["start_idx"], block["end_idx"])
            R_yaw, forward_meta = estimate_yaw_align_forward(
                acc_world,
                static_indices=static_slice,
                min_motion_ms2=forward_min_motion_ms2,
            )
            R = R_yaw @ R
            forward_meta["frame_alignment"] = "gravity_plus_forward"
        elif alignment_mode == "section_horizontal_frame":
            R = R_section_yaw @ R
            forward_meta = dict(section_frame_meta)
            forward_meta["frame_alignment"] = "section_horizontal_frame"
            forward_meta["applied_sensor"] = sensor

        df_cal = _apply_rotation_to_columns(df, R, ACC_COLS, GYRO_COLS)

        out_csv = calibrated_dir / f"{sensor}.csv"
        write_dataframe(df_cal, out_csv)

        calib_quality = "good"
        gravity_residual = float(block["gravity_residual"])
        n_static_samples = int(block["n_static_samples"])
        if gravity_residual > 0.5 or n_static_samples < 100:
            calib_quality = "marginal"
        if gravity_residual > 1.0 or n_static_samples < 30:
            calib_quality = "poor"
        if alignment_mode in {"gravity_plus_forward", "section_horizontal_frame"} and forward_meta.get("fallback"):
            if calib_quality == "good":
                calib_quality = "marginal"

        result[sensor] = {
            "sensor": sensor,
            "static_window_seconds": static_window_seconds,
            "g_hat_sensor_frame": block["g_hat"].tolist(),
            "gravity_residual_m_per_s2": gravity_residual,
            "n_static_samples": n_static_samples,
            "rotation_matrix": R.tolist(),
            "gyro_unit": "rad_per_s",
            "frame_alignment": alignment_mode,
            "forward_frame_meta": forward_meta if alignment_mode != "gravity_only" else {},
            "calibration_quality": calib_quality,
        }

    # Write calibration.json (both sensors)
    cal_json_path = calibrated_dir / "calibration.json"
    with cal_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if alignment_mode == "section_horizontal_frame":
        section_frame_path = calibrated_dir / "section_frame.json"
        with section_frame_path.open("w", encoding="utf-8") as f:
            json.dump(section_frame_meta, f, indent=2)

    if write_plots and result:
        from .plots import plot_calibration_diagnostics
        plot_calibration_diagnostics(
            section_path=section_path,
            calibrated_dir=calibrated_dir,
            calibration=result,
        )

    return result


def _resolve_section_dir(arg: str) -> Path | None:
    """Resolve a CLI argument to an existing section directory.

    Accepted forms:
    - A direct path to a section dir (absolute or relative), e.g. ``data/sections/2026-02-26_r2s1``
    - A section folder name, e.g. ``2026-02-26_r2s1`` (resolved under ``data/sections``)
    """
    raw = arg.strip().rstrip("/")
    if not raw:
        return None

    p = Path(raw)
    if p.is_dir():
        return p.resolve()

    # Accept bare section folder names under data/sections/
    from common.paths import sections_root

    cand = sections_root() / raw
    if cand.is_dir():
        return cand.resolve()

    return None


def calibrate_sections_from_args(
    name: str,
    *,
    all_sections: bool = False,
    static_window_seconds: float = 3.0,
    variance_threshold: float = 0.5,
    frame_alignment: str = "gravity_only",
    forward_min_motion_ms2: float = 0.35,
) -> list[Path]:
    """Calibrate section(s) based on CLI-style arguments."""
    if all_sections:
        from common.paths import iter_sections_for_recording

        recording_name = name.strip().rstrip("/")
        if "/" in recording_name or "\\" in recording_name:
            raise ValueError(
                "When using --all-sections, pass a recording folder name like "
                "'2026-02-26_r2' (not a path)."
            )

        section_dirs = iter_sections_for_recording(recording_name)
        if not section_dirs:
            raise FileNotFoundError(f"No sections found for recording: {recording_name}")
    else:
        sec_dir = _resolve_section_dir(name)
        if sec_dir is None:
            raise ValueError(
                "Specify a section directory like 'data/sections/2026-02-26_r2s1' "
                "(or just '2026-02-26_r2s1'), or use --all-sections with a recording "
                "name like '2026-02-26_r2'."
            )
        section_dirs = [sec_dir]

    done: list[Path] = []
    for sec_path in section_dirs:
        calibrate_section(
            sec_path,
            static_window_seconds=static_window_seconds,
            variance_threshold=variance_threshold,
            frame_alignment=frame_alignment,
            forward_min_motion_ms2=forward_min_motion_ms2,
        )
        done.append(sec_path)
    return done


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(prog="python -m calibration.calibrate")
    parser.add_argument("name", help="Section path or recording name with --all-sections")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument("--static-window", type=float, default=3.0)
    parser.add_argument("--variance-threshold", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    calibrate_sections_from_args(
        args.name,
        all_sections=args.all_sections,
        static_window_seconds=args.static_window,
        variance_threshold=args.variance_threshold,
    )
