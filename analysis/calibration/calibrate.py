"""Ride-level world-frame calibration: static correction and gravity alignment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
) -> dict[str, Any]:
    """Calibrate a section: apply static calibration (Arduino), align to world frame.

    Reads sporsa.csv and arduino.csv from section_path.
    Writes calibrated/*.csv, calibration.json, and plots to section_path/calibrated/.

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

    for sensor in ("sporsa", "arduino"):
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

        R = _compute_rotation_matrix(g_hat)
        df_cal = _apply_rotation_to_columns(df, R, ACC_COLS, GYRO_COLS)

        out_csv = calibrated_dir / f"{sensor}.csv"
        write_dataframe(df_cal, out_csv)

        result[sensor] = {
            "sensor": sensor,
            "static_window_seconds": static_window_seconds,
            "g_hat_sensor_frame": g_hat.tolist(),
            "gravity_residual_m_per_s2": gravity_residual,
            "n_static_samples": n_static_samples,
            "rotation_matrix": R.tolist(),
            "gyro_unit": "rad_per_s",
        }

    # Write calibration.json (both sensors)
    cal_json_path = calibrated_dir / "calibration.json"
    with cal_json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if write_plots and result:
        from .plots import plot_calibration_diagnostics
        plot_calibration_diagnostics(
            section_path=section_path,
            calibrated_dir=calibrated_dir,
            calibration=result,
        )

    return result


def _parse_section_arg(arg: str) -> tuple[str, str | None]:
    """Parse CLI argument into (recording_name, section_name or None)."""
    arg = arg.strip().rstrip("/")
    parts = arg.replace("\\", "/").split("/")
    recording_name = parts[0]
    section_name: str | None = None
    for i, p in enumerate(parts):
        if p == "sections" and i + 1 < len(parts):
            section_name = parts[i + 1]
            break
    return recording_name, section_name


def calibrate_sections_from_args(
    name: str,
    *,
    all_sections: bool = False,
    static_window_seconds: float = 3.0,
    variance_threshold: float = 0.5,
) -> list[Path]:
    """Calibrate section(s) based on CLI-style arguments."""
    recording_name, section_name = _parse_section_arg(name)
    rec_dir = recording_dir(recording_name)
    if not rec_dir.exists():
        raise FileNotFoundError(f"Recording directory not found: {rec_dir}")

    sections_root = rec_dir / "sections"
    if not sections_root.exists():
        raise FileNotFoundError(f"No sections directory: {sections_root}")

    if all_sections:
        section_dirs = sorted(
            d for d in sections_root.iterdir()
            if d.is_dir() and d.name.startswith("section_")
        )
        if not section_dirs:
            raise FileNotFoundError(f"No sections found in {sections_root}")
    else:
        if section_name is None:
            raise ValueError(
                "Specify a section (e.g. 2026-02-26_5/sections/section_1) or use --all-sections"
            )
        sec_dir = sections_root / section_name
        if not sec_dir.exists():
            raise FileNotFoundError(f"Section not found: {sec_dir}")
        section_dirs = [sec_dir]

    done: list[Path] = []
    for sec_path in section_dirs:
        calibrate_section(
            sec_path,
            static_window_seconds=static_window_seconds,
            variance_threshold=variance_threshold,
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
