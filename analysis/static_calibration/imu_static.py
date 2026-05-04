"""IMU static helpers to estimate per-sensor static calibration values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import data_root, write_csv
from parser.arduino import parse_arduino_log
from parser.sporsa import parse_sporsa_log

GRAVITY = 9.81
AXES = ("x", "y", "z")
FACE_ORDER = ("x+", "x-", "y+", "y-", "z+", "z-")

_FACE_TO_PARSED_STEM: dict[str, str] = {
    "x+": "x_pos",
    "x-": "x_neg",
    "y+": "y_pos",
    "y-": "y_neg",
    "z+": "z_pos",
    "z-": "z_neg",
}


def calibration_data_dir() -> Path:
    """Directory for static calibration assets: ``<data_root>/_calibrations``."""

    return data_root() / "_calibrations"


def calibration_sensor_dir(sensor: str) -> Path:
    """Return per-sensor static-calibration directory."""
    return calibration_data_dir() / sensor


def default_calibration_raw_logs(sensor: str) -> list[Path]:
    """Default raw logs from ``<calibration_data_dir>/<sensor>/raw``."""
    root = calibration_sensor_dir(sensor)
    return sorted((root / "raw").glob("*.txt"))


def _parse_sensor_log(path: Path, sensor: str) -> pd.DataFrame:
    """Parse a sensor raw log into canonical IMU columns."""
    if sensor == "arduino":
        return parse_arduino_log(path)
    if sensor == "sporsa":
        return parse_sporsa_log(path)
    raise ValueError(f"Unsupported sensor for static calibration: {sensor!r}")


def _trim_by_time(df: pd.DataFrame, trim_fraction: float) -> pd.DataFrame:
    """Keep the central ``1 - 2 * trim_fraction`` of the timeline by timestamp."""

    if df.empty or trim_fraction <= 0:
        return df
    ts = df["timestamp"].to_numpy(dtype=float)
    t0, t1 = float(ts.min()), float(ts.max())
    span = t1 - t0
    if span <= 0:
        return df
    margin = span * trim_fraction
    lo, hi = t0 + margin, t1 - margin
    if hi <= lo:
        return df
    return df[(df["timestamp"] >= lo) & (df["timestamp"] <= hi)].reset_index(drop=True)


def _dominant_face(acc_mean: np.ndarray) -> str:
    """Return dominant face."""
    axis_index = int(np.argmax(np.abs(acc_mean)))
    sign = "+" if acc_mean[axis_index] >= 0 else "-"
    return f"{AXES[axis_index]}{sign}"


def summarize_stationary_recording(
    df: pd.DataFrame,
    stem: str,
    *,
    trim_fraction: float = 0.05,
) -> dict[str, Any]:
    """Compute mean IMU and timestamps over the (trimmed) stationary segment."""

    work = _trim_by_time(df, trim_fraction)
    if work.empty:
        raise ValueError(f"{stem}: no samples after time trim.")

    acc = work[["ax", "ay", "az"]].dropna()
    gyro = work[["gx", "gy", "gz"]].dropna()
    if acc.empty:
        raise ValueError(f"{stem}: no accelerometer samples.")
    if gyro.empty:
        raise ValueError(f"{stem}: no gyroscope samples.")

    acc_mean = acc.mean().to_numpy(dtype=float)
    dominant = _dominant_face(acc_mean)
    t_start = int(work["timestamp"].iloc[0])
    t_end = int(work["timestamp"].iloc[-1])

    mag_cols = [c for c in ("mx", "my", "mz") if c in work.columns]
    mean_mag: dict[str, float] | None = None
    n_samples_mag = 0
    if len(mag_cols) == 3:
        mag = work[mag_cols].dropna()
        if not mag.empty:
            mean_mag = {axis: float(mag[f"m{axis}"].mean()) for axis in AXES}
            n_samples_mag = int(len(mag))

    return {
        "stem": stem,
        "dominant_face": dominant,
        "n_samples_acc": int(len(acc)),
        "n_samples_gyro": int(len(gyro)),
        "n_samples_mag": n_samples_mag,
        "mean_acc": {axis: float(acc[f"a{axis}"].mean()) for axis in AXES},
        "mean_gyro": {axis: float(gyro[f"g{axis}"].mean()) for axis in AXES},
        "mean_mag": mean_mag,
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "selected_duration_seconds": float((t_end - t_start) / 1000.0),
    }


def _weighted_mean(values: list[tuple[float, int]]) -> float:
    """Return weighted mean."""
    if not values:
        raise ValueError("Need at least one value to compute a weighted mean.")
    samples = np.array([value for value, _ in values], dtype=float)
    weights = np.array([weight for _, weight in values], dtype=float)
    return float(np.average(samples, weights=weights))


def _estimate_accelerometer_axis(
    summaries: list[dict[str, Any]],
    axis: str,
    warnings: list[str],
) -> tuple[float, float]:
    """Estimate accelerometer axis."""
    positive = [s for s in summaries if s["dominant_face"] == f"{axis}+"]
    negative = [s for s in summaries if s["dominant_face"] == f"{axis}-"]
    zero_faces = [s for s in summaries if s["dominant_face"][0] != axis]

    if positive and negative:
        positive_mean = _weighted_mean(
            [(s["mean_acc"][axis], s["n_samples_acc"]) for s in positive]
        )
        negative_mean = _weighted_mean(
            [(s["mean_acc"][axis], s["n_samples_acc"]) for s in negative]
        )
        bias = (positive_mean + negative_mean) / 2.0
        scale = (2.0 * GRAVITY) / (positive_mean - negative_mean)
    else:
        if not zero_faces:
            raise ValueError(f"Axis {axis}: missing zero-g faces for fallback bias estimation.")

        bias = _weighted_mean(
            [(s["mean_acc"][axis], s["n_samples_acc"]) for s in zero_faces]
        )
        one_sided = positive or negative
        if not one_sided:
            raise ValueError(f"Axis {axis}: missing both {axis}+ and {axis}- recordings.")

        expected_sign = 1.0 if positive else -1.0
        measured = _weighted_mean(
            [(s["mean_acc"][axis], s["n_samples_acc"]) for s in one_sided]
        )
        scale = (expected_sign * GRAVITY) / (measured - bias)
        missing_face = f"{axis}-" if positive else f"{axis}+"
        warnings.append(
            f"Missing {missing_face} recording for accelerometer axis {axis}; used one-sided fallback."
        )

    if scale <= 0:
        raise ValueError(f"Axis {axis}: computed non-positive accelerometer scale {scale}.")
    return float(bias), float(scale)


def _estimate_mag_hard_iron(mag_means: list[np.ndarray]) -> np.ndarray:
    """Estimate mag hard iron."""
    M = np.array(mag_means, dtype=float)
    n = len(M)
    if n < 4:
        return np.zeros(3)
    # Linear system: [2*m | 1] @ [cx, cy, cz, (||c||² - r²)]ᵀ = ||m||²
    A = np.hstack([2.0 * M, np.ones((n, 1))])
    b = np.sum(M ** 2, axis=1)
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:3]


def estimate_calibration_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate accelerometer bias/scale, gyroscope bias, and magnetometer hard-iron from per-recording summaries."""

    if not summaries:
        raise ValueError("Need at least one recording summary.")

    warnings: list[str] = []
    face_counts = {face: sum(1 for s in summaries if s["dominant_face"] == face) for face in FACE_ORDER}

    for face, count in face_counts.items():
        if count == 0:
            warnings.append(f"No recording was classified as {face}.")
        elif count > 1:
            warnings.append(
                f"Multiple recordings were classified as {face}; means were combined with sample-count weighting."
            )

    acc_bias: dict[str, float] = {}
    acc_scale: dict[str, float] = {}
    for axis in AXES:
        bias, scale = _estimate_accelerometer_axis(summaries, axis, warnings)
        acc_bias[axis] = bias
        acc_scale[axis] = scale

    gyro_bias = {
        axis: _weighted_mean(
            [(s["mean_gyro"][axis], s["n_samples_gyro"]) for s in summaries]
        )
        for axis in AXES
    }

    # Magnetometer hard-iron estimation from all faces with valid mag readings.
    mag_hi: dict[str, float] | None = None
    mag_means = [
        np.array([s["mean_mag"]["x"], s["mean_mag"]["y"], s["mean_mag"]["z"]])
        for s in summaries
        if s.get("mean_mag") is not None
    ]
    if len(mag_means) >= 4:
        hi = _estimate_mag_hard_iron(mag_means)
        mag_hi = {"x": float(hi[0]), "y": float(hi[1]), "z": float(hi[2])}
    else:
        warnings.append(
            f"Magnetometer hard-iron not estimated: only {len(mag_means)} faces had mag readings (need ≥ 4)."
        )

    return {
        "gravity_m_s2": GRAVITY,
        "accelerometer": {"bias": acc_bias, "scale": acc_scale},
        "gyroscope": {"bias_deg_s": gyro_bias},
        "magnetometer": {"hard_iron_bias": mag_hi},
        "face_counts": face_counts,
        "warnings": warnings,
    }


def write_parsed_csvs(
    log_paths: list[Path],
    parsed_dir: Path,
    *,
    sensor: str,
    trim_fraction: float = 0.05,
) -> dict[str, Path]:
    """Write parsed csvs."""

    parsed_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for log_number, path in enumerate(sorted(Path(p) for p in log_paths)):
        df = _parse_sensor_log(path, sensor)

        work = _trim_by_time(df, trim_fraction)
        acc = work[["ax", "ay", "az"]].dropna()
        if acc.empty:
            raise ValueError(f"{path}: cannot infer face: no accelerometer samples.")
        face = _dominant_face(acc.mean().to_numpy(dtype=float))
        label = _FACE_TO_PARSED_STEM.get(face) or "unknown"
        stem = f"log{log_number}({label})"
        out = parsed_dir / f"{stem}.csv"
        write_csv(df, out)
        mapping[stem] = out
    return mapping


def load_calibration(calibration_path: Path) -> dict[str, Any]:
    """Load a calibration JSON file."""

    with calibration_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_calibration_to_dataframe(df: pd.DataFrame, calibration: dict[str, Any]) -> pd.DataFrame:
    """Apply accelerometer scale/bias and gyroscope bias to a DataFrame."""

    calibrated = df.copy()
    for axis in AXES:
        acc_column = f"a{axis}"
        gyro_column = f"g{axis}"
        if acc_column in calibrated.columns:
            calibrated[acc_column] = (
                pd.to_numeric(calibrated[acc_column], errors="coerce")
                - calibration["accelerometer"]["bias"][axis]
            ) * calibration["accelerometer"]["scale"][axis]
        if gyro_column in calibrated.columns:
            calibrated[gyro_column] = (
                pd.to_numeric(calibrated[gyro_column], errors="coerce")
                - calibration["gyroscope"]["bias_deg_s"][axis]
            )
    return calibrated
