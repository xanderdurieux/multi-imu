"""Static Arduino IMU calibration from six axis-aligned stationary recordings.

Each raw log should be ~stationary with the board in one of six approximate
orientations so gravity lies primarily along +X, -X, +Y, -Y, +Z, or -Z. Means
are taken over the full recording after an optional time trim at the edges to
ignore placement transients.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import write_csv
from parser.arduino import parse_arduino_log

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
    """Directory for static calibration assets: ``analysis/data/calibrations``."""

    return Path(__file__).resolve().parents[1] / "data" / "calibrations"


def default_calibration_raw_logs() -> list[Path]:
    """Default raw Arduino ``*.txt`` logs: ``<calibration_data_dir>/raw``, else that root."""

    root = calibration_data_dir()
    logs = sorted((root / "raw").glob("*.txt"))
    return logs if logs else sorted(root.glob("*.txt"))


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

    return {
        "stem": stem,
        "dominant_face": dominant,
        "n_samples_acc": int(len(acc)),
        "n_samples_gyro": int(len(gyro)),
        "mean_acc": {axis: float(acc[f"a{axis}"].mean()) for axis in AXES},
        "mean_gyro": {axis: float(gyro[f"g{axis}"].mean()) for axis in AXES},
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "selected_duration_seconds": float((t_end - t_start) / 1000.0),
    }


def _weighted_mean(values: list[tuple[float, int]]) -> float:
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


def estimate_calibration_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate accelerometer bias/scale and gyroscope bias from per-recording summaries."""

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

    return {
        "gravity_m_s2": GRAVITY,
        "accelerometer": {"bias": acc_bias, "scale": acc_scale},
        "gyroscope": {"bias_deg_s": gyro_bias},
        "face_counts": face_counts,
        "warnings": warnings,
    }


def write_parsed_csvs(
    log_paths: list[Path],
    parsed_dir: Path,
    *,
    trim_fraction: float = 0.05,
) -> dict[str, Path]:
    """Parse each raw log and write ``calib_<orientation>.csv`` under ``parsed_dir``.

    Orientation is inferred from mean acceleration (same trim as calibration).
    Unclassified faces use ``calib_unknown_<raw_stem>.csv``.
    """

    parsed_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, Path] = {}
    for log_number, path in enumerate(sorted(Path(p) for p in log_paths)):
        df = parse_arduino_log(path)
        work = _trim_by_time(df, trim_fraction)
        acc = work[["ax", "ay", "az"]].dropna()
        if acc.empty:
            raise ValueError(f"{path}: cannot infer face: no accelerometer samples.")
        face = _dominant_face(acc.mean().to_numpy(dtype=float))
        label = _FACE_TO_PARSED_STEM.get(face)
        stem = f"log{log_number}({label if label else "unknown"})"
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
