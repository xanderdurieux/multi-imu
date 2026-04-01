"""Orientation estimation pipeline for calibrated IMU sections.

Runs the configured orientation method(s), scores them, selects the best
result, and flattens it into ``orientation/``.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import iter_sections_for_recording, read_csv, write_csv
from common.quaternion import euler_from_quat
from .complementary import ComplementaryFilter
from .madgwick import MadgwickFilter

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_DEG_PER_RAD = 180.0 / np.pi
_RAD_PER_DEG = np.pi / 180.0
_ALL_METHODS = ["madgwick", "complementary"]
_DEFAULT_METHOD = "madgwick"
_METHOD_PARAMS: dict[str, dict[str, float]] = {
    "madgwick": {"beta": 0.1},
    "complementary": {"alpha": 0.98},
}


@dataclass
class OrientationStats:
    """Quality statistics for a single sensor + method."""

    sensor: str
    method: str
    gravity_alignment: float
    pitch_std_deg: float
    roll_std_deg: float
    score: float
    quality: str


def _build_filter(method: str, *, sample_rate_hz: float) -> MadgwickFilter | ComplementaryFilter:
    if method == "madgwick":
        return MadgwickFilter(beta=_METHOD_PARAMS["madgwick"]["beta"], sample_rate_hz=sample_rate_hz)
    if method == "complementary":
        return ComplementaryFilter(
            alpha=_METHOD_PARAMS["complementary"]["alpha"],
            sample_rate_hz=sample_rate_hz,
        )
    raise ValueError(f"Unknown orientation method: {method}")


def run_orientation_filters(
    df: pd.DataFrame,
    *,
    sensor_name: str,
    sample_rate_hz: float = 100.0,
    methods: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run orientation filters on calibrated IMU data."""
    if methods is None:
        methods = list(_ALL_METHODS)

    has_world = all(c in df.columns for c in ("ax_world", "ay_world", "az_world"))
    acc_cols = ("ax_world", "ay_world", "az_world") if has_world else ("ax", "ay", "az")

    has_world_gyro = all(c in df.columns for c in ("gx_world", "gy_world", "gz_world"))
    gyro_cols = ("gx_world", "gy_world", "gz_world") if has_world_gyro else ("gx", "gy", "gz")

    timestamps = df["timestamp"].to_numpy(dtype=float)
    acc_arr = df[list(acc_cols)].to_numpy(dtype=float)
    gyro_arr = df[list(gyro_cols)].to_numpy(dtype=float)
    gyro_rad_arr = gyro_arr * _RAD_PER_DEG

    n_samples = len(timestamps)
    if n_samples == 0:
        empty = pd.DataFrame(
            columns=["timestamp", "qw", "qx", "qy", "qz", "yaw_deg", "pitch_deg", "roll_deg"]
        )
        return {method: empty.copy() for method in methods}

    dt_nominal = 1.0 / sample_rate_hz
    dt_arr = np.diff(timestamps, prepend=timestamps[0] - dt_nominal * 1000.0) / 1000.0
    dt_arr = np.clip(dt_arr, dt_nominal * 0.1, dt_nominal * 10.0)
    first_acc = acc_arr[0]

    results: dict[str, pd.DataFrame] = {}
    for method in methods:
        filt = _build_filter(method, sample_rate_hz=sample_rate_hz)
        filt.initialize_from_acc(first_acc)

        rows: list[dict[str, float]] = []
        pending_dt = 0.0
        for i in range(n_samples):
            gyro_i = gyro_rad_arr[i]
            acc_i = acc_arr[i]
            pending_dt += float(dt_arr[i])

            if not np.all(np.isfinite(gyro_i)):
                # No gyro this step — repeat last quaternion, carry dt forward so
                # the next real gyro sample integrates over the full elapsed time.
                q = filt._q.copy()
            else:
                q = filt.update(acc_i, gyro_i, dt=pending_dt)
                pending_dt = 0.0
            yaw_r, pitch_r, roll_r = euler_from_quat(q)
            rows.append(
                {
                    "timestamp": timestamps[i],
                    "qw": float(q[0]),
                    "qx": float(q[1]),
                    "qy": float(q[2]),
                    "qz": float(q[3]),
                    "yaw_deg": float(yaw_r * _DEG_PER_RAD),
                    "pitch_deg": float(pitch_r * _DEG_PER_RAD),
                    "roll_deg": float(roll_r * _DEG_PER_RAD),
                }
            )

        results[method] = pd.DataFrame(rows)
        log.debug("%s/%s: processed %d samples", sensor_name, method, n_samples)

    return results


def _compute_stats(
    df_raw: pd.DataFrame,
    df_orient: pd.DataFrame,
    sensor: str,
    method: str,
) -> OrientationStats:
    """Compute quality statistics for one sensor + method output."""
    acc_col = "az_world" if "az_world" in df_raw.columns else "az"
    az = df_raw[acc_col].dropna().to_numpy(dtype=float)
    if len(az) < 2:
        gravity_alignment = 0.0
    else:
        mean_az = float(np.mean(az))
        std_az = float(np.std(az))
        if abs(mean_az) < 1e-6:
            gravity_alignment = 0.0
        else:
            gravity_alignment = float(np.clip(1.0 - (std_az / mean_az) ** 2, 0.0, 1.0))

    pitch_std = float(df_orient["pitch_deg"].std()) if "pitch_deg" in df_orient.columns else 999.0
    roll_std = float(df_orient["roll_deg"].std()) if "roll_deg" in df_orient.columns else 999.0
    if not np.isfinite(pitch_std):
        pitch_std = 999.0
    if not np.isfinite(roll_std):
        roll_std = 999.0
    score = float(gravity_alignment - 0.0025 * (pitch_std + roll_std))

    if pitch_std >= 999.0 or roll_std >= 999.0:
        quality = "poor"
    elif gravity_alignment < 0.8:
        quality = "poor"
    elif gravity_alignment < 0.95:
        quality = "marginal"
    else:
        quality = "good"

    return OrientationStats(
        sensor=sensor,
        method=method,
        gravity_alignment=round(gravity_alignment, 4),
        pitch_std_deg=round(pitch_std, 4),
        roll_std_deg=round(roll_std, 4),
        score=round(score, 4),
        quality=quality,
    )


def _stats_to_dict(stats: OrientationStats) -> dict[str, Any]:
    return {
        "gravity_alignment": stats.gravity_alignment,
        "pitch_std_deg": stats.pitch_std_deg,
        "roll_std_deg": stats.roll_std_deg,
        "score": stats.score,
        "quality": stats.quality,
    }


def _score_method(method_summary: dict[str, Any]) -> float:
    sensor_scores = [
        float(method_summary.get(sensor, {}).get("score", float("-inf")))
        for sensor in _SENSORS
        if method_summary.get(sensor)
    ]
    if not sensor_scores:
        return float("-inf")
    return min(sensor_scores)


def _select_best_orientation_method(
    method_summaries: dict[str, dict[str, Any]],
    *,
    preferred_method: str = _DEFAULT_METHOD,
) -> str:
    """Select the best orientation method from per-method summaries."""
    if not method_summaries:
        raise ValueError("No orientation methods to select from.")

    available = [m for m, summary in method_summaries.items() if any(summary.get(s) for s in _SENSORS)]
    if not available:
        return preferred_method

    best = max(available, key=lambda method: (_score_method(method_summaries[method]), method == preferred_method))
    if preferred_method in available:
        pref_score = _score_method(method_summaries[preferred_method])
        best_score = _score_method(method_summaries[best])
        if pref_score >= best_score - 0.01:
            return preferred_method
    return best


def process_section_orientation(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    canonical_variant: str = _DEFAULT_METHOD,
    variants: list[str] | None = None,
) -> dict[str, Any]:
    """Run orientation method(s) for one section and flatten the selected result."""
    methods = list(variants) if variants is not None else list(_ALL_METHODS)
    orient_dir = section_dir / "orientation"
    all_methods_json = orient_dir / "all_methods.json"
    stats_json = orient_dir / "orientation_stats.json"

    if all_methods_json.exists() and stats_json.exists() and not force:
        log.info(
            "Orientation already exists for %s — skipping (use force=True to overwrite)",
            section_dir.name,
        )
        return json.loads(stats_json.read_text(encoding="utf-8"))

    orient_dir.mkdir(parents=True, exist_ok=True)

    sensor_raw: dict[str, pd.DataFrame] = {}
    for sensor in _SENSORS:
        cal_csv = section_dir / "calibrated" / f"{sensor}.csv"
        if not cal_csv.exists():
            log.warning("Calibrated CSV not found for sensor '%s' in %s", sensor, section_dir.name)
            continue
        df = read_csv(cal_csv)
        if df.empty:
            log.warning("Empty calibrated CSV for %s/%s", section_dir.name, sensor)
            continue
        sensor_raw[sensor] = df

    method_summaries: dict[str, dict[str, Any]] = {}
    created_at = datetime.now(UTC).isoformat()

    for method in methods:
        method_dir = orient_dir / method
        if method_dir.exists():
            shutil.rmtree(method_dir)
        method_dir.mkdir(parents=True, exist_ok=True)

        method_summary: dict[str, Any] = {
            "method": method,
            "parameters": dict(_METHOD_PARAMS.get(method, {})),
            "created_at_utc": created_at,
        }

        for sensor, df in sensor_raw.items():
            result_map = run_orientation_filters(
                df,
                sensor_name=sensor,
                sample_rate_hz=sample_rate_hz,
                methods=[method],
            )
            df_orient = result_map[method]
            out_csv = method_dir / f"{sensor}.csv"
            write_csv(df_orient, out_csv)

            stats = _compute_stats(df, df_orient, sensor, method)
            method_summary[sensor] = _stats_to_dict(stats)
            log.info(
                "Wrote orientation %s/%s → %s (%d rows)",
                sensor,
                method,
                out_csv,
                len(df_orient),
            )

        (method_dir / "orientation.json").write_text(
            json.dumps(method_summary, indent=2),
            encoding="utf-8",
        )
        method_summaries[method] = method_summary

    selected_method = _select_best_orientation_method(
        method_summaries,
        preferred_method=canonical_variant,
    )

    for sensor in _SENSORS:
        selected_csv = orient_dir / selected_method / f"{sensor}.csv"
        if selected_csv.exists():
            shutil.copy2(selected_csv, orient_dir / f"{sensor}.csv")

    # Keep only flattened outputs in orientation/ to match other auto stages.
    for method in methods:
        method_dir = orient_dir / method
        if method_dir.is_dir():
            shutil.rmtree(method_dir)

    selected_summary = method_summaries[selected_method]
    flat_stats = {
        "selected_method": selected_method,
        "parameters": selected_summary.get("parameters", {}),
        "created_at_utc": created_at,
    }
    for sensor in _SENSORS:
        flat_stats[sensor] = selected_summary.get(sensor, {})
    stats_json.write_text(json.dumps(flat_stats, indent=2), encoding="utf-8")

    all_methods = {
        "selected_method": selected_method,
        "methods": method_summaries,
        "created_at_utc": created_at,
    }
    all_methods_json.write_text(json.dumps(all_methods, indent=2), encoding="utf-8")
    log.info("Wrote orientation outputs → %s (selected=%s)", orient_dir, selected_method)
    return flat_stats


def process_recording_orientation(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    canonical_variant: str = _DEFAULT_METHOD,
    variants: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the orientation stage for all sections of a recording."""
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'", recording_name)
        return []

    results: list[dict[str, Any]] = []
    for sec_dir in section_dirs:
        log.info("Processing orientation for section %s ...", sec_dir.name)
        try:
            stats = process_section_orientation(
                sec_dir,
                sample_rate_hz=sample_rate_hz,
                force=force,
                canonical_variant=canonical_variant,
                variants=variants,
            )
            results.append(stats)
        except Exception as exc:
            log.error("Failed to process orientation for %s: %s", sec_dir.name, exc)
    return results
