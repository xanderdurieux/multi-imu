"""Calibration pipeline: process one section or all sections of a recording."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    project_relative_path,
    read_csv,
    write_csv,
)
from parser.calibration_segments import load_section_calibration_segments
from .core import (
    OpeningSequence,
    SensorIntrinsics,
    SensorAlignment,
    SectionCalibration,
    detect_protocol_landmarks,
    estimate_sensor_intrinsics,
    estimate_sensor_alignment,
    apply_calibration,
)

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_ARDUINO_SENSOR = "arduino"
_NO_PROTOCOL_FALLBACK_MS = 5_000.0  # first 5 s used as pseudo-static window when no protocol


def _build_opening_sequence(seg) -> OpeningSequence:
    peak_ts = list(seg.peak_ms) if seg.peak_ms else []
    pre_end_ms = seg.start_ms + seg.static_pre_ms if seg.static_pre_ms > 0 else seg.start_ms
    post_start_ms = seg.end_ms - seg.static_post_ms if seg.static_post_ms > 0 else seg.end_ms
    return OpeningSequence(
        tap_times_ms=peak_ts,
        pre_static_start_ms=seg.start_ms,
        pre_static_end_ms=pre_end_ms,
        post_static_start_ms=post_start_ms,
        post_static_end_ms=seg.end_ms,
        n_taps=len(peak_ts),
    )


def calibrate_section(
    section_dir: Path,
    *,
    force: bool = False,
    output_subdir: str = "calibrated",
    static_calibration_path: Path | None = None,
) -> SectionCalibration:
    """Calibrate both sensors for one section and write outputs.

    Flow:
    1. Detect opening routine from parser-stage calibration segments.
    2. Estimate intrinsics (gyro bias; acc from static cal if available).
    3. Estimate gravity-only alignment from the post-tap static window.
    4. Apply calibration and write CSVs + calibration.json.
    """
    cal_dir = section_dir / output_subdir
    cal_json_path = cal_dir / "calibration.json"

    if cal_json_path.exists() and not force:
        log.info("Calibration already exists for %s — skipping", section_dir.name)
        return SectionCalibration.from_dict(json.loads(cal_json_path.read_text(encoding="utf-8")))

    cal_dir.mkdir(parents=True, exist_ok=True)

    static_cal: dict[str, Any] | None = None
    resolved_static = static_calibration_path
    if resolved_static is None:
        from common.paths import calibrations_root
        default = calibrations_root() / "arduino_imu_calibration.json"
        if default.exists():
            resolved_static = default

    if resolved_static and resolved_static.exists():
        try:
            from static_calibration.imu_static import load_calibration
            static_cal = load_calibration(resolved_static)
        except Exception as exc:
            log.warning("Could not load static calibration: %s", exc)

    all_intrinsics: dict[str, SensorIntrinsics] = {}
    all_alignment: dict[str, SensorAlignment] = {}
    all_opening: dict[str, OpeningSequence] = {}
    all_closing: dict[str, OpeningSequence] = {}
    all_quality_tags: list[str] = []
    protocol_detected_any = False

    for sensor in _SENSORS:
        csv_path = section_dir / f"{sensor}.csv"
        if not csv_path.exists():
            log.warning("Sensor %s not found in %s — using identity calibration", sensor, section_dir.name)
            all_intrinsics[sensor] = SensorIntrinsics(quality="poor", quality_tags=[f"missing_{sensor}"])
            all_alignment[sensor] = SensorAlignment()
            all_quality_tags.append(f"missing_{sensor}")
            continue

        df = read_csv(csv_path).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        sensor_static_cal = static_cal if sensor == _ARDUINO_SENSOR else None

        try:
            opening_seg, closing_seg = _cal_sensor(
                df=df,
                sensor=sensor,
                section_dir=section_dir,
                static_cal=sensor_static_cal,
                all_intrinsics=all_intrinsics,
                all_alignment=all_alignment,
                all_quality_tags=all_quality_tags,
            )
            if opening_seg is not None:
                all_opening[sensor] = _build_opening_sequence(opening_seg)
                protocol_detected_any = True
            if closing_seg is not None:
                all_closing[sensor] = _build_opening_sequence(closing_seg)
        except Exception as exc:
            log.error("Calibration failed for %s/%s: %s", section_dir.name, sensor, exc)
            all_intrinsics[sensor] = SensorIntrinsics(quality="poor", quality_tags=["estimation_failed"])
            all_alignment[sensor] = SensorAlignment()

    for sensor in _SENSORS:
        csv_path = section_dir / f"{sensor}.csv"
        if not csv_path.exists():
            continue
        df = read_csv(csv_path).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if sensor in all_intrinsics and sensor in all_alignment:
            cal_df = apply_calibration(df, all_intrinsics[sensor], all_alignment[sensor])
        else:
            cal_df = df.copy()
        out_csv = cal_dir / f"{sensor}.csv"
        write_csv(cal_df, out_csv)
        log.info("Wrote calibrated %s → %s (%d rows)", sensor, project_relative_path(out_csv), len(cal_df))

    qualities = [v.quality for v in all_intrinsics.values()]
    align_residuals = [v.gravity_residual_ms2 for v in all_alignment.values()]
    if "poor" in qualities or any(r > 2.0 for r in align_residuals if r < 99.0):
        overall = "poor"
    elif "marginal" in qualities or any(r > 1.0 for r in align_residuals if r < 99.0):
        overall = "marginal"
    else:
        overall = "good"

    section_cal = SectionCalibration(
        protocol_detected=protocol_detected_any,
        opening_sequence=all_opening,
        closing_sequence=all_closing,
        intrinsics=all_intrinsics,
        alignment=all_alignment,
        quality={"overall": overall, "tags": sorted(set(all_quality_tags))},
        provenance={"created_at_utc": datetime.now(UTC).isoformat()},
    )
    cal_json_path.write_text(json.dumps(section_cal.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote calibration.json → %s (quality=%s)", project_relative_path(cal_dir), overall)
    return section_cal


def _cal_sensor(
    *,
    df: pd.DataFrame,
    sensor: str,
    section_dir: Path,
    static_cal: dict[str, Any] | None,
    all_intrinsics: dict,
    all_alignment: dict,
    all_quality_tags: list,
) -> tuple[Any, Any]:
    """Estimate intrinsics and alignment for one sensor; update the shared dicts."""
    section_name = section_dir.name

    try:
        segs = load_section_calibration_segments(section_dir, sensor)
    except Exception as exc:
        log.warning("%s/%s: could not load section calibration segments (%s)", section_name, sensor, exc)
        segs = []

    opening_seg, static_ranges, closing_seg = detect_protocol_landmarks(segs)

    if not static_ranges:
        t0 = float(df["timestamp"].iloc[0])
        static_ranges = [(t0, t0 + _NO_PROTOCOL_FALLBACK_MS)]
        all_quality_tags.append("no_protocol")

    intrinsics = estimate_sensor_intrinsics(df, static_ranges, static_cal=static_cal)
    all_intrinsics[sensor] = intrinsics
    all_quality_tags.extend(intrinsics.quality_tags)

    alignment = estimate_sensor_alignment(df, static_ranges[-1])
    all_alignment[sensor] = alignment

    log.info(
        "%s/%s: quality=%s, gravity_residual=%.3f m/s²",
        section_name, sensor, intrinsics.quality, alignment.gravity_residual_ms2,
    )
    return opening_seg, closing_seg


def calibrate_recording_sections(
    recording_name: str,
    *,
    force: bool = False,
    output_subdir: str = "calibrated",
    static_calibration_path: Path | None = None,
) -> list[SectionCalibration]:
    """Calibrate all sections for a recording."""
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for %s", recording_name)
        return []

    results: list[SectionCalibration] = []
    for sec_dir in section_dirs:
        log.info("Calibrating section %s ...", sec_dir.name)
        try:
            cal = calibrate_section(sec_dir, force=force, output_subdir=output_subdir,
                                    static_calibration_path=static_calibration_path)
            results.append(cal)
        except Exception as exc:
            log.error("Failed to calibrate %s: %s", sec_dir.name, exc)
    return results
