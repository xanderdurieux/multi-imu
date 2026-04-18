"""Protocol-aware calibration pipeline: process one section or all sections of a recording."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import re

import pandas as pd

from common.paths import iter_sections_for_recording, project_relative_path, read_csv, write_csv
from parser.calibration_segments import CalibrationSegment, load_calibration_segments_from_json
from .core import (
    OpeningSequence,
    SensorIntrinsics,
    SensorAlignment,
    SectionCalibration,
    detect_protocol_landmarks,
    estimate_sensor_intrinsics,
    estimate_sensor_alignment,
    _find_first_stable_window,
    apply_calibration,
)

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")

# For Arduino the mount changes after the opening routine, so we look for a
# post-mount stable window.  Sporsa stays in place, so its opening static
# windows are also the alignment source.
_ARDUINO_SENSOR = "arduino"
_SPORSA_SENSOR = "sporsa"


def _build_opening_sequence(
    seg: Any,  # CalibrationSegment
) -> OpeningSequence:
    peak_ts = list(seg.peak_ms) if seg.peak_ms else []
    pre_end_ms = peak_ts[0] if peak_ts else seg.end_ms
    post_start_ms = peak_ts[-1] if peak_ts else seg.start_ms
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
    sample_rate_hz: float | None = None,  # kept for CLI compat, no longer used
) -> SectionCalibration:
    """Calibrate both sensors for one section and write outputs.

    Protocol-aware flow:
    1. Detect opening routine (calibration sequence) in each sensor's data.
    2. Estimate intrinsics (gyro bias; acc from static cal if available) from
       the opening static windows.
    3. Estimate alignment:
       - Sporsa: uses opening static windows directly (no mount change).
       - Arduino: finds the first stable post-mount window after the opening
         routine ends.
    4. Apply calibration and write calibrated CSVs.
    5. Write ``calibration.json`` with the new schema.

    Outputs written:
    - ``<section_dir>/<output_subdir>/sporsa.csv``
    - ``<section_dir>/<output_subdir>/arduino.csv``
    - ``<section_dir>/<output_subdir>/calibration.json``

    Parameters
    ----------
    section_dir:
        Path to the section directory (e.g. ``data/sections/2026-02-26_r1s1``).
    force:
        If True, overwrite existing outputs.
    output_subdir:
        Subdirectory name within ``section_dir`` for calibration outputs.
    static_calibration_path:
        Optional path to a static hardware calibration JSON file.  When
        provided, its accelerometer and gyroscope bias/scale values are used
        for the intrinsics of the Arduino sensor.
    """
    cal_dir = section_dir / output_subdir
    cal_json_path = cal_dir / "calibration.json"

    if cal_json_path.exists() and not force:
        log.info(
            "Calibration already exists for %s — skipping (use force=True to overwrite)",
            section_dir.name,
        )
        data = json.loads(cal_json_path.read_text(encoding="utf-8"))
        return SectionCalibration.from_dict(data)

    cal_dir.mkdir(parents=True, exist_ok=True)

    # Load static calibration reference if available.
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
            log.debug("Loaded static calibration reference from %s", resolved_static)
        except Exception as exc:
            log.warning("Could not load static calibration reference: %s", exc)

    # Derive recording name from section directory name (e.g. "2026-02-26_r5s1" → "2026-02-26_r5")
    recording_name = re.sub(r"s\d+$", "", section_dir.name)

    # Per-sensor calibration
    all_intrinsics: dict[str, SensorIntrinsics] = {}
    all_alignment: dict[str, SensorAlignment] = {}
    all_quality_tags: list[str] = []
    protocol_detected_any = False
    opening_sequence: OpeningSequence | None = None
    fallback_used = False

    for sensor in _SENSORS:
        csv_path = section_dir / f"{sensor}.csv"
        if not csv_path.exists():
            log.warning(
                "Sensor %s not found in %s — using identity calibration",
                sensor, section_dir.name,
            )
            all_intrinsics[sensor] = SensorIntrinsics(
                quality="poor", quality_tags=[f"missing_{sensor}"]
            )
            all_alignment[sensor] = SensorAlignment()
            all_quality_tags.append(f"missing_{sensor}")
            continue

        df = read_csv(csv_path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Resolve static calibration: only apply hardware ref to Arduino
        sensor_static_cal = static_cal if sensor == _ARDUINO_SENSOR else None

        try:
            opening_seg = _cal_sensor(
                df=df,
                sensor=sensor,
                section_dir=section_dir,
                cal_dir=cal_dir,
                static_cal=sensor_static_cal,
                all_intrinsics=all_intrinsics,
                all_alignment=all_alignment,
                all_quality_tags=all_quality_tags,
                recording_name=recording_name,
            )
            if opening_seg is not None and opening_sequence is None:
                opening_sequence = _build_opening_sequence(opening_seg)
                protocol_detected_any = True
            elif not protocol_detected_any and all_intrinsics.get(sensor):
                intrin = all_intrinsics[sensor]
                if "fallback_static" not in intrin.quality_tags:
                    protocol_detected_any = True

        except Exception as exc:
            log.error(
                "Calibration failed for %s/%s: %s", section_dir.name, sensor, exc
            )
            all_intrinsics[sensor] = SensorIntrinsics(
                quality="poor", quality_tags=["estimation_failed"]
            )
            all_alignment[sensor] = SensorAlignment()
            fallback_used = True

    # Apply and write calibrated CSVs
    for sensor in _SENSORS:
        csv_path = section_dir / f"{sensor}.csv"
        if not csv_path.exists():
            continue
        df = read_csv(csv_path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if sensor in all_intrinsics and sensor in all_alignment:
            cal_df = apply_calibration(df, all_intrinsics[sensor], all_alignment[sensor])
        else:
            cal_df = df.copy()

        out_csv = cal_dir / f"{sensor}.csv"
        write_csv(cal_df, out_csv)
        log.info("Wrote calibrated %s → %s (%d rows)", sensor, project_relative_path(out_csv), len(cal_df))

    # Overall quality
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
        opening_sequence=opening_sequence,
        intrinsics=all_intrinsics,
        alignment=all_alignment,
        quality={"overall": overall, "tags": sorted(set(all_quality_tags))},
        provenance={
            "created_at_utc": datetime.now(UTC).isoformat(),
            "fallback_used": fallback_used,
        },
    )

    cal_json_path.write_text(json.dumps(section_cal.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote calibration.json → %s (quality=%s)", project_relative_path(cal_dir), overall)
    return section_cal


def _cal_sensor(
    *,
    df: pd.DataFrame,
    sensor: str,
    section_dir: Path,
    cal_dir: Path,
    static_cal: dict[str, Any] | None,
    all_intrinsics: dict,
    all_alignment: dict,
    all_quality_tags: list,
    recording_name: str,
) -> CalibrationSegment | None:
    """Estimate intrinsics and alignment for one sensor and update the dicts.

    Returns the opening :class:`CalibrationSegment` if one was found, else ``None``.
    """
    section_name = section_dir.name

    # Load pre-detected calibration segments from the parser-stage JSON.
    try:
        segs = load_calibration_segments_from_json(recording_name, sensor)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        log.warning("%s/%s: could not load calibration segments (%s) — using fallback", section_name, sensor, exc)
        segs = []

    opening_seg, static_ranges = detect_protocol_landmarks(segs)

    fallback_used = False
    if not static_ranges:
        # Fallback: use first 5 s as a static window
        ts = df["timestamp"].to_numpy(dtype=float)
        if ts.size > 0:
            end_ms = ts[0] + 5000.0
            if int((ts <= end_ms).sum()) > 10:
                static_ranges = [(float(ts[0]), float(end_ms))]
                fallback_used = True
                all_quality_tags.append(f"fallback_static_{sensor}")
                log.debug(
                    "%s/%s: no opening sequence detected — using first 5 s as static",
                    section_name, sensor,
                )
            else:
                all_quality_tags.append(f"no_static_{sensor}")
                log.warning("%s/%s: no static window found", section_name, sensor)
        if not static_ranges:
            all_intrinsics[sensor] = SensorIntrinsics(
                quality="poor", quality_tags=["no_static_found"]
            )
            all_alignment[sensor] = SensorAlignment()
            return None

    # Intrinsics
    intrinsics = estimate_sensor_intrinsics(df, static_ranges, static_cal=static_cal)
    if fallback_used:
        intrinsics.quality_tags.append("fallback_static")
    all_intrinsics[sensor] = intrinsics
    all_quality_tags.extend(intrinsics.quality_tags)

    # Alignment window
    if sensor == _ARDUINO_SENSOR and opening_seg is not None:
        # Arduino: find first stable post-mount window after the opening routine
        post_mount_window = _find_first_stable_window(
            df,
            after_ms=opening_seg.end_ms,
            min_duration_ms=3000.0,
            threshold_ms2=1.5,
        )
        if post_mount_window is not None:
            alignment_window = post_mount_window
            log.debug(
                "%s/%s: using post-mount window [%.1f, %.1f] ms for alignment",
                section_name, sensor, *alignment_window,
            )
        else:
            # Fallback: use opening static range for alignment
            alignment_window = static_ranges[0]
            all_quality_tags.append(f"no_post_mount_window_{sensor}")
            log.warning(
                "%s/%s: no post-mount stable window found — using opening static for alignment",
                section_name, sensor,
            )
    else:
        # Sporsa (no mount change): use opening static window for alignment
        alignment_window = static_ranges[0]

    alignment = estimate_sensor_alignment(df, alignment_window, full_df=df)
    all_alignment[sensor] = alignment

    log.info(
        "%s/%s: intrinsics quality=%s, alignment yaw_source=%s residual=%.3f",
        section_name, sensor, intrinsics.quality,
        alignment.yaw_source, alignment.gravity_residual_ms2,
    )
    return opening_seg


# ---------------------------------------------------------------------------
# Recording-level helpers
# ---------------------------------------------------------------------------


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
            cal = calibrate_section(
                sec_dir,
                force=force,
                output_subdir=output_subdir,
                static_calibration_path=static_calibration_path,
            )
            results.append(cal)
        except Exception as exc:
            log.error("Failed to calibrate %s: %s", sec_dir.name, exc)

    return results
