"""Calibration pipeline: process one section or all sections of a recording."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.csv_schema import load_dataframe, write_dataframe
from common.paths import iter_sections_for_recording, sections_root
from .core import (
    CalibrationParams,
    SectionCalibration,
    estimate_calibration,
    apply_calibration,
)

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_FRAME_ALIGNMENT_DEFAULT = "gravity_only"


def calibrate_section(
    section_dir: Path,
    *,
    frame_alignment: str = _FRAME_ALIGNMENT_DEFAULT,
    sample_rate_hz: float = 100.0,
    force: bool = False,
) -> SectionCalibration:
    """Calibrate both sensors for one section and write outputs.

    Outputs written:
    - ``<section_dir>/calibrated/sporsa.csv``
    - ``<section_dir>/calibrated/arduino.csv``
    - ``<section_dir>/calibrated/calibration.json``

    Parameters
    ----------
    section_dir:
        Path to the section directory (e.g. ``data/sections/2026-02-26_r1s1``).
    frame_alignment:
        ``"gravity_only"`` or ``"gravity_plus_forward"``.
    sample_rate_hz:
        Approximate sampling rate for calibration-segment detection.
    force:
        If True, overwrite existing outputs.
    """
    cal_dir = section_dir / "calibrated"
    cal_json = cal_dir / "calibration.json"

    if cal_json.exists() and not force:
        log.info("Calibration already exists for %s — skipping (use force=True to overwrite)", section_dir.name)
        data = json.loads(cal_json.read_text(encoding="utf-8"))
        return SectionCalibration.from_dict(data)

    cal_dir.mkdir(parents=True, exist_ok=True)

    sensor_params: dict[str, CalibrationParams] = {}
    all_quality_tags: list[str] = []

    for sensor in _SENSORS:
        csv_path = section_dir / f"{sensor}.csv"
        if not csv_path.exists():
            log.warning("Sensor %s not found in %s — using identity calibration", sensor, section_dir.name)
            sensor_params[sensor] = CalibrationParams(
                quality_tags=[f"missing_{sensor}"], quality="poor"
            )
            all_quality_tags.append(f"missing_{sensor}")
            continue

        df = load_dataframe(csv_path)
        try:
            params = estimate_calibration(
                df,
                sample_rate_hz=sample_rate_hz,
                frame_alignment=frame_alignment,
            )
        except Exception as exc:
            log.error("Calibration failed for %s/%s: %s", section_dir.name, sensor, exc)
            params = CalibrationParams(
                quality_tags=["estimation_failed"], quality="poor", fallback_used=True
            )

        sensor_params[sensor] = params
        all_quality_tags.extend(params.quality_tags)

        # Apply and write calibrated CSV.
        cal_df = apply_calibration(df, params)
        out_csv = cal_dir / f"{sensor}.csv"
        write_dataframe(cal_df, out_csv)
        log.info("Wrote calibrated %s → %s (%d rows)", sensor, out_csv, len(cal_df))

    # Overall quality.
    qualities = [p.quality for p in sensor_params.values()]
    if "poor" in qualities:
        overall = "poor"
    elif "marginal" in qualities:
        overall = "marginal"
    else:
        overall = "good"

    section_cal = SectionCalibration(
        sporsa=sensor_params.get("sporsa", CalibrationParams()),
        arduino=sensor_params.get("arduino", CalibrationParams()),
        frame_alignment=frame_alignment,
        calibration_quality=overall,
        quality_tags=sorted(set(all_quality_tags)),
        created_at_utc=datetime.now(UTC).isoformat(),
    )

    cal_json.write_text(json.dumps(section_cal.to_dict(), indent=2), encoding="utf-8")
    log.info("Wrote calibration.json → %s (quality=%s)", cal_dir, overall)
    return section_cal


def calibrate_recording_sections(
    recording_name: str,
    *,
    frame_alignment: str = _FRAME_ALIGNMENT_DEFAULT,
    sample_rate_hz: float = 100.0,
    force: bool = False,
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
                frame_alignment=frame_alignment,
                sample_rate_hz=sample_rate_hz,
                force=force,
            )
            results.append(cal)
        except Exception as exc:
            log.error("Failed to calibrate %s: %s", sec_dir.name, exc)

    return results
