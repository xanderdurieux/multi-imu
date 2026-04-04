"""Protocol-aware calibration pipeline: process one section or all sections of a recording."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import iter_sections_for_recording, read_csv, write_csv
from .core import (
    OpeningSequence,
    SensorIntrinsics,
    SensorAlignment,
    SectionCalibration,
    detect_protocol_landmarks,
    estimate_sensor_intrinsics,
    estimate_sensor_alignment,
    _find_first_stable_window,
    _ms_from_idx,
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
    df: pd.DataFrame,
    seg: Any,  # CalibrationSegment
) -> OpeningSequence:
    pre_end = seg.peak_indices[0] if seg.peak_indices else seg.end_idx
    post_start = seg.peak_indices[-1] if seg.peak_indices else seg.start_idx
    tap_times_ms = [_ms_from_idx(df, idx) for idx in seg.peak_indices]
    return OpeningSequence(
        pre_static_range=[seg.start_idx, pre_end],
        tap_cluster=list(seg.peak_indices),
        tap_times_ms=tap_times_ms,
        post_static_range=[post_start, seg.end_idx],
        pre_static_start_ms=_ms_from_idx(df, seg.start_idx),
        pre_static_end_ms=_ms_from_idx(df, pre_end),
        post_static_start_ms=_ms_from_idx(df, post_start),
        post_static_end_ms=_ms_from_idx(df, seg.end_idx),
        n_taps=len(seg.peak_indices),
    )


def calibrate_section(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    output_subdir: str = "calibrated",
    static_calibration_path: Path | None = None,
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
    sample_rate_hz:
        Approximate sampling rate for calibration-segment detection.
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
        from common.paths import data_root
        default = data_root() / "calibrations" / "arduino_imu_calibration.json"
        if default.exists():
            resolved_static = default

    if resolved_static and resolved_static.exists():
        try:
            from static_calibration.imu_static import load_calibration
            static_cal = load_calibration(resolved_static)
            log.debug("Loaded static calibration reference from %s", resolved_static)
        except Exception as exc:
            log.warning("Could not load static calibration reference: %s", exc)

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
            _cal_sensor(
                df=df,
                sensor=sensor,
                section_dir=section_dir,
                cal_dir=cal_dir,
                sample_rate_hz=sample_rate_hz,
                static_cal=sensor_static_cal,
                all_intrinsics=all_intrinsics,
                all_alignment=all_alignment,
                all_quality_tags=all_quality_tags,
            )
            # Record protocol detection result and opening sequence
            # (use whichever sensor successfully detected the protocol first)
            if not protocol_detected_any and all_intrinsics.get(sensor):
                intrin = all_intrinsics[sensor]
                if "fallback_static" not in intrin.quality_tags:
                    protocol_detected_any = True

            # Capture opening sequence from first sensor that yields one
            if opening_sequence is None and sensor in all_intrinsics:
                seg, _ = detect_protocol_landmarks(df, sample_rate_hz=sample_rate_hz)
                if seg is not None:
                    opening_sequence = _build_opening_sequence(df, seg)
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
    sample_rate_hz: float,
    static_cal: dict[str, Any] | None,
    all_intrinsics: dict,
    all_alignment: dict,
    all_quality_tags: list,
) -> None:
    """Estimate intrinsics and alignment for one sensor and update the dicts."""
    section_name = section_dir.name

    # Detect protocol landmarks
    opening_seg, static_ranges = detect_protocol_landmarks(df, sample_rate_hz=sample_rate_hz)

    fallback_used = False
    if not static_ranges:
        # Fallback: use first 5 s as static
        ts = df["timestamp"].to_numpy(dtype=float)
        if ts.size > 0:
            mask = ts <= ts[0] + 5000.0
            n = int(mask.sum())
            if n > 10:
                static_ranges = [(0, n)]
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
            return

    # Intrinsics
    intrinsics = estimate_sensor_intrinsics(df, static_ranges, static_cal=static_cal)
    if fallback_used:
        intrinsics.quality_tags.append("fallback_static")
    all_intrinsics[sensor] = intrinsics
    all_quality_tags.extend(intrinsics.quality_tags)

    # Alignment window
    if sensor == _ARDUINO_SENSOR and opening_seg is not None:
        # Arduino: find first stable post-mount window
        after_idx = opening_seg.end_idx
        post_mount_window = _find_first_stable_window(
            df,
            after_idx=after_idx,
            sample_rate_hz=sample_rate_hz,
            min_duration_s=3.0,
            threshold_ms2=1.5,
        )
        if post_mount_window is not None:
            alignment_window = post_mount_window
            log.debug(
                "%s/%s: using post-mount window [%d, %d] for alignment",
                section_name, sensor, *alignment_window,
            )
        else:
            # Fallback: use opening static ranges for alignment
            alignment_window = static_ranges[0]
            all_quality_tags.append(f"no_post_mount_window_{sensor}")
            log.warning(
                "%s/%s: no post-mount stable window found — using opening static for alignment",
                section_name, sensor,
            )
    else:
        # Sporsa (no mount change): use opening static windows for alignment
        alignment_window = static_ranges[0]

    alignment = estimate_sensor_alignment(
        df,
        alignment_window,
        sample_rate_hz=sample_rate_hz,
        full_df=df,  # use full section for PCA
    )
    all_alignment[sensor] = alignment

    log.info(
        "%s/%s: intrinsics quality=%s, alignment yaw_source=%s residual=%.3f",
        section_name, sensor, intrinsics.quality,
        alignment.yaw_source, alignment.gravity_residual_ms2,
    )


# ---------------------------------------------------------------------------
# Recording-level helpers
# ---------------------------------------------------------------------------


def calibrate_recording_sections(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
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
                sample_rate_hz=sample_rate_hz,
                force=force,
                output_subdir=output_subdir,
                static_calibration_path=static_calibration_path,
            )
            results.append(cal)
        except Exception as exc:
            log.error("Failed to calibrate %s: %s", sec_dir.name, exc)

    return results
