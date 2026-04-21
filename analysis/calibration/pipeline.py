"""Protocol-aware calibration pipeline: process one section or all sections of a recording."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import re

import numpy as np
import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    project_relative_path,
    read_csv,
    recording_stage_dir,
    write_csv,
)
from parser.calibration_segments import CalibrationSegment, load_calibration_segments_from_json
from sync.model import apply_linear_time_transform
from .core import (
    OpeningSequence,
    SensorIntrinsics,
    SensorAlignment,
    SectionCalibration,
    detect_protocol_landmarks,
    estimate_sensor_intrinsics,
    estimate_sensor_alignment,
    _gravity_from_ranges,
    _gyro_bias_from_ranges,
    _GYRO_BIAS_DRIFT_THRESHOLD_DEG_S,
    _segment_static_ranges,
    apply_calibration,
)

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")

# For Arduino the mount changes after the opening routine, so we look for a
# post-mount stable window.  Sporsa stays in place, so its opening static
# windows are also the alignment source.
_ARDUINO_SENSOR = "arduino"
_SPORSA_SENSOR = "sporsa"


def _load_arduino_sync_params(recording_name: str) -> dict[str, float] | None:
    """Return the sync-model parameters used to map Arduino raw millis → synced ms.

    Returns ``None`` when no ``sync_info.json`` is available yet (e.g. when
    calibration is run before the sync stage has produced output).
    """
    sync_path = recording_stage_dir(recording_name, "synced") / "sync_info.json"
    if not sync_path.exists():
        return None
    try:
        data = json.loads(sync_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("Could not parse %s: %s", sync_path, exc)
        return None
    try:
        return {
            "offset_seconds": float(data["offset_seconds"]),
            "drift_seconds_per_second": float(data["drift_seconds_per_second"]),
            "target_origin_seconds": float(data["target_time_origin_seconds"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Incomplete sync model in %s: %s", sync_path, exc)
        return None


def _transform_segments_to_synced(
    segs: list[CalibrationSegment],
    sync_params: dict[str, float],
) -> list[CalibrationSegment]:
    """Map segment timestamps from raw Arduino millis to the synced frame.

    Durations (``static_pre_ms``, ``static_post_ms``) are preserved on the
    synced axis by adjusting them for the linear drift factor, so that
    ``end_ms - static_post_ms`` continues to mark the genuine start of the
    post-tap static flank.
    """
    if not segs:
        return segs

    import numpy as np

    starts = np.array([s.start_ms for s in segs], dtype=float)
    ends = np.array([s.end_ms for s in segs], dtype=float)
    synced_starts = apply_linear_time_transform(starts, **sync_params)
    synced_ends = apply_linear_time_transform(ends, **sync_params)
    # Linear transform preserves durations up to the drift factor (1 + a).
    drift_factor = 1.0 + float(sync_params["drift_seconds_per_second"])

    out: list[CalibrationSegment] = []
    for seg, s_ms, e_ms in zip(segs, synced_starts, synced_ends):
        peaks: list[float] = []
        if seg.peak_ms:
            peak_arr = np.array(seg.peak_ms, dtype=float)
            peaks = apply_linear_time_transform(peak_arr, **sync_params).tolist()
        out.append(
            CalibrationSegment(
                start_ms=float(s_ms),
                end_ms=float(e_ms),
                peak_ms=peaks,
                peak_strength=seg.peak_strength,
                static_pre_ms=seg.static_pre_ms * drift_factor,
                static_post_ms=seg.static_post_ms * drift_factor,
            )
        )
    return out


def _clip_segments_to_range(
    segs: list[CalibrationSegment],
    t_min: float,
    t_max: float,
) -> list[CalibrationSegment]:
    """Clip each segment's boundaries and static-flank durations to ``[t_min, t_max]``.

    Segments that end before ``t_min`` or start after ``t_max`` are dropped.
    For a segment that straddles either boundary, the pre-/post-static
    duration is shortened so ``start_ms + static_pre_ms`` and
    ``end_ms - static_post_ms`` remain within the clipped span.  Tap peaks
    outside the clipped range are removed.
    """
    out: list[CalibrationSegment] = []
    for seg in segs:
        if seg.end_ms < t_min or seg.start_ms > t_max:
            continue

        new_start = max(seg.start_ms, t_min)
        new_end = min(seg.end_ms, t_max)
        pre_cut = new_start - seg.start_ms
        post_cut = seg.end_ms - new_end
        new_static_pre = max(0.0, seg.static_pre_ms - pre_cut)
        new_static_post = max(0.0, seg.static_post_ms - post_cut)

        new_peaks = [p for p in seg.peak_ms if new_start <= p <= new_end]

        out.append(
            CalibrationSegment(
                start_ms=new_start,
                end_ms=new_end,
                peak_ms=new_peaks,
                peak_strength=seg.peak_strength,
                static_pre_ms=new_static_pre,
                static_post_ms=new_static_post,
            )
        )
    return out


def _build_opening_sequence(
    seg: Any,  # CalibrationSegment
) -> OpeningSequence:
    """Build the protocol-level opening summary using the genuinely-static flanks.

    The static flank boundaries come from ``static_pre_ms``/``static_post_ms``
    (parser-side detection), not from the tap peak timestamps, because the
    span between the last peak and the end of the segment can include tap
    settling motion.
    """
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
    all_opening: dict[str, OpeningSequence] = {}
    all_closing: dict[str, OpeningSequence] = {}
    all_quality_tags: list[str] = []
    protocol_detected_any = False
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
            opening_seg, closing_seg = _cal_sensor(
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
            if opening_seg is not None:
                all_opening[sensor] = _build_opening_sequence(opening_seg)
                protocol_detected_any = True
            elif not protocol_detected_any and all_intrinsics.get(sensor):
                intrin = all_intrinsics[sensor]
                if "fallback_static" not in intrin.quality_tags:
                    protocol_detected_any = True
            if closing_seg is not None:
                all_closing[sensor] = _build_opening_sequence(closing_seg)

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
        opening_sequence=all_opening,
        closing_sequence=all_closing,
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
) -> tuple[CalibrationSegment | None, CalibrationSegment | None]:
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

    # Arduino calibration segments are stored in raw Arduino millis (pre-sync);
    # the section CSV uses synced timestamps.  Map the segments through the
    # recording's sync model so the static ranges match the CSV timestamps.
    if sensor == _ARDUINO_SENSOR and segs:
        sync_params = _load_arduino_sync_params(recording_name)
        if sync_params is None:
            log.warning(
                "%s/%s: no sync_info.json — Arduino calibration segments are in raw millis "
                "and will not match the synced CSV timestamps; using fallback",
                section_name, sensor,
            )
            all_quality_tags.append(f"sync_missing_{sensor}")
            segs = []
        else:
            segs = _transform_segments_to_synced(segs, sync_params)

    # Calibration segments are detected over the full recording; clip any that
    # straddle the section's time span so downstream code only references
    # samples that actually exist in the section CSV.
    if segs:
        ts = df["timestamp"].to_numpy(dtype=float)
        segs = _clip_segments_to_range(segs, float(ts[0]), float(ts[-1]))

    opening_seg, static_ranges, closing_seg = detect_protocol_landmarks(segs)

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
            return None, closing_seg

    # Intrinsics
    intrinsics = estimate_sensor_intrinsics(df, static_ranges, static_cal=static_cal)
    if fallback_used:
        intrinsics.quality_tags.append("fallback_static")
    all_intrinsics[sensor] = intrinsics
    all_quality_tags.extend(intrinsics.quality_tags)

    # Closing-sequence quality checks (independent validation at end-of-session)
    if closing_seg is not None:
        closing_ranges = _segment_static_ranges(closing_seg)
        if closing_ranges:
            # Gyro bias drift: re-estimate from closing window and compare.
            closing_gyro_bias = _gyro_bias_from_ranges(df, closing_ranges)
            drift = float(np.linalg.norm(
                np.array(intrinsics.gyro_bias) - closing_gyro_bias
            ))
            if drift > _GYRO_BIAS_DRIFT_THRESHOLD_DEG_S:
                tag = f"closing_gyro_drift_{drift:.2f}deg_s"
                intrinsics.quality_tags.append(tag)
                all_quality_tags.append(tag)
                if intrinsics.quality == "good":
                    intrinsics.quality = "marginal"
                log.debug(
                    "%s/%s: gyro bias drifted %.2f °/s over session",
                    section_name, sensor, drift,
                )

            # Closing static gravity residual: held-out validation of acc calibration.
            _, closing_residual = _gravity_from_ranges(df, closing_ranges)
            if np.isfinite(closing_residual):
                if closing_residual >= 1.5:
                    tag = "closing_static_poor"
                    if intrinsics.quality != "poor":
                        intrinsics.quality = "marginal"
                elif closing_residual >= 0.5:
                    tag = "closing_static_marginal"
                else:
                    tag = None
                if tag:
                    intrinsics.quality_tags.append(tag)
                    all_quality_tags.append(tag)
                log.debug(
                    "%s/%s: closing static gravity residual %.3f m/s²",
                    section_name, sensor, closing_residual,
                )

    # Alignment window: use the post-tap static flank for both sensors.
    # This is the window immediately after the taps, before motion starts.
    alignment_window = static_ranges[-1]

    alignment = estimate_sensor_alignment(df, alignment_window, full_df=df)
    all_alignment[sensor] = alignment

    log.info(
        "%s/%s: intrinsics quality=%s, alignment yaw_source=%s residual=%.3f",
        section_name, sensor, intrinsics.quality,
        alignment.yaw_source, alignment.gravity_residual_ms2,
    )
    return opening_seg, closing_seg


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
