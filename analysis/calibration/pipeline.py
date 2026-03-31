"""Calibration pipeline: process one section or all sections of a recording."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import iter_sections_for_recording, read_csv, sections_root, write_csv
from .core import (
    CalibrationParams,
    SectionCalibration,
    estimate_calibration,
    apply_calibration,
)

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")
_FRAME_ALIGNMENT_DEFAULT = "gravity_only"
_ALL_METHODS: list[str] = ["gravity_only", "gravity_plus_forward"]


def calibrate_section(
    section_dir: Path,
    *,
    frame_alignment: str = _FRAME_ALIGNMENT_DEFAULT,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    output_subdir: str = "calibrated",
) -> SectionCalibration:
    """Calibrate both sensors for one section and write outputs.

    Outputs written:
    - ``<section_dir>/<output_subdir>/sporsa.csv``
    - ``<section_dir>/<output_subdir>/arduino.csv``
    - ``<section_dir>/<output_subdir>/calibration.json``

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
    output_subdir:
        Subdirectory name within ``section_dir`` for calibration outputs.
        Defaults to ``"calibrated"``.
    """
    cal_dir = section_dir / output_subdir
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

        df = read_csv(csv_path)
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
        write_csv(cal_df, out_csv)
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


# ---------------------------------------------------------------------------
# Multi-method selection (mirrors sync pipeline pattern)
# ---------------------------------------------------------------------------

def _score_calibration(cal: SectionCalibration) -> float:
    """Compute a quality score for a calibration result (lower = better).

    Based on the worst-sensor gravity residual, penalising fallback usage.
    For ``gravity_plus_forward`` a bonus is applied when forward confidence is
    high, since a well-estimated forward direction improves the rotation.
    """
    residuals = [
        cal.sporsa.gravity_residual_ms2,
        cal.arduino.gravity_residual_ms2,
    ]
    # Use the worst sensor as the bottleneck.
    valid = [r for r in residuals if r == r and r < 99.0]  # exclude NaN / sentinel
    score = max(valid) if valid else 99.0

    # Penalise fallback static segments.
    if cal.sporsa.fallback_used or cal.arduino.fallback_used:
        score += 2.0

    # Small bonus for well-estimated forward direction (gravity_plus_forward).
    forward_conf = max(cal.sporsa.forward_confidence, cal.arduino.forward_confidence)
    if cal.frame_alignment == "gravity_plus_forward" and forward_conf >= 0.3:
        score -= forward_conf * 0.5  # up to -0.5 bonus

    return score


def _select_best_calibration_method(
    method_results: dict[str, SectionCalibration],
) -> str:
    """Return the name of the best calibration method.

    Prefers ``gravity_plus_forward`` when the forward direction was reliably
    estimated (``forward_confidence >= 0.3``) and its score is not worse than
    ``gravity_only`` by more than a small margin.  Falls back to whichever
    available method has the lowest score.
    """
    if not method_results:
        raise ValueError("No calibration methods to select from.")

    scores = {m: _score_calibration(cal) for m, cal in method_results.items()}

    go = method_results.get("gravity_only")
    gp = method_results.get("gravity_plus_forward")

    if go is not None and gp is not None:
        forward_conf = max(gp.sporsa.forward_confidence, gp.arduino.forward_confidence)
        if forward_conf >= 0.3 and scores["gravity_plus_forward"] <= scores["gravity_only"] + 0.5:
            return "gravity_plus_forward"
        return "gravity_only"

    return min(method_results.keys(), key=lambda m: scores[m])


def _method_summary(cal: SectionCalibration) -> dict[str, Any]:
    """Compact per-method metrics for all_methods.json."""
    return {
        "frame_alignment": cal.frame_alignment,
        "calibration_quality": cal.calibration_quality,
        "quality_tags": cal.quality_tags,
        "sporsa": {
            "gravity_residual_ms2": cal.sporsa.gravity_residual_ms2,
            "forward_confidence": cal.sporsa.forward_confidence,
            "quality": cal.sporsa.quality,
            "fallback_used": cal.sporsa.fallback_used,
        },
        "arduino": {
            "gravity_residual_ms2": cal.arduino.gravity_residual_ms2,
            "forward_confidence": cal.arduino.forward_confidence,
            "quality": cal.arduino.quality,
            "fallback_used": cal.arduino.fallback_used,
        },
    }


def calibrate_section_all_methods(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    static_calibration_path: Path | None = None,
) -> tuple[SectionCalibration, str]:
    """Run all calibration methods, select the best, and flatten it to ``calibrated/``.

    Each method writes to a temporary ``calibrated/<method>/`` subdirectory.
    The selected result's files are copied to ``calibrated/`` and the method
    subdirectories are removed, leaving a clean structure identical to a
    single-method run plus ``calibrated/all_methods.json``.

    Parameters
    ----------
    section_dir:
        Path to the section directory.
    sample_rate_hz:
        Sampling rate in Hz.
    force:
        Overwrite existing outputs when True.
    static_calibration_path:
        Optional path to a static calibration JSON (e.g. from
        ``static_calibration`` package).  When provided, its parameters are
        included in ``all_methods.json`` as a reference / fact-check entry but
        are not used to override the selected dynamic result.

    Returns
    -------
    (best SectionCalibration, selected method name)
    """
    cal_dir = section_dir / "calibrated"
    all_methods_json = cal_dir / "all_methods.json"

    if all_methods_json.exists() and not force:
        log.info(
            "Calibration (all methods) already exists for %s — skipping (use force=True to overwrite)",
            section_dir.name,
        )
        summary = json.loads(all_methods_json.read_text(encoding="utf-8"))
        selected = summary.get("selected_method", _ALL_METHODS[0])
        cal_json = cal_dir / "calibration.json"
        return SectionCalibration.from_dict(json.loads(cal_json.read_text(encoding="utf-8"))), selected

    method_results: dict[str, SectionCalibration] = {}

    for method in _ALL_METHODS:
        # Each method gets a temporary subfolder inside calibrated/.
        subdir = f"calibrated/{method}"
        log.info("calibration %s: running method=%s", section_dir.name, method)
        try:
            cal = calibrate_section(
                section_dir,
                frame_alignment=method,
                sample_rate_hz=sample_rate_hz,
                force=force,
                output_subdir=subdir,
            )
            method_results[method] = cal
            log.info("calibration %s: %s done (quality=%s)", section_dir.name, method, cal.calibration_quality)
        except Exception as exc:
            log.warning("calibration %s: method=%s failed: %s", section_dir.name, method, exc)

    if not method_results:
        raise RuntimeError(f"All calibration methods failed for {section_dir.name}")

    selected = _select_best_calibration_method(method_results)
    log.info(
        "calibration %s: selected method=%s (score=%.3f)",
        section_dir.name,
        selected,
        _score_calibration(method_results[selected]),
    )

    # Copy selected method's files to the calibrated/ root.
    src_dir = cal_dir / selected
    for filename in (*(f"{s}.csv" for s in _SENSORS), "calibration.json"):
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, cal_dir / filename)

    # Prune method subdirectories — leave calibrated/ clean.
    for method in _ALL_METHODS:
        method_subdir = cal_dir / method
        if method_subdir.is_dir():
            shutil.rmtree(method_subdir)

    # Build all_methods summary.
    all_methods_data: dict[str, Any] = {
        "selected_method": selected,
        "score": round(_score_calibration(method_results[selected]), 4),
        "methods": {m: _method_summary(cal) for m, cal in method_results.items()},
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    # Optionally embed static calibration as a reference.
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
            all_methods_data["static_calibration_reference"] = {
                "path": str(resolved_static),
                "accelerometer_bias": static_cal.get("accelerometer", {}).get("bias"),
                "accelerometer_scale": static_cal.get("accelerometer", {}).get("scale"),
                "gyroscope_bias_deg_s": static_cal.get("gyroscope", {}).get("bias_deg_s"),
                "warnings": static_cal.get("warnings", []),
            }
        except Exception as exc:
            log.warning("Could not load static calibration reference: %s", exc)

    all_methods_json.write_text(json.dumps(all_methods_data, indent=2), encoding="utf-8")
    log.info("Wrote calibration all_methods.json → %s", cal_dir)

    return method_results[selected], selected


def calibrate_recording_sections(
    recording_name: str,
    *,
    frame_alignment: str = _FRAME_ALIGNMENT_DEFAULT,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    output_subdir: str = "calibrated",
) -> list[SectionCalibration]:
    """Calibrate all sections for a recording (single method)."""
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
                output_subdir=output_subdir,
            )
            results.append(cal)
        except Exception as exc:
            log.error("Failed to calibrate %s: %s", sec_dir.name, exc)

    return results


def calibrate_recording_sections_all_methods(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
    static_calibration_path: Path | None = None,
) -> list[SectionCalibration]:
    """Run all calibration methods for every section of a recording.

    See :func:`calibrate_section_all_methods` for per-section details.
    """
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for %s", recording_name)
        return []

    results: list[SectionCalibration] = []
    for sec_dir in section_dirs:
        log.info("Calibrating (all methods) section %s ...", sec_dir.name)
        try:
            cal, _ = calibrate_section_all_methods(
                sec_dir,
                sample_rate_hz=sample_rate_hz,
                force=force,
                static_calibration_path=static_calibration_path,
            )
            results.append(cal)
        except Exception as exc:
            log.error("Failed to calibrate (all methods) %s: %s", sec_dir.name, exc)

    return results
