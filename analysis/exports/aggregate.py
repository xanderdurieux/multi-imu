"""Aggregate parsed, calibration, and sync parameters from sections/recordings."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pandas as pd

from common.paths import (
    project_relative_path,
    recording_sort_key,
    section_sort_key,
    recordings_root,
    sections_root,
)

log = logging.getLogger(__name__)

_ALL_SYNC_METHODS = ["multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only"]


def _sync_method_row(data: dict, method: str) -> dict:
    """Normalize one method summary from ``sync_info.json``."""
    methods = data.get("methods") if isinstance(data.get("methods"), dict) else {}
    m = methods.get(method, {}) or {}
    return {
        "available": bool(m.get("available", False)),
        "offset_seconds": m.get("offset_seconds"),
        "corr_offset_and_drift": m.get("corr_offset_and_drift"),
        "drift_ppm": m.get("drift_ppm"),
        "drift_source": m.get("drift_source", ""),
        "calibration_span_s": m.get("calibration_span_s"),
        "calibration_n_windows": m.get("calibration_n_anchors"),
        "calibration_fit_r2": m.get("calibration_fit_r2"),
    }


def _session_and_suffix(name: str) -> tuple[str, str]:
    """Split ``'2026-02-26_r4'`` into session ``'2026-02-26'`` and suffix ``'r4'``."""
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1][:1] == "r" and parts[1][1:].isdigit():
        return parts[0], parts[1]
    return name, ""


# ---------------------------------------------------------------------------
# Calibration parameter aggregation
# ---------------------------------------------------------------------------

def aggregate_calibration_params(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate calibration params."""
    root = sections_root()
    if not root.exists():
        log.warning("sections_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    rows: list[dict] = []

    for section_dir in sorted(root.iterdir(), key=section_sort_key):
        if not section_dir.is_dir():
            continue

        if recording_names is not None:
            if not any(section_dir.name.startswith(rec) for rec in recording_names):
                continue

        cal_json = section_dir / "calibrated" / "calibration.json"
        if not cal_json.exists():
            log.debug("No calibration.json for section %s", section_dir.name)
            continue

        try:
            data = json.loads(cal_json.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Failed to read %s: %s", project_relative_path(cal_json), exc)
            continue

        # Derive recording name from section folder name (strip trailing sN).
        rec_name = section_dir.name.rsplit("s", 1)[0] if "s" in section_dir.name else section_dir.name
        session, suffix = _session_and_suffix(rec_name)

        quality_block = data.get("quality", {})
        provenance = data.get("provenance", {})

        row: dict = {
            "section_id": section_dir.name,
            "recording_name": rec_name,
            "session": session,
            "recording_suffix": suffix,
            "protocol_detected": bool(data.get("protocol_detected", False)),
            "calibration_quality": quality_block.get("overall", ""),
            "quality_tags": "|".join(quality_block.get("tags", [])),
            "fallback_used": bool(provenance.get("fallback_used", False)),
        }

        intrinsics_block = data.get("intrinsics", {})
        alignment_block = data.get("alignment", {})

        for sensor in ("sporsa", "arduino"):
            intr = intrinsics_block.get(sensor, {})
            aln = alignment_block.get(sensor, {})

            bias_gx, bias_gy, bias_gz = _unpack3(intr.get("gyro_bias"), 0.0)
            row[f"{sensor}_gyro_bias_x"] = bias_gx
            row[f"{sensor}_gyro_bias_y"] = bias_gy
            row[f"{sensor}_gyro_bias_z"] = bias_gz

            acc_bias = intr.get("acc_bias")
            bias_ax, bias_ay, bias_az = _unpack3(acc_bias, float("nan")) if acc_bias is not None else (float("nan"),) * 3
            row[f"{sensor}_acc_bias_x"] = bias_ax
            row[f"{sensor}_acc_bias_y"] = bias_ay
            row[f"{sensor}_acc_bias_z"] = bias_az

            acc_scale = intr.get("acc_scale")
            sc_x, sc_y, sc_z = _unpack3(acc_scale, float("nan")) if acc_scale is not None else (float("nan"),) * 3
            row[f"{sensor}_acc_scale_x"] = sc_x
            row[f"{sensor}_acc_scale_y"] = sc_y
            row[f"{sensor}_acc_scale_z"] = sc_z

            row[f"{sensor}_intrinsics_quality"] = intr.get("quality", "")
            row[f"{sensor}_intrinsics_residual_ms2"] = intr.get("static_residual_ms2")

            grav_x, grav_y, grav_z = _unpack3(aln.get("gravity_estimate"), 0.0)
            row[f"{sensor}_gravity_estimate_x"] = grav_x
            row[f"{sensor}_gravity_estimate_y"] = grav_y
            row[f"{sensor}_gravity_estimate_z"] = grav_z
            row[f"{sensor}_gravity_residual_ms2"] = aln.get("gravity_residual_ms2")
            row[f"{sensor}_yaw_source"] = aln.get("yaw_source", "")
            row[f"{sensor}_yaw_confidence"] = aln.get("yaw_confidence")
            row[f"{sensor}_alignment_window_start_ms"] = aln.get("alignment_window_start_ms")
            row[f"{sensor}_alignment_window_end_ms"] = aln.get("alignment_window_end_ms")

        rows.append(row)
        log.debug("Loaded calibration for section %s", section_dir.name)

    if not rows:
        log.warning("No calibration.json files found under %s", project_relative_path(root))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Aggregated calibration params: %d sections", len(df))
    return df


# ---------------------------------------------------------------------------
# Orientation stats aggregation
# ---------------------------------------------------------------------------

def aggregate_orientation_stats(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate orientation stats."""
    root = sections_root()
    if not root.exists():
        log.warning("sections_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    rows: list[dict] = []

    for section_dir in sorted(root.iterdir(), key=section_sort_key):
        if not section_dir.is_dir():
            continue

        if recording_names is not None:
            if not any(section_dir.name.startswith(rec) for rec in recording_names):
                continue

        stats_path = section_dir / "orientation" / "orientation_stats.json"
        if not stats_path.exists():
            log.debug("No orientation_stats.json for section %s", section_dir.name)
            continue

        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Failed to read %s: %s", project_relative_path(stats_path), exc)
            continue

        rec_name = section_dir.name.rsplit("s", 1)[0] if "s" in section_dir.name else section_dir.name
        session, suffix = _session_and_suffix(rec_name)
        row: dict = {
            "section_id": section_dir.name,
            "recording_name": rec_name,
            "session": session,
            "recording_suffix": suffix,
        }

        sensors_block = data.get("sensors", {})
        for sensor in ("sporsa", "arduino"):
            sensor_data = sensors_block.get(sensor, {})
            residual = sensor_data.get("gravity_residual_ms2", sensor_data.get("selected_residual_ms2"))
            row[f"{sensor}_residual_ms2"] = residual
            row[f"{sensor}_quality"] = sensor_data.get("quality", "")

        rows.append(row)
        log.debug("Loaded orientation stats for section %s", section_dir.name)

    if not rows:
        log.warning("No orientation_stats.json files found under %s", project_relative_path(root))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Aggregated orientation stats: %d sections", len(df))
    return df


def _unpack3(value, default: float) -> tuple[float, float, float]:
    """Return unpack3."""
    try:
        lst = list(value)
        return float(lst[0]), float(lst[1]), float(lst[2])
    except (TypeError, ValueError, IndexError):
        return default, default, default


def _as_float(value) -> float | None:
    """Convert float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Sync parameter aggregation
# ---------------------------------------------------------------------------

def aggregate_sync_params(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate sync params."""
    root = recordings_root()
    if not root.exists():
        log.warning("recordings_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    rows: list[dict] = []

    targets = sorted([root / name for name in recording_names], key=recording_sort_key) if recording_names is not None else sorted(root.iterdir(), key=recording_sort_key)

    for rec_dir in targets:
        if not rec_dir.is_dir():
            continue

        synced_dir = rec_dir / "synced"
        if not synced_dir.is_dir():
            log.debug("No synced/ directory for recording %s", rec_dir.name)
            continue

        row = _build_sync_row(rec_dir.name, synced_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        log.warning("No sync data found under %s", project_relative_path(root))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Aggregated sync params: %d recordings", len(df))
    return df


def _build_sync_row(recording_name: str, synced_dir: Path) -> dict | None:
    """Build one row of sync metadata for a recording."""
    session, suffix = _session_and_suffix(recording_name)
    row: dict = {
        "recording_name": recording_name,
        "session": session,
        "recording_suffix": suffix,
    }

    sync_info_path = synced_dir / "sync_info.json"
    if sync_info_path.exists():
        try:
            info = json.loads(sync_info_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Failed to read %s: %s", project_relative_path(sync_info_path), exc)
            info = {}

        calibration = info.get("calibration") if isinstance(info.get("calibration"), dict) else {}
        anchors = calibration.get("anchors") if isinstance(calibration.get("anchors"), list) else []

        row["selected_method"] = info.get("selected_method", "")
        row["offset_seconds"] = info.get("offset_seconds")
        drift = info.get("drift_seconds_per_second")
        row["drift_seconds_per_second"] = drift
        row["drift_ppm"] = (drift * 1e6) if drift is not None else None
        corr = (info.get("correlation") or {})
        row["corr_offset_only"] = corr.get("offset_only")
        row["corr_offset_and_drift"] = corr.get("offset_and_drift")
        row["drift_source"] = info.get("drift_source", "")

        for method in _ALL_SYNC_METHODS:
            m = _sync_method_row(info, method)
            row[f"{method}_available"] = bool(m.get("available", False))
            row[f"{method}_offset_seconds"] = m.get("offset_seconds")
            row[f"{method}_corr_offset_and_drift"] = m.get("corr_offset_and_drift")
            row[f"{method}_drift_ppm"] = m.get("drift_ppm")
            row[f"{method}_drift_source"] = m.get("drift_source", "")
            if method == "signal_only":
                row[f"{method}_cal_span_s"] = m.get("calibration_span_s")
                row[f"{method}_cal_n_windows"] = m.get("calibration_n_windows")
                row[f"{method}_cal_fit_r2"] = m.get("calibration_fit_r2")
                row[f"{method}_cal_has_anchors"] = False
            else:
                row[f"{method}_cal_span_s"] = calibration.get("anchor_span_s")
                row[f"{method}_cal_n_windows"] = calibration.get("n_anchors")
                row[f"{method}_cal_fit_r2"] = m.get("calibration_fit_r2")
                row[f"{method}_cal_has_anchors"] = bool(anchors)
    else:
        log.debug("No sync data for recording %s", recording_name)
        return None

    log.debug("Loaded sync params for recording %s", recording_name)
    return row


# ---------------------------------------------------------------------------
# Parsed parameter aggregation
# ---------------------------------------------------------------------------

def aggregate_parsed_params(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate parsed params."""
    root = recordings_root()
    if not root.exists():
        log.warning("recordings_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    rows: list[dict] = []

    targets = sorted([root / name for name in recording_names], key=recording_sort_key) if recording_names is not None else sorted(root.iterdir(), key=recording_sort_key)

    for rec_dir in targets:
        if not rec_dir.is_dir():
            continue

        stats_path = rec_dir / "parsed" / "recording_stats.json"
        if not stats_path.exists():
            log.debug("No recording_stats.json for %s", rec_dir.name)
            continue

        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Failed to read %s: %s", project_relative_path(stats_path), exc)
            continue

        row: dict = {
            "recording_name": data.get("recording_name", rec_dir.name),
            "session_name": data.get("session_name", ""),
            "quality_category": data.get("quality_category", ""),
            "quality_reason": data.get("quality_reason", ""),
            "sporsa_segments": data.get("sporsa_segments"),
            "arduino_segments": data.get("arduino_segments"),
        }

        for sensor in ("sporsa", "arduino"):
            stream = data.get("streams", {}).get(sensor, {})
            for k, v in stream.items():
                row[f"{sensor}_{k}"] = v

        rows.append(row)
        log.debug("Loaded parsed stats for recording %s", rec_dir.name)

    if not rows:
        log.warning("No recording_stats.json files found under %s", project_relative_path(root))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    log.info("Aggregated parsed params: %d recordings", len(df))
    return df
