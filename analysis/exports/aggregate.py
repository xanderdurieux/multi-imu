"""Aggregate calibration and sync parameters from sections/recordings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from common.paths import (
    project_relative_path,
    recordings_root,
    sections_root,
)

log = logging.getLogger(__name__)

_ALL_SYNC_METHODS = ["sda", "lida", "calibration", "online", "adaptive"]


# ---------------------------------------------------------------------------
# Calibration parameter aggregation
# ---------------------------------------------------------------------------

def aggregate_calibration_params(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Collect calibration.json (schema v2) from all sections into a flat DataFrame.

    One row per section.  Per-sensor columns (sporsa / arduino):

    Intrinsics
        ``<sensor>_gyro_bias_{x,y,z}``,
        ``<sensor>_acc_bias_{x,y,z}`` (may be NaN when not estimated),
        ``<sensor>_acc_scale_{x,y,z}`` (may be NaN when not estimated),
        ``<sensor>_intrinsics_quality``, ``<sensor>_intrinsics_residual_ms2``.

    Alignment
        ``<sensor>_gravity_estimate_{x,y,z}``,
        ``<sensor>_gravity_residual_ms2``,
        ``<sensor>_yaw_source``,
        ``<sensor>_yaw_confidence``,
        ``<sensor>_alignment_window_start_ms``,
        ``<sensor>_alignment_window_end_ms``.

    Section-level
        ``section_id``, ``recording_name``,
        ``protocol_detected``,
        ``calibration_quality`` (alias: ``quality_overall``),
        ``quality_tags``,
        ``fallback_used``.
    """
    root = sections_root()
    if not root.exists():
        log.warning("sections_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    rows: list[dict] = []

    for section_dir in sorted(root.iterdir()):
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
        except Exception as exc:
            log.warning("Failed to read %s: %s", project_relative_path(cal_json), exc)
            continue

        # Derive recording name from section folder name (strip trailing sN).
        rec_name = section_dir.name.rsplit("s", 1)[0] if "s" in section_dir.name else section_dir.name

        quality_block = data.get("quality", {})
        provenance = data.get("provenance", {})

        row: dict = {
            "section_id": section_dir.name,
            "recording_name": rec_name,
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


def _unpack3(value, default: float) -> tuple[float, float, float]:
    """Unpack a 3-element list, filling with default on failure."""
    try:
        lst = list(value)
        return float(lst[0]), float(lst[1]), float(lst[2])
    except Exception:
        return default, default, default


def _as_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sync parameter aggregation
# ---------------------------------------------------------------------------

def aggregate_sync_params(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Collect sync parameters from all recordings into a flat DataFrame.

    Reads ``<recording>/synced/all_methods.json`` (preferred) and falls back
    to ``sync_info.json`` when only a single-method result is available.

    One row per recording with columns:
    ``recording_name``, ``selected_method``, ``offset_seconds``,
    ``drift_seconds_per_second``, ``drift_ppm``,
    ``corr_offset_only``, ``corr_offset_and_drift``.

    Per-method quality columns are also included:
    ``<method>_available``, ``<method>_corr_offset_and_drift``,
    ``<method>_drift_ppm``, ``<method>_drift_source``.
    """
    root = recordings_root()
    if not root.exists():
        log.warning("recordings_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    rows: list[dict] = []

    targets = sorted(root.iterdir()) if recording_names is None else [
        root / name for name in recording_names
    ]

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
    row: dict = {"recording_name": recording_name}

    # Prefer all_methods.json (written by selection step).
    all_methods_path = synced_dir / "all_methods.json"
    sync_info_path = synced_dir / "sync_info.json"

    if all_methods_path.exists():
        try:
            data = json.loads(all_methods_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to read %s: %s", project_relative_path(all_methods_path), exc)
            data = {}

        row["selected_method"] = data.get("selected_method", "")
        row["selected_stage"] = data.get("selected_stage", "")

        for method in _ALL_SYNC_METHODS:
            m = data.get(method, {}) or {}
            row[f"{method}_available"] = bool(m.get("available", False))
            row[f"{method}_corr_offset_and_drift"] = m.get("corr_offset_and_drift")
            row[f"{method}_drift_ppm"] = m.get("drift_ppm")
            row[f"{method}_drift_source"] = m.get("drift_source", "")
            row[f"{method}_cal_span_s"] = m.get("calibration_span_s")
            row[f"{method}_cal_n_windows"] = m.get("calibration_n_windows")
            row[f"{method}_cal_fit_r2"] = m.get("calibration_fit_r2")
            anchors = m.get("calibration_anchors")
            row[f"{method}_cal_has_anchors"] = bool(isinstance(anchors, list) and len(anchors) > 0)

    # Also pull the selected method's offset / correlation from sync_info.json.
    if sync_info_path.exists():
        try:
            info = json.loads(sync_info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to read %s: %s", project_relative_path(sync_info_path), exc)
            info = {}

        row["offset_seconds"] = info.get("offset_seconds")
        drift = info.get("drift_seconds_per_second")
        row["drift_seconds_per_second"] = drift
        row["drift_ppm"] = (drift * 1e6) if drift is not None else None
        corr = (info.get("correlation") or {})
        row["corr_offset_only"] = corr.get("offset_only")
        row["corr_offset_and_drift"] = corr.get("offset_and_drift")
        row["drift_source"] = info.get("drift_source", "")
        row["signal_mode"] = info.get("signal_mode")
        row["calibration_usage_strategy"] = info.get("calibration_usage_strategy")
        row["segment_aware_used"] = info.get("segment_aware_used")

        if "selected_method" not in row:
            row["selected_method"] = ""
    elif "selected_method" not in row:
        log.debug("No sync data for recording %s", recording_name)
        return None

    log.debug("Loaded sync params for recording %s", recording_name)
    return row
