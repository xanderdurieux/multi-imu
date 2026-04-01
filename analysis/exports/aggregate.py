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

logger = logging.getLogger(__name__)

_ALL_SYNC_METHODS = ["sda", "lida", "calibration", "online", "adaptive"]


# ---------------------------------------------------------------------------
# Calibration parameter aggregation
# ---------------------------------------------------------------------------

def aggregate_calibration_params(
    recording_names: list[str] | None = None,
) -> pd.DataFrame:
    """Collect calibration.json from all sections into a flat DataFrame.

    One row per section with columns for both sensors (sporsa / arduino):
    ``<sensor>_acc_bias_{x,y,z}``, ``<sensor>_gyro_bias_{x,y,z}``,
    ``<sensor>_gravity_residual_ms2``, ``<sensor>_forward_confidence``,
    ``<sensor>_quality``, ``<sensor>_fallback_used``.

    Plus section-level columns:
    ``section_id``, ``recording_name``, ``frame_alignment``,
    ``calibration_quality``, ``quality_tags``.

    If present in ``calibrated/all_methods.json``, static-reference values are
    also exported:
    ``static_acc_bias_{x,y,z}``, ``static_acc_scale_{x,y,z}``,
    ``static_gyro_bias_deg_s_{x,y,z}``, ``static_reference_path``,
    ``static_reference_warnings``.
    """
    root = sections_root()
    if not root.exists():
        logger.warning("sections_root does not exist: %s", project_relative_path(root))
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
            logger.debug("No calibration.json for section %s", section_dir.name)
            continue

        try:
            data = json.loads(cal_json.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", project_relative_path(cal_json), exc)
            continue

        # Derive recording name from section folder name (strip trailing sN).
        rec_name = section_dir.name.rsplit("s", 1)[0] if "s" in section_dir.name else section_dir.name

        row: dict = {
            "section_id": section_dir.name,
            "recording_name": rec_name,
            "frame_alignment": data.get("frame_alignment", ""),
            "calibration_quality": data.get("calibration_quality", ""),
            "quality_tags": "|".join(data.get("quality_tags", [])),
            "selected_method": _read_all_methods_selected(section_dir),
        }

        static_ref = _read_static_calibration_reference(section_dir)
        if static_ref:
            row.update(static_ref)

        for sensor in ("sporsa", "arduino"):
            s = data.get(sensor, {})
            bias_ax, bias_ay, bias_az = _unpack3(s.get("acc_bias"), 0.0)
            bias_gx, bias_gy, bias_gz = _unpack3(s.get("gyro_bias"), 0.0)
            grav_x, grav_y, grav_z = _unpack3(s.get("gravity_vector_body"), 0.0)

            row[f"{sensor}_acc_bias_x"] = bias_ax
            row[f"{sensor}_acc_bias_y"] = bias_ay
            row[f"{sensor}_acc_bias_z"] = bias_az
            row[f"{sensor}_gyro_bias_x"] = bias_gx
            row[f"{sensor}_gyro_bias_y"] = bias_gy
            row[f"{sensor}_gyro_bias_z"] = bias_gz
            row[f"{sensor}_gravity_x"] = grav_x
            row[f"{sensor}_gravity_y"] = grav_y
            row[f"{sensor}_gravity_z"] = grav_z
            row[f"{sensor}_gravity_residual_ms2"] = s.get("gravity_residual_ms2")
            row[f"{sensor}_forward_confidence"] = s.get("forward_confidence")
            row[f"{sensor}_quality"] = s.get("quality", "")
            row[f"{sensor}_fallback_used"] = s.get("fallback_used", False)

        rows.append(row)
        logger.debug("Loaded calibration for section %s", section_dir.name)

    if not rows:
        logger.warning("No calibration.json files found under %s", project_relative_path(root))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Aggregated calibration params: %d sections", len(df))
    return df


def _read_all_methods_selected(section_dir: Path) -> str:
    """Read selected_method from calibrated/all_methods.json if present."""
    path = section_dir / "calibrated" / "all_methods.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("selected_method", ""))
    except Exception:
        return ""


def _read_static_calibration_reference(section_dir: Path) -> dict:
    """Read static calibration reference values from calibrated/all_methods.json."""
    path = section_dir / "calibrated" / "all_methods.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    ref = data.get("static_calibration_reference")
    if not isinstance(ref, dict):
        return {}

    out: dict[str, float | str] = {
        "static_reference_path": str(ref.get("path", "")),
        "static_reference_warnings": "|".join(ref.get("warnings", []) or []),
    }

    acc_bias = ref.get("accelerometer_bias") or {}
    acc_scale = ref.get("accelerometer_scale") or {}
    gyro_bias = ref.get("gyroscope_bias_deg_s") or {}
    for axis in ("x", "y", "z"):
        out[f"static_acc_bias_{axis}"] = _as_float(acc_bias.get(axis))
        out[f"static_acc_scale_{axis}"] = _as_float(acc_scale.get(axis))
        out[f"static_gyro_bias_deg_s_{axis}"] = _as_float(gyro_bias.get(axis))
    return out


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
        logger.warning("recordings_root does not exist: %s", project_relative_path(root))
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
            logger.debug("No synced/ directory for recording %s", rec_dir.name)
            continue

        row = _build_sync_row(rec_dir.name, synced_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        logger.warning("No sync data found under %s", project_relative_path(root))
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info("Aggregated sync params: %d recordings", len(df))
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
            logger.warning("Failed to read %s: %s", project_relative_path(all_methods_path), exc)
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

    # Also pull the selected method's offset / correlation from sync_info.json.
    if sync_info_path.exists():
        try:
            info = json.loads(sync_info_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", project_relative_path(sync_info_path), exc)
            info = {}

        row["offset_seconds"] = info.get("offset_seconds")
        drift = info.get("drift_seconds_per_second")
        row["drift_seconds_per_second"] = drift
        row["drift_ppm"] = (drift * 1e6) if drift is not None else None
        corr = (info.get("correlation") or {})
        row["corr_offset_only"] = corr.get("offset_only")
        row["corr_offset_and_drift"] = corr.get("offset_and_drift")
        row["drift_source"] = info.get("drift_source", "")

        if "selected_method" not in row:
            row["selected_method"] = ""
    elif "selected_method" not in row:
        logger.debug("No sync data for recording %s", recording_name)
        return None

    logger.debug("Loaded sync params for recording %s", recording_name)
    return row
