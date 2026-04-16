"""Shared path resolution and CSV I/O helpers."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common.signals import vector_norm


# ---------------------------------------------------------------------------
# CSV schema and shared I/O
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    "timestamp",
    "ax",
    "ay",
    "az",
    "acc_norm",
    "gx",
    "gy",
    "gz",
    "gyro_norm",
    "mx",
    "my",
    "mz",
    "mag_norm",
]

_ACC_AXIS_COLS = ("ax", "ay", "az")
_GYRO_AXIS_COLS = ("gx", "gy", "gz")
_MAG_AXIS_COLS = ("mx", "my", "mz")

_IMU_VALUE_COLUMNS = [col for col in CSV_COLUMNS if col != "timestamp"]


def _looks_like_imu_dataframe(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    return "timestamp" in columns and any(col in columns for col in _IMU_VALUE_COLUMNS)


def _normalize_imu_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in CSV_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    extra = [c for c in out.columns if c not in CSV_COLUMNS]
    out = out[CSV_COLUMNS + extra]

    for col in CSV_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if all(c in out.columns for c in _ACC_AXIS_COLS):
        out["acc_norm"] = vector_norm(out, _ACC_AXIS_COLS)
    if all(c in out.columns for c in _GYRO_AXIS_COLS):
        out["gyro_norm"] = vector_norm(out, _GYRO_AXIS_COLS)
    if all(c in out.columns for c in _MAG_AXIS_COLS):
        out["mag_norm"] = vector_norm(out, _MAG_AXIS_COLS)
      
    return out


def read_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a CSV from disk"""
    path = Path(csv_path)
    df = pd.read_csv(path)
    if _looks_like_imu_dataframe(df):
        return _normalize_imu_dataframe(df)
    return df


def write_csv(df: pd.DataFrame, csv_path: Path | str) -> None:
    """Write a CSV to disk."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = _normalize_imu_dataframe(df) if _looks_like_imu_dataframe(df) else df.copy()
    out.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def read_json_file(path: Path | str) -> Any:
    """Load a JSON file (UTF-8). Returns the decoded object (dict, list, etc.)."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def write_json_file(
    path: Path | str,
    data: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    default: Callable[..., Any] | None = None,
) -> None:
    """Write *data* as JSON to *path* (UTF-8)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    opts: dict[str, Any] = {}
    if indent is not None:
        opts["indent"] = indent
    if sort_keys:
        opts["sort_keys"] = True
    if default is not None:
        opts["default"] = default
    p.write_text(json.dumps(data, **opts), encoding="utf-8")


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert *df* rows to JSON-native Python objects (handles NumPy scalars)."""
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


# ---------------------------------------------------------------------------
# Project and data roots
# ---------------------------------------------------------------------------

def analysis_root() -> Path:
    """Return the analysis project root directory."""
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the root directory containing project data folders."""
    override = os.environ.get("MULTI_IMU_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return analysis_root() / "data"


def calibrations_root() -> Path:
    """Return the directory containing calibration assets (JSON, raw logs)."""
    return data_root() / "_calibrations"


def configs_root() -> Path:
    """Return the directory containing configuration files."""
    return data_root() / "_configs"


def labels_root() -> Path:
    """Return the directory that holds repo-level recording label CSVs."""
    return data_root() / "_labels"


def sessions_root() -> Path:
    """Return the directory containing raw session input folders."""
    return data_root() / "_sessions"


def recordings_root() -> Path:
    """Return the directory containing all processed recording folders."""
    return data_root() / "recordings"


def sections_root() -> Path:
    """Return the directory containing all processed per-section folders."""
    return data_root() / "sections"


def exports_root() -> Path:
    """Return the directory containing exported dataset artifacts."""
    return data_root() / "exports"


def evaluation_root() -> Path:
    """Return the directory containing evaluation artifacts."""
    return data_root() / "evaluation"


def project_relative_path(path: Path | str) -> str:
    """Return a display path relative to the data root."""
    p = Path(path).expanduser().resolve()
    return str(p.relative_to(data_root()))


# ---------------------------------------------------------------------------
# Recording and section naming
# ---------------------------------------------------------------------------

def session_input_dir(session_name: str) -> Path:
    """Return path to the raw input directory for a session (date)."""
    return sessions_root() / session_name


def recording_dir(recording_name: str) -> Path:
    """Return path to the root directory of a recording.

    Recording folder naming is ``<session_name>_r<recording_idx>``.
    """
    return recordings_root() / recording_name


def section_dir(section_name: str) -> Path:
    """Return path to the root directory of a section.

    Section folder naming is ``<recording_name>s<section_idx>``.
    """
    return sections_root() / section_name


def recording_stage_dir(recording_name: str, stage: str) -> Path:
    """Return path to a stage directory within a recording.

    Example stages: ``parsed``, ``synced``.
    """
    return recording_dir(recording_name) / stage


def section_stage_dir(section_name: str, stage: str) -> Path:
    """Return path to a stage directory within a section.

    Example stages: ``calibrated``, ``orientation``, ``derived``, ``features``, ``events``.
    """
    return section_dir(section_name) / stage


def parse_section_folder_name(section_folder_name: str) -> tuple[str, int]:
    """Parse ``<recording_name>s<section_idx>`` into (recording_name, section_idx)."""
    name = section_folder_name.strip().rstrip("/")
    if "s" not in name:
        raise ValueError(f"Not a section folder name (missing 's'): {section_folder_name!r}")
    rec_part, _, s_part = name.rpartition("s")
    if not rec_part or not s_part.isdigit():
        raise ValueError(f"Invalid section folder name: {section_folder_name!r}")
    return rec_part, int(s_part)


def iter_sections_for_recording(recording_name: str) -> list[Path]:
    """List section directories for a recording (sorted by section_idx)."""
    prefix = f"{recording_name}s"
    root = sections_root()
    if not root.exists():
        return []
    out: list[tuple[int, Path]] = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        try:
            _rec, sec_idx = parse_section_folder_name(d.name)
        except ValueError:
            continue
        out.append((sec_idx, d))
    return [p for _i, p in sorted(out, key=lambda t: t[0])]


# ---------------------------------------------------------------------------
# Directory resolution
# ---------------------------------------------------------------------------

def resolve_data_dir(target: str | Path) -> Path:
    """Resolve a recording or section name (with optional stage) to an absolute directory path.

    Accepted string forms:
    - ``<recording_name>``               → recording root directory
    - ``<recording_name>/<stage>``       → recording stage directory
    - ``<section_name>``                 → section root directory
    - ``<section_name>/<stage>``         → section stage directory

    A ``Path`` object may be passed for programmatic callers that already hold
    a concrete directory — it is validated and returned as-is.
    """
    if isinstance(target, Path):
        if target.is_dir():
            return target.resolve()
        raise FileNotFoundError(f"Directory not found: {target}")

    name, sep, stage = str(target).strip().rstrip("/").partition("/")
    if not name:
        raise FileNotFoundError("Empty data directory reference.")

    try:
        parse_section_folder_name(name)
        path = section_stage_dir(name, stage) if sep else section_dir(name)
    except ValueError:
        path = recording_stage_dir(name, stage) if sep else recording_dir(name)

    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return path


# ---------------------------------------------------------------------------
# CSV lookup helpers
# ---------------------------------------------------------------------------

def list_csv_files(directory: Path | str) -> list[Path]:
    """Return all CSV files directly inside *directory*, sorted by name."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def sensor_csv(ref: str, sensor_name: str) -> Path:
    """Return ``<sensor_name>.csv`` inside the resolved directory.

    *ref* is any string accepted by ``resolve_data_dir``.
    """
    directory = resolve_data_dir(ref)
    csv_path = directory / f"{sensor_name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Label path helpers
# ---------------------------------------------------------------------------

def recording_labels_csv(recording_name: str) -> Path:
    """Return the label CSV path for a recording."""
    return labels_root() / f"labels_intervals_{recording_name}.csv"


def section_labels_csv(sec_dir: Path | str) -> Path:
    """Return the label CSV path for a section."""
    return Path(sec_dir) / "labels" / "labels.csv"


# ---------------------------------------------------------------------------
# Config path helpers
# ---------------------------------------------------------------------------

def default_workflow_config_path() -> Path:
    """Return the path to the default workflow config file."""
    return configs_root() / "workflow.default.json"


def cal_segments_config_path() -> Path:
    """Return the path to per-sensor calibration-segment detection parameters (JSON)."""
    return configs_root() / "cal_segments_args.json"


def calibration_segments_json_path(recording_name: str) -> Path:
    """Return the path to the calibration-segments JSON for a recording."""
    return recording_stage_dir(recording_name, "parsed") / "calibration_segments.json"


def load_workflow_config_data(override_path: Path | str | None = None) -> dict:
    """Load workflow config data with default-as-base merging.

    When ``override_path`` is provided, its keys overwrite keys from the
    default workflow config.
    """
    default_payload = read_json_file(default_workflow_config_path())
    if override_path is None:
        return dict(default_payload)
    override_payload = read_json_file(override_path)
    merged = dict(default_payload)
    merged.update(override_payload)
    return merged

