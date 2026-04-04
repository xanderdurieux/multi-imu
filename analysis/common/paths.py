"""Shared path resolution and CSV I/O helpers."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# CSV schema and shared I/O
# ---------------------------------------------------------------------------

CSV_COLUMNS: list[str] = [
    "timestamp",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
]

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

    return out


def read_csv(csv_path: Path | str) -> pd.DataFrame:
    """Load a CSV from disk using the project's single shared entry point."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    if _looks_like_imu_dataframe(df):
        return _normalize_imu_dataframe(df)
    return df


def write_csv(df: pd.DataFrame, csv_path: Path | str) -> None:
    """Write a CSV to disk using the project's single shared entry point."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = _normalize_imu_dataframe(df) if _looks_like_imu_dataframe(df) else df.copy()
    out.to_csv(path, index=False)


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


def sessions_root() -> Path:
    """Return the directory containing raw session input folders."""
    return data_root() / "sessions"


def recordings_root() -> Path:
    """Return the directory containing all processed recording folders."""
    return data_root() / "recordings"


def sections_root() -> Path:
    """Return the directory containing all processed per-section folders."""
    return data_root() / "sections"


def labels_root() -> Path:
    """Return the directory that holds repo-level recording label CSVs."""
    return data_root() / "labels"


def exports_root() -> Path:
    """Return the directory containing exported dataset artifacts."""
    return data_root() / "exports"


def evaluation_root() -> Path:
    """Return the directory containing evaluation artifacts."""
    return data_root() / "evaluation"


def configs_root() -> Path:
    """Return the directory containing configuration files."""
    return analysis_root() / "configs"


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

    Recording folder naming is ``<session>_r<idx>``.
    """
    return recordings_root() / recording_name


def recording_stage_dir(recording_name: str, stage: str) -> Path:
    """Return path to a stage directory within a recording.

    Example stages: ``parsed``, ``synced``.
    """
    return recording_dir(recording_name) / stage


def section_dir(recording_name: str, section_idx: int) -> Path:
    """Return the directory for one section.

    Section folder naming is ``<recording_name>s<section_idx>``.

    Example:
        recording_name='2026-02-26_r2', section_idx=1 -> ``data/sections/2026-02-26_r2s1/``
    """
    return sections_root() / f"{recording_name}s{section_idx}"


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
    """Resolve a directory reference inside the project data tree.

    Accepted references are:
    - an existing directory path
    - ``<recording>/<stage>`` for recording stage directories
    - ``<recording>s<section_idx>`` for section directories
    """
    if isinstance(target, Path):
        path = target.expanduser()
        if path.is_dir():
            return path.resolve()
        raise FileNotFoundError(f"Directory not found: {path}")

    ref = str(target).strip().rstrip("/").replace("\\", "/")
    if not ref:
        raise FileNotFoundError("Empty data directory reference.")

    direct = Path(ref).expanduser()
    if direct.is_dir():
        return direct.resolve()

    parts = ref.split("/", 1)
    if len(parts) == 2:
        stage_path = recording_stage_dir(parts[0], parts[1])
        if stage_path.is_dir():
            return stage_path.resolve()

    try:
        parse_section_folder_name(ref)
    except ValueError:
        pass
    else:
        section_path = sections_root() / ref
        if section_path.is_dir():
            return section_path.resolve()

    raise FileNotFoundError(f"Could not resolve data directory reference: {target!r}")


# ---------------------------------------------------------------------------
# CSV lookup helpers
# ---------------------------------------------------------------------------

def list_csv_files(directory: Path | str) -> list[Path]:
    """Return all CSV files directly inside *directory*, sorted by name."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def find_csv_in_dir(directory: Path | str, name_token: str) -> Path:
    """Find ``<name_token>.csv`` directly inside *directory*."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")
    csv_path = path / f"{name_token}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return csv_path


def resolve_sensor_csv(stage_ref: str | Path, sensor_name: str) -> Path:
    """Resolve a stage reference and find the matching sensor CSV inside it."""
    return find_csv_in_dir(resolve_data_dir(stage_ref), sensor_name)


def find_sensor_csv(recording_name: str, stage: str, sensor_name: str) -> Path:
    """Find a single sensor CSV within a recording stage directory."""
    return find_csv_in_dir(recording_stage_dir(recording_name, stage), sensor_name)


def section_sensor_csv(section_dir: Path | str, sensor_name: str) -> Path:
    """Return the canonical sensor CSV path for a section."""
    path = Path(section_dir) / f"{sensor_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Section sensor CSV not found: {path}")
    return path


# ---------------------------------------------------------------------------
# Label path helpers
# ---------------------------------------------------------------------------

def recording_labels_csv(recording_name: str) -> Path:
    """Return the canonical label CSV path for a recording.

    Canonical location: ``data/labels/labels_intervals_<slug>.csv``.
    The file may not exist yet (e.g. before labeling is done).
    """
    return labels_root() / f"labels_intervals_{recording_name}.csv"


def section_labels_csv(section_dir: Path | str) -> Path:
    """Return the canonical label CSV path for a section directory.

    Canonical location: ``<section_dir>/labels/labels.csv``.
    The file may not exist yet (e.g. before label transfer is run).
    """
    return Path(section_dir) / "labels" / "labels.csv"


# ---------------------------------------------------------------------------
# Config path helpers
# ---------------------------------------------------------------------------

def default_workflow_config_path() -> Path:
    """Return the path to the default workflow config file."""
    return data_root() / "configs" / "workflow.default.json"


def read_json_file(path: Path | str) -> dict:
    """Read a JSON file and return a dictionary payload."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


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

