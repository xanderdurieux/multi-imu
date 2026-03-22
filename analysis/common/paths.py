"""Path utilities for locating data directories."""

from __future__ import annotations

from pathlib import Path


def analysis_root() -> Path:
    """Return the analysis project root directory."""
    return Path(__file__).resolve().parents[1]


def _data_root() -> Path:
    return analysis_root() / "data"


def sessions_root() -> Path:
    """Return the directory containing raw session input folders."""
    return _data_root() / "sessions"


def recordings_root() -> Path:
    """Return the directory containing all processed recording folders."""
    return _data_root() / "recordings"


def session_input_dir(session_name: str) -> Path:
    """Return path to the raw input directory for a session (date)."""
    return sessions_root() / session_name


def recording_dir(recording_name: str) -> Path:
    """Return path to the root directory of a recording (e.g. '2026-02-26_5')."""
    return recordings_root() / recording_name


def recording_stage_dir(recording_name: str, stage: str) -> Path:
    """Return path to a stage directory within a recording.

    Example stages: ``parsed``, ``synced_lida``, ``synced_cal``, ``orientation``.
    """
    return recording_dir(recording_name) / stage


def find_sensor_csv(
    recording_name: str,
    stage: str,
    sensor_name: str,
) -> Path:
    """Find a single CSV for a given sensor within a recording stage directory.

    Looks for ``*.csv`` files under
    ``data/recordings/<recording_name>/<stage>/`` whose filename contains
    ``sensor_name`` (case-insensitive). Raises if none or more than one match.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    csv_files = list(stage_dir.glob("*.csv"))
    token = sensor_name.lower()

    exact = [f for f in csv_files if f.stem.lower() == token]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        names = ", ".join(sorted(f.name for f in exact))
        raise ValueError(f"Multiple CSV files named like '{sensor_name}.csv' in {stage_dir}: {names}")

    matching = [f for f in csv_files if token in f.name.lower()]
    if not matching:
        raise FileNotFoundError(
            f"No CSV file containing '{sensor_name}' in {stage_dir}"
        )
    if len(matching) > 1:
        names = ", ".join(sorted(f.name for f in matching))
        raise ValueError(
            f"Multiple files matching '{sensor_name}' in {stage_dir}: {names}"
        )

    return matching[0]
