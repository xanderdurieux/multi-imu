"""Path utilities for locating data directories."""

from __future__ import annotations

from pathlib import Path


def analysis_root() -> Path:
    """Return the analysis project root directory."""
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the data directory containing raw/ and processed/ subdirectories."""
    return analysis_root() / "data"


def session_stage_dir(session_name: str, stage: str) -> Path:
    """Return path to a specific stage directory for a session (e.g., parsed, synced)."""
    return data_root() / session_name / stage


def raw_session_dir(session_name: str) -> Path:
    """Return path to raw log files directory for a session."""
    return session_stage_dir(session_name, "raw")


def parsed_session_dir(session_name: str) -> Path:
    """Return path to processed CSV files directory for a session."""
    return session_stage_dir(session_name, "parsed")


def synced_session_dir(session_name: str) -> Path:
    """Return path to synced CSV files directory for a session."""
    return session_stage_dir(session_name, "synced")


def find_sensor_csv(
    session_name: str,
    stage: str,
    sensor_name: str,
) -> Path:
    """
    Find a single CSV file for a given sensor within a session stage.

    Looks for ``*.csv`` files under ``data/<session_name>/<stage>/`` whose
    filename contains ``sensor_name`` (case-insensitive). Raises if none or
    more than one match is found.
    """
    session_dir = session_stage_dir(session_name, stage)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    csv_files = list(session_dir.glob("*.csv"))
    matching_files = [f for f in csv_files if sensor_name.lower() in f.name.lower()]

    if not matching_files:
        raise FileNotFoundError(f"No CSV file found containing '{sensor_name}' in {session_dir}")

    if len(matching_files) > 1:
        names = ", ".join(sorted(f.name for f in matching_files))
        raise ValueError(f"Multiple files found matching '{sensor_name}' in {session_dir}: {names}")

    return matching_files[0]


