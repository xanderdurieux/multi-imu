"""Path utilities for locating data directories."""

from __future__ import annotations

from pathlib import Path


def analysis_root() -> Path:
    """Return the analysis project root directory."""
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the data directory containing raw/ and processed/ subdirectories."""
    return analysis_root() / "data"


def raw_session_dir(session_name: str) -> Path:
    """Return path to raw log files directory for a session."""
    return data_root() / "raw" / session_name


def processed_session_dir(session_name: str) -> Path:
    """Return path to processed CSV files directory for a session."""
    return data_root() / "processed" / session_name


