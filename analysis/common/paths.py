"""
Helpers for locating data folders within the analysis project.

"""

from __future__ import annotations

from pathlib import Path


def analysis_root() -> Path:
    """Return the root folder of the analysis project."""
    # This file lives in analysis/common/, so the parent of its parent is analysis/
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Return the root folder containing raw/ and processed/ data."""
    return analysis_root() / "data"


def raw_session_dir(session_name: str) -> Path:
    """Folder where raw logs for a given session are stored."""
    return data_root() / "raw" / session_name


def processed_session_dir(session_name: str) -> Path:
    """Folder where processed CSVs for a given session are written."""
    return data_root() / "processed" / session_name


