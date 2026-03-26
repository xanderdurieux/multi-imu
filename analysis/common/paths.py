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
    """Return path to the root directory of a recording.

    Recording folder naming is ``<session>_r<idx>``.
    """
    return recordings_root() / recording_name


def sections_root() -> Path:
    """Return the directory containing all processed per-section folders."""
    return _data_root() / "sections"


def section_dir(recording_name: str, section_idx: int) -> Path:
    """Return the directory for one section.

    New section folder naming encodes both the recording index and the section index:
    ``<recording_name>s<section_idx>``.

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


def recording_name_prefix_for_session(session_name: str) -> str:
    """Return the recording name prefix for a given session date."""
    # recordings are named like: <session_name>_r<idx>
    return f"{session_name}_r"


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
