"""Session-level plotting utilities for the thesis pipeline.

For a given session (date), this module iterates all matching recording
directories under ``data/recordings/`` and regenerates plots for each
relevant processing stage.  The resulting figures are used to visually
assess data quality, synchronisation, calibration, and orientation before
including recordings in motion or incident analysis.

Stages plotted per recording
------------------------------
- ``parsed``:
  Sensor + comparison plots, timing quality (interval histograms,
  interval timeline, Arduino clock drift).
- ``synced``:
  Flat selected-stream CSVs plus multi-method comparison PNGs
  (acc/gyro/mag norms, metrics) and ``sporsa`` vs ``arduino`` overlays.
- ``sections/section_*``:
  Sensor + comparison plots for each section.
- ``calibrated``:
  World-frame calibration diagnostics.
- ``orientation``:
  Euler angles, gravity-compensated acceleration, sensor comparison, and
  relative (head vs handlebar) orientation.

During a full sync run, methods write to ``synced/sda/`` … temporarily;
after selection those folders are removed and only flat ``synced/*.csv``
and comparison figures remain.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

from common import recording_stage_dir, recordings_root
from common.paths import iter_sections_for_recording, parse_section_folder_name
from visualization import (
    plot_calibration,
    plot_comparison,
    plot_orientation,
    plot_sensor,
    plot_sync,
    plot_timing,
)


SENSOR_NAMES: tuple[str, str] = ("sporsa", "arduino")

# Stages to skip entirely during session plotting.
_SKIP_STAGES: frozenset[str] = frozenset()


def _iter_session_recordings(session_name: str) -> Iterable[str]:
    """Yield recording names that belong to ``session_name`` (prefix match)."""
    root = recordings_root()
    if not root.exists():
        raise FileNotFoundError(f"Recordings root not found: {root}")
    prefix = f"{session_name}_"
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(prefix):
            yield entry.name


def _iter_recording_stages(recording_name: str) -> Iterable[str]:
    """Yield stage identifiers for a recording.

    The ``sections`` directory (now a top-level ``data/sections/`` root) is
    expanded into virtual stages ``sections/section_N``.
    """
    rec_dir = recordings_root() / recording_name
    for child in sorted(rec_dir.iterdir()):
        if not child.is_dir() or child.name in _SKIP_STAGES:
            continue
        yield child.name

    # Add per-section virtual stages.
    section_entries: list[tuple[int, str]] = []
    for sec_dir in iter_sections_for_recording(recording_name):
        _rec, sec_idx = parse_section_folder_name(sec_dir.name)
        section_entries.append((sec_idx, f"sections/section_{sec_idx}"))
    for _sec_idx, stage in sorted(section_entries, key=lambda t: t[0]):
        yield stage


def _plot_sensor_and_comparison(recording_name: str, stage: str) -> None:
    """Run sensor + comparison plots for a given ``<recording_name>/<stage>``."""
    stage_ref = f"{recording_name}/{stage}"
    for sensor_name in SENSOR_NAMES:
        try:
            plot_sensor.main([stage_ref, sensor_name])
            plot_sensor.main([stage_ref, sensor_name, "--norm", "--acc"])
        except SystemExit:
            pass

    try:
        plot_comparison.main([stage_ref])
        plot_comparison.main([stage_ref, "--norm"])
    except SystemExit:
        pass


def plot_recording(recording_name: str, stage_filter: Optional[str] = None) -> None:
    """Generate all plots for a single recording across its stages."""
    for stage in _iter_recording_stages(recording_name):
        if stage_filter is not None and stage != stage_filter:
            continue

        # --- Orientation stage ---
        if stage == "orientation":
            plot_orientation.plot_orientation_stage(recording_name, stage)
            continue

        # --- Calibration stage ---
        if stage == "calibrated":
            plot_calibration.plot_calibration_stage(recording_name)
            continue

        # --- Parsed stage: sensor plots + timing quality ---
        if stage == "parsed":
            _plot_sensor_and_comparison(recording_name, stage)
            plot_timing.plot_timing_stage(recording_name)
            continue

        # --- Best selected sync stage: sensor plots + sync quality ---
        if stage == "synced":
            _plot_sensor_and_comparison(recording_name, stage)
            plot_sync.plot_sync_stage(recording_name)
            continue

        # --- Sections: sensor + comparison plots ---
        _plot_sensor_and_comparison(recording_name, stage)


def plot_session(session_name: str, stage_filter: Optional[str] = None) -> None:
    """Generate plots for all recordings belonging to ``session_name``."""
    for recording_name in _iter_session_recordings(session_name):
        plot_recording(recording_name, stage_filter=stage_filter)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_session",
        description="Generate plots for all recordings of a session (date).",
    )
    parser.add_argument(
        "session_names",
        nargs="+",
        help="One or more session names (dates) whose recordings will be plotted.",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        default=None,
        help="Only plot this stage (e.g. parsed, synced, orientation).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    for session_name in args.session_names:
        plot_session(session_name, stage_filter=args.stage)


if __name__ == "__main__":
    main()
