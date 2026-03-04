"""Session-level plotting utilities.

For a given session (date), iterates all matching recording directories
under ``data/recordings/`` and generates plots for each stage found.

Stage directories plotted per recording:
- ``parsed``, ``synced_lida``, ``synced_cal``: sensor + comparison plots.
- ``orientation``: orientation plots.
"""

from __future__ import annotations

import argparse
from typing import Iterable, Optional

from common import recording_stage_dir, recordings_root
from visualization import plot_calibration, plot_comparison, plot_orientation, plot_sensor


SENSOR_NAMES: tuple[str, str] = ("sporsa", "arduino")
SKIP_STAGES: frozenset[str] = frozenset()


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

    For most stages (``parsed``, ``synced_lida``, ``synced_cal``, ``orientation``) the stage name
    is returned directly.  The ``sections`` directory is treated specially:
    each of its subdirectories (``section_1``, ``section_2``, …) is yielded as
    the virtual stage ``sections/section_N``.
    """
    rec_dir = recordings_root() / recording_name
    for child in sorted(rec_dir.iterdir()):
        if not child.is_dir() or child.name in SKIP_STAGES:
            continue
        if child.name == "sections":
            for section_dir in sorted(child.iterdir()):
                if section_dir.is_dir():
                    yield f"sections/{section_dir.name}"
        else:
            yield child.name


def plot_recording(recording_name: str, stage_filter: Optional[str] = None) -> None:
    """Generate all plots for a single recording across its stages."""
    for stage in _iter_recording_stages(recording_name):
        if stage_filter is not None and stage != stage_filter:
            continue

        print(f"[{recording_name}/{stage}]")

        if stage == "orientation":
            plot_orientation.plot_orientation_stage(recording_name, stage)
            continue

        if stage == "calibrated":
            plot_calibration.plot_calibration_stage(recording_name)
            continue

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
