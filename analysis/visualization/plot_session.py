"""Session-level plotting utilities.

For a given session, this module:

- Iterates over all stage directories under ``data/<session_name>/`` except ``full/``.
- For each stage, generates:
  - Comparison plots between Sporsa and Arduino IMUs (axes and norm/magnitude).
  - Per-sensor plots for Sporsa and Arduino.
  - A dedicated comparison plot of the accelerometer norm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import argparse

from common import data_root
from visualization import plot_comparison, plot_sensor


SENSOR_NAMES: tuple[str, str] = ("sporsa", "arduino")


def _iter_session_stages(session_name: str) -> Iterable[str]:
    """Yield stage names for a session, skipping ``full/``."""
    session_dir = data_root() / session_name
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    for child in sorted(session_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name == "raw" or child.name == "full":
            continue
        yield child.name


def plot_session(session_name: str, stage_filter: Optional[str] = None) -> None:
    """Generate all requested plots for a single session across its stages."""
    for stage in _iter_session_stages(session_name):
        if stage_filter is not None and stage != stage_filter:
            continue

        session_name_stage = f"{session_name}/{stage}"
        print(f"[plot_session] Processing {session_name_stage} ...")

        # Comparison plots: axes and norm/magnitude.
        plot_comparison.main(
            [session_name_stage],
        )
        plot_comparison.main(
            [session_name_stage, "--norm"],
        )

        # Per-sensor plots (axes) for Sporsa and Arduino.
        for sensor_name in SENSOR_NAMES:
            # Per-sensor axes plots.
            plot_sensor.main([session_name_stage, sensor_name])

            # Accelerometer norm comparison using plot_sensor, then keep only the 'acc' subplot.
            plot_sensor.main([session_name_stage, sensor_name, "--norm", "--acc"])


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for session-level plotting."""
    parser = argparse.ArgumentParser(
        prog="python -m visualization.plot_session",
        description="Generate plots for one or more sessions (optionally for a single stage).",
    )
    parser.add_argument(
        "session_names",
        nargs="+",
        help="One or more session names under data/<session_name>/",
    )
    parser.add_argument(
        "--stage",
        dest="stage",
        help="Only plot this stage (e.g. parsed, synced, parsed_orientation).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point using argparse, with optional stage filter."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    for session_name in args.session_names:
        plot_session(session_name, stage_filter=args.stage)


if __name__ == "__main__":
    main()

