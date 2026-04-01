"""CLI entry point for the calibration stage.

Usage::

    python -m calibration <section_name>
    python -m calibration <section_name> --frame gravity_plus_forward
    python -m calibration --recording 2026-02-26_r1 --all-sections
"""

from __future__ import annotations

import argparse
import logging
import sys

from common.paths import parse_section_folder_name, project_relative_path, section_dir
from .pipeline import calibrate_section, calibrate_recording_sections


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m calibration",
        description="Estimate and apply IMU calibration for one or more sections.",
    )
    parser.add_argument(
        "section_name",
        nargs="?",
        help="Section folder name (e.g. 2026-02-26_r1s1).",
    )
    parser.add_argument(
        "--recording",
        help="Recording name to calibrate all its sections (e.g. 2026-02-26_r1).",
    )
    parser.add_argument(
        "--frame",
        default="gravity_only",
        choices=["gravity_only", "gravity_plus_forward"],
        help="Frame alignment mode (default: gravity_only).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Approximate sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing calibration outputs.",
    )
    args = parser.parse_args(argv)

    if args.recording:
        results = calibrate_recording_sections(
            args.recording,
            frame_alignment=args.frame,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
        )
        print(f"Calibrated {len(results)} section(s) for recording '{args.recording}'.")
    elif args.section_name:
        try:
            recording_name, section_idx = parse_section_folder_name(args.section_name)
            section_path = section_dir(recording_name, section_idx)
        except ValueError:
            print(f"Invalid section name: {args.section_name}", file=sys.stderr)
            sys.exit(1)
        if not section_path.exists():
            print(f"Section not found: {project_relative_path(section_path)}", file=sys.stderr)
            sys.exit(1)
        cal = calibrate_section(
            section_path,
            frame_alignment=args.frame,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
        )
        print(f"Quality: {cal.calibration_quality}")
        if cal.quality_tags:
            print(f"Tags: {', '.join(cal.quality_tags)}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
