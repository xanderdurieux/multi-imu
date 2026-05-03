"""Command-line entry point for calibration."""

from __future__ import annotations

import argparse
import logging
import sys

from common.paths import parse_section_folder_name, project_relative_path, section_dir
from .pipeline import calibrate_section, calibrate_recording_sections


def main(argv: list[str] | None = None) -> None:
    """Run the command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m calibration",
        description="Protocol-aware IMU calibration: gyro bias, acc correction, gravity alignment.",
    )
    parser.add_argument("section_name", nargs="?", help="Section folder name (e.g. 2026-02-26_r1s1).")
    parser.add_argument("--recording", help="Calibrate all sections for a recording.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing calibration outputs.")
    args = parser.parse_args(argv)

    if args.recording:
        results = calibrate_recording_sections(args.recording, force=args.force)
        print(f"Calibrated {len(results)} section(s) for recording '{args.recording}'.")
    elif args.section_name:
        try:
            parse_section_folder_name(args.section_name)
        except ValueError:
            print(f"Invalid section name: {args.section_name}", file=sys.stderr)
            sys.exit(1)
        section_path = section_dir(args.section_name)
        if not section_path.exists():
            print(f"Section not found: {project_relative_path(section_path)}", file=sys.stderr)
            sys.exit(1)
        cal = calibrate_section(section_path, force=args.force)
        print(f"Protocol detected: {cal.protocol_detected}")
        print(f"Overall quality: {cal.quality.get('overall', '?')}")
        tags = cal.quality.get("tags", [])
        if tags:
            print(f"Tags: {', '.join(tags)}")
        for sensor, alignment in cal.alignment.items():
            print(
                f"  {sensor}: gravity_residual={alignment.gravity_residual_ms2:.3f} m/s²"
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
