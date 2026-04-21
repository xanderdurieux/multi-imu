"""CLI entry point for the orientation estimation stage.

Usage::

    python -m orientation <section_name>
    python -m orientation <section_name> --force
    python -m orientation --recording <recording_name>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from common.paths import parse_section_folder_name, project_relative_path, section_dir
from .pipeline import process_section_orientation, process_recording_orientation


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m orientation",
        description="Run Mahony orientation filter on calibrated IMU data.",
    )
    parser.add_argument("section_name", nargs="?", help="Section folder name (e.g. 2026-02-26_r1s1).")
    parser.add_argument("--recording", metavar="RECORDING", help="Process all sections for a recording.")
    parser.add_argument("--sample-rate-hz", type=float, default=100.0, metavar="HZ")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args(argv)

    if args.recording:
        results = process_recording_orientation(
            args.recording,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
        )
        print(f"Processed {len(results)} section(s) for recording '{args.recording}'.")
        for i, stats in enumerate(results, start=1):
            print(f"  Section {i}: {json.dumps(stats.get('sensors', {}), indent=4)}")
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
        stats = process_section_orientation(
            section_path,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
        )
        for sensor, s in stats.get("sensors", {}).items():
            print(
                f"  {sensor}: quality={s['quality']}"
                f"  gravity_residual={s.get('gravity_residual_ms2', '?')} m/s²"
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
