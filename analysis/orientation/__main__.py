"""CLI entry point for the orientation estimation stage.

Usage::

    python -m orientation <section_name>
    python -m orientation <section_name> --force
    python -m orientation --recording <recording_name>
    python -m orientation --recording <recording_name> --variants madgwick complementary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from common.paths import parse_section_folder_name, section_dir
from .pipeline import process_section_orientation, process_recording_orientation


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m orientation",
        description="Run orientation filters on calibrated IMU data.",
    )
    parser.add_argument(
        "section_name",
        nargs="?",
        help="Section folder name (e.g. 2026-02-26_r1s1).",
    )
    parser.add_argument(
        "--recording",
        metavar="RECORDING",
        help="Recording name to process all its sections (e.g. 2026-02-26_r1).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        metavar="HZ",
        help="Sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["madgwick", "complementary"],
        choices=["madgwick", "complementary"],
        metavar="VARIANT",
        help="Filter variants to run (default: madgwick complementary).",
    )
    parser.add_argument(
        "--canonical-variant",
        default="madgwick",
        choices=["madgwick", "complementary"],
        metavar="VARIANT",
        help="Canonical (primary) variant to record in stats (default: madgwick).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing orientation outputs.",
    )
    args = parser.parse_args(argv)

    if args.recording:
        results = process_recording_orientation(
            args.recording,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
            canonical_variant=args.canonical_variant,
            variants=args.variants,
        )
        print(f"Processed {len(results)} section(s) for recording '{args.recording}'.")
        for i, stats in enumerate(results, start=1):
            summary = {
                "selected_method": stats.get("selected_method"),
                "sporsa": stats.get("sporsa", {}),
                "arduino": stats.get("arduino", {}),
            }
            print(f"  Section {i}: {json.dumps(summary, indent=4)}")
    elif args.section_name:
        try:
            recording_name, section_idx = parse_section_folder_name(args.section_name)
            section_path = section_dir(recording_name, section_idx)
        except ValueError:
            print(f"Invalid section name: {args.section_name}", file=sys.stderr)
            sys.exit(1)
        if not section_path.exists():
            print(f"Section not found: {section_path}", file=sys.stderr)
            sys.exit(1)
        stats = process_section_orientation(
            section_path,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
            canonical_variant=args.canonical_variant,
            variants=args.variants,
        )
        print(f"Selected method: {stats.get('selected_method', '?')}")
        for sensor in ("sporsa", "arduino"):
            sensor_stats = stats.get(sensor, {})
            if not sensor_stats:
                print(f"  {sensor}: not processed (CSV missing?)")
                continue
            print(
                f"  {sensor}: quality={sensor_stats['quality']}"
                f"  score={sensor_stats['score']:.3f}"
                f"  gravity_alignment={sensor_stats['gravity_alignment']:.3f}"
                f"  pitch_std={sensor_stats['pitch_std_deg']:.2f}°"
                f"  roll_std={sensor_stats['roll_std_deg']:.2f}°"
            )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
