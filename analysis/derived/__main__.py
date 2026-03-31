"""CLI entry point for the derived signal computation stage.

Usage::

    python -m derived <section_name>
    python -m derived <section_name> --sample-rate-hz 50
    python -m derived --recording 2026-02-26_r1
    python -m derived --recording 2026-02-26_r1 --force
"""

from __future__ import annotations

import argparse
import logging
import sys

from common.paths import parse_section_folder_name, section_dir
from .pipeline import process_section_derived, process_recording_derived


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m derived",
        description="Compute derived physical signals from calibrated IMU data.",
    )
    parser.add_argument(
        "section_name",
        nargs="?",
        help="Section folder name (e.g. 2026-02-26_r1s1).",
    )
    parser.add_argument(
        "--recording",
        help="Recording name to process all its sections (e.g. 2026-02-26_r1).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Nominal sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing derived signal outputs.",
    )
    args = parser.parse_args(argv)

    if args.recording:
        results = process_recording_derived(
            args.recording,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
        )
        n_ok = sum(results)
        print(f"Derived signals: {n_ok}/{len(results)} section(s) succeeded for recording '{args.recording}'.")
        if n_ok < len(results):
            sys.exit(1)
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
        ok = process_section_derived(
            section_path,
            sample_rate_hz=args.sample_rate_hz,
            force=args.force,
        )
        if ok:
            print(f"Derived signals computed successfully for '{args.section_name}'.")
        else:
            print(f"Failed to compute derived signals for '{args.section_name}'.", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
