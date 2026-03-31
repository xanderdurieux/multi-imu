"""CLI entry point for the feature extraction stage.

Usage::

    python -m features <section_name>
    python -m features <section_name> --window-s 2.0 --hop-s 1.0
    python -m features --recording 2026-02-26_r1
    python -m features --recording 2026-02-26_r1 --force
"""

from __future__ import annotations

import argparse
import logging
import sys

from common.paths import sections_root
from .pipeline import (
    extract_features_for_section,
    process_recording_features,
)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m features",
        description="Extract sliding-window features from calibrated dual-IMU section data.",
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
        "--window-s",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="Window length in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--hop-s",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Hop (stride) length in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum sporsa samples per window (default: 10).",
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
        help="Overwrite existing feature outputs.",
    )
    args = parser.parse_args(argv)

    kwargs = dict(
        window_s=args.window_s,
        hop_s=args.hop_s,
        min_samples=args.min_samples,
        sample_rate_hz=args.sample_rate_hz,
        force=args.force,
    )

    if args.recording:
        df = process_recording_features(args.recording, **kwargs)
        if df.empty:
            print(
                f"No features extracted for recording '{args.recording}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Features: {len(df)} windows extracted for recording '{args.recording}'."
        )

    elif args.section_name:
        section_dir = sections_root() / args.section_name
        if not section_dir.exists():
            print(f"Section not found: {section_dir}", file=sys.stderr)
            sys.exit(1)
        df = extract_features_for_section(section_dir, **kwargs)
        if df.empty:
            print(
                f"No features extracted for '{args.section_name}'.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Features: {len(df)} windows extracted for '{args.section_name}'."
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
