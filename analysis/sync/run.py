"""Command-line entry point for the sync package."""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import synchronize_recording_all_methods, synchronize_session


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m sync",
        description="Run all sync methods on parsed CSVs, select the best result, and write synced/ outputs.",
    )
    parser.add_argument("name", help="Recording name or session date prefix.")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process all recordings whose folder name starts with NAME + '_'.",
    )
    args = parser.parse_args(list(argv if argv is not None else sys.argv[1:]))
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if args.all_recordings:
        synchronize_session(args.name)
    else:
        synchronize_recording_all_methods(args.name)
