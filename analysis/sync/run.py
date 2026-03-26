"""Command-line entry for the sync package."""

from __future__ import annotations

import argparse
import logging
import sys

from .pipeline import synchronize_recording_all_methods, synchronize_session


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m sync",
        description=(
            "Run all four sync methods on parsed CSVs, pick the best, copy to synced/, "
            "write comparison plots, and remove per-method subfolders."
        ),
    )
    parser.add_argument(
        "name",
        help="Recording (e.g. 2026-02-26_r5) or session date with --all (e.g. 2026-02-26).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process every recording whose folder name starts with NAME + '_'.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.all_recordings:
        synchronize_session(args.name)
    else:
        synchronize_recording_all_methods(args.name)


if __name__ == "__main__":
    main()
