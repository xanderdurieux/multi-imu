"""Command-line entry point for exports."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.paths import project_relative_path
from exports.pipeline import run_exports


def _build_parser() -> argparse.ArgumentParser:
    """Build parser."""
    parser = argparse.ArgumentParser(
        prog="python -m exports",
        description="Aggregate section features and write split export tables.",
    )
    parser.add_argument(
        "--recording",
        metavar="NAME",
        action="append",
        dest="recordings",
        default=None,
        help="Recording name to include (may be repeated). Default: all recordings.",
    )
    parser.add_argument(
        "--quality",
        metavar="LABEL",
        default="marginal",
        dest="min_quality_label",
        choices=["poor", "marginal", "good"],
        help="Minimum quality label filter (default: marginal).",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        dest="output_dir",
        help="Output directory (default: data/exports/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing exports.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    paths = run_exports(
        recording_names=args.recordings,
        output_dir=output_dir,
        min_quality_label=args.min_quality_label,
        force=args.force,
    )

    if paths:
        print(f"Export complete. Files written:")
        for name, path in paths.items():
            print(f"  {name}: {project_relative_path(path)}")
    else:
        print("No data exported.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
