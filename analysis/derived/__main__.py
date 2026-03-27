"""CLI for derived signal computation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .compute import derive_section_signals


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m derived")
    ap.add_argument("section", type=str, help="Section folder path")
    ap.add_argument("--orientation-variant", default="complementary_orientation")
    ap.add_argument("--no-normalized", action="store_true", help="Disable robust-z outputs")
    args = ap.parse_args()

    derive_section_signals(
        Path(args.section),
        orientation_variant=args.orientation_variant,
        include_normalized=not args.no_normalized,
    )


if __name__ == "__main__":
    main()
