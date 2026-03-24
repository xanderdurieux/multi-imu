"""CLI entry for orientation: uv run -m orientation.estimate."""

from __future__ import annotations

import argparse
import logging
import sys

# When run as python -m orientation, run estimate
if __name__ == "__main__":
    from .estimate import estimate_sections_from_args
    parser = argparse.ArgumentParser(prog="python -m orientation.estimate")
    parser.add_argument("name", help="Section path or recording with --all-sections")
    parser.add_argument("--all-sections", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    estimate_sections_from_args(args.name, all_sections=args.all_sections)
