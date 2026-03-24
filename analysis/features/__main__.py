"""CLI entry for features: uv run -m features.extract or features.aggregate."""

from __future__ import annotations

import sys

# python -m features runs extract by default
# python -m features.aggregate for aggregation
if __name__ == "__main__":
    from .extract import extract_from_args
    import argparse
    import logging
    parser = argparse.ArgumentParser(prog="python -m features.extract")
    parser.add_argument("name", help="Section path, recording, or session")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument("--all", action="store_true", dest="all_recordings")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--hop", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    extract_from_args(
        args.name,
        all_sections=args.all_sections,
        all_recordings=args.all_recordings,
        window_s=args.window,
        hop_s=args.hop,
    )
