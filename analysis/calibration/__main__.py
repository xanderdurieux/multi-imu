"""CLI entry for calibration: uv run -m calibration (dispatches to calibrate or validate)."""

from __future__ import annotations

# When run as "python -m calibration", run calibrate by default.
# Use "python -m calibration.validate" for validation.
if __name__ == "__main__":
    from .calibrate import calibrate_sections_from_args
    import argparse
    import logging
    parser = argparse.ArgumentParser(prog="python -m calibration")
    parser.add_argument("name", help="Section path or recording name with --all-sections")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument("--static-window", type=float, default=3.0)
    parser.add_argument("--variance-threshold", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    calibrate_sections_from_args(
        args.name,
        all_sections=args.all_sections,
        static_window_seconds=args.static_window,
        variance_threshold=args.variance_threshold,
    )
