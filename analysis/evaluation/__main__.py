"""Run evaluation report: ``uv run python -m evaluation <features_fused.csv> [out_dir] [--config cfg.json]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .experiments import run_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run thesis-oriented dual-IMU experiments.")
    parser.add_argument("features_csv", type=Path, help="Path to fused feature CSV")
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        help="Output directory (default: <csv_parent>/evaluation_report)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config path for feature subsets/experiment settings",
    )
    args = parser.parse_args()

    out = args.out_dir if args.out_dir is not None else args.features_csv.parent / "evaluation_report"
    run_evaluation_report(args.features_csv, out, config_path=args.config)


if __name__ == "__main__":
    main()
