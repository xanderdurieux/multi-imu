"""Run evaluation report: ``uv run python -m evaluation <features_fused.csv> [out_dir] [--config cfg.json]``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .experiments import run_evaluation_report, run_primary_thesis_bundle


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed override (falls back to config/default).",
    )
    parser.add_argument(
        "--primary",
        action="store_true",
        help="Run the thesis primary bundle (main comparison + compact supporting ablations).",
    )
    args = parser.parse_args()

    out = args.out_dir if args.out_dir is not None else args.features_csv.parent / "evaluation_report"
    env_seed = os.environ.get("MULTI_IMU_EVALUATION_SEED")
    effective_seed = args.seed if args.seed is not None else (int(env_seed) if env_seed else 42)
    if args.primary:
        run_primary_thesis_bundle(args.features_csv, out, config_path=args.config, random_state=effective_seed)
    else:
        run_evaluation_report(args.features_csv, out, config_path=args.config, random_state=effective_seed)


if __name__ == "__main__":
    main()
