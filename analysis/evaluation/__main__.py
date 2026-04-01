"""CLI entry point for the evaluation package.

Usage::

    python -m evaluation [--features data/exports/features_fused.csv]
                         [--output outputs/evaluation]
                         [--seed 42]
                         [--quality marginal]
                         [--label scenario_label]
                         [--group section_id]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.paths import evaluation_root, fused_features_csv
from evaluation.experiments import run_evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Train and evaluate classification models on feature tables.",
    )
    parser.add_argument(
        "--features",
        metavar="PATH",
        default=str(fused_features_csv()),
        help="Path to feature CSV (default: data/exports/features_fused.csv).",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        default=str(evaluation_root()),
        help="Output directory (default: data/evaluation/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--quality",
        metavar="LABEL",
        default="marginal",
        dest="min_quality",
        choices=["poor", "marginal", "good"],
        help="Minimum quality label filter (default: marginal).",
    )
    parser.add_argument(
        "--label",
        metavar="COL",
        default="scenario_label",
        dest="label_col",
        help="Target label column (default: scenario_label).",
    )
    parser.add_argument(
        "--group",
        metavar="COL",
        default="section_id",
        dest="group_col",
        help="Group column for CV splitting (default: section_id).",
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
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        summary = run_evaluation(
            features_path=Path(args.features),
            output_dir=Path(args.output),
            label_col=args.label_col,
            group_col=args.group_col,
            seed=args.seed,
            min_quality=args.min_quality,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Evaluation complete.")
    print(f"  Windows evaluated : {summary['n_windows']}")
    print(f"  Classes           : {', '.join(summary['classes'])}")
    if summary.get("results"):
        best = max(summary["results"], key=lambda k: summary["results"][k]["accuracy"])
        r = summary["results"][best]
        print(
            f"  Best result       : {best}  "
            f"acc={r['accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
