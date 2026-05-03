"""CLI entry point for the evaluation package.

Two modes:

* **Single run** (default) — one ``label_col`` × one ``min_quality``::

    python -m evaluation [--features data/exports/features_fused.csv]
                         [--output outputs/evaluation]
                         [--seed 42]
                         [--quality marginal]
                         [--label scenario_label_activity]
                         [--group section_id]
                         [--exclude-non-riding]
                         [--no-permutation]

* **Sweep** — cross-product of label schemes × quality filters::

    python -m evaluation --sweep [--qualities marginal good]
                                 [--label auto|<col>]
                                 [--primary-quality marginal]
                                 [--no-permutation]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.paths import evaluation_root, exports_root
from evaluation.experiments import run_evaluation
from evaluation.sweep import DEFAULT_LABEL_COLS, DEFAULT_QUALITIES, run_evaluation_sweep


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Train and evaluate classification models on feature tables.",
    )
    parser.add_argument(
        "--features",
        metavar="PATH",
        default=str(exports_root() / "features_fused.csv"),
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
        help="Minimum quality label filter for single-run mode (default: marginal).",
    )
    parser.add_argument(
        "--label",
        metavar="COL",
        default="scenario_label_activity",
        dest="label_col",
        help=(
            "Target label column. In sweep mode, pass 'auto' to evaluate every "
            "label scheme (default: scenario_label_activity)."
        ),
    )
    parser.add_argument(
        "--group",
        metavar="COL",
        default="section_id",
        dest="group_col",
        help="Group column for CV splitting (default: section_id).",
    )
    parser.add_argument(
        "--exclude-non-riding",
        action="store_true",
        default=False,
        help="Drop windows with scenario_label_binary=non_riding before evaluation.",
    )
    parser.add_argument(
        "--no-permutation",
        dest="permutation",
        action="store_false",
        default=True,
        help="Skip permutation importance (faster but loses the model-agnostic ranking).",
    )
    parser.add_argument(
        "--permutation-models",
        nargs="+",
        default=["random_forest"],
        choices=["random_forest", "hist_gradient_boosting", "logistic_regression"],
        help="Models to compute permutation importance for (default: random_forest).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run a sweep across the label_col × min_quality grid.",
    )
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=list(DEFAULT_QUALITIES),
        choices=["poor", "marginal", "good"],
        help=f"Quality filters to sweep (default: {list(DEFAULT_QUALITIES)}).",
    )
    parser.add_argument(
        "--primary-quality",
        default="marginal",
        choices=["poor", "marginal", "good"],
        help="Quality filter to compute permutation importance for (default: marginal).",
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
        if args.sweep:
            label_cols = (
                list(DEFAULT_LABEL_COLS)
                if args.label_col == "auto"
                else [args.label_col]
            )
            sweep_summary = run_evaluation_sweep(
                features_path=Path(args.features),
                output_dir=Path(args.output),
                label_cols=label_cols,
                qualities=args.qualities,
                primary_quality=args.primary_quality,
                seed=args.seed,
                exclude_non_riding=args.exclude_non_riding,
                no_plots=False,
                permutation_models=tuple(args.permutation_models),
            )
            print(
                f"Sweep complete. {sweep_summary['n_runs_ok']}/{sweep_summary['n_runs']} "
                "run(s) succeeded."
            )
            return 0

        summary = run_evaluation(
            features_path=Path(args.features),
            output_dir=Path(args.output),
            label_col=args.label_col,
            group_col=args.group_col,
            seed=args.seed,
            min_quality=args.min_quality,
            exclude_non_riding=args.exclude_non_riding,
            compute_permutation_importance=args.permutation,
            permutation_models=tuple(args.permutation_models),
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Evaluation complete.")
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
