"""Command-line entry point for evaluation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from common.paths import evaluation_root, exports_root
from evaluation.event_contrasts import run_event_contrast_evaluation
from evaluation.experiments import resolve_evaluation_models, run_evaluation
from evaluation.label_grid import DEFAULT_QUALITIES, all_label_cols, run_label_grid_evaluation
from evaluation.two_stage_events import run_two_stage_event_evaluation


def _build_parser() -> argparse.ArgumentParser:
    """Build parser."""
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
        help="Drop windows with scenario_label_safety=non_riding before evaluation.",
    )
    parser.add_argument(
        "--no-permutation",
        dest="permutation",
        action="store_false",
        default=True,
        help="Skip permutation importance (faster but loses the model-agnostic ranking).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--evaluation-models",
        nargs="+",
        default=["auto"],
        metavar="MODEL",
        help=(
            "Classifier(s) for CV: random_forest, hist_gradient_boosting, "
            "logistic_regression, or auto alone for all three (default: auto)."
        ),
    )
    parser.add_argument(
        "--permutation-models",
        nargs="+",
        default=["random_forest"],
        metavar="MODEL",
        help=(
            "Models for permutation importance (subset of evaluation models "
            "recommended); auto alone runs all three; none alone disables "
            "permutation importance (default: random_forest)."
        ),
    )
    parser.add_argument(
        "--event-contrast-models",
        nargs="+",
        default=["hist_gradient_boosting", "logistic_regression"],
        metavar="MODEL",
        help=(
            "Classifier(s) for event contrasts: random_forest, "
            "hist_gradient_boosting, logistic_regression, or auto alone "
            "(default: hist_gradient_boosting logistic_regression)."
        ),
    )
    parser.add_argument(
        "--two-stage-event-models",
        nargs="+",
        default=["hist_gradient_boosting", "logistic_regression"],
        metavar="MODEL",
        help=(
            "Classifier(s) for two-stage event evaluation: random_forest, "
            "hist_gradient_boosting, logistic_regression, or auto alone "
            "(default: hist_gradient_boosting logistic_regression)."
        ),
    )
    parser.add_argument(
        "--two-stage-tasks",
        nargs="+",
        default=["core"],
        metavar="TASK",
        help=(
            "Two-stage event task(s): core, all, or explicit task names "
            "(turning, deceleration, high_effort, posture)."
        ),
    )
    parser.add_argument(
        "--two-stage-target-recall",
        type=float,
        default=0.90,
        help="Detector target recall for threshold selection (default: 0.90).",
    )
    parser.add_argument(
        "--two-stage-hop-s",
        type=float,
        default=1.0,
        help="Window hop in seconds for candidate interval merging (default: 1.0).",
    )
    parser.add_argument(
        "--label-grid",
        action="store_true",
        default=False,
        dest="label_grid",
        help="Run evaluation across all label schemes × quality filters.",
    )
    parser.add_argument(
        "--event-contrasts",
        action="store_true",
        default=False,
        help="Run label-derived event contrast evaluation under output/event_contrasts.",
    )
    parser.add_argument(
        "--two-stage-events",
        action="store_true",
        default=False,
        help="Run hierarchical detector-plus-contrast evaluation under output/two_stage_events.",
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
    """Run the command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        eval_models = resolve_evaluation_models(args.evaluation_models)
        perm_models = resolve_evaluation_models(args.permutation_models)
        event_models = resolve_evaluation_models(args.event_contrast_models)
        two_stage_models = resolve_evaluation_models(args.two_stage_event_models)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if not eval_models:
        print("Error: --evaluation-models cannot be none", file=sys.stderr)
        return 1
    if not event_models:
        print("Error: --event-contrast-models cannot be none", file=sys.stderr)
        return 1
    if not two_stage_models:
        print("Error: --two-stage-event-models cannot be none", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.event_contrasts:
            summary = run_event_contrast_evaluation(
                features_path=Path(args.features),
                output_dir=Path(args.output),
                models=event_models,
                min_quality=args.min_quality,
                seed=args.seed,
                no_plots=args.no_plots,
            )
            print("Event contrast evaluation complete.")
            print(f"  Metric rows       : {summary['n_metric_rows']}")
            print(f"  Output directory  : {summary['output_dir']}")
            if summary.get("skipped"):
                print(f"  Skipped entries   : {len(summary['skipped'])}")
            return 0

        if args.two_stage_events:
            summary = run_two_stage_event_evaluation(
                features_path=Path(args.features),
                output_dir=Path(args.output),
                task_names=args.two_stage_tasks,
                models=two_stage_models,
                min_quality=args.min_quality,
                seed=args.seed,
                target_recall=args.two_stage_target_recall,
                hop_s=args.two_stage_hop_s,
                no_plots=args.no_plots,
            )
            print("Two-stage event evaluation complete.")
            print(f"  Metric rows       : {summary['n_metric_rows']}")
            print(f"  Output directory  : {summary['output_dir']}")
            if summary.get("skipped"):
                print(f"  Skipped entries   : {len(summary['skipped'])}")
            return 0

        if args.label_grid:
            label_cols = (
                list(all_label_cols())
                if args.label_col == "auto"
                else [args.label_col]
            )
            grid_summary = run_label_grid_evaluation(
                features_path=Path(args.features),
                output_dir=Path(args.output) / "label_grid",
                label_cols=label_cols,
                qualities=args.qualities,
                primary_quality=args.primary_quality,
                seed=args.seed,
                exclude_non_riding=args.exclude_non_riding,
                no_plots=args.no_plots,
                evaluation_models=eval_models,
                permutation_models=perm_models,
            )
            print(
                f"Label-grid complete. {grid_summary['n_runs_ok']}/{grid_summary['n_runs']} "
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
            no_plots=args.no_plots,
            compute_permutation_importance=args.permutation,
            evaluation_models=eval_models,
            permutation_models=perm_models,
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
