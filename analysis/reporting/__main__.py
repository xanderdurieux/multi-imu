"""Command-line entry point for reporting."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    """Return setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m reporting",
        description="Generate thesis reporting figures, tables, and summary.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        help="Root output directory (default: data/report/)",
    )
    parser.add_argument(
        "--features",
        metavar="CSV",
        help="Path to features_fused.csv (default: data/exports/features_fused.csv)",
    )
    parser.add_argument(
        "--evaluation",
        metavar="DIR",
        help="Evaluation output directory (default: data/evaluation/)",
    )
    parser.add_argument(
        "--context",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Context seconds shown on each side of the feature window (default: 5.0)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        metavar="LABEL",
        help="Scenario labels to generate signal examples for (default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    from reporting.pipeline import run_report

    kwargs: dict = {"context_s": args.context}
    if args.output:
        kwargs["output_dir"] = Path(args.output)
    if args.features:
        kwargs["features_path"] = Path(args.features)
    if args.evaluation:
        kwargs["evaluation_dir"] = Path(args.evaluation)
    if args.scenarios:
        kwargs["scenarios"] = args.scenarios

    try:
        summary = run_report(**kwargs)
        print(f"\nReport generated successfully:")
        print(f"  Output directory : {summary['output_dir']}")
        print(f"  Windows          : {summary['n_windows']}")
        print(f"  Classes          : {summary['n_classes']}")
        print(f"  Signal plots     : {summary['n_signal_plots']}")
        for stage, n in summary.get("n_stage_figures", {}).items():
            print(f"  {stage.capitalize()} figures : {n}")
        print(f"  Evaluation figs  : {summary['n_eval_figures']}")
        return 0
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logging.getLogger(__name__).exception("Report generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
