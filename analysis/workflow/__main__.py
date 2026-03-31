"""Workflow entrypoint: run the full pipeline from a config file.

Usage::

    python -m workflow configs/workflow.thesis.json
    python -m workflow configs/workflow.thesis.json --force
    python -m workflow configs/workflow.thesis.json --stage calibration orientation
    python -m workflow configs/workflow.thesis.json --list-stages
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import load_workflow_config, WorkflowConfig
from .runner import ALL_STAGES, run_pipeline


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    argv = list(argv if argv is not None else sys.argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m workflow",
        description="Run the dual-IMU cycling pipeline end-to-end from a config file.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to workflow config JSON (e.g. configs/workflow.thesis.json).",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        metavar="STAGE",
        help=f"Run only these stages. Available: {', '.join(ALL_STAGES)}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing stage outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="Print available stages and exit.",
    )
    parser.add_argument(
        "--generate-config",
        metavar="PATH",
        help="Write a default config template to PATH and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_stages:
        print("Available pipeline stages:")
        for s in ALL_STAGES:
            print(f"  {s}")
        return

    if args.generate_config:
        out = Path(args.generate_config)
        out.parent.mkdir(parents=True, exist_ok=True)
        cfg = WorkflowConfig()
        out.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        print(f"Config template written → {out}")
        return

    if not args.config:
        parser.print_help()
        sys.exit(1)

    try:
        cfg = load_workflow_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading config: {exc}", file=sys.stderr)
        sys.exit(1)

    # CLI overrides.
    if args.force:
        cfg.force = True
    if args.no_plots:
        cfg.no_plots = True
    if args.stage:
        cfg.stages = list(args.stage)

    print(f"\n{'━' * 60}")
    print(f"  Dual-IMU Cycling Pipeline")
    print(f"  Config: {args.config}")
    print(f"  Stages: {cfg.stages or ALL_STAGES}")
    print(f"{'━' * 60}\n")

    summary = run_pipeline(cfg)

    print(f"\n{'━' * 60}")
    print("  Stage summary:")
    for stage, r in summary.get("stage_results", {}).items():
        ok = r.get("ok", 0)
        failed = r.get("failed", 0)
        skipped = r.get("skipped", 0)
        status = "✓" if failed == 0 else "✗"
        print(f"  {status} {stage:<14}  ok={ok}  failed={failed}  skipped={skipped}")
    print(f"{'━' * 60}\n")


if __name__ == "__main__":
    main()
