"""Workflow entrypoint: run the full pipeline from a config file.

Usage::

    python -m workflow configs/workflow.thesis.json
    python -m workflow configs/workflow.thesis.json --force
    python -m workflow configs/workflow.thesis.json --stage calibration orientation
    python -m workflow configs/workflow.thesis.json --list-stages
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from common.paths import (
    data_root,
    default_workflow_config_path,
    project_relative_path,
    write_json_file,
)
from .config import (
    WorkflowConfig,
    known_stages,
    load_workflow_config,
)
from .runner import run_pipeline


def _configure_logging(log_to_file: bool) -> Path | None:
    """Log to terminal, and optionally to a per-run workflow log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_path: Path | None = None
    if log_to_file:
        logs_dir = data_root() / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return log_path


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])

    stage_names = known_stages()
    parser = argparse.ArgumentParser(
        prog="python -m workflow",
        description="Run the dual-IMU cycling pipeline end-to-end from a config file.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        help=(
            "Path to workflow config JSON override "
            f"(merged over default: {project_relative_path(default_workflow_config_path())})."
        ),
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        metavar="STAGE",
        help=f"Run only these stages. Available: {', '.join(stage_names)}",
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
        for s in stage_names:
            print(f"  {s}")
        return

    if args.generate_config:
        out = Path(args.generate_config)
        cfg = WorkflowConfig()
        write_json_file(out, cfg.to_dict(), indent=2)
        print(f"Config template written → {project_relative_path(out)}")
        return

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

    log_path = _configure_logging(cfg.log_to_file)

    print(f"\n{'━' * 60}")
    print(f"  Dual-IMU Cycling Pipeline")
    cfg_display = (
        project_relative_path(args.config)
        if args.config
        else project_relative_path(default_workflow_config_path())
    )
    print(f"  Config: {cfg_display}")
    print(f"  Stages: {cfg.stages}")
    if log_path is not None:
        print(f"  Log file: {project_relative_path(log_path)}")
    else:
        print("  Log file: disabled")
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
