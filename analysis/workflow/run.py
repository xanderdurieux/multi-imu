"""Main reproducible workflow entry point.

Usage:
    uv run python -m workflow --config configs/workflow.thesis.json
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from pipeline.run import run_pipeline

from .config import WorkflowConfig, load_config, merge_cli_overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m workflow",
        description="Run the thesis workflow using a single reproducible config file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/workflow.thesis.json"),
        help="Path to workflow JSON config (default: configs/workflow.thesis.json).",
    )
    parser.add_argument("--session", default=None, help="Optional session override.")
    parser.add_argument(
        "--recording",
        action="append",
        dest="recordings",
        metavar="ID",
        help="Optional recording override (repeatable).",
    )
    parser.add_argument("--data-root", default=None, help="Optional data root override.")
    parser.add_argument(
        "--sync-method",
        choices=("best", "sda", "lida", "calibration", "online"),
        default=None,
        help="Optional synchronization override.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Disable plots for this run.")
    parser.add_argument("--force", action="store_true", help="Force re-run for this run.")
    return parser


def _resolve_paths(cfg: WorkflowConfig, config_path: Path) -> WorkflowConfig:
    base = config_path.parent

    def _maybe_resolve(value: str | None) -> str | None:
        if value is None:
            return None
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((base / p).resolve())

    return WorkflowConfig(
        **{
            **cfg.to_dict(),
            "data_root": _maybe_resolve(cfg.data_root) or cfg.data_root,
            "labels_path": _maybe_resolve(cfg.labels_path),
            "event_config_path": _maybe_resolve(cfg.event_config_path),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(
        cfg,
        session=args.session,
        recordings=args.recordings,
        data_root=args.data_root,
        sync_method=args.sync_method,
        no_plots=True if args.no_plots else None,
        force=True if args.force else None,
    )
    cfg = _resolve_paths(cfg, args.config.resolve())

    os.environ["MULTI_IMU_DATA_ROOT"] = cfg.data_root

    run_pipeline(
        session=cfg.session,
        only_recordings=frozenset(cfg.recordings) if cfg.recordings else None,
        force=cfg.force,
        sync_mode=cfg.sync_method,
        split_stage=cfg.split_stage,
        no_plots=cfg.no_plots,
        orientation_filter=cfg.orientation_filter,
        labels_path=Path(cfg.labels_path) if cfg.labels_path else None,
        frame_alignment=cfg.frame_alignment,
        skip_exports=cfg.skip_exports,
        event_config_path=Path(cfg.event_config_path) if cfg.event_config_path else None,
        event_centered_features=cfg.event_centered_features,
        min_event_confidence=cfg.min_event_confidence,
    )


if __name__ == "__main__":
    main()
