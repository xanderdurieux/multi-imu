"""Main reproducible workflow entry point.

Usage:
    uv run python -m workflow --config configs/workflow.thesis.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--log-format",
        choices=("human", "json"),
        default="human",
        help="Logging format. Default is human-readable.",
    )
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


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for field in ("run_id", "recording_id", "section_id", "step", "status", "reason"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging(log_format: str) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    handler = logging.StreamHandler()
    if log_format == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(handler)


def _git_commit_sha(cwd: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_provenance_manifest(
    *,
    analysis_root: Path,
    config_path: Path,
    cfg: WorkflowConfig,
    run_id: str,
    selected_recordings: list[str],
    output_root: Path,
) -> Path:
    started_at = os.environ.get("MULTI_IMU_WORKFLOW_STARTED_AT", "")
    ended_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest_dir = output_root / "provenance"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"workflow_run_{run_id}.json"
    payload = {
        "run_id": run_id,
        "started_at_utc": started_at,
        "finished_at_utc": ended_at,
        "analysis_root": str(analysis_root),
        "git_commit_sha": _git_commit_sha(analysis_root),
        "config_path": str(config_path),
        "config_values": cfg.to_dict(),
        "dataset_root": cfg.data_root,
        "selected_session": cfg.session,
        "selected_recordings": selected_recordings,
        "seeds": {
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "evaluation_seed": cfg.evaluation_seed,
            "numpy_random_seed": cfg.evaluation_seed,
        },
        "output_directories": {
            "recordings_root": str(output_root / "recordings"),
            "sections_root": str(output_root / "sections"),
            "exports_root": str(output_root / "exports"),
            "run_summary_json": str(output_root / "pipeline_run_summary.json"),
            "run_summary_sections_csv": str(output_root / "pipeline_run_summary_sections.csv"),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest = manifest_dir / "workflow_run_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_format)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["MULTI_IMU_RUN_ID"] = run_id
    os.environ["MULTI_IMU_WORKFLOW_STARTED_AT"] = dt.datetime.now(dt.timezone.utc).isoformat()
    logging.info("Workflow run started (run_id=%s)", run_id)

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
    os.environ["MULTI_IMU_EVALUATION_SEED"] = str(cfg.evaluation_seed)

    statuses = run_pipeline(
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
    selected_recordings = [s.recording_id for s in statuses]
    output_root = Path(cfg.data_root).resolve()
    manifest_path = _write_provenance_manifest(
        analysis_root=Path(__file__).resolve().parents[1],
        config_path=args.config.resolve(),
        cfg=cfg,
        run_id=run_id,
        selected_recordings=selected_recordings,
        output_root=output_root,
    )
    logging.info("Workflow run finished (run_id=%s, recordings=%d)", run_id, len(selected_recordings))
    logging.info("Wrote provenance manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
