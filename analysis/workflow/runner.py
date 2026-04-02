"""Pipeline runner: execute stages in order from a WorkflowConfig."""

from __future__ import annotations

import json
import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.paths import (
    analysis_root,
    data_root,
    evaluation_root,
    exports_root,
    iter_sections_for_recording,
    project_relative_path,
    recording_stage_dir,
    recordings_root,
)
from .config import WorkflowConfig

_ALL_ORIENTATION_FILTERS = ["madgwick", "complementary"]

log = logging.getLogger(__name__)

def _collect_recordings(cfg: WorkflowConfig) -> list[str]:
    """Resolve the list of recording names to process."""
    explicit = list(cfg.recordings)
    if explicit:
        return explicit

    root = recordings_root()
    if not root.exists():
        return []

    recordings: list[str] = []
    for session in cfg.sessions:
        prefix = f"{session}_r"
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith(prefix):
                recordings.append(d.name)

    if not recordings and not cfg.sessions:
        # Default: all recordings.
        recordings = sorted(d.name for d in root.iterdir() if d.is_dir())

    return recordings


def _collect_sections(recordings: list[str]) -> list[Path]:
    """Return all section dirs for the given recordings."""
    sections: list[Path] = []
    for rec in recordings:
        sections.extend(iter_sections_for_recording(rec))
    return sections


def _run_recording_stage_plots(recording_name: str, stage: str) -> None:
    """Generate recording-level plots for a given stage output."""
    if stage == "synced":
        from visualization.plot_sync import plot_sync_stage
        plot_sync_stage(recording_stage_dir(recording_name, stage))
    elif stage == "parsed":
        from visualization.plot_comparison import plot_stage_data
        plot_stage_data(recording_stage_dir(recording_name, stage))


def _run_section_stage_plots(section_dir: Path, stage: str) -> None:
    """Generate section-level plots for a given stage output."""
    if stage == "calibration":
        from visualization.plot_calibration import plot_calibration_stage
        plot_calibration_stage(section_dir)
    elif stage == "orientation":
        from visualization.plot_orientation import plot_orientation_stage
        plot_orientation_stage(section_dir)
    elif stage == "derived":
        from visualization.plot_derived import plot_derived_stage
        plot_derived_stage(section_dir)
    elif stage == "features":
        from visualization.plot_features import plot_features_stage
        from visualization.plot_labels import plot_labels
        plot_features_stage(section_dir)
        plot_labels(section_dir, stage="sensor")
        plot_labels(section_dir, stage="features")


def _run_stage(stage: str, cfg: WorkflowConfig, recordings: list[str]) -> dict[str, Any]:
    """Run a single pipeline stage for all recordings/sections."""
    result: dict[str, Any] = {"stage": stage, "ok": 0, "failed": 0, "skipped": 0}

    if stage == "parse":
        from parser.session import process_session
        sessions = cfg.sessions or []
        for session in sessions:
            try:
                process_session(session, plot=not cfg.no_plots)
                result["ok"] += 1
            except Exception as exc:
                log.error("parse failed for %s: %s", session, exc)
                result["failed"] += 1

    elif stage == "sync":
        from sync.pipeline import synchronize_recording_all_methods, synchronize_recording_chosen_method
        for rec in recordings:
            try:
                if cfg.sync_method == "auto":
                    synchronize_recording_all_methods(rec)
                else:
                    synchronize_recording_chosen_method(rec, cfg.sync_method)
                result["ok"] += 1
                if not cfg.no_plots:
                    try:
                        _run_recording_stage_plots(rec, "synced")
                    except Exception as exc:
                        log.warning("sync plots failed for %s: %s", rec, exc)
            except Exception as exc:
                log.error("sync failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "split":
        from parser.split_sections import split_recording
        for rec in recordings:
            split_input_dir = recording_stage_dir(rec, cfg.split_stage)
            if not split_input_dir.is_dir():
                log.warning(
                    "split skipped for %s: missing input stage directory %s",
                    rec,
                    project_relative_path(split_input_dir),
                )
                result["skipped"] += 1
                continue
            try:
                split_recording(
                    rec,
                    stage=cfg.split_stage,
                    plot=not cfg.no_plots,
                )
                result["ok"] += 1
            except Exception as exc:
                log.error("split failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "calibration":
        from calibration.pipeline import calibrate_recording_sections
        for rec in recordings:
            try:
                cals = calibrate_recording_sections(
                    rec,
                    sample_rate_hz=cfg.sample_rate_hz,
                    force=cfg.force,
                )
                result["ok"] += len(cals)
                if not cfg.no_plots:
                    for section_dir in iter_sections_for_recording(rec):
                        try:
                            _run_section_stage_plots(section_dir, "calibration")
                        except Exception as exc:
                            log.warning("calibration plots failed for %s: %s", section_dir.name, exc)
            except Exception as exc:
                log.error("calibration failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "orientation":
        from orientation.pipeline import process_recording_orientation
        canonical = _ALL_ORIENTATION_FILTERS[0] if cfg.orientation_filter == "auto" else cfg.orientation_filter
        variants = _ALL_ORIENTATION_FILTERS if cfg.orientation_filter == "auto" else [cfg.orientation_filter]
        for rec in recordings:
            try:
                results = process_recording_orientation(
                    rec,
                    canonical_variant=canonical,
                    variants=variants,
                    sample_rate_hz=cfg.sample_rate_hz,
                    force=cfg.force,
                )
                result["ok"] += len(results)
                if not cfg.no_plots:
                    for section_dir in iter_sections_for_recording(rec):
                        try:
                            _run_section_stage_plots(section_dir, "orientation")
                        except Exception as exc:
                            log.warning("orientation plots failed for %s: %s", section_dir.name, exc)
            except Exception as exc:
                log.error("orientation failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "derived":
        from derived.pipeline import process_recording_derived
        for rec in recordings:
            try:
                ok_list = process_recording_derived(
                    rec,
                    sample_rate_hz=cfg.sample_rate_hz,
                    force=cfg.force,
                )
                result["ok"] += sum(ok_list)
                result["failed"] += sum(not x for x in ok_list)
                if not cfg.no_plots:
                    for section_dir in iter_sections_for_recording(rec):
                        try:
                            _run_section_stage_plots(section_dir, "derived")
                        except Exception as exc:
                            log.warning("derived plots failed for %s: %s", section_dir.name, exc)
            except Exception as exc:
                log.error("derived failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "events":
        from events.pipeline import process_recording_events
        from events.config import EventConfig
        event_cfg = None
        if cfg.event_config_path:
            p = Path(cfg.event_config_path)
            if p.exists():
                event_cfg = EventConfig.load(p)
        for rec in recordings:
            try:
                process_recording_events(rec, config=event_cfg, force=cfg.force)
                result["ok"] += 1
            except Exception as exc:
                log.error("events failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "features":
        from features.pipeline import process_recording_features
        for rec in recordings:
            try:
                df = process_recording_features(
                    rec,
                    window_s=cfg.window_s,
                    hop_s=cfg.hop_s,
                    force=cfg.force,
                )
                result["ok"] += 1
                if not cfg.no_plots:
                    for section_dir in iter_sections_for_recording(rec):
                        try:
                            _run_section_stage_plots(section_dir, "features")
                        except Exception as exc:
                            log.warning("features plots failed for %s: %s", section_dir.name, exc)
            except Exception as exc:
                log.error("features failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "exports" and not cfg.skip_exports:
        from exports.pipeline import run_exports
        try:
            run_exports(
                recordings,
                min_quality_label=cfg.min_quality_label,
                force=cfg.force,
                no_plots=cfg.no_plots,
            )
            result["ok"] += 1
        except Exception as exc:
            log.error("exports failed: %s", exc)
            result["failed"] += 1

    elif stage == "evaluation":
        from evaluation.experiments import run_evaluation
        fused = exports_root() / "features_fused.csv"
        out = evaluation_root()
        if fused.exists():
            try:
                run_evaluation(
                    fused,
                    output_dir=out,
                    seed=cfg.evaluation_seed,
                    min_quality=cfg.min_quality_label,
                    no_plots=cfg.no_plots,
                )
                result["ok"] += 1
            except Exception as exc:
                log.error("evaluation failed: %s", exc)
                result["failed"] += 1
        else:
            log.warning("features_fused.csv not found — skipping evaluation")
            result["skipped"] += 1

    elif stage == "reporting":
        from reporting.bundle import generate_report_bundle
        out = data_root() / "thesis_report_bundle"
        try:
            generate_report_bundle(out, force=cfg.force)
            result["ok"] += 1
        except Exception as exc:
            log.error("reporting failed: %s", exc)
            result["failed"] += 1

    return result


def run_pipeline(cfg: WorkflowConfig) -> dict[str, Any]:
    """Execute the full pipeline from a WorkflowConfig.

    Returns a summary dict with per-stage results.
    """
    stages = list(cfg.stages)
    recordings = _collect_recordings(cfg)
    log.info("Pipeline: %d recording(s), stages=%s", len(recordings), stages)

    summary: dict[str, Any] = {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "recordings": recordings,
        "stages": stages,
        "stage_results": {},
    }

    for stage in stages:
        log.info("-------------------------------- Stage: %s --------------------------------", stage)
        
        try:
            recordings = _collect_recordings(cfg)
            r = _run_stage(stage, cfg, recordings)
        except Exception as exc:
            log.error("Stage %s crashed: %s\n%s", stage, exc, traceback.format_exc())
            r = {"stage": stage, "ok": 0, "failed": 1, "error": str(exc)}
        summary["stage_results"][stage] = r
        log.info("Stage %s: ok=%d failed=%d", stage, r.get("ok", 0), r.get("failed", 0))

    summary["recordings"] = recordings
    summary["finished_at_utc"] = datetime.now(UTC).isoformat()

    # Write run summary.
    out_path = data_root() / "pipeline_run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    log.info("Wrote pipeline run summary: %s", project_relative_path(out_path))

    return summary
