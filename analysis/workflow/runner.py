"""Pipeline runner: execute stages in order from a WorkflowConfig."""

from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.paths import (
    data_root,
    evaluation_root,
    exports_root,
    iter_sections_for_recording,
    project_relative_path,
    recording_stage_dir,
    recordings_root,
    write_json_file,
)
from orientation.pipeline import (
    DEFAULT_CANONICAL_ORIENTATION_METHOD,
    DEFAULT_ORIENTATION_VARIANTS,
)
from visualization.stage_plots import (
    plot_recording_pipeline_stage,
    plot_section_pipeline_stage,
)

from .config import WorkflowConfig

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
                    synced_dir = recording_stage_dir(rec, "synced")
                    if not synced_dir.is_dir():
                        # When no sync method succeeded we won't have a synced/ dir.
                        # Don't treat that as a plotting failure.
                        log.info("sync plots skipped for %s: missing %s", rec, project_relative_path(synced_dir))
                    else:
                        try:
                            plot_recording_pipeline_stage(rec, "synced")
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
                            plot_section_pipeline_stage(section_dir, "calibration")
                        except Exception as exc:
                            log.warning("calibration plots failed for %s: %s", section_dir.name, exc)
            except Exception as exc:
                log.error("calibration failed for %s: %s", rec, exc)
                result["failed"] += 1

    elif stage == "orientation":
        from orientation.pipeline import process_recording_orientation
        canonical = (
            DEFAULT_CANONICAL_ORIENTATION_METHOD
            if cfg.orientation_filter == "auto"
            else cfg.orientation_filter
        )
        variants = (
            list(DEFAULT_ORIENTATION_VARIANTS)
            if cfg.orientation_filter == "auto"
            else [cfg.orientation_filter]
        )
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
                            plot_section_pipeline_stage(section_dir, "orientation")
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
                            plot_section_pipeline_stage(section_dir, "derived")
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
                            plot_section_pipeline_stage(section_dir, "features")
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

    elif stage == "dataset_summary":
        # Reads: data/exports/features_fused.csv
        # Writes: data/report/tables/ and data/report/figures/dataset/
        # Position: after "exports", before "evaluation".  Safe to re-run at
        # any time — all outputs are deterministically derived from the CSV.
        from reporting.dataset_summary import run_dataset_summary
        fused = exports_root() / "features_fused.csv"
        out = data_root() / "report"
        if fused.exists():
            try:
                run_dataset_summary(
                    fused,
                    output_dir=out,
                    min_quality=cfg.min_quality_label,
                    no_plots=cfg.no_plots,
                )
                result["ok"] += 1
            except Exception as exc:
                log.error("dataset_summary failed: %s", exc)
                result["failed"] += 1
        else:
            log.warning("features_fused.csv not found — skipping dataset_summary")
            result["skipped"] += 1

    elif stage == "evaluation":
        from evaluation.experiments import run_evaluation
        fused = exports_root() / "features_fused.csv"
        out = evaluation_root()
        if fused.exists():
            try:
                run_evaluation(
                    fused,
                    output_dir=out,
                    label_col=cfg.evaluation_label_col,
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

    elif stage == "report":
        from reporting.pipeline import run_report
        fused = exports_root() / "features_fused.csv"
        if fused.exists():
            try:
                run_report(
                    output_dir=data_root() / "report",
                    features_path=fused,
                    evaluation_dir=evaluation_root(),
                    no_plots=cfg.no_plots,
                )
                result["ok"] += 1
            except Exception as exc:
                log.error("report failed: %s", exc)
                result["failed"] += 1
        else:
            log.warning("features_fused.csv not found — skipping report")
            result["skipped"] += 1

    elif stage == "thesis_bundle":
        from reporting.thesis_bundle import run_thesis_bundle
        fused = exports_root() / "features_fused.csv"
        if fused.exists():
            try:
                run_thesis_bundle(
                    output_dir=data_root() / "report" / "thesis_bundle",
                    features_path=fused,
                    evaluation_dir=evaluation_root(),
                    exports_dir=exports_root(),
                    report_dir=data_root() / "report",
                )
                result["ok"] += 1
            except Exception as exc:
                log.error("thesis_bundle failed: %s", exc)
                result["failed"] += 1
        else:
            log.warning("features_fused.csv not found — skipping thesis_bundle")
            result["skipped"] += 1

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
    write_json_file(out_path, summary, indent=2, default=str)
    log.info("Wrote pipeline run summary: %s", project_relative_path(out_path))

    return summary
