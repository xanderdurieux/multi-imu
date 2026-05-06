"""Pipeline runner: execute stages in order from a WorkflowConfig."""

from __future__ import annotations

import logging
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from common.paths import (
    data_root,
    evaluation_root,
    exports_root,
    iter_sections_for_recording,
    project_relative_path,
    read_csv,
    recording_sort_key,
    recording_stage_dir,
    recordings_root,
    write_csv,
    write_json_file,
)
from visualization.stage_plots import (
    plot_recording_pipeline_stage,
    plot_section_pipeline_stage,
)

from .config import WorkflowConfig

log = logging.getLogger(__name__)

from evaluation.label_grid import resolve_label_cols

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
        for d in sorted(root.iterdir(), key=recording_sort_key):
            if d.is_dir() and d.name.startswith(prefix):
                recordings.append(d.name)

    if not recordings and not cfg.sessions:
        # Default: all recordings.
        recordings = sorted((d.name for d in root.iterdir() if d.is_dir()), key=recording_sort_key)

    return recordings


def _collect_sections(recordings: list[str]) -> list[Path]:
    """Return all section dirs for the given recordings."""
    sections: list[Path] = []
    for rec in recordings:
        sections.extend(iter_sections_for_recording(rec))
    return sections


def _recording_from_section_id(section_id: str) -> str:
    """Return recording name from a section id such as ``2026-02-26_r1s2``."""
    name = str(section_id)
    if "s" not in name:
        return name
    return name.rsplit("s", 1)[0]


def _session_from_recording(recording_name: str) -> str:
    """Return session name from a recording name such as ``2026-02-26_r1``."""
    name = str(recording_name)
    return name.rsplit("_", 1)[0] if "_" in name else name


def _evaluation_scope_configured(cfg: WorkflowConfig) -> bool:
    """Return whether evaluation should use a scoped feature-table copy."""
    return any(
        (
            cfg.evaluation_sessions,
            cfg.evaluation_exclude_sessions,
            cfg.evaluation_recordings,
            cfg.evaluation_exclude_recordings,
        )
    )


def _feature_recording_series(df: pd.DataFrame) -> pd.Series:
    """Return per-row recording names from a feature table."""
    if "recording_name" in df.columns:
        return df["recording_name"].astype(str)
    if "section_id" in df.columns:
        return df["section_id"].astype(str).map(_recording_from_section_id)
    raise ValueError("features table must contain recording_name or section_id")


def _feature_session_series(df: pd.DataFrame, recordings: pd.Series) -> pd.Series:
    """Return per-row session names from a feature table."""
    if "session" in df.columns:
        return df["session"].astype(str)
    return recordings.map(_session_from_recording)


def _prepare_evaluation_features(
    fused: Path,
    cfg: WorkflowConfig,
    *,
    method: str,
    output_dir: Path,
) -> Path | None:
    """Return the feature CSV to evaluate, preserving the full exported feature set."""
    if not _evaluation_scope_configured(cfg):
        return fused

    df = read_csv(fused)
    if df.empty:
        log.warning("features_fused.csv is empty — skipping evaluation")
        return None

    recordings = _feature_recording_series(df)
    sessions = _feature_session_series(df, recordings)

    include_mask = pd.Series(True, index=df.index)
    include_parts: list[pd.Series] = []
    if cfg.evaluation_sessions:
        include_sessions = {str(s) for s in cfg.evaluation_sessions}
        include_parts.append(sessions.isin(include_sessions))
    if cfg.evaluation_recordings:
        include_recordings = {str(r) for r in cfg.evaluation_recordings}
        include_parts.append(recordings.isin(include_recordings))
    if include_parts:
        include_mask = include_parts[0].copy()
        for part in include_parts[1:]:
            include_mask |= part

    exclude_mask = pd.Series(False, index=df.index)
    if cfg.evaluation_exclude_sessions:
        exclude_sessions = {str(s) for s in cfg.evaluation_exclude_sessions}
        exclude_mask |= sessions.isin(exclude_sessions)
    if cfg.evaluation_exclude_recordings:
        exclude_recordings = {str(r) for r in cfg.evaluation_exclude_recordings}
        exclude_mask |= recordings.isin(exclude_recordings)

    scoped = df.loc[include_mask & ~exclude_mask].copy()
    scoped_recordings = _feature_recording_series(scoped) if not scoped.empty else pd.Series(dtype=str)
    scoped_sessions = (
        _feature_session_series(scoped, scoped_recordings)
        if not scoped.empty
        else pd.Series(dtype=str)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "features_scoped.csv"
    metadata_path = output_dir / "features_scoped_metadata.json"

    metadata = {
        "source": project_relative_path(fused),
        "output": project_relative_path(out_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_method": method,
        "filters": {
            "evaluation_sessions": list(cfg.evaluation_sessions),
            "evaluation_exclude_sessions": list(cfg.evaluation_exclude_sessions),
            "evaluation_recordings": list(cfg.evaluation_recordings),
            "evaluation_exclude_recordings": list(cfg.evaluation_exclude_recordings),
        },
        "n_rows_source": int(len(df)),
        "n_rows_scoped": int(len(scoped)),
        "n_sessions_source": int(sessions.nunique()),
        "n_sessions_scoped": int(scoped_sessions.nunique()),
        "n_recordings_source": int(recordings.nunique()),
        "n_recordings_scoped": int(scoped_recordings.nunique()),
    }

    write_json_file(metadata_path, metadata, indent=2)
    if scoped.empty:
        log.warning(
            "%s evaluation scope produced 0 rows from %s; metadata written to %s",
            method,
            project_relative_path(fused),
            project_relative_path(metadata_path),
        )
        return None

    write_csv(scoped, out_path)
    log.info(
        "%s evaluation scope: %d/%d rows, %d recording(s), %d session(s) -> %s",
        method,
        len(scoped),
        len(df),
        metadata["n_recordings_scoped"],
        metadata["n_sessions_scoped"],
        project_relative_path(out_path),
    )
    return out_path


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
                    label_set=cfg.label_set,
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
        orient_method = cfg.orientation_filter  # "auto" or specific method name
        for rec in recordings:
            try:
                results = process_recording_orientation(
                    rec,
                    method=orient_method,
                    sample_rate_hz=cfg.sample_rate_hz,
                    force=cfg.force,
                )
                result["ok"] += len(results)
                # Plots are now generated inside process_recording_orientation.
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

    elif stage == "features":
        from features.pipeline import process_recording_features
        for rec in recordings:
            try:
                df = process_recording_features(
                    rec,
                    window_s=cfg.window_s,
                    hop_s=cfg.hop_s,
                    label_set=cfg.label_set,
                    event_aligned=cfg.event_aligned,
                    n_lags=cfg.lag_features_n_lags,
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
            # Exports should write the full feature set (no quality filtering)
            # so evaluation strategies can be changed without re-running features.
            run_exports(
                recordings,
                min_quality_label=None,
                force=cfg.force,
                no_plots=cfg.no_plots,
            )
            result["ok"] += 1
        except Exception as exc:
            log.error("exports failed: %s", exc)
            result["failed"] += 1
        if not cfg.no_plots:
            try:
                from common.paths import sections_root
                from reporting.stage_summaries import generate_all_stage_summaries
                generate_all_stage_summaries(
                    exports_dir=exports_root(),
                    sections_root_dir=sections_root(),
                    output_dir=data_root() / "report" / "figures",
                )
            except Exception as exc:
                log.warning("stage summary plots failed during exports: %s", exc)

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
                    min_quality=cfg.evaluation_min_quality,
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
        from evaluation.experiments import resolve_evaluation_models
        fused = exports_root() / "features_fused.csv"
        out = evaluation_root()
        if fused.exists():
            seen_methods: set[str] = set()
            for method in cfg.evaluation_methods:
                if method in seen_methods:
                    continue
                seen_methods.add(method)

                if method == "label_grid":
                    from evaluation.label_grid import run_label_grid_evaluation
                    method_out = out / "label_grid"
                    eval_features = _prepare_evaluation_features(
                        fused,
                        cfg,
                        method=method,
                        output_dir=method_out,
                    )
                    if eval_features is None:
                        result["skipped"] += 1
                        continue
                    label_cols = resolve_label_cols(cfg.label_grid_label_cols)
                    # Always include the primary quality so permutation importance
                    # is produced even when the user narrows the quality axis.
                    qualities = list(cfg.label_grid_quality_sweep)
                    if cfg.evaluation_min_quality not in qualities:
                        qualities = [cfg.evaluation_min_quality] + qualities
                    try:
                        eval_models = resolve_evaluation_models(cfg.label_grid_models)
                        perm_models = resolve_evaluation_models(cfg.label_grid_permutation_models)
                        log.info(
                            "Running label-grid evaluation: label_cols=%s × qualities=%s "
                            "models=%s permutation=%s features=%s -> %s",
                            label_cols,
                            qualities,
                            eval_models,
                            perm_models,
                            project_relative_path(eval_features),
                            project_relative_path(method_out),
                        )
                        grid_summary = run_label_grid_evaluation(
                            eval_features,
                            output_dir=method_out,
                            label_cols=label_cols,
                            qualities=qualities,
                            primary_quality=cfg.evaluation_min_quality,
                            seed=cfg.evaluation_seed,
                            exclude_non_riding=cfg.label_grid_exclude_non_riding,
                            no_plots=cfg.no_plots,
                            evaluation_models=eval_models,
                            permutation_models=perm_models,
                            save_trained_models=cfg.label_grid_save_trained_models,
                            force=cfg.force,
                        )
                        result["ok"] += int(grid_summary.get("n_runs_ok", 0))
                        result["failed"] += int(
                            grid_summary.get("n_runs", 0) - grid_summary.get("n_runs_ok", 0)
                        )
                    except Exception as exc:
                        log.error("label-grid evaluation failed: %s", exc)
                        result["failed"] += 1

                elif method == "event_contrasts":
                    try:
                        from evaluation.event_contrasts import run_event_contrast_evaluation
                        method_out = out / "event_contrasts"
                        eval_features = _prepare_evaluation_features(
                            fused,
                            cfg,
                            method=method,
                            output_dir=method_out,
                        )
                        if eval_features is None:
                            result["skipped"] += 1
                            continue
                        event_models = resolve_evaluation_models(cfg.event_contrast_models)
                        log.info(
                            "Running event contrast evaluation: models=%s features=%s -> %s",
                            event_models,
                            project_relative_path(eval_features),
                            project_relative_path(method_out),
                        )
                        summary = run_event_contrast_evaluation(
                            eval_features,
                            output_dir=out,
                            models=event_models,
                            group_col="recording_name",
                            min_quality=cfg.evaluation_min_quality,
                            seed=cfg.evaluation_seed,
                            no_plots=cfg.no_plots,
                            force=cfg.force,
                        )
                        result["ok"] += 1 if int(summary.get("n_metric_rows", 0)) > 0 else 0
                        result["skipped"] += 1 if int(summary.get("n_metric_rows", 0)) == 0 else 0
                    except Exception as exc:
                        log.error("event contrast evaluation failed: %s", exc)
                        result["failed"] += 1

                elif method == "two_stage_events":
                    try:
                        from evaluation.two_stage_events import run_two_stage_event_evaluation
                        method_out = out / "two_stage_events"
                        eval_features = _prepare_evaluation_features(
                            fused,
                            cfg,
                            method=method,
                            output_dir=method_out,
                        )
                        if eval_features is None:
                            result["skipped"] += 1
                            continue
                        two_stage_models = resolve_evaluation_models(cfg.two_stage_event_models)
                        log.info(
                            "Running two-stage event evaluation: tasks=%s models=%s features=%s -> %s",
                            cfg.two_stage_event_tasks,
                            two_stage_models,
                            project_relative_path(eval_features),
                            project_relative_path(method_out),
                        )
                        summary = run_two_stage_event_evaluation(
                            eval_features,
                            output_dir=out,
                            task_names=cfg.two_stage_event_tasks,
                            models=two_stage_models,
                            min_quality=cfg.evaluation_min_quality,
                            seed=cfg.evaluation_seed,
                            target_recall=cfg.two_stage_target_recall,
                            hop_s=cfg.hop_s,
                            no_plots=cfg.no_plots,
                            force=cfg.force,
                        )
                        result["ok"] += 1 if int(summary.get("n_metric_rows", 0)) > 0 else 0
                        result["skipped"] += 1 if int(summary.get("n_metric_rows", 0)) == 0 else 0
                    except Exception as exc:
                        log.error("two-stage event evaluation failed: %s", exc)
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
    """Run all configured workflow stages."""
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
