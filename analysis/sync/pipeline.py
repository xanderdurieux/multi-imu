"""Recording-level sync I/O, method orchestration, and CLI entrypoint.

Workflow per recording:

1. Run each strategy in ``SYNC_METHODS`` order; write its outputs to
   ``synced/<method>/`` (``sporsa.csv``, ``arduino.csv``, ``sync_info.json``,
   ``sync_metadata.json``).
2. Load all per-method ``sync_info.json`` files and pick the winner via
   ``selection.select_best_sync_method``.
3. Copy the winner into flat ``synced/`` and prune the per-method dirs.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from common import recordings_root
from common.paths import (
    project_relative_path,
    recording_stage_dir,
    sensor_csv,
    write_csv,
)

from .model import apply_sync_model
from .selection import (
    METHOD_STAGES,
    SYNC_METHODS,
    SyncSelectionResult,
    compare_sync_models,
    extract_quality,
    method_label,
    method_stage,
    select_best_sync_method,
)
from .signals import compute_sync_correlations, load_stream
from .strategies import (
    estimate_multi_anchor,
    estimate_one_anchor_adaptive,
    estimate_one_anchor_prior,
    estimate_signal_only,
)

log = logging.getLogger(__name__)

_STAGE_IN = "parsed"

_STRATEGIES = {
    "multi_anchor": estimate_multi_anchor,
    "one_anchor_adaptive": estimate_one_anchor_adaptive,
    "one_anchor_prior": estimate_one_anchor_prior,
    "signal_only": estimate_signal_only,
}


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MethodResult:
    method: str
    ok: bool
    error: Optional[str] = None


@dataclass
class RecordingResult:
    recording_name: str
    method_results: list[MethodResult] = field(default_factory=list)
    selection: Optional[SyncSelectionResult] = None
    selection_error: Optional[str] = None

    @property
    def succeeded(self) -> list[str]:
        return [r.method for r in self.method_results if r.ok]

    @property
    def failed(self) -> list[str]:
        return [r.method for r in self.method_results if not r.ok]


# ---------------------------------------------------------------------------
# Per-method execution
# ---------------------------------------------------------------------------


def _drop_alignment_columns(df):
    drop = [c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
            if c in df.columns]
    return df.drop(columns=drop) if drop else df


def _method_summary_block(
    model, meta: dict[str, Any], correlation: dict[str, Any]
) -> dict[str, Any]:
    calibration = meta.get("calibration") or {}
    return {
        "available": True,
        "offset_seconds": model.offset_seconds,
        "drift_seconds_per_second": model.drift_seconds_per_second,
        "drift_ppm": model.drift_seconds_per_second * 1e6,
        "drift_source": meta.get("drift_source"),
        "corr_offset_and_drift": (correlation or {}).get("offset_and_drift"),
        "calibration_span_s": calibration.get("anchor_span_s"),
        "calibration_n_anchors": calibration.get("n_anchors"),
        "calibration_fit_r2": calibration.get("fit_r2"),
    }


def _build_sync_info(
    *, method: str, model, meta: dict[str, Any], correlation: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "selected_method": method,
        "target_time_origin_seconds": model.target_time_origin_seconds,
        "offset_seconds": model.offset_seconds,
        "drift_seconds_per_second": model.drift_seconds_per_second,
        "drift_ppm": model.drift_seconds_per_second * 1e6,
        "drift_source": meta.get("drift_source"),
        "correlation": dict(correlation),
        "methods": {method: _method_summary_block(model, meta, correlation)},
    }
    calibration = meta.get("calibration")
    if isinstance(calibration, dict):
        payload["calibration"] = calibration
    return payload


def _build_sync_metadata(
    *, reference_csv: Path, target_csv: Path, meta: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "reference_csv": str(reference_csv),
        "target_csv": str(target_csv),
        "sync_method": meta.get("sync_method"),
    }
    hyperparameters = meta.get("hyperparameters")
    if isinstance(hyperparameters, dict):
        payload["hyperparameters"] = hyperparameters
    for key in ("coarse_alignment_score", "window_refinement"):
        value = meta.get(key)
        if value is not None:
            payload[key] = value
    return payload


def _run_method(
    recording_name: str,
    stage_in: str,
    method: str,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
) -> None:
    """Run a single strategy and write outputs under ``synced/<method>/``."""
    ref_csv = sensor_csv(f"{recording_name}/{stage_in}", reference_sensor)
    tgt_csv = sensor_csv(f"{recording_name}/{stage_in}", target_sensor)
    out_dir = recording_stage_dir(recording_name, method_stage(method))
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "sync %s/%s: ref=%s tgt=%s",
        recording_name, method_stage(method), ref_csv.name, tgt_csv.name,
    )

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    model, meta = _STRATEGIES[method](
        ref_df, tgt_df,
        recording_name=recording_name,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
    )

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    sample_rate_hz = float(
        (meta.get("hyperparameters") or {}).get("resample_rate_hz", 100.0)
    )
    correlation = compute_sync_correlations(
        ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz,
    )

    info_payload = _build_sync_info(
        method=method, model=model, meta=meta, correlation=correlation,
    )
    metadata_payload = _build_sync_metadata(
        reference_csv=ref_csv, target_csv=tgt_csv, meta=meta,
    )

    shutil.copy2(ref_csv, out_dir / f"{reference_sensor}.csv")
    write_csv(_drop_alignment_columns(aligned_df), out_dir / f"{target_sensor}.csv")
    (out_dir / "sync_info.json").write_text(
        json.dumps(info_payload, indent=2), encoding="utf-8"
    )
    (out_dir / "sync_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Selection flattening
# ---------------------------------------------------------------------------


def _prune_method_stage_directories(recording_name: str) -> None:
    for method in SYNC_METHODS:
        path = recording_stage_dir(recording_name, method_stage(method))
        if path.is_dir():
            shutil.rmtree(path)
            log.debug("sync pruned: %s", project_relative_path(path))


def _apply_selection(
    recording_name: str,
    result: SyncSelectionResult,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
) -> Path:
    """Copy the selected method's outputs into flat ``synced/`` and prune."""
    src_dir = recording_stage_dir(recording_name, result.stage)
    out_dir = recording_stage_dir(recording_name, "synced")
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in (
        f"{reference_sensor}.csv",
        f"{target_sensor}.csv",
        "sync_metadata.json",
    ):
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    (out_dir / "sync_info.json").write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )

    log.info(
        "sync %s/synced: selected method=%s (stage=%s)",
        recording_name, result.method, result.stage,
    )
    _prune_method_stage_directories(recording_name)
    return out_dir


# ---------------------------------------------------------------------------
# Public orchestration API
# ---------------------------------------------------------------------------


def synchronize_recording_all_methods(recording_name: str) -> RecordingResult:
    """Run all sync methods, select the best, and flatten into ``synced/``."""
    result = RecordingResult(recording_name=recording_name)
    log.info("sync start: %s", recording_name)

    for method in SYNC_METHODS:
        log.info("sync %s: running %s", recording_name, method_label(method))
        try:
            _run_method(recording_name, _STAGE_IN, method)
            result.method_results.append(MethodResult(method=method, ok=True))
            log.info("sync %s: %s done", recording_name, method_label(method))
        except Exception as exc:
            result.method_results.append(
                MethodResult(method=method, ok=False, error=str(exc))
            )
            log.warning(
                "sync %s: %s failed: %s", recording_name, method_label(method), exc
            )
            log.debug(traceback.format_exc())

    if not result.succeeded:
        log.warning("sync %s: no method succeeded; skipping selection", recording_name)
        return result

    try:
        selection = select_best_sync_method(recording_name)
        result.selection = selection
        log.info(
            "sync %s: selected %s (stage=%s)",
            recording_name, selection.method, selection.stage,
        )
        _apply_selection(recording_name, selection)
    except Exception as exc:
        result.selection_error = str(exc)
        log.warning("Selection failed for %s: %s", recording_name, exc)

    return result


def synchronize_recording_chosen_method(
    recording_name: str,
    method: str,
    *,
    stage_in: str = _STAGE_IN,
    quiet: bool = False,
) -> SyncSelectionResult:
    """Run a single sync method and flatten its output into ``synced/``."""
    if method not in METHOD_STAGES:
        raise ValueError(
            f"Unknown sync method {method!r}; expected one of {SYNC_METHODS}"
        )
    if not quiet:
        log.info(
            "sync start: %s (single method=%s)",
            recording_name, method_label(method),
        )

    _run_method(recording_name, stage_in, method)
    comparison = compare_sync_models(recording_name)
    qualities = {m: extract_quality(m, comparison[m]) for m in SYNC_METHODS}
    result = SyncSelectionResult(
        recording_name=recording_name,
        method=method,
        stage=method_stage(method),
        qualities=qualities,
        comparison=comparison,
    )
    _apply_selection(recording_name, result)
    return result


def _synchronize_session(session_name: str) -> list[RecordingResult]:
    """Run sync for every recording in a session (CLI helper)."""
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith(f"{session_name}_")
    )
    if not recordings:
        log.warning("No recordings matching prefix '%s_'.", session_name)
        return []

    log.info("sync session %s: %d recording(s)", session_name, len(recordings))
    results = [synchronize_recording_all_methods(rec) for rec in recordings]

    log.info("sync session summary: %s", session_name)
    for r in results:
        ok = ", ".join(method_label(m) for m in r.succeeded) or "none"
        sel = f"-> {r.selection.method}" if r.selection else "no selection"
        log.info("  %-22s  ok=[%s]  %s", r.recording_name, ok, sel)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m sync",
        description="Run synchronization on parsed recordings.",
    )
    parser.add_argument(
        "name",
        help="Recording name (e.g. 2026-02-26_r5) or session date with --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process every recording whose folder starts with NAME + '_'.",
    )
    parser.add_argument(
        "--method",
        choices=list(SYNC_METHODS),
        default=None,
        help="Run only this method (default: run all, pick best).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.all_recordings:
        _synchronize_session(args.name)
    elif args.method:
        synchronize_recording_chosen_method(args.name, args.method)
    else:
        synchronize_recording_all_methods(args.name)
