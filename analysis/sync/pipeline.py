"""High-level sync orchestration, method selection, and CLI entrypoints."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from common import recordings_root
from common.paths import project_relative_path, recording_stage_dir

from .methods import (
    METHOD_LABELS,
    METHOD_STAGES,
    SYNC_METHODS,
    method_label,
    method_stage,
    synchronize_recording,
    synchronize_recording_adaptive,
    synchronize_recording_from_calibration,
    synchronize_recording_online,
    synchronize_recording_sda,
)

log = logging.getLogger(__name__)

_STAGE_IN = "parsed"
ALL_METHODS: list[str] = ["calibration", "lida", "sda", "online", "adaptive"]
SyncMethodName = Literal["sda", "lida", "calibration", "online", "adaptive"]


@dataclass
class MethodResult:
    """Outcome of running one sync method on one recording."""

    method: str
    ok: bool
    error: Optional[str] = None


@dataclass
class RecordingResult:
    """Outcome of running all requested methods on one recording."""

    recording_name: str
    method_results: list[MethodResult] = field(default_factory=list)
    selection: Optional["SyncSelectionResult"] = None
    selection_error: Optional[str] = None

    @property
    def succeeded(self) -> list[str]:
        return [r.method for r in self.method_results if r.ok]

    @property
    def failed(self) -> list[str]:
        return [r.method for r in self.method_results if not r.ok]


@dataclass(frozen=True)
class SyncMethodQuality:
    """Per-method quality summary extracted from ``sync_info.json``."""

    method: SyncMethodName
    stage: str
    available: bool
    corr_offset_and_drift: Optional[float]
    drift_ppm: Optional[float]
    drift_source: Optional[str]
    calibration_span_s: Optional[float]
    calibration_open_score: Optional[float]
    calibration_close_score: Optional[float]
    calibration_n_windows: Optional[int]
    calibration_fit_r2: Optional[float]
    calibration_anchors: Optional[list[dict[str, Any]]]


@dataclass(frozen=True)
class SyncSelectionResult:
    """Final selection result for one recording."""

    recording_name: str
    method: SyncMethodName
    stage: str
    qualities: dict[str, SyncMethodQuality]

    @property
    def metrics(self) -> dict[str, Any]:
        def _q(q: SyncMethodQuality) -> dict[str, Any]:
            return {
                "stage": q.stage,
                "available": q.available,
                "corr_offset_and_drift": q.corr_offset_and_drift,
                "drift_ppm": q.drift_ppm,
                "drift_source": q.drift_source,
                "calibration_span_s": q.calibration_span_s,
                "calibration_open_score": q.calibration_open_score,
                "calibration_close_score": q.calibration_close_score,
                "calibration_n_windows": q.calibration_n_windows,
                "calibration_fit_r2": q.calibration_fit_r2,
                "calibration_anchors": q.calibration_anchors,
            }

        return {
            "recording": self.recording_name,
            "selected_method": self.method,
            "selected_stage": self.stage,
            **{m: _q(q) for m, q in self.qualities.items()},
        }


_METHOD_RUNNERS = {
    "sda": lambda rec, stage: synchronize_recording_sda(rec, stage),
    "lida": lambda rec, stage: synchronize_recording(rec, stage),
    "calibration": lambda rec, stage: synchronize_recording_from_calibration(rec, stage),
    "online": lambda rec, stage: synchronize_recording_online(rec, stage),
    "adaptive": lambda rec, stage: synchronize_recording_adaptive(rec, stage),
}

CHOSEN_SYNC_METHODS: tuple[str, ...] = SYNC_METHODS


def _load_sync_info(recording_name: str, stage: str) -> Optional[dict]:
    path = recording_stage_dir(recording_name, stage) / "sync_info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_sync_models(recording_name: str) -> dict[str, Any]:
    """Load ``sync_info.json`` from all available method directories."""
    result: dict[str, Any] = {"recording": recording_name}
    for method, stage in METHOD_STAGES.items():
        result[method] = _load_sync_info(recording_name, stage)
    return result


def _fmt(value: Optional[float], fmt: str, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value:{fmt}}{suffix}"


def _corr_pair(info: Optional[dict]) -> tuple[str, str]:
    if info is None:
        return "N/A", "N/A"
    corr = info.get("correlation", {}) or {}
    return _fmt(corr.get("offset_only"), ".4f"), _fmt(corr.get("offset_and_drift"), ".4f")


def print_comparison(result: dict[str, Any]) -> None:
    """Pretty-print a comparison result for all four methods."""
    rec = result["recording"]
    col_w = 14

    log.info("sync comparison for %s", rec)

    header = f"  {'Metric':<34}"
    for method in ALL_METHODS:
        header += f" {method_label(method):>{col_w}}"
    log.info(header)
    log.info("  %s", "─" * 68)

    def _row(label: str, values: list[str]) -> None:
        line = f"  {label:<34}"
        for value in values:
            line += f" {value:>{col_w}}"
        log.info(line)

    _row("Offset (s)", [
        _fmt(result[m]["offset_seconds"] if result[m] else None, "18.3f")
        for m in ALL_METHODS
    ])
    _row("Drift (ppm)", [
        _fmt((result[m]["drift_seconds_per_second"] * 1e6) if result[m] else None, ".1f")
        for m in ALL_METHODS
    ])

    corr_pairs = [_corr_pair(result[m]) for m in ALL_METHODS]
    _row("Corr (offset only)", [pair[0] for pair in corr_pairs])
    _row("Corr (offset + drift)", [pair[1] for pair in corr_pairs])

    cal_info = result.get("calibration")
    if cal_info and "calibration" in cal_info:
        cal_block = cal_info["calibration"]
        padding = [""] * (len(ALL_METHODS) - 1)
        _row("Cal span (s)", padding + [_fmt(cal_block.get("calibration_span_s"), ".1f")])
        _row("Cal score open / close", padding + [
            f"{_fmt(cal_block.get('opening', {}).get('score'), '.3f')} / "
            f"{_fmt(cal_block.get('closing', {}).get('score'), '.3f')}"
        ])

    log.info("%s", "─" * 70)


def _extract_quality(method: str, info: Optional[dict]) -> SyncMethodQuality:
    stage = method_stage(method)
    if info is None:
        return SyncMethodQuality(
            method=method,
            stage=stage,
            available=False,
            corr_offset_and_drift=None,
            drift_ppm=None,
            drift_source=None,
            calibration_span_s=None,
            calibration_open_score=None,
            calibration_close_score=None,
            calibration_n_windows=None,
            calibration_fit_r2=None,
            calibration_anchors=None,
        )

    corr = (info.get("correlation") or {}).get("offset_and_drift")
    drift = info.get("drift_seconds_per_second")
    cal_block = info.get("calibration") if isinstance(info.get("calibration"), dict) else None
    return SyncMethodQuality(
        method=method,
        stage=stage,
        available=True,
        corr_offset_and_drift=corr,
        drift_ppm=(drift * 1e6) if drift is not None else None,
        drift_source=info.get("drift_source"),
        calibration_span_s=cal_block.get("calibration_span_s") if cal_block else None,
        calibration_open_score=cal_block.get("opening", {}).get("score") if cal_block else None,
        calibration_close_score=cal_block.get("closing", {}).get("score") if cal_block else None,
        calibration_n_windows=cal_block.get("n_windows_used") if cal_block else None,
        calibration_fit_r2=cal_block.get("fit_r2") if cal_block else None,
        calibration_anchors=cal_block.get("anchors") if cal_block else None,
    )


def _calibration_passes_quality(
    q: SyncMethodQuality,
    *,
    min_cal_span_s: float = 60.0,
    min_cal_score: float = 0.5,
    min_corr: float = 0.2,
    max_drift_ppm: float = 5_000.0,
) -> bool:
    if not q.available:
        return False
    if q.calibration_span_s is None or q.calibration_span_s < min_cal_span_s:
        return False
    if (q.calibration_open_score or 0.0) < min_cal_score:
        return False
    if q.drift_source != "duration_ratio" and (q.calibration_close_score or 0.0) < min_cal_score:
        return False
    if q.corr_offset_and_drift is None or q.corr_offset_and_drift < min_corr:
        return False
    if q.drift_ppm is not None and abs(q.drift_ppm) > max_drift_ppm:
        return False
    return True


def select_best_sync_method(recording_name: str) -> SyncSelectionResult:
    """Select the best sync method for a recording."""
    comparison = compare_sync_models(recording_name)
    qualities = {method: _extract_quality(method, comparison[method]) for method in ALL_METHODS}
    available = [method for method in ALL_METHODS if qualities[method].available]
    if not available:
        raise RuntimeError(
            f"No sync_info.json found in any stage for recording '{recording_name}'. Run a sync method first."
        )

    cal_q = qualities["calibration"]
    if _calibration_passes_quality(cal_q):
        chosen = "calibration"
    else:
        best_corr = -1.0
        chosen = available[0]
        for method in ALL_METHODS:
            if method not in available:
                continue
            corr = qualities[method].corr_offset_and_drift or -1.0
            if corr > best_corr:
                best_corr = corr
                chosen = method

    return SyncSelectionResult(
        recording_name=recording_name,
        method=chosen,
        stage=method_stage(chosen),
        qualities=qualities,
    )


def prune_method_stage_directories(recording_name: str) -> None:
    """Remove all method-specific output directories after flattening ``synced/``."""
    for method in SYNC_METHODS:
        path = recording_stage_dir(recording_name, method_stage(method))
        if path.is_dir():
            shutil.rmtree(path)
            log.debug("sync pruned: %s", project_relative_path(path))


def apply_selection(
    recording_name: str,
    result: SyncSelectionResult,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
) -> Path:
    """Copy the selected result into flat ``synced/`` and remove method subdirectories."""
    src_dir = recording_stage_dir(recording_name, result.stage)
    out_dir = recording_stage_dir(recording_name, "synced")
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in (f"{reference_sensor}.csv", f"{target_sensor}.csv", "sync_info.json"):
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    all_methods_path = out_dir / "all_methods.json"
    all_methods_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    log.info(
        "sync %s/synced: selected method=%s (stage=%s)",
        recording_name,
        result.method,
        result.stage,
    )
    log.info(f"{project_relative_path(out_dir / f"{reference_sensor}.csv")}")
    log.info(f"{project_relative_path(out_dir / f"{target_sensor}.csv")}")
    log.info(f"{project_relative_path(out_dir / "sync_info.json")}")
    log.info(f"{project_relative_path(all_methods_path)}")

    prune_method_stage_directories(recording_name)
    return out_dir


def print_selection_result(result: SyncSelectionResult) -> None:
    """Print a short summary of the selected method and per-method metrics."""
    log.info("Recording  : %s", result.recording_name)
    log.info("Selected   : %s (stage: %s)", result.method, result.stage)
    for method in ALL_METHODS:
        q = result.qualities[method]
        if not q.available:
            log.info("  %s  unavailable", f"{method_label(method):<14}")
            continue
        corr = _fmt(q.corr_offset_and_drift, ".4f")
        drift = _fmt(q.drift_ppm, ".1f")
        marker = " <- selected" if method == result.method else ""
        log.info("  %-14s  corr=%s  drift=%s ppm%s", method_label(method), corr, drift, marker)


def synchronize_recording_all_methods(recording_name: str) -> RecordingResult:
    """Run all four sync methods for a recording, then select and flatten the best result."""
    result = RecordingResult(recording_name=recording_name)
    log.info("sync start: %s", recording_name)

    for method in SYNC_METHODS:
        runner = _METHOD_RUNNERS[method]
        log.info("sync %s: running %s", recording_name, method_label(method))
        try:
            runner(recording_name, _STAGE_IN)
            result.method_results.append(MethodResult(method=method, ok=True))
            log.info("sync %s: %s done", recording_name, method_label(method))
        except Exception as exc:
            result.method_results.append(MethodResult(method=method, ok=False, error=str(exc)))
            log.warning("sync %s: %s failed: %s", recording_name, method_label(method), exc)
            log.debug("Traceback for %s/%s:\n%s", recording_name, method, traceback.format_exc())

    if not result.succeeded:
        log.warning("sync %s: no method succeeded; skipping selection", recording_name)
        return result

    try:
        comparison = compare_sync_models(recording_name)
        print_comparison(comparison)
        selection = select_best_sync_method(recording_name)
        result.selection = selection
        log.info("sync %s: selected %s (stage=%s)", recording_name, selection.method, selection.stage)
        apply_selection(recording_name, selection)
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
    """Run a single sync method, then flatten its output into ``synced/``."""
    if method not in _METHOD_RUNNERS:
        raise ValueError(f"Unknown sync method {method!r}; expected one of {CHOSEN_SYNC_METHODS}")

    if not quiet:
        log.info("sync start: %s (single method=%s)", recording_name, method_label(method))

    _METHOD_RUNNERS[method](recording_name, stage_in)
    comparison = compare_sync_models(recording_name)
    qualities = {m: _extract_quality(m, comparison[m]) for m in ALL_METHODS}
    result = SyncSelectionResult(
        recording_name=recording_name,
        method=method,
        stage=method_stage(method),
        qualities=qualities,
    )
    apply_selection(recording_name, result)
    return result


def synchronize_session(session_name: str) -> list[RecordingResult]:
    """Run synchronization for every ``session_name_*`` recording."""
    root = recordings_root()
    recordings = sorted(
        d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(f"{session_name}_")
    )
    if not recordings:
        log.warning("No recordings found matching prefix '%s_'.", session_name)
        return []

    log.info("sync session %s: %d recording(s) found", session_name, len(recordings))
    results: list[RecordingResult] = []
    for rec in recordings:
        results.append(synchronize_recording_all_methods(rec))
    _print_session_summary(session_name, results)
    return results


def _print_session_summary(session_name: str, results: list[RecordingResult]) -> None:
    log.info("sync session summary: %s", session_name)
    for result in results:
        ok_str = ", ".join(method_label(m) for m in result.succeeded) or "none"
        fail_str = ", ".join(method_label(m) for m in result.failed)
        sel_str = f"-> {result.selection.method}" if result.selection else "no selection"
        line = f"  {result.recording_name:<22}  ok=[{ok_str}]  {sel_str}"
        if fail_str:
            line += f"  failed=[{fail_str}]"
        log.info(line)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for ``python -m sync``."""
    argv = list(argv if argv is not None else sys.argv[1:])
    parser = argparse.ArgumentParser(
        prog="python -m sync",
        description=(
            "Run all four sync methods on parsed CSVs, pick the best, and copy the selected "
            "result into synced/."
        ),
    )
    parser.add_argument(
        "name",
        help="Recording (e.g. 2026-02-26_r5) or session date with --all (e.g. 2026-02-26).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process every recording whose folder name starts with NAME + '_'.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.all_recordings:
        synchronize_session(args.name)
    else:
        synchronize_recording_all_methods(args.name)
