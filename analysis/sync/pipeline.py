"""Run all synchronisation methods, compare them, and keep the best result."""

from __future__ import annotations

import json
import logging
import shutil
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import recording_stage_dir, recordings_root

from . import plotting
from .sync_cal import synchronize_recording as synchronize_recording_cal
from .sync_lida import synchronize_recording as synchronize_recording_lida
from .sync_online import synchronize_recording as synchronize_recording_online
from .sync_sda import synchronize_recording as synchronize_recording_sda

log = logging.getLogger(__name__)
_STAGE_IN = "parsed"
ALL_METHODS = ("sda", "lida", "cal", "online")
METHOD_LABELS = {
    "sda": "SDA only",
    "lida": "SDA + LIDA",
    "cal": "Calibration",
    "online": "Online",
}
METHOD_STAGES = {method: f"synced/{method}" for method in ALL_METHODS}
_METHOD_RUNNERS = {
    "sda": synchronize_recording_sda,
    "lida": synchronize_recording_lida,
    "cal": synchronize_recording_cal,
    "online": synchronize_recording_online,
}


@dataclass(frozen=True)
class MethodQuality:
    method: str
    stage: str
    available: bool
    corr_offset_and_drift: float | None
    drift_ppm: float | None
    drift_source: str | None
    calibration_span_s: float | None
    calibration_open_score: float | None
    calibration_close_score: float | None


@dataclass(frozen=True)
class SyncSelectionResult:
    recording_name: str
    method: str
    stage: str
    qualities: dict[str, MethodQuality]

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "recording": self.recording_name,
            "selected_method": self.method,
            "selected_stage": self.stage,
            **{
                method: {
                    "stage": quality.stage,
                    "available": quality.available,
                    "corr_offset_and_drift": quality.corr_offset_and_drift,
                    "drift_ppm": quality.drift_ppm,
                    "drift_source": quality.drift_source,
                    "calibration_span_s": quality.calibration_span_s,
                    "calibration_open_score": quality.calibration_open_score,
                    "calibration_close_score": quality.calibration_close_score,
                }
                for method, quality in self.qualities.items()
            },
        }


@dataclass(frozen=True)
class MethodResult:
    method: str
    ok: bool
    error: str | None = None


@dataclass
class RecordingResult:
    recording_name: str
    method_results: list[MethodResult] = field(default_factory=list)
    selection: SyncSelectionResult | None = None
    selection_error: str | None = None

    @property
    def succeeded(self) -> list[str]:
        return [result.method for result in self.method_results if result.ok]

    @property
    def failed(self) -> list[str]:
        return [result.method for result in self.method_results if not result.ok]


def compare_sync_models(recording_name: str) -> dict[str, Any]:
    """Load sync_info.json from every available method directory."""
    result: dict[str, Any] = {"recording": recording_name}
    for method, stage in METHOD_STAGES.items():
        path = recording_stage_dir(recording_name, stage) / "sync_info.json"
        result[method] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return result


def _extract_quality(method: str, info: dict[str, Any] | None) -> MethodQuality:
    stage = METHOD_STAGES[method]
    if info is None:
        return MethodQuality(method, stage, False, None, None, None, None, None, None)
    correlation = (info.get("correlation") or {}).get("offset_and_drift")
    drift = info.get("drift_seconds_per_second")
    calibration = info.get("calibration") if isinstance(info.get("calibration"), dict) else {}
    return MethodQuality(
        method=method,
        stage=stage,
        available=True,
        corr_offset_and_drift=correlation,
        drift_ppm=(float(drift) * 1e6) if drift is not None else None,
        drift_source=info.get("drift_source"),
        calibration_span_s=calibration.get("calibration_span_s"),
        calibration_open_score=(calibration.get("opening") or {}).get("score"),
        calibration_close_score=(calibration.get("closing") or {}).get("score"),
    )


def _calibration_passes_quality(
    quality: MethodQuality,
    *,
    min_cal_span_s: float = 60.0,
    min_cal_score: float = 0.5,
    min_corr: float = 0.2,
    max_drift_ppm: float = 5_000.0,
) -> bool:
    if not quality.available:
        return False
    if quality.calibration_span_s is None or quality.calibration_span_s < min_cal_span_s:
        return False
    if (quality.calibration_open_score or 0.0) < min_cal_score:
        return False
    if quality.drift_source != "duration_ratio" and (quality.calibration_close_score or 0.0) < min_cal_score:
        return False
    if quality.corr_offset_and_drift is None or quality.corr_offset_and_drift < min_corr:
        return False
    if quality.drift_ppm is not None and abs(quality.drift_ppm) > max_drift_ppm:
        return False
    return True


def select_best_sync_method(recording_name: str) -> SyncSelectionResult:
    """Select the best method for a recording from saved sync_info files."""
    comparison = compare_sync_models(recording_name)
    qualities = {method: _extract_quality(method, comparison[method]) for method in ALL_METHODS}
    available = [method for method in ALL_METHODS if qualities[method].available]
    if not available:
        raise RuntimeError(f"No sync_info.json found for recording '{recording_name}'.")

    if _calibration_passes_quality(qualities["cal"]):
        chosen = "cal"
    else:
        chosen = max(available, key=lambda method: (qualities[method].corr_offset_and_drift or -1.0, -ALL_METHODS.index(method)))

    return SyncSelectionResult(
        recording_name=recording_name,
        method=chosen,
        stage=METHOD_STAGES[chosen],
        qualities=qualities,
    )


def print_comparison(comparison: dict[str, Any]) -> None:
    """Pretty-print the main per-method metrics."""
    print(f"\n{'─' * 70}")
    print(f"  Recording : {comparison['recording']}")
    print(f"{'─' * 70}")
    header = f"  {'Metric':<28}" + "".join(f" {METHOD_LABELS[m]:>12}" for m in ALL_METHODS)
    print(header)
    print(f"  {'─' * 68}")

    def row(label: str, values: list[str]) -> None:
        print(f"  {label:<28}" + "".join(f" {value:>12}" for value in values))

    def fmt(value, spec: str) -> str:
        return "N/A" if value is None else format(value, spec)

    row("Offset (s)", [fmt(comparison[m].get("offset_seconds") if comparison[m] else None, ".3f") for m in ALL_METHODS])
    row("Drift (ppm)", [fmt((comparison[m].get("drift_seconds_per_second") * 1e6) if comparison[m] else None, ".1f") for m in ALL_METHODS])
    row(
        "Corr (offset+drift)",
        [fmt(((comparison[m].get("correlation") or {}).get("offset_and_drift") if comparison[m] else None), ".4f") for m in ALL_METHODS],
    )


def prune_method_stage_directories(recording_name: str) -> None:
    """Delete synced/<method>/ directories after flattening the chosen output."""
    for stage in METHOD_STAGES.values():
        path = recording_stage_dir(recording_name, stage)
        if path.is_dir():
            shutil.rmtree(path)


def apply_selection(
    recording_name: str,
    result: SyncSelectionResult,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
) -> Path:
    """Copy the chosen method into synced/ and write comparison artifacts."""
    source_dir = recording_stage_dir(recording_name, result.stage)
    out_dir = recording_stage_dir(recording_name, "synced")
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = compare_sync_models(recording_name)
    for filename in (f"{reference_sensor}.csv", f"{target_sensor}.csv", "sync_info.json"):
        src = source_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)
    (out_dir / "all_methods.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    plotting.plot_methods_norm_grid(recording_name, reference_sensor=reference_sensor, target_sensor=target_sensor, out_dir=out_dir)
    plotting.plot_method_scores(recording_name, result, comparison=comparison, out_dir=out_dir)
    plotting.plot_synced_norm_overlay(
        recording_name,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
        selected_method_key=result.method,
        out_dir=out_dir,
    )
    prune_method_stage_directories(recording_name)
    return out_dir


def print_selection_result(result: SyncSelectionResult) -> None:
    """Print a compact summary of the chosen method and available scores."""
    print(f"Recording  : {result.recording_name}")
    print(f"Selected   : {result.method} (stage: {result.stage})")
    for method in ALL_METHODS:
        quality = result.qualities[method]
        if not quality.available:
            print(f"  {METHOD_LABELS[method]:<12} unavailable")
            continue
        corr = quality.corr_offset_and_drift
        drift = quality.drift_ppm
        marker = " ← selected" if method == result.method else ""
        corr_text = f'{corr:.4f}' if corr is not None else 'N/A'
        drift_text = f'{drift:.1f}' if drift is not None else 'N/A'
        print(f"  {METHOD_LABELS[method]:<12} corr={corr_text} drift={drift_text}{marker}")


def synchronize_recording_all_methods(recording_name: str) -> RecordingResult:
    """Run all methods on parsed/ input, choose the best, and flatten synced/."""
    result = RecordingResult(recording_name=recording_name)
    print(f"\n{'━' * 60}\n  {recording_name}\n{'━' * 60}")
    for method in ALL_METHODS:
        print(f"  [{METHOD_LABELS[method]}] running …", end="", flush=True)
        try:
            _METHOD_RUNNERS[method](recording_name, _STAGE_IN)
            result.method_results.append(MethodResult(method=method, ok=True))
            print(" done")
        except Exception as exc:
            result.method_results.append(MethodResult(method=method, ok=False, error=str(exc)))
            print(f" FAILED: {exc}")
            log.debug("Traceback for %s/%s\n%s", recording_name, method, traceback.format_exc())

    if not result.succeeded:
        print(f"  No method succeeded for {recording_name}.")
        return result

    try:
        comparison = compare_sync_models(recording_name)
        print_comparison(comparison)
        result.selection = select_best_sync_method(recording_name)
        apply_selection(recording_name, result.selection)
    except Exception as exc:
        result.selection_error = str(exc)
        log.warning("Selection failed for %s: %s", recording_name, exc)
    return result


def synchronize_session(session_name: str) -> list[RecordingResult]:
    """Run the full sync pipeline for every recording with the given session prefix."""
    recordings = sorted(
        path.name
        for path in recordings_root().iterdir()
        if path.is_dir() and path.name.startswith(f"{session_name}_")
    )
    if not recordings:
        log.warning("No recordings found matching prefix '%s_'.", session_name)
        return []

    results = [synchronize_recording_all_methods(recording) for recording in recordings]
    print(f"\n{'━' * 60}\n  Session summary: {session_name}\n{'━' * 60}")
    for result in results:
        ok_methods = ", ".join(METHOD_LABELS[m] for m in result.succeeded) or "none"
        failed_methods = ", ".join(METHOD_LABELS[m] for m in result.failed) or "none"
        selected = result.selection.method if result.selection else "none"
        print(f"  {result.recording_name:<22} ok=[{ok_methods}] selected={selected} failed=[{failed_methods}]")
    print(f"{'━' * 60}")
    return results
