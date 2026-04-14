"""Run all synchronization tiers, compare results, and select the best."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from common.paths import recording_stage_dir

from .model import SyncModel

log = logging.getLogger(__name__)

# Ordered from strongest to weakest tier.
SYNC_METHODS: tuple[str, ...] = (
    "multi_anchor",
    "one_anchor_adaptive",
    "one_anchor_prior",
    "signal_only",
)

METHOD_STAGES: dict[str, str] = {
    "multi_anchor": "synced/multi_anchor",
    "one_anchor_adaptive": "synced/one_anchor_adaptive",
    "one_anchor_prior": "synced/one_anchor_prior",
    "signal_only": "synced/signal_only",
}

METHOD_LABELS: dict[str, str] = {
    "multi_anchor": "Multi-anchor protocol",
    "one_anchor_adaptive": "One-anchor adaptive",
    "one_anchor_prior": "One-anchor prior",
    "signal_only": "Signal-only",
}

SyncMethodName = Literal[
    "multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only"
]


def method_stage(method: str) -> str:
    try:
        return METHOD_STAGES[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sync method {method!r}; expected one of {SYNC_METHODS}"
        ) from exc


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


# ---------------------------------------------------------------------------
# Quality extraction from sync_info.json
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncMethodQuality:
    """Per-method quality summary extracted from sync_info.json."""

    method: str
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
    """Final selection outcome for one recording."""

    recording_name: str
    method: str
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


def _load_sync_info(recording_name: str, stage: str) -> Optional[dict]:
    path = recording_stage_dir(recording_name, stage) / "sync_info.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_quality(method: str, info: Optional[dict]) -> SyncMethodQuality:
    """Build a SyncMethodQuality from a sync_info.json dict (or None)."""
    stage = method_stage(method)
    if info is None:
        return SyncMethodQuality(
            method=method, stage=stage, available=False,
            corr_offset_and_drift=None, drift_ppm=None, drift_source=None,
            calibration_span_s=None, calibration_open_score=None,
            calibration_close_score=None, calibration_n_windows=None,
            calibration_fit_r2=None, calibration_anchors=None,
        )

    corr = (info.get("correlation") or {}).get("offset_and_drift")
    drift = info.get("drift_seconds_per_second")
    cal = info.get("calibration") if isinstance(info.get("calibration"), dict) else None

    return SyncMethodQuality(
        method=method, stage=stage, available=True,
        corr_offset_and_drift=corr,
        drift_ppm=(drift * 1e6) if drift is not None else None,
        drift_source=info.get("drift_source"),
        calibration_span_s=cal.get("anchor_span_s") if cal else None,
        calibration_open_score=(cal.get("opening") or {}).get("score") if cal else None,
        calibration_close_score=(cal.get("closing") or {}).get("score") if cal else None,
        calibration_n_windows=cal.get("n_anchors") if cal else None,
        calibration_fit_r2=cal.get("fit_r2") if cal else None,
        calibration_anchors=cal.get("anchors") if cal else None,
    )


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------


def _multi_anchor_passes(
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


def compare_sync_models(recording_name: str) -> dict[str, Any]:
    """Load sync_info.json from all available method directories."""
    result: dict[str, Any] = {"recording": recording_name}
    for method, stage in METHOD_STAGES.items():
        result[method] = _load_sync_info(recording_name, stage)
    return result


def select_best_sync_method(recording_name: str) -> SyncSelectionResult:
    """Select the best sync method for a recording based on tier priority."""
    comparison = compare_sync_models(recording_name)
    qualities = {
        m: extract_quality(m, comparison[m]) for m in SYNC_METHODS
    }
    available = [m for m in SYNC_METHODS if qualities[m].available]
    if not available:
        raise RuntimeError(
            f"No sync results for recording '{recording_name}'. Run sync first."
        )

    # Prefer multi_anchor if quality gates pass.
    if _multi_anchor_passes(qualities["multi_anchor"]):
        chosen = "multi_anchor"
    else:
        # Fall through tier order.  Prefer tier priority, but penalise
        # methods with implausible drift or negative correlation.
        max_plausible_drift_ppm = 5_000.0
        min_useful_corr = 0.0

        def _score(q: SyncMethodQuality) -> float:
            if not q.available:
                return -999.0
            corr = q.corr_offset_and_drift or -1.0
            if q.drift_ppm is not None and abs(q.drift_ppm) > max_plausible_drift_ppm:
                corr -= 0.5
            return corr

        best_score = -999.0
        chosen = available[0]
        for method in SYNC_METHODS:
            if method not in available:
                continue
            s = _score(qualities[method])
            if s > best_score:
                best_score = s
                chosen = method

    return SyncSelectionResult(
        recording_name=recording_name,
        method=chosen,
        stage=method_stage(chosen),
        qualities=qualities,
    )


def print_comparison(result: dict[str, Any]) -> None:
    """Pretty-print a multi-method comparison table."""
    rec = result["recording"]
    col_w = 16
    log.info("sync comparison for %s", rec)

    header = f"  {'Metric':<34}"
    for method in SYNC_METHODS:
        header += f" {method_label(method):>{col_w}}"
    log.info(header)
    log.info("  %s", "─" * 80)

    def _fmt(value, fmt_str, suffix=""):
        return "N/A" if value is None else f"{value:{fmt_str}}{suffix}"

    def _row(label, values):
        line = f"  {label:<34}"
        for v in values:
            line += f" {v:>{col_w}}"
        log.info(line)

    _row("Offset (s)", [
        _fmt(result[m]["offset_seconds"] if result[m] else None, "18.3f")
        for m in SYNC_METHODS
    ])
    _row("Drift (ppm)", [
        _fmt(
            (result[m]["drift_seconds_per_second"] * 1e6) if result[m] else None,
            ".1f",
        )
        for m in SYNC_METHODS
    ])

    for m in SYNC_METHODS:
        info = result[m]
        if info is None:
            continue
        corr = info.get("correlation") or {}
        _row(f"Corr offset+drift ({m})", [
            _fmt(corr.get("offset_and_drift"), ".4f")
        ])


def print_selection_result(result: SyncSelectionResult) -> None:
    """Print a short summary of the selected method."""
    log.info("Recording  : %s", result.recording_name)
    log.info("Selected   : %s (stage: %s)", result.method, result.stage)
    for method in SYNC_METHODS:
        q = result.qualities[method]
        if not q.available:
            log.info("  %-22s  unavailable", method_label(method))
            continue
        corr = f"{q.corr_offset_and_drift:.4f}" if q.corr_offset_and_drift is not None else "N/A"
        drift = f"{q.drift_ppm:.1f}" if q.drift_ppm is not None else "N/A"
        marker = " <- selected" if method == result.method else ""
        log.info(
            "  %-22s  corr=%s  drift=%s ppm%s",
            method_label(method), corr, drift, marker,
        )
