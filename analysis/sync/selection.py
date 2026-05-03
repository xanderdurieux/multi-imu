"""Selection helpers for align arduino timestamps to the sporsa reference clock."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional

from common.paths import recording_stage_dir

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
    "multi_anchor": "Multi-anchor",
    "one_anchor_adaptive": "One-anchor adaptive",
    "one_anchor_prior": "One-anchor prior",
    "signal_only": "Signal-only",
}

SyncMethodName = Literal[
    "multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only"
]


def method_stage(method: str) -> str:
    """Return method stage."""
    try:
        return METHOD_STAGES[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sync method {method!r}; expected one of {SYNC_METHODS}"
        ) from exc


def method_label(method: str) -> str:
    """Return method label."""
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
    offset_seconds: Optional[float]
    corr_offset_and_drift: Optional[float]
    drift_ppm: Optional[float]
    drift_source: Optional[str]
    calibration_span_s: Optional[float]
    calibration_n_windows: Optional[int]
    calibration_fit_r2: Optional[float]
    calibration_anchors: Optional[list[dict[str, Any]]]
    target_time_origin_seconds: Optional[float]
    drift_seconds_per_second: Optional[float]


@dataclass(frozen=True)
class SyncSelectionResult:
    """Final selection outcome for one recording."""

    recording_name: str
    method: str
    stage: str
    qualities: dict[str, SyncMethodQuality]
    comparison: Optional[dict[str, Any]] = None

    @property
    def metrics(self) -> dict[str, Any]:
        """Return metrics."""
        selected_info = self._selected_info()
        shared = build_shared_sync_context(self.comparison or {})
        if shared["target_time_origin_seconds"] is None:
            shared = _shared_fallback_from_qualities(self.qualities)

        methods_out: dict[str, Any] = {}
        for m, q in self.qualities.items():
            methods_out[m] = {
                "available": q.available,
                "offset_seconds": q.offset_seconds,
                "drift_seconds_per_second": q.drift_seconds_per_second,
                "drift_ppm": q.drift_ppm,
                "drift_source": q.drift_source,
                "corr_offset_and_drift": q.corr_offset_and_drift,
                "calibration_span_s": q.calibration_span_s,
                "calibration_n_anchors": q.calibration_n_windows,
                "calibration_fit_r2": q.calibration_fit_r2,
            }

        payload: dict[str, Any] = {
            "selected_method": self.method,
            "target_time_origin_seconds": selected_info.get("target_time_origin_seconds"),
            "offset_seconds": selected_info.get("offset_seconds"),
            "drift_seconds_per_second": selected_info.get("drift_seconds_per_second"),
            "drift_ppm": self.qualities[self.method].drift_ppm,
            "drift_source": self.qualities[self.method].drift_source,
            "methods": methods_out,
        }
        correlation = selected_info.get("correlation")
        if isinstance(correlation, dict):
            payload["correlation"] = correlation
        calibration = shared.get("calibration")
        if isinstance(calibration, dict):
            payload["calibration"] = calibration
        return payload

    def _selected_info(self) -> dict[str, Any]:
        """Return selected info."""
        info = (self.comparison or {}).get(self.method)
        return info if isinstance(info, dict) else {}


def _load_sync_info(recording_name: str, stage: str) -> Optional[dict]:
    """Load sync info."""
    path = recording_stage_dir(recording_name, stage) / "sync_info.json"
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return info if isinstance(info, dict) else None


def build_shared_sync_context(comparison: dict[str, Any]) -> dict[str, Any]:
    """Recording-level calibration block shared by all anchor-based sync methods."""
    empty_calibration: dict[str, Any] = {
        "n_anchors": 0,
        "anchor_span_s": 0.0,
        "anchors": [],
    }
    empty: dict[str, Any] = {
        "target_time_origin_seconds": None,
        "calibration": empty_calibration,
    }
    for m in SYNC_METHODS:
        if m not in comparison:
            continue
        info = comparison[m]
        if not isinstance(info, dict):
            continue
        cal = info.get("calibration")
        if isinstance(cal, dict) and cal.get("anchors"):
            return {
                "target_time_origin_seconds": info.get("target_time_origin_seconds"),
                "calibration": cal,
            }
    for m in SYNC_METHODS:
        info = comparison.get(m)
        if not isinstance(info, dict):
            continue
        if info.get("target_time_origin_seconds") is None:
            continue
        cal = info.get("calibration")
        return {
            "target_time_origin_seconds": info.get("target_time_origin_seconds"),
            "calibration": cal if isinstance(cal, dict) else empty_calibration,
        }
    return empty


def _shared_fallback_from_qualities(
    qualities: dict[str, SyncMethodQuality],
) -> dict[str, Any]:
    """Return shared fallback from qualities."""
    empty_calibration: dict[str, Any] = {
        "n_anchors": 0,
        "anchor_span_s": 0.0,
        "anchors": [],
    }
    for m in SYNC_METHODS:
        q = qualities[m]
        if not q.available or not q.calibration_anchors:
            continue
        anchors = q.calibration_anchors
        cal: dict[str, Any] = {
            "n_anchors": q.calibration_n_windows or len(anchors),
            "anchor_span_s": q.calibration_span_s or 0.0,
            "anchors": anchors,
        }
        if q.calibration_fit_r2 is not None:
            cal["fit_r2"] = q.calibration_fit_r2
        return {
            "target_time_origin_seconds": q.target_time_origin_seconds,
            "calibration": cal,
        }
    for m in SYNC_METHODS:
        q = qualities[m]
        if q.available and q.target_time_origin_seconds is not None:
            return {
                "target_time_origin_seconds": q.target_time_origin_seconds,
                "calibration": empty_calibration,
            }
    return {"target_time_origin_seconds": None, "calibration": empty_calibration}


def extract_quality(method: str, info: Optional[dict]) -> SyncMethodQuality:
    """Build a SyncMethodQuality from a sync_info.json dict (or None)."""
    stage = method_stage(method)
    if info is None:
        return SyncMethodQuality(
            method=method, stage=stage, available=False,
            offset_seconds=None,
            corr_offset_and_drift=None, drift_ppm=None, drift_source=None,
            calibration_span_s=None, calibration_n_windows=None,
            calibration_fit_r2=None, calibration_anchors=None,
            target_time_origin_seconds=None,
            drift_seconds_per_second=None,
        )

    corr = (info.get("correlation") or {}).get("offset_and_drift")
    drift = info.get("drift_seconds_per_second")
    cal = info.get("calibration") if isinstance(info.get("calibration"), dict) else None

    return SyncMethodQuality(
        method=method, stage=stage, available=True,
        offset_seconds=info.get("offset_seconds"),
        corr_offset_and_drift=corr,
        drift_ppm=(drift * 1e6) if drift is not None else None,
        drift_source=info.get("drift_source"),
        calibration_span_s=cal.get("anchor_span_s") if cal else None,
        calibration_n_windows=cal.get("n_anchors") if cal else None,
        calibration_fit_r2=cal.get("fit_r2") if cal else None,
        calibration_anchors=cal.get("anchors") if cal else None,
        target_time_origin_seconds=info.get("target_time_origin_seconds"),
        drift_seconds_per_second=float(drift) if drift is not None else None,
    )


# ---------------------------------------------------------------------------
# Selection policy
# ---------------------------------------------------------------------------

_MAX_DRIFT_PPM = 2_000.0


def _boundary_anchor_scores(anchors: Optional[list[dict[str, Any]]]) -> tuple[float | None, float | None]:
    """Return boundary anchor scores."""
    if not anchors:
        return None, None

    opening = anchors[0] if isinstance(anchors[0], dict) else None
    closing = anchors[-1] if isinstance(anchors[-1], dict) else None

    open_score = opening.get("score") if opening else None
    close_score = closing.get("score") if closing else None
    return open_score, close_score


def _multi_anchor_passes(
    q: SyncMethodQuality,
    *,
    min_cal_span_s: float = 60.0,
    min_cal_score: float = 0.5,
    min_corr: float = 0.2,
    max_drift_ppm: float = _MAX_DRIFT_PPM,
) -> bool:
    """Return multi anchor passes."""
    if not q.available:
        return False
    if q.calibration_span_s is None or q.calibration_span_s < min_cal_span_s:
        return False
    open_score, close_score = _boundary_anchor_scores(q.calibration_anchors)
    if (open_score or 0.0) < min_cal_score:
        return False
    if (close_score or 0.0) < min_cal_score:
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
        def _score(q: SyncMethodQuality) -> float:
            """Return score."""
            if not q.available:
                return -999.0
            corr = q.corr_offset_and_drift or -1.0
            if q.drift_ppm is not None and abs(q.drift_ppm) > _MAX_DRIFT_PPM:
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
        comparison=comparison,
    )
