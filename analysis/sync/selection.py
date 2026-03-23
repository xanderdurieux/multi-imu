"""Comparison and selection of the best synchronization method for a recording.

This module compares all four sync methods — SDA-only, SDA+LIDA, calibration-window,
and online (opening-anchor) — and selects the best one based on calibration quality
and acc_norm correlation scores.

Methods and their output directories
--------------------------------------
- ``sda``         → ``synced/sda/``    (offset-only, no drift)
- ``lida``        → ``synced/lida/``   (SDA + LIDA, offset + drift)
- ``calibration`` → ``synced/cal/``    (calibration-window anchors, offset + drift)
- ``online``      → ``synced/online/`` (opening anchor + pre-characterised drift)

Selection priority
------------------
1. ``calibration`` if it passes quality checks (span, scores, drift, correlation).
2. Otherwise, whichever available method has the highest
   ``correlation.offset_and_drift``, with ties broken by:
   ``lida > sda > online``.

The selected method's CSVs and ``sync_info.json`` are copied to flat ``synced/``;
per-method subfolders are removed. ``all_methods.json`` and comparison PNGs
summarise every method's metrics.

Usage
-----

.. code-block:: python

    from sync.selection import compare_sync_models, select_best_sync_method

    cmp = compare_sync_models("2026-02-26_5")
    print_comparison(cmp)

    result = select_best_sync_method("2026-02-26_5")
    print(result.method)   # e.g. "calibration"
    print(result.stage)    # e.g. "synced/cal" (intermediate; flattened to synced/ after apply)

Command line::

    python -m sync 2026-02-26_5
    python -m sync 2026-02-26 --all
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from common import recording_stage_dir

from . import plots as sync_plots

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

# Ordered from most-preferred to least-preferred for tie-breaking.
ALL_METHODS: list[str] = ["calibration", "lida", "sda", "online"]

METHOD_STAGES: dict[str, str] = {
    "sda": "synced/sda",
    "lida": "synced/lida",
    "calibration": "synced/cal",
    "online": "synced/online",
}

METHOD_LABELS: dict[str, str] = {
    "sda": "SDA only",
    "lida": "SDA + LIDA",
    "calibration": "Calibration",
    "online": "Online",
}

SyncMethodName = Literal["sda", "lida", "calibration", "online"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_sync_info(recording_name: str, stage: str) -> Optional[dict]:
    """Load ``sync_info.json`` from *stage* directory, or return ``None``."""
    path = recording_stage_dir(recording_name, stage) / "sync_info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sensor_csv(recording_name: str, stage: str, sensor: str) -> Optional[pd.DataFrame]:
    """Load a sensor CSV from *stage*, returning ``None`` if absent."""
    csv_path = recording_stage_dir(recording_name, stage) / f"{sensor}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return None
    return df


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_sync_models(recording_name: str) -> dict:
    """Load sync_info.json from all available method directories.

    Returns a dict with:
    - ``"recording"``          — recording name
    - ``"sda"``                — raw dict from ``synced/sda/sync_info.json``    (or None)
    - ``"lida"``               — raw dict from ``synced/lida/sync_info.json``   (or None)
    - ``"calibration"``        — raw dict from ``synced/cal/sync_info.json``    (or None)
    - ``"online"``             — raw dict from ``synced/online/sync_info.json`` (or None)
    """
    result: dict[str, Any] = {"recording": recording_name}
    for method, stage in METHOD_STAGES.items():
        result[method] = _load_sync_info(recording_name, stage)
    return result


def _fmt(value: Optional[float], fmt: str, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def _corr_pair(info: Optional[dict]) -> tuple[str, str]:
    if info is None:
        return "N/A", "N/A"
    corr = info.get("correlation", {}) or {}
    return _fmt(corr.get("offset_only"), ".4f"), _fmt(corr.get("offset_and_drift"), ".4f")


def print_comparison(result: dict) -> None:
    """Pretty-print a comparison result for all 4 methods to stdout."""
    rec = result["recording"]
    col_w = 14

    print(f"\n{'─' * 70}")
    print(f"  Recording : {rec}")
    print(f"{'─' * 70}")

    header = f"  {'Metric':<34}"
    for m in ALL_METHODS:
        header += f" {METHOD_LABELS[m]:>{col_w}}"
    print(header)
    print(f"  {'─' * 68}")

    def _row(label: str, values: list[str]) -> None:
        line = f"  {label:<34}"
        for v in values:
            line += f" {v:>{col_w}}"
        print(line)

    # Offset
    _row("Offset (s)", [
        _fmt(result[m]["offset_seconds"] if result[m] else None, "18.3f")
        for m in ALL_METHODS
    ])

    # Drift ppm
    _row("Drift (ppm)", [
        _fmt(
            (result[m]["drift_seconds_per_second"] * 1e6) if result[m] else None,
            ".1f",
        )
        for m in ALL_METHODS
    ])

    # Correlations
    corr_pairs = [_corr_pair(result[m]) for m in ALL_METHODS]
    _row("Corr (offset only)", [p[0] for p in corr_pairs])
    _row("Corr (offset + drift)", [p[1] for p in corr_pairs])

    # Calibration details (calibration method only)
    cal_info = result.get("calibration")
    if cal_info and "calibration" in cal_info:
        cal_block = cal_info["calibration"]
        span = cal_block.get("calibration_span_s")
        open_score = cal_block.get("opening", {}).get("score")
        close_score = cal_block.get("closing", {}).get("score")
        padding = [""] * (len(ALL_METHODS) - 1)
        _row("Cal span (s)", padding + [_fmt(span, ".1f")])
        _row("Cal score open / close", padding + [
            f"{_fmt(open_score, '.3f')} / {_fmt(close_score, '.3f')}"
        ])

    print(f"{'─' * 70}")


# ---------------------------------------------------------------------------
# Quality dataclasses
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class SyncSelectionResult:
    """Final selection result for one recording."""

    recording_name: str
    method: SyncMethodName
    stage: str
    qualities: dict[str, SyncMethodQuality]

    @property
    def metrics(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all methods and the choice."""

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
            }

        return {
            "recording": self.recording_name,
            "selected_method": self.method,
            "selected_stage": self.stage,
            **{m: _q(q) for m, q in self.qualities.items()},
        }


def _extract_quality(method: str, info: Optional[dict]) -> SyncMethodQuality:
    """Convert a raw sync_info dict into a compact quality summary."""
    stage = METHOD_STAGES[method]

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
        )

    corr = (info.get("correlation") or {}).get("offset_and_drift")
    drift = info.get("drift_seconds_per_second")
    drift_ppm = drift * 1e6 if drift is not None else None
    drift_source: Optional[str] = info.get("drift_source")

    cal_block = info.get("calibration")
    if isinstance(cal_block, dict):
        span = cal_block.get("calibration_span_s")
        open_score = cal_block.get("opening", {}).get("score")
        close_score = cal_block.get("closing", {}).get("score")
    else:
        span = open_score = close_score = None

    return SyncMethodQuality(
        method=method,
        stage=stage,
        available=True,
        corr_offset_and_drift=corr,
        drift_ppm=drift_ppm,
        drift_source=drift_source,
        calibration_span_s=span,
        calibration_open_score=open_score,
        calibration_close_score=close_score,
    )


# ---------------------------------------------------------------------------
# Selection logic
# ---------------------------------------------------------------------------

def _calibration_passes_quality(
    q: SyncMethodQuality,
    *,
    min_cal_span_s: float = 60.0,
    min_cal_score: float = 0.5,
    min_corr: float = 0.2,
    max_drift_ppm: float = 5_000.0,
) -> bool:
    """Return True when calibration-sync meets all quality requirements.

    When drift was estimated from the recording-duration ratio
    (``drift_source == "duration_ratio"``), the closing score is considered
    unreliable and is skipped; only the opening score is checked.
    """
    if not q.available:
        return False
    if q.calibration_span_s is None or q.calibration_span_s < min_cal_span_s:
        return False
    if (q.calibration_open_score or 0.0) < min_cal_score:
        return False
    if q.drift_source != "duration_ratio":
        if (q.calibration_close_score or 0.0) < min_cal_score:
            return False
    if q.corr_offset_and_drift is None or q.corr_offset_and_drift < min_corr:
        return False
    if q.drift_ppm is not None and abs(q.drift_ppm) > max_drift_ppm:
        return False
    return True


def select_best_sync_method(recording_name: str) -> SyncSelectionResult:
    """Select the best sync method for *recording_name*.

    Reads ``sync_info.json`` from all four stage directories and applies:

    1. If calibration-sync passes quality checks → choose calibration.
    2. Otherwise pick the method with the highest
       ``correlation.offset_and_drift``.  Ties are broken by the preference
       order: calibration > lida > sda > online.

    Raises ``RuntimeError`` if no sync_info.json is found in any stage.
    """
    cmp = compare_sync_models(recording_name)
    qualities = {m: _extract_quality(m, cmp[m]) for m in ALL_METHODS}

    available = [m for m in ALL_METHODS if qualities[m].available]
    if not available:
        raise RuntimeError(
            f"No sync_info.json found in any stage for recording '{recording_name}'. "
            "Run at least one sync method first."
        )

    # Priority 1: calibration passes quality
    cal_q = qualities["calibration"]
    if _calibration_passes_quality(cal_q):
        chosen = "calibration"
    else:
        # Priority 2: highest correlation (preference order breaks ties)
        best_corr = -1.0
        chosen = available[0]
        for m in ALL_METHODS:
            if m not in available:
                continue
            corr = qualities[m].corr_offset_and_drift or -1.0
            if corr > best_corr:
                best_corr = corr
                chosen = m

    return SyncSelectionResult(
        recording_name=recording_name,
        method=chosen,
        stage=METHOD_STAGES[chosen],
        qualities=qualities,
    )


# ---------------------------------------------------------------------------
# Apply selection: copy winner to synced/
# ---------------------------------------------------------------------------

def prune_method_stage_directories(recording_name: str) -> None:
    """Remove all ``synced/{sda,lida,cal,online}/`` trees after the winner was copied flat."""
    for _method, stage in METHOD_STAGES.items():
        path = recording_stage_dir(recording_name, stage)
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[{recording_name}/synced] removed {path.name}/")


def apply_selection(
    recording_name: str,
    result: SyncSelectionResult,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
) -> Path:
    """Copy the winner into flat ``synced/``, write ``all_methods.json``, plots, then prune method subdirs."""
    src_dir = recording_stage_dir(recording_name, result.stage)
    out_dir = recording_stage_dir(recording_name, "synced")
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_snapshot = compare_sync_models(recording_name)

    sync_plots.plot_methods_norm_grid(
        recording_name,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
        out_dir=out_dir,
    )

    for filename in (f"{reference_sensor}.csv", f"{target_sensor}.csv", "sync_info.json"):
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    all_methods_path = out_dir / "all_methods.json"
    all_methods_path.write_text(
        json.dumps(result.metrics, indent=2), encoding="utf-8"
    )

    print(f"[{recording_name}/synced] Selected method: {result.method} (stage: {result.stage})")
    print(f"[{recording_name}/synced] {reference_sensor}.csv")
    print(f"[{recording_name}/synced] {target_sensor}.csv")
    print(f"[{recording_name}/synced] sync_info.json")
    print(f"[{recording_name}/synced] all_methods.json")

    sync_plots.plot_method_scores(
        recording_name,
        result,
        comparison=comparison_snapshot,
        out_dir=out_dir,
    )
    sync_plots.plot_synced_norm_overlay(
        recording_name,
        reference_sensor=reference_sensor,
        target_sensor=target_sensor,
        selected_method_key=result.method,
        out_dir=out_dir,
    )

    prune_method_stage_directories(recording_name)

    return out_dir


def print_selection_result(result: SyncSelectionResult) -> None:
    """Print a short summary of which method was selected and per-method metrics."""
    print(f"Recording  : {result.recording_name}")
    print(f"Selected   : {result.method} (stage: {result.stage})")
    print()
    for m in ALL_METHODS:
        q = result.qualities[m]
        if not q.available:
            print(f"  {METHOD_LABELS[m]:<14}  unavailable")
            continue
        corr = _fmt(q.corr_offset_and_drift, ".4f")
        drift = _fmt(q.drift_ppm, ".1f")
        marker = " ← selected" if m == result.method else ""
        print(f"  {METHOD_LABELS[m]:<14}  corr={corr}  drift={drift} ppm{marker}")
