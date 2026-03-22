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

The selected method's files are copied to ``synced/`` together with a
``all_methods.json`` summary of every method's metrics.

Usage
-----

.. code-block:: python

    from sync.selection import compare_sync_models, select_best_sync_method

    cmp = compare_sync_models("2026-02-26_5")
    print_comparison(cmp)

    result = select_best_sync_method("2026-02-26_5")
    print(result.method)   # e.g. "calibration"
    print(result.stage)    # e.g. "synced_cal"

Command line::

    python -m sync.selection 2026-02-26_5
    python -m sync.selection 2026-02-26_5 --plot
    python -m sync.selection 2026-02-26_5 --all
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import recording_stage_dir, recordings_root

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
# Overlay plot
# ---------------------------------------------------------------------------

def plot_sync_comparison(
    recording_name: str,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = 10.0,
    zoom_s: float = 90.0,
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Generate an acc_norm overlay plot comparing all available sync methods.

    Produces one row per available method (up to 4), with two columns:
    left = full recording, right = first ``zoom_s`` seconds.

    Returns the path to the saved PNG, or ``None`` if no method data exists.
    """
    from .core import add_vector_norms, remove_dropouts, resample_stream

    def _load(recording_name: str, stage: str, sensor: str) -> Optional[pd.DataFrame]:
        df = _load_sensor_csv(recording_name, stage, sensor)
        if df is None or df.empty:
            return None
        df = add_vector_norms(df)
        df = remove_dropouts(df)
        return resample_stream(df, sample_rate_hz)

    panels = []
    for method, stage in METHOD_STAGES.items():
        ref_df = _load(recording_name, stage, reference_sensor)
        tgt_df = _load(recording_name, stage, target_sensor)
        if ref_df is not None or tgt_df is not None:
            panels.append((method, METHOD_LABELS[method], ref_df, tgt_df))

    if not panels:
        log.warning("No sync data found for any method – skipping comparison plot.")
        return None

    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), sharey="row")
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(f"{recording_name}: acc_norm alignment per sync method", fontsize=12)

    for row_idx, (method, label, ref_df, tgt_df) in enumerate(panels):
        for col, zoom in enumerate([False, True]):
            ax = axes[row_idx][col]

            def _plot(df: Optional[pd.DataFrame], name: str, color: str) -> None:
                if df is None or df.empty:
                    return
                ts = df["timestamp"].to_numpy(float)
                t0 = float(ref_df["timestamp"].iloc[0]) if ref_df is not None else ts[0]
                t_s = (ts - t0) / 1000.0 if ts.mean() > 1e9 else ts - t0
                ax.plot(t_s, df["acc_norm"].to_numpy(float), lw=0.6, alpha=0.85,
                        label=name, color=color)

            _plot(ref_df, reference_sensor, "steelblue")
            _plot(tgt_df, target_sensor, "tomato")

            if zoom and ref_df is not None:
                ax.set_xlim(0, zoom_s)

            ax.set_xlabel("Time (s)" if row_idx == n_rows - 1 else "")
            ax.set_ylabel("acc_norm (m/s²)" if col == 0 else "")
            title_suffix = f"first {zoom_s:.0f} s" if zoom else "full recording"
            ax.set_title(f"{label} — {title_suffix}", fontsize=9)
            if row_idx == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if out_dir is None:
        out_dir = recordings_root() / recording_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sync_method_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{recording_name}] Comparison plot saved → {out_path.name}")
    return out_path


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

def apply_selection(
    recording_name: str,
    result: SyncSelectionResult,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    plot: bool = True,
) -> Path:
    """Copy the selected method's outputs to ``synced/`` and write a summary JSON.

    The ``synced/`` directory will contain:

    - ``<reference_sensor>.csv``  — reference sensor data
    - ``<target_sensor>.csv``     — synchronised target sensor data
    - ``sync_info.json``          — the winning method's sync model
    - ``all_methods.json``        — comparison metrics for all methods
    - ``sync_method_comparison.png`` — overlay plot (if *plot* is True)

    Returns the path to the ``synced/`` directory.
    """
    src_dir = recording_stage_dir(recording_name, result.stage)
    out_dir = recording_stage_dir(recording_name, "synced")
    out_dir.mkdir(parents=True, exist_ok=True)

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

    if plot:
        plot_sync_comparison(recording_name, out_dir=out_dir)
        from visualization import plot_comparison
        stage_ref = f"{recording_name}/synced"
        try:
            plot_comparison.main([stage_ref])
            plot_comparison.main([stage_ref, "--norm"])
        except SystemExit:
            pass

    return out_dir


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def compare_all_recordings(
    session_prefix: str,
    *,
    plot: bool = False,
) -> list[dict]:
    """Run :func:`compare_sync_models` for every recording matching *session_prefix*.

    Returns a list of comparison dicts (one per recording).
    """
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith(f"{session_prefix}_")
    )
    if not recordings:
        log.warning("No recordings found matching prefix '%s_'", session_prefix)
        return []

    results = []
    for rec in recordings:
        result = compare_sync_models(rec)
        results.append(result)
        print_comparison(result)
        if plot:
            plot_sync_comparison(rec)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Usage::

        python -m sync.selection <recording_name> [--apply] [--plot] [--all]
    """
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m sync.selection",
        description=(
            "Compare all sync methods and select the best for a recording. "
            "Optionally copy the winner to synced/ with a summary JSON."
        ),
    )
    parser.add_argument(
        "recording_name",
        help="Recording name (e.g. '2026-02-26_5') or session prefix when --all is used.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the selected method's outputs to synced/.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process all recordings whose name starts with RECORDING_NAME followed by '_'.",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if args.all_recordings:
        compare_all_recordings(args.recording_name, plot=args.plot)
    else:
        cmp = compare_sync_models(args.recording_name)
        print_comparison(cmp)
        result = select_best_sync_method(args.recording_name)
        print()
        print_selection_result(result)
        if args.apply:
            apply_selection(args.recording_name, result, plot=args.plot)


if __name__ == "__main__":
    main()
