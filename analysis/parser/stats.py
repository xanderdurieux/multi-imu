"""
Per-recording statistics utilities.

Computes timing/statistics for processed IMU CSV streams (DataFrames) and writes
the results to ``recording_stats.json`` in the recording folder.

Key metrics:
- sampling rate and inter-sample interval distribution
- jitter / variability (IQR, outlier rate)
- gap / packet-loss estimate (heuristic from median interval)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import (
    read_csv,
    read_json_file,
    recording_stage_dir,
)


def _interval_summary(interval_ms: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(interval_ms, errors="coerce").dropna()
    if s.empty:
        return {"median_ms": None, "std_ms": None}
    return {
        "median_ms": float(s.median()),
        "std_ms": float(s.std(ddof=1)) if s.shape[0] >= 2 else 0.0,
    }


def _estimate_missing_samples(
    interval_ms: pd.Series, expected_ms: float, *, gap_factor: float = 1.5
) -> dict[str, Any]:
    s = pd.to_numeric(interval_ms, errors="coerce").dropna()
    if s.empty or not np.isfinite(expected_ms) or expected_ms <= 0:
        return {"gap_count": 0, "missing_samples": 0, "threshold_ms": None}

    gap_threshold = gap_factor * expected_ms
    gaps = s[s > gap_threshold]
    missing = 0
    for dt_ms in gaps.to_numpy(dtype=float):
        k = int(round(dt_ms / expected_ms)) - 1
        if k > 0:
            missing += k

    return {
        "threshold_ms": float(gap_threshold),
        "gap_count": int(gaps.shape[0]),
        "missing_samples": int(missing),
    }


def compute_stream_timing_stats(df: pd.DataFrame, *, timestamp_col: str = "timestamp") -> dict[str, Any]:
    """Compact timing stats (rate, interval jitter, gaps) for a single stream."""
    if timestamp_col not in df.columns:
        return {"error": f"missing column: {timestamp_col}"}

    ts = pd.to_numeric(df[timestamp_col], errors="coerce").dropna()
    ts = ts.sort_values().reset_index(drop=True)

    if ts.empty:
        return {
            "start_timestamp_ms": None,
            "end_timestamp_ms": None,
            "duration_s": None,
            "interval_ms": _interval_summary(pd.Series([], dtype=float)),
            "rate_hz": None,
            "gaps": {"gap_count": 0, "missing_samples": 0, "threshold_ms": None},
        }

    start_ms = float(ts.iloc[0])
    end_ms = float(ts.iloc[-1])

    if ts.shape[0] < 2:
        return {
            "start_timestamp_ms": start_ms,
            "end_timestamp_ms": end_ms,
            "duration_s": 0.0,
            "interval_ms": _interval_summary(pd.Series([], dtype=float)),
            "rate_hz": None,
            "gaps": {"gap_count": 0, "missing_samples": 0, "threshold_ms": None},
        }

    intervals = ts.diff().iloc[1:]
    interval_stats = _interval_summary(intervals)
    median_ms = interval_stats.get("median_ms")
    rate_hz = (1000.0 / median_ms) if median_ms and median_ms > 0 else None
    gaps = _estimate_missing_samples(intervals, float(median_ms) if median_ms else float("nan"))
    duration_s = float((end_ms - start_ms) / 1000.0)

    return {
        "start_timestamp_ms": start_ms,
        "end_timestamp_ms": end_ms,
        "duration_s": duration_s,
        "interval_ms": interval_stats,
        "rate_hz": float(rate_hz) if rate_hz is not None else None,
        "gaps": gaps,
    }


@dataclass(frozen=True)
class StreamTimingStats:
    present: bool
    num_samples: int | None
    duration_s: float | None
    median_dt_ms: float | None
    iqr_dt_ms: float | None
    outlier_rate_pct: float | None
    max_gap_ms: float | None
    non_monotonic_steps: int | None
    estimated_missing_samples: int | None
    estimated_missing_rate_pct: float | None


@dataclass(frozen=True)
class RecordingQualitySummary:
    recording_name: str
    session_name: str
    sporsa: StreamTimingStats
    arduino: StreamTimingStats
    sporsa_segments: int | None
    arduino_segments: int | None
    quality_category: str
    quality_reason: str


def _round_or_none(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _stream_missing_stats() -> StreamTimingStats:
    return StreamTimingStats(
        present=False,
        num_samples=None,
        duration_s=None,
        median_dt_ms=None,
        iqr_dt_ms=None,
        outlier_rate_pct=None,
        max_gap_ms=None,
        non_monotonic_steps=None,
        estimated_missing_samples=None,
        estimated_missing_rate_pct=None,
    )


def compute_stream_timing_summary(
    df: pd.DataFrame, *, timestamp_col: str = "timestamp"
) -> StreamTimingStats:
    """Expanded timing summary for a parsed stream."""
    if timestamp_col not in df.columns:
        return _stream_missing_stats()

    timing = compute_stream_timing_stats(df, timestamp_col=timestamp_col)
    if timing.get("start_timestamp_ms") is None:
        return _stream_missing_stats()

    ts = pd.to_numeric(df[timestamp_col], errors="coerce").dropna().reset_index(drop=True)
    if ts.empty:
        return _stream_missing_stats()

    if ts.shape[0] < 2:
        return StreamTimingStats(
            present=True,
            num_samples=int(ts.shape[0]),
            duration_s=_round_or_none(timing.get("duration_s"), digits=3),
            median_dt_ms=None,
            iqr_dt_ms=None,
            outlier_rate_pct=0.0,
            max_gap_ms=None,
            non_monotonic_steps=0,
            estimated_missing_samples=0,
            estimated_missing_rate_pct=0.0,
        )

    dt = ts.diff().iloc[1:].to_numpy(dtype=float)
    finite_dt = dt[np.isfinite(dt)]
    positive_dt = finite_dt[finite_dt > 0]
    if positive_dt.size == 0:
        return StreamTimingStats(
            present=True,
            num_samples=int(ts.shape[0]),
            duration_s=_round_or_none(timing.get("duration_s"), digits=3),
            median_dt_ms=None,
            iqr_dt_ms=None,
            outlier_rate_pct=None,
            max_gap_ms=None,
            non_monotonic_steps=int(np.sum(finite_dt <= 0)),
            estimated_missing_samples=0,
            estimated_missing_rate_pct=None,
        )

    q25, q75 = np.quantile(positive_dt, [0.25, 0.75])
    outlier_mask = positive_dt > (5.0 * float(np.median(positive_dt)))
    missing_samples = timing.get("gaps", {}).get("missing_samples", 0) or 0
    estimated_total = int(ts.shape[0]) + int(missing_samples)

    return StreamTimingStats(
        present=True,
        num_samples=int(ts.shape[0]),
        duration_s=_round_or_none(timing.get("duration_s"), digits=3),
        median_dt_ms=_round_or_none(timing.get("interval_ms", {}).get("median_ms"), digits=3),
        iqr_dt_ms=_round_or_none(float(q75 - q25), digits=3),
        outlier_rate_pct=_round_or_none(100.0 * float(np.mean(outlier_mask)), digits=4),
        max_gap_ms=_round_or_none(float(np.max(positive_dt)), digits=3),
        non_monotonic_steps=int(np.sum(finite_dt <= 0)),
        estimated_missing_samples=int(missing_samples),
        estimated_missing_rate_pct=_round_or_none(
            100.0 * float(missing_samples) / estimated_total if estimated_total > 0 else 0.0,
            digits=3,
        ),
    )


def _session_name_from_recording(recording_name: str) -> str:
    session_name = recording_name.rpartition("_r")[0]
    return session_name if session_name else recording_name


def _load_segment_count(recording_name: str, sensor: str) -> int | None:
    path = recording_stage_dir(recording_name, "parsed") / "calibration_segments.json"
    if not path.exists():
        return None
    payload = read_json_file(path)
    value = payload.get("sensors", {}).get(sensor, {}).get("num_segments")
    if value is None:
        return None
    return int(value)


def _classify_quality(
    *,
    sporsa: StreamTimingStats,
    arduino: StreamTimingStats,
    sporsa_segments: int | None,
    arduino_segments: int | None,
) -> tuple[str, str]:
    if not sporsa.present:
        return "limited", "missing_sporsa_stream"

    max_duration_s = max(
        value for value in (sporsa.duration_s, arduino.duration_s) if value is not None
    )
    sporsa_seg = int(sporsa_segments or 0)
    arduino_seg = int(arduino_segments or 0)
    arduino_loss_pct = float(arduino.estimated_missing_rate_pct or 0.0)
    arduino_outlier_pct = float(arduino.outlier_rate_pct or 0.0)

    if (
        sporsa_seg >= 2
        and arduino_seg >= 2
        and max_duration_s >= 180.0
        and arduino_loss_pct <= 10.0
        and arduino_outlier_pct <= 0.25
    ):
        return "good", "full_protocol_and_stable_timing"

    if max_duration_s < 60.0:
        return "limited", "short_recording"

    if sporsa_seg == 0 and arduino_seg == 0:
        return "limited", "missing_calibration_protocol"

    return "usable", "partial_protocol_or_dropout"


def compute_recording_quality_summary(recording_name: str) -> RecordingQualitySummary:
    """Quality summary for one parsed recording."""
    parsed_dir = recording_stage_dir(recording_name, "parsed")
    sporsa_csv = parsed_dir / "sporsa.csv"
    arduino_csv = parsed_dir / "arduino.csv"

    sporsa = (
        compute_stream_timing_summary(read_csv(sporsa_csv))
        if sporsa_csv.exists()
        else _stream_missing_stats()
    )
    arduino = (
        compute_stream_timing_summary(read_csv(arduino_csv))
        if arduino_csv.exists()
        else _stream_missing_stats()
    )

    sporsa_segments = _load_segment_count(recording_name, "sporsa")
    arduino_segments = _load_segment_count(recording_name, "arduino")
    quality_category, quality_reason = _classify_quality(
        sporsa=sporsa,
        arduino=arduino,
        sporsa_segments=sporsa_segments,
        arduino_segments=arduino_segments,
    )

    return RecordingQualitySummary(
        recording_name=recording_name,
        session_name=_session_name_from_recording(recording_name),
        sporsa=sporsa,
        arduino=arduino,
        sporsa_segments=sporsa_segments,
        arduino_segments=arduino_segments,
        quality_category=quality_category,
        quality_reason=quality_reason,
    )


def write_recording_stats(recording_name: str, stage: str = "parsed") -> Path:
    """Compute quality summary and write recording_stats.json."""
    summary = compute_recording_quality_summary(recording_name)
    payload = {
        "recording_name": summary.recording_name,
        "session_name": summary.session_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "sporsa_segments": summary.sporsa_segments,
        "arduino_segments": summary.arduino_segments,
        "quality_category": summary.quality_category,
        "quality_reason": summary.quality_reason,
        "streams": {
            "sporsa": asdict(summary.sporsa),
            "arduino": asdict(summary.arduino),
        },
    }
    path = recording_stage_dir(recording_name, stage) / "recording_stats.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.stats",
        description="Compute per-recording stats and write to recording_stats.json.",
    )
    parser.add_argument(
        "recording_name",
        nargs="?",
        help="Recording name under data/recordings/ (e.g. 2026-02-26_r5).",
    )
    parser.add_argument(
        "--stage",
        default="parsed",
        help="Stage to compute stats for (default: parsed).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    if not args.recording_name:
        raise SystemExit("Pass a recording name (e.g. 2026-02-26_r1).")

    out = write_recording_stats(args.recording_name, args.stage)
    print(out)


if __name__ == "__main__":
    main()
