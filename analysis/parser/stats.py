"""
Session statistics utilities.

Computes timing/statistics for processed IMU CSV streams (DataFrames) and writes
the results to ``session_stats.json`` in the recording folder.

Key metrics:
- sampling rate and inter-sample interval distribution
- jitter/variability
- gap / packet-loss estimate (heuristic from median interval)
- (Arduino) mapping between device timestamp and host received timestamp to
  estimate drift between clocks
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

from common import load_dataframe, recording_dir, recording_stage_dir


def _interval_summary(interval_ms: pd.Series) -> dict[str, Any]:
    """
    Compact interval summary focused on jitter/variability.

    Kept intentionally simple so stream quality is easy to interpret
    without exposing too many low-level statistics.
    """
    s = pd.to_numeric(interval_ms, errors="coerce").dropna()
    if s.empty:
        return {
            "median_ms": None,
            "std_ms": None,
        }
    return {
        "median_ms": float(s.median()),
        "std_ms": float(s.std(ddof=1)) if s.shape[0] >= 2 else 0.0,
    }


def _estimate_missing_samples(
    interval_ms: pd.Series, expected_ms: float, *, gap_factor: float = 1.5
) -> dict[str, Any]:
    """
    Heuristic packet-loss estimate, simplified:
    - choose expected interval as median inter-sample interval
    - count "gaps" when an interval exceeds gap_factor * expected
    - estimate missing samples by rounding interval/expected - 1 within those gaps
    """
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
    """
    Compute compact timing stats (rate, interval jitter, gaps) for a single stream.

    This returns:
    - start_timestamp_ms / end_timestamp_ms
    - duration_s
    - interval_ms: median and std
    - rate_hz: approximate sampling rate
    - gaps: simple missing-sample estimate
    """
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


def estimate_clock_drift(
    *,
    device_ts_ms: pd.Series,
    received_ts_ms: pd.Series,
) -> dict[str, Any] | None:
    """
    Estimate linear clock mapping drift between device timestamps and received timestamps.

    We fit:
        received_rel_ms ~= a + b * device_rel_s
    where device_rel_s = (device_ts_ms - device_ts_ms[0]) / 1000.

    slope_ms_per_second is expected to be ~1000.0 if both clocks tick at the same rate.

    We report drift as a fractional error relative to ideal (1000 ms/s):
        drift_seconds_per_second = slope_ms_per_second / 1000 - 1
        drift_ppm = drift_seconds_per_second * 1e6
    """
    x_ms = pd.to_numeric(device_ts_ms, errors="coerce")
    y_ms = pd.to_numeric(received_ts_ms, errors="coerce")
    mask = x_ms.notna() & y_ms.notna()
    x_ms = x_ms[mask]
    y_ms = y_ms[mask]
    if x_ms.shape[0] < 3:
        return None

    x0 = float(x_ms.iloc[0])
    y0 = float(y_ms.iloc[0])
    x = (x_ms.to_numpy(dtype=float) - x0) / 1000.0  # seconds
    y = (y_ms.to_numpy(dtype=float) - y0)  # ms

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None

    slope_ms_per_second, intercept = np.polyfit(x, y, 1)  # y ~= intercept + slope*x
    y_hat = intercept + slope_ms_per_second * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None

    drift_seconds_per_second = float(slope_ms_per_second) / 1000.0 - 1.0
    return {
        "num_points": int(x.shape[0]),
        "device_origin_ms": x0,
        "received_origin_ms": y0,
        "slope_ms_per_second": float(slope_ms_per_second),
        "drift_seconds_per_second": float(drift_seconds_per_second),
        "drift_ppm": float(drift_seconds_per_second * 1e6),
        "offset_ms_at_origin": float(intercept),
        "fit_r2": float(r2) if r2 is not None else None,
    }




def compute_file_stats(csv_path: Path) -> dict[str, Any]:
    df = load_dataframe(csv_path)
    out: dict[str, Any] = {
        "num_samples": int(df.shape[0]),
        "timing": compute_stream_timing_stats(df, timestamp_col="timestamp"),
    }

    # Optional Arduino-only received timestamp stats + drift mapping.
    if "timestamp_received" in df.columns:
        drift = estimate_clock_drift(device_ts_ms=df["timestamp"], received_ts_ms=df["timestamp_received"])
        if drift is not None:
            # Keep only the least redundant drift info.
            out["device_to_received_clock"] = {
                "drift_ppm": drift.get("drift_ppm"),
                "fit_r2": drift.get("fit_r2"),
            }

    return out


@dataclass(frozen=True)
class RecordingStats:
    recording_name: str
    generated_at_utc: str
    stage_dir: str
    streams: dict[str, dict[str, Any]]


def compute_recording_stats(recording_name: str, stage: str = "parsed") -> RecordingStats:
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    streams: dict[str, dict[str, Any]] = {}
    for csv_path in sorted(stage_dir.glob("*.csv")):
        streams[csv_path.stem] = compute_file_stats(csv_path)

    return RecordingStats(
        recording_name=recording_name,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        stage_dir=str(stage_dir),
        streams=streams,
    )


def write_recording_stats(recording_name: str, stage: str = "parsed") -> Path:
    stats = compute_recording_stats(recording_name, stage)
    path = recording_dir(recording_name) / "session_stats.json"
    path.write_text(json.dumps(asdict(stats), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.stats",
        description="Compute IMU CSV timing statistics and write recording/session_stats.json.",
    )
    parser.add_argument(
        "recording_name",
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
    out = write_recording_stats(args.recording_name, args.stage)
    print(out)


if __name__ == "__main__":
    main()

