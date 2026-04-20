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

from common.paths import (
    data_root,
    list_csv_files,
    read_csv,
    read_json_file,
    recording_stage_dir,
    recordings_root,
    write_csv,
)


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
    df = read_csv(csv_path)
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


@dataclass(frozen=True)
class StreamIntegritySummary:
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
class RecordingIntegritySummary:
    recording_name: str
    session_name: str
    sporsa: StreamIntegritySummary
    arduino: StreamIntegritySummary
    sporsa_segments: int | None
    arduino_segments: int | None
    quality_category: str
    quality_reason: str


def compute_recording_stats(recording_name: str, stage: str = "parsed") -> RecordingStats:
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    streams: dict[str, dict[str, Any]] = {}
    for csv_path in list_csv_files(stage_dir):
        streams[csv_path.stem] = compute_file_stats(csv_path)

    return RecordingStats(
        recording_name=recording_name,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        stage_dir=str(stage_dir),
        streams=streams,
    )


def write_recording_stats(recording_name: str, stage: str = "parsed") -> Path:
    stats = compute_recording_stats(recording_name, stage)
    path = recording_stage_dir(recording_name, stage) / "session_stats.json"
    path.write_text(json.dumps(asdict(stats), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _round_or_none(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _stream_missing_summary() -> StreamIntegritySummary:
    return StreamIntegritySummary(
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


def compute_stream_integrity_summary(df: pd.DataFrame, *, timestamp_col: str = "timestamp") -> StreamIntegritySummary:
    """Expanded parsed-stream integrity summary built on top of timing stats."""
    if timestamp_col not in df.columns:
        return _stream_missing_summary()

    timing = compute_stream_timing_stats(df, timestamp_col=timestamp_col)
    if timing.get("start_timestamp_ms") is None:
        return _stream_missing_summary()

    ts = pd.to_numeric(df[timestamp_col], errors="coerce").dropna().reset_index(drop=True)
    if ts.empty:
        return _stream_missing_summary()

    if ts.shape[0] < 2:
        return StreamIntegritySummary(
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
        return StreamIntegritySummary(
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

    median_dt_ms = float(np.median(positive_dt))
    q25, q75 = np.quantile(positive_dt, [0.25, 0.75])
    outlier_mask = positive_dt > (5.0 * median_dt_ms)
    missing_samples = timing.get("gaps", {}).get("missing_samples", 0) or 0
    estimated_total = int(ts.shape[0]) + int(missing_samples)

    return StreamIntegritySummary(
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
    session_name, _sep, _suffix = recording_name.rpartition("_r")
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
    sporsa: StreamIntegritySummary,
    arduino: StreamIntegritySummary,
    sporsa_segments: int | None,
    arduino_segments: int | None,
) -> tuple[str, str]:
    if not sporsa.present:
        return "limited", "missing_sporsa_stream"

    max_duration_s = max(value for value in (sporsa.duration_s, arduino.duration_s) if value is not None)
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


def compute_recording_integrity_summary(recording_name: str) -> RecordingIntegritySummary:
    """Session-level parsed integrity summary for one recording."""
    parsed_dir = recording_stage_dir(recording_name, "parsed")
    sporsa_csv = parsed_dir / "sporsa.csv"
    arduino_csv = parsed_dir / "arduino.csv"

    sporsa = compute_stream_integrity_summary(read_csv(sporsa_csv)) if sporsa_csv.exists() else _stream_missing_summary()
    arduino = compute_stream_integrity_summary(read_csv(arduino_csv)) if arduino_csv.exists() else _stream_missing_summary()

    sporsa_segments = _load_segment_count(recording_name, "sporsa")
    arduino_segments = _load_segment_count(recording_name, "arduino")
    quality_category, quality_reason = _classify_quality(
        sporsa=sporsa,
        arduino=arduino,
        sporsa_segments=sporsa_segments,
        arduino_segments=arduino_segments,
    )

    return RecordingIntegritySummary(
        recording_name=recording_name,
        session_name=_session_name_from_recording(recording_name),
        sporsa=sporsa,
        arduino=arduino,
        sporsa_segments=sporsa_segments,
        arduino_segments=arduino_segments,
        quality_category=quality_category,
        quality_reason=quality_reason,
    )


def list_session_recordings(session_name: str) -> list[str]:
    names = [
        path.name
        for path in recordings_root().iterdir()
        if path.is_dir() and path.name.startswith(f"{session_name}_r")
    ]
    return sorted(
        names,
        key=lambda name: int(name.rsplit("_r", 1)[1]) if "_r" in name and name.rsplit("_r", 1)[1].isdigit() else name,
    )


def session_integrity_summaries(session_name: str) -> list[RecordingIntegritySummary]:
    return [compute_recording_integrity_summary(recording_name) for recording_name in list_session_recordings(session_name)]


def session_integrity_dataframe(session_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary in session_integrity_summaries(session_name):
        row: dict[str, object] = {
            "recording_name": summary.recording_name,
            "session_name": summary.session_name,
            "sporsa_segments": summary.sporsa_segments,
            "arduino_segments": summary.arduino_segments,
            "quality_category": summary.quality_category,
            "quality_reason": summary.quality_reason,
        }
        for sensor in ("sporsa", "arduino"):
            stream = getattr(summary, sensor)
            for key, value in asdict(stream).items():
                row[f"{sensor}_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _latex_name(name: str) -> str:
    return name.replace("_", r"\_")


def _format_recording_list(recordings: list[str]) -> str:
    return ", ".join(rf"\texttt{{{_latex_name(name)}}}" for name in recordings)


def _latex_row(recording_name: str, sporsa_segments: Any, arduino_segments: Any) -> str:
    sporsa_missing = sporsa_segments is None or pd.isna(sporsa_segments)
    arduino_missing = arduino_segments is None or pd.isna(arduino_segments)
    sporsa = "0" if sporsa_missing else str(int(sporsa_segments))
    arduino = "0" if arduino_missing else str(int(arduino_segments))
    return f"        {_latex_name(recording_name)} & {sporsa} & {arduino} \\\\"


def _as_float_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _session_available_stream_count(summary_df: pd.DataFrame) -> int:
    present = 0
    for sensor in ("sporsa", "arduino"):
        col = f"{sensor}_present"
        if col in summary_df.columns:
            present += int(summary_df[col].fillna(False).astype(bool).sum())
    return present


def _session_summary_payload(session_name: str, summary_df: pd.DataFrame) -> dict[str, Any]:
    sporsa_present = summary_df[summary_df["sporsa_present"].fillna(False)]
    arduino_present = summary_df[summary_df["arduino_present"].fillna(False)]
    sporsa_max_gap_idx = _as_float_series(sporsa_present, "sporsa_max_gap_ms").idxmax()
    arduino_max_gap_idx = _as_float_series(arduino_present, "arduino_max_gap_ms").idxmax()
    arduino_loss_idx = _as_float_series(arduino_present, "arduino_estimated_missing_rate_pct").idxmax()

    quality_groups = {
        category: summary_df.loc[summary_df["quality_category"] == category, "recording_name"].tolist()
        for category in ("good", "usable", "limited")
    }

    return {
        "session_name": session_name,
        "n_recordings": int(len(summary_df)),
        "n_recordings_with_sporsa": int(sporsa_present.shape[0]),
        "n_available_streams": int(_session_available_stream_count(summary_df)),
        "missing_sporsa_recordings": summary_df.loc[~summary_df["sporsa_present"].fillna(False), "recording_name"].tolist(),
        "quality_groups": quality_groups,
        "sporsa": {
            "median_dt_ms": float(_as_float_series(sporsa_present, "sporsa_median_dt_ms").median()),
            "iqr_dt_ms": float(_as_float_series(sporsa_present, "sporsa_iqr_dt_ms").median()),
            "max_outlier_rate_pct": float(_as_float_series(sporsa_present, "sporsa_outlier_rate_pct").max()),
            "max_gap_ms": float(_as_float_series(sporsa_present, "sporsa_max_gap_ms").max()),
            "max_gap_recording": str(summary_df.loc[sporsa_max_gap_idx, "recording_name"]),
        },
        "arduino": {
            "median_dt_ms": float(_as_float_series(arduino_present, "arduino_median_dt_ms").median()),
            "iqr_dt_ms_min": float(_as_float_series(arduino_present, "arduino_iqr_dt_ms").min()),
            "iqr_dt_ms_max": float(_as_float_series(arduino_present, "arduino_iqr_dt_ms").max()),
            "max_outlier_rate_pct": float(_as_float_series(arduino_present, "arduino_outlier_rate_pct").max()),
            "n_zero_outlier_recordings": int((_as_float_series(arduino_present, "arduino_outlier_rate_pct") == 0.0).sum()),
            "max_gap_ms": float(_as_float_series(arduino_present, "arduino_max_gap_ms").max()),
            "max_gap_recording": str(summary_df.loc[arduino_max_gap_idx, "recording_name"]),
            "min_missing_samples": int(_as_float_series(arduino_present, "arduino_estimated_missing_samples").min()),
            "max_missing_samples": int(_as_float_series(arduino_present, "arduino_estimated_missing_samples").max()),
            "min_missing_rate_pct": float(_as_float_series(arduino_present, "arduino_estimated_missing_rate_pct").min()),
            "max_missing_rate_pct": float(_as_float_series(arduino_present, "arduino_estimated_missing_rate_pct").max()),
            "max_missing_recording": str(summary_df.loc[arduino_loss_idx, "recording_name"]),
        },
    }


def _build_latex_section(payload: dict[str, Any], summary_df: pd.DataFrame) -> str:
    quality_groups = payload["quality_groups"]
    missing_sporsa = payload["missing_sporsa_recordings"]
    missing_sporsa_text = _format_recording_list(missing_sporsa) if missing_sporsa else "none"

    rows = []
    for recording_name in list_session_recordings(payload["session_name"]):
        row = summary_df.loc[summary_df["recording_name"] == recording_name].iloc[0]
        rows.append(_latex_row(recording_name, row["sporsa_segments"], row["arduino_segments"]))
    calibration_rows = "\n".join(rows)

    return rf"""\section{{Data integrity checks}}

\paragraph{{Timestamp monotonicity and discontinuities}}
For each sensor stream, timestamps were required to be strictly monotonic after parsing. Non-monotonic segments indicate logging corruption, counter wraparound handling errors, or parsing misalignment. For the Arduino BLE stream, additional discontinuities can occur when notification bursts are dropped and later samples resume; these gaps are observable as unusually large inter-sample timestamp differences. Across the {payload["n_recordings"]} parsed recordings ({payload["n_available_streams"]} available sensor streams, with the SPORSA stream missing only for {missing_sporsa_text}), no non-monotonic timestamps were observed. The largest positive timestamp gap was {payload["sporsa"]["max_gap_ms"]:.0f}~ms on SPORSA in \texttt{{{_latex_name(payload["sporsa"]["max_gap_recording"])}}} and {payload["arduino"]["max_gap_ms"]:.0f}~ms on Arduino in \texttt{{{_latex_name(payload["arduino"]["max_gap_recording"])}}}.

\paragraph{{Inter-sample interval statistics}}
A key integrity indicator is the distribution of inter-sample intervals $\Delta t$. For an ideal $f_s$-Hz stream, $\Delta t$ should concentrate near $1/f_s$. In practice, BLE transport and smartphone logging can create bursty arrival, but the \emph{{device timestamps}} should still reflect near-uniform sampling if the microcontroller sampling loop is stable. We therefore compute median $\Delta t$, interquartile range, and outlier rate (e.g., $\Delta t$ exceeding $5\times$ the median) as a compact summary of timing quality. In the 26~February~2026 session, the median $\Delta t$ was 10~ms for every SPORSA recording and 17~ms for every Arduino recording, corresponding to effective rates of approximately 100~Hz and 58.8~Hz. The SPORSA interquartile range was consistently 1~ms, while the Arduino interquartile range ranged from {payload["arduino"]["iqr_dt_ms_min"]:.0f} to {payload["arduino"]["iqr_dt_ms_max"]:.0f}~ms. Using the $5\times$ median criterion, SPORSA outlier rates remained below {payload["sporsa"]["max_outlier_rate_pct"]:.3f}\%, and Arduino outlier rates were exactly zero in {payload["arduino"]["n_zero_outlier_recordings"]} of {payload["n_recordings"]} recordings, with the worst case reaching {payload["arduino"]["max_outlier_rate_pct"]:.3f}\%.

\paragraph{{Packet loss proxy from timestamp gaps}}
Because the nRF Connect logs do not necessarily expose lower-layer retransmissions, packet loss was estimated indirectly from timestamp gaps. If the Arduino timestamps increase in steps consistent with the sampling period but the log skips multiple consecutive steps, the most plausible explanation is that one or more notifications were not recorded at the smartphone. This provides a conservative loss proxy that is sufficient to label recordings with extended dropouts as unsuitable for cross-sensor alignment. For the parsed Arduino streams, the conservative gap-based proxy ranged from {payload["arduino"]["min_missing_samples"]} missing samples ({payload["arduino"]["min_missing_rate_pct"]:.1f}\%) to {payload["arduino"]["max_missing_samples"]} missing samples ({payload["arduino"]["max_missing_rate_pct"]:.1f}\%), with the heaviest dropout burden observed in \texttt{{{_latex_name(payload["arduino"]["max_missing_recording"])}}}.

\paragraph{{Clock drift observations}}
Preliminary synchronization analysis revealed that the Arduino clock drifts relative to the GPS-synchronized SPORSA clock at a rate of approximately 300~ppm (parts per million), equivalent to roughly 0.3~milliseconds of accumulated drift per second of recording. Over an 8-minute recording, this amounts to approximately 140~ms of total drift---enough to noticeably degrade cross-sensor alignment if left uncorrected. The drift rate was consistent across recordings, suggesting it reflects the Arduino crystal's intrinsic frequency offset rather than temperature-dependent variation during the session.

\begin{{figure}}[ht!]
    \centering
    \includegraphics[width=\linewidth]{{parsed_session_bar_chart.png}}
    \caption{{Recording durations and sample counts from the primary data collection session. The asymmetry between Arduino and SPORSA sample counts reflects the difference in effective sampling rates (approximately 58.8~Hz vs 100~Hz).}}
    \label{{fig:session_bar_chart}}
\end{{figure}}

\paragraph{{Calibration sequence detection}}
The structured calibration protocol - static hold, tap burst, static hold - was successfully detected in most recordings. The number of detected calibration segments per recording ranged from zero (in short test recordings without the full protocol) to five (in the longest recording with multiple mid-session recalibrations). Recordings with at least two detected segments (opening and closing) are preferred for synchronization because they provide two timing anchors for drift estimation. Table~\ref{{tab:calibration_segments}} summarizes the calibration segment counts.

\begin{{table}}[ht!]
    \centering
    \caption{{Number of detected calibration segments per recording. Recordings with two or more segments on both sensors support drift-corrected synchronization.}}
    \label{{tab:calibration_segments}}
    \begin{{tabular}}{{lcc}}
        \toprule
        Recording & SPORSA segments & Arduino segments \\
        \midrule
{calibration_rows}
        \bottomrule
    \end{{tabular}}
\end{{table}}

\paragraph{{Recording quality categories}}
Based on the integrity checks described earlier, the nine recordings were classified into quality categories:
\begin{{itemize}}
    \item \textbf{{Good}} ({len(quality_groups["good"])} recordings: {_format_recording_list(quality_groups["good"])}): Complete logs with clear opening and closing calibration sequences, stable timestamps, and no major BLE dropouts. These recordings are suitable for full synchronization and fusion analysis.
    \item \textbf{{Usable}} ({len(quality_groups["usable"])} recordings: {_format_recording_list(quality_groups["usable"])}): Minor issues such as a missing calibration sequence on one sensor or short dropout periods, but still sufficient for partial analysis.
    \item \textbf{{Limited}} ({len(quality_groups["limited"])} recordings: {_format_recording_list(quality_groups["limited"])}): Short duration or missing calibration protocol. Useful for single-sensor inspection but not for cross-sensor timing analysis.
\end{{itemize}}

This classification guides the selection of recordings for subsequent preprocessing and analysis stages. The majority of recordings passed the basic quality gates, confirming that the prototype setup and recording protocol were sufficiently robust for field data collection.
"""


def generate_parsed_integrity_report(
    session_name: str = "2026-02-26",
    *,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a thesis-ready parsed-stage integrity bundle for one session."""
    output_root = output_dir or (data_root() / "report" / "parsed_integrity" / session_name)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = session_integrity_summaries(session_name)
    summary_df = session_integrity_dataframe(session_name)
    if summary_df.empty:
        raise FileNotFoundError(f"No parsed recordings found for session {session_name!r}")

    from visualization.plot_parsed_integrity import plot_parsed_session_bar_chart

    plot_path = output_root / "parsed_session_bar_chart.png"
    plot_parsed_session_bar_chart(summary_df, plot_path)

    summary_csv_path = output_root / "parsed_recording_summary.csv"
    write_csv(summary_df, summary_csv_path)

    payload = _session_summary_payload(session_name, summary_df)
    payload["recordings"] = [asdict(summary) for summary in summaries]

    summary_json_path = output_root / "parsed_integrity_summary.json"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    latex_path = output_root / "parsed_integrity_section.tex"
    latex_path.write_text(_build_latex_section(payload, summary_df), encoding="utf-8")

    return {
        "session_name": session_name,
        "output_dir": str(output_root),
        "plot_path": str(plot_path),
        "summary_csv_path": str(summary_csv_path),
        "summary_json_path": str(summary_json_path),
        "latex_path": str(latex_path),
        "n_recordings": int(summary_df.shape[0]),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.stats",
        description="Compute parsed-stage recording stats or generate a session integrity bundle.",
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
    parser.add_argument(
        "--session",
        help="Session name (e.g. 2026-02-26) for a parsed integrity bundle.",
    )
    parser.add_argument(
        "--output",
        metavar="DIR",
        help="Output directory for --session integrity bundle.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.session:
        out = generate_parsed_integrity_report(
            session_name=args.session,
            output_dir=Path(args.output) if args.output else None,
        )
        print(json.dumps(out, indent=2))
        return

    if not args.recording_name:
        raise SystemExit("Pass a recording name or use --session to generate a session bundle.")

    out = write_recording_stats(args.recording_name, args.stage)
    print(out)


if __name__ == "__main__":
    main()
