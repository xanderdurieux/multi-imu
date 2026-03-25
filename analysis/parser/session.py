"""High-level parser for converting raw IMU logs to processed CSV streams."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Optional
import json

from common import (
    load_dataframe,
    recording_dir,
    recording_stage_dir,
    session_input_dir,
    write_dataframe,
)

from visualization.plot_session import plot_recording
from visualization.plot_calibration_segments import (
    plot_calibration_segments_from_detection,
)

from .arduino import parse_arduino_log
from .sporsa import parse_sporsa_log
from .stats import compute_stream_timing_stats, write_recording_stats


def _process_file(sensor_type: str, src: Path, dst: Path):
    """Dispatch one raw log file to the matching parser and write CSV output.

    Returns the parsed DataFrame on success, or None if the file is skipped.
    """
    if sensor_type == "arduino":
        df = parse_arduino_log(src)
    elif sensor_type == "sporsa":
        df = parse_sporsa_log(src)
    else:
        return None

    write_dataframe(df, dst)
    return df


def _extract_recording_number(filename: str) -> Optional[int]:
    """Extract recording number from a filename.

    Supported patterns (case-insensitive):
    - ``...session10...``
    - ``log3...``
    Falls back to the last integer found in the filename.
    """
    lower = filename.lower()
    m = re.search(r"(?:session|log)\s*(\d+)", lower)
    if m:
        return int(m.group(1))

    nums = re.findall(r"\d+", lower)
    if nums:
        return int(nums[-1])

    return None


def process_session(session_name: str) -> None:
    """Parse a session and write processed CSV streams + plots + stats.

    Input layout::

        data/sessions/<session_name>/{arduino,sporsa}/*.txt

    Output layout (one folder per recording number)::

        data/recordings/<session_name>_r<N>/
            session_stats.json                 # per-recording stream timing stats
            parsed/
                sporsa.csv
                arduino.csv
                *.png                          # per-recording sensor/comparison plots
                <sensor>_calibration_segments.png

    Additionally, a session-wide summary JSON (no file paths) is written to::

        data/sessions/<session_name>/session_stats.json

    This aggregates per-recording parsed durations, section statistics, and
    calibration-segment summaries for each sensor.
    """
    in_root = session_input_dir(session_name)
    if not in_root.is_dir():
        print(f"Sessions input folder not found: {in_root}")
        return

    sensor_dirs = {
        "arduino": in_root / "arduino",
        "sporsa": in_root / "sporsa",
    }

    recordings: dict[int, dict[str, Path]] = {}
    for sensor, sdir in sensor_dirs.items():
        if not sdir.is_dir():
            continue
        for src in sorted(sdir.glob("*.txt")):
            n = _extract_recording_number(src.name)
            if n is None:
                print(f"[{session_name}] skipping (no recording nr): {src.name}")
                continue
            recordings.setdefault(n, {})
            if sensor in recordings[n]:
                print(f"[{session_name}] warning: duplicate {sensor} for rec {n}: {src.name}")
                continue
            recordings[n][sensor] = src

    if not recordings:
        print(f"[{session_name}] No recordings found under {in_root}")
        return

    session_summaries: list[dict] = []

    for n in sorted(recordings.keys()):
        recording_name = f"{session_name}_r{n}"
        out_dir = recording_stage_dir(recording_name, "parsed")
        out_dir.mkdir(parents=True, exist_ok=True)

        pair = recordings[n]
        parsed_nonempty: dict[str, bool] = {}
        for sensor in ("sporsa", "arduino"):
            src = pair.get(sensor)
            if src is None:
                continue
            dst = out_dir / f"{sensor}.csv"
            print(f"[{recording_name}/parsed] {sensor}: {src.name} -> {dst.name}")
            df = _process_file(sensor, src, dst)
            parsed_nonempty[sensor] = bool(df is not None and not df.empty)

        # Per-recording timing stats at recording root (see write_recording_stats).
        stats_path = write_recording_stats(recording_name)
        print(f"[{recording_name}] stats: {stats_path.name}")

        # Per-recording sensor/comparison plots for the parsed stage.
        plot_recording(recording_name, stage_filter="parsed")

        # Calibration-segment diagnostic plots per sensor (sporsa, arduino) and
        # compact, path-free summaries for the session stats JSON.
        calibration_segments_summary: dict[str, dict] = {}
        for sensor_name in ("sporsa", "arduino"):
            csv_path = out_dir / f"{sensor_name}.csv"
            if not csv_path.exists():
                continue
            df_sensor = load_dataframe(csv_path)
            if df_sensor.empty:
                continue

            # Use slightly more permissive defaults for the diagnostic plots so
            # that short or early calibration sequences (e.g. at the very start
            # of the recording) are still visualised.
            _, info_df, _ = plot_calibration_segments_from_detection(
                df_sensor,
                sample_rate_hz=100.0,
                static_min_s=2.0,
                static_threshold=1.5,
                peak_min_height=2.5,
                peak_min_count=5,
                peak_max_count=20,
                peak_max_gap_s=3.0,
                static_gap_max_s=8.0,
                out_path=out_dir / f"{sensor_name}_calibration_segments.png",
            )

            if info_df.empty or "segment_index" not in info_df.columns:
                calibration_segments_summary[sensor_name] = {
                    "num_segments": 0,
                    "total_duration_s": 0.0,
                    "segments": [],
                }
                continue

            # Per-segment timing and peak summaries.
            segments_list: list[dict] = []
            total_duration_s = 0.0
            for _, row in info_df.iterrows():
                start_t = float(row.get("start_time_s", 0.0))
                end_t = float(row.get("end_time_s", start_t))
                duration_s = max(0.0, end_t - start_t)
                total_duration_s += duration_s

                seg = {
                    "index": int(row.get("segment_index", 0)),
                    "start_time_s": start_t,
                    "end_time_s": end_t,
                    "duration_s": duration_s,
                    "n_peaks": int(row.get("n_peaks", 0)),
                }
                segments_list.append(seg)

            calibration_segments_summary[sensor_name] = {
                "num_segments": len(segments_list),
                "total_duration_s": total_duration_s,
                "segments": segments_list,
            }

        # Number of sections already created for this recording (if any).
        from common.paths import iter_sections_for_recording

        section_dirs = iter_sections_for_recording(recording_name)
        n_sections = len(section_dirs)

        # Extract a compact per-recording summary for the session-wide JSON.
        stats_json = {}
        try:
            stats_json = json.loads(stats_path.read_text(encoding="utf-8"))
        except Exception:
            stats_json = {}

        durations_s: dict[str, float | None] = {}
        num_samples: dict[str, int | None] = {}
        streams = stats_json.get("streams", {}) or {}
        for stream_name, stream_stats in streams.items():
            timing = stream_stats.get("timing", {}) or {}
            durations_s[stream_name] = timing.get("duration_s")
            num_samples[stream_name] = stream_stats.get("num_samples")

        rec_summary: dict = {
            "recording_name": recording_name,
            "parsed": {
                "durations_s": durations_s,
                "num_samples": num_samples,
            },
            "sections": {
                "count": n_sections,
            },
        }
        if calibration_segments_summary:
            rec_summary["calibration_segments"] = calibration_segments_summary

        session_summaries.append(rec_summary)

        # Optional section-level timing summaries (if sections already exist).
        if n_sections > 0:
            section_details: list[dict] = []
            from common.paths import parse_section_folder_name, section_id_for_idx

            for section_dir in section_dirs:
                section_streams: dict[str, dict] = {}
                for sensor_name in ("sporsa", "arduino"):
                    csv_path = section_dir / f"{sensor_name}.csv"
                    if not csv_path.exists():
                        continue
                    df_sec = load_dataframe(csv_path)
                    if df_sec.empty:
                        continue
                    timing = compute_stream_timing_stats(df_sec, timestamp_col="timestamp")
                    section_streams[sensor_name] = {
                        "num_samples": int(df_sec.shape[0]),
                        "timing": timing,
                    }

                if section_streams:
                    _rec_name, sec_idx = parse_section_folder_name(section_dir.name)
                    section_details.append(
                        {
                            # Keep legacy section id in the JSON summary (e.g. "section_1").
                            "name": section_id_for_idx(sec_idx),
                            "streams": section_streams,
                        }
                    )

            if section_details:
                rec_summary["sections"]["details"] = section_details

    # Write session-wide summary JSON across all recordings (without paths).
    session_stats = {
        "session_name": session_name,
        "num_recordings": len(session_summaries),
        "recordings": session_summaries,
    }
    out_session_stats = in_root / "session_stats.json"
    out_session_stats.write_text(
        json.dumps(session_stats, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"[{session_name}] session stats: sessions/{session_name}/{out_session_stats.name}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.session",
        description="Parse all recordings for a session date into CSVs, plots, and stats.",
    )
    parser.add_argument(
        "session_name",
        help="Date folder under data/sessions/ (e.g. 2026-02-26).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    process_session(args.session_name)


if __name__ == "__main__":
    main()
