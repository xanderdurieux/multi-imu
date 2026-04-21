"""High-level parser for converting raw IMU logs to processed CSV streams."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Optional

from common import (
    recording_stage_dir,
    read_csv,
    session_input_dir,
    write_csv,
)
from common.paths import project_relative_path

from visualization.plot_calibration_segments import (
    plot_calibration_segments_from_detection,
)
from visualization.plot_sensor import plot_sensor_data

from .calibration_segments import (
    export_calibration_segments_json,
    find_calibration_segments,
    load_cal_seg_params,
)
from .arduino import parse_arduino_log
from .sporsa import parse_sporsa_log
from .stats import write_recording_stats

log = logging.getLogger(__name__)


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

    write_csv(df, dst)
    return df


def _extract_recording_number(filename: str) -> Optional[int]:
    """Extract recording number from a filename."""
    lower = filename.lower()
    m = re.search(r"(?:session|log)\s*(\d+)", lower)
    if m:
        return int(m.group(1))

    nums = re.findall(r"\d+", lower)
    if nums:
        return int(nums[-1])

    return None


def process_session(session_name: str, *, plot: bool = True) -> None:
    """Parse a session and write processed CSV streams + calibration segments + plots."""
    in_root = session_input_dir(session_name)
    if not in_root.is_dir():
        log.warning("Sessions input folder not found: %s", project_relative_path(in_root))
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
                log.info("parse %s: skipping file without recording number: %s", session_name, src.name)
                continue
            recordings.setdefault(n, {})
            if sensor in recordings[n]:
                log.warning(
                    "parse %s: duplicate %s for recording %s, ignoring %s",
                    session_name, sensor, n, src.name,
                )
                continue
            recordings[n][sensor] = src

    if not recordings:
        log.warning("parse %s: no recordings found under %s", session_name, project_relative_path(in_root))
        return

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
            log.info(
                "parse %s/parsed: %s source=%s output=%s",
                recording_name,
                sensor,
                src.name,
                project_relative_path(dst),
            )
            df = _process_file(sensor, src, dst)
            parsed_nonempty[sensor] = bool(df is not None and not df.empty)

            if plot:
                plot_sensor_data(dst, output_path=out_dir / f"{sensor}.png")

        # Calibration-segment detection and JSON export.
        cal_json_payload: dict = {"recording_name": recording_name, "sensors": {}}
        for sensor_name in ("sporsa", "arduino"):
            csv_path = out_dir / f"{sensor_name}.csv"
            if not csv_path.exists():
                continue
            df_sensor = read_csv(csv_path)
            if df_sensor.empty:
                continue

            df_work = (
                df_sensor.dropna(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if df_work.empty:
                continue

            params = load_cal_seg_params(sensor_name)
            segments = find_calibration_segments(df_work, sensor=sensor_name)

            if plot:
                plot_calibration_segments_from_detection(
                    df_work,
                    segments,
                    out_path=out_dir / f"{sensor_name}_cal_segments.png",
                    sensor=sensor_name,
                )

            export_calibration_segments_json(
                recording_name, sensor_name, segments, params,
                existing=cal_json_payload,
            )
            log.info(
                "parse %s: wrote calibration segments for %s (%d found)",
                recording_name, sensor_name, len(segments),
            )

        # Per-recording timing stats — written after calibration_segments.json so
        # segment counts are available for quality classification.
        stats_path = write_recording_stats(recording_name)
        log.info("parse %s: wrote stats %s", recording_name, project_relative_path(stats_path))

        if plot:
            from visualization.plot_parsed_stats import (
                plot_interval_distribution,
                plot_packet_loss_received,
                plot_timestamp_continuity,
            )
            dfs = {
                s: read_csv(out_dir / f"{s}.csv")
                for s in ("sporsa", "arduino")
                if (out_dir / f"{s}.csv").exists()
            }
            if dfs:
                plot_timestamp_continuity(dfs, out_dir / "parsed_timing.png")
                plot_interval_distribution(dfs, out_dir / "parsed_intervals.png")
            if "arduino" in dfs and "timestamp_received" in dfs["arduino"].columns:
                plot_packet_loss_received(dfs["arduino"], out_dir / "parsed_packet_loss.png")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser",
        description="Parse all recordings for a session date into CSVs, plots, and stats.",
    )
    parser.add_argument(
        "session_name",
        help="Date folder under data/_sessions/ (e.g. 2026-02-26).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating parsed-stage diagnostic plots.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    process_session(args.session_name, plot=not args.no_plot)


if __name__ == "__main__":
    main()
