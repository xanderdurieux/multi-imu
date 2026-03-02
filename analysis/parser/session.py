"""High-level parser for converting raw IMU logs to processed CSV streams."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Optional

from common import recording_stage_dir, session_input_dir, write_dataframe

from visualization.plot_session import plot_recording

from .arduino import parse_arduino_log
from .sporsa import parse_sporsa_log
from .stats import write_recording_stats


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

        data/recordings/<session_name>_<N>/parsed/
            sporsa.csv
            arduino.csv
            session_stats.json
            *.png
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

    for n in sorted(recordings.keys()):
        recording_name = f"{session_name}_{n}"
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

        stats_path = write_recording_stats(recording_name)
        print(f"[{recording_name}/parsed] stats: {stats_path.name}")

        plot_recording(recording_name, stage_filter="parsed")


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
