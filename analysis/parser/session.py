"""High-level parser for converting raw IMU logs to processed CSV streams."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from common import parsed_session_dir, raw_session_dir, write_dataframe

from .arduino import parse_arduino_log
from .sporsa import parse_sporsa_log
from .stats import write_session_stats


def _classify_file(path: Path) -> Optional[str]:
    """
    Return a sensor type string ("arduino" or "sporsa") based on file name,
    or None if the file should be ignored.
    """
    name = path.stem.lower()
    if "arduino" or "Log" in name:
        return "arduino"
    if "sporsa" in name:
        return "sporsa"
    return None


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


def process_session(session_name: str) -> None:
    """Parse all raw sensor logs for a session and write processed CSV streams + stats."""
    raw_dir = raw_session_dir(session_name)
    out_dir = parsed_session_dir(session_name)

    if not raw_dir.is_dir():
        print(f"Raw session folder not found: {raw_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sorted(raw_dir.iterdir()):
        if not src.is_file():
            continue

        sensor_type = _classify_file(src)
        if sensor_type is None:
            continue

        dst = out_dir / f"{src.stem}.csv"
        print(f"[{session_name}] {sensor_type}: {src.name} -> {dst.name}")
        df = _process_file(sensor_type, src, dst)
        if df is None:
            continue

    stats_path = write_session_stats(session_name)
    print(f"[{session_name}] stats: {stats_path.name}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for session processing."""
    parser = argparse.ArgumentParser(
        prog="python -m parser.session",
        description="Parse all recognized raw logs for one session into processed CSVs.",
    )
    parser.add_argument("session_name", help="Session folder under data/raw.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    process_session(args.session_name)


if __name__ == "__main__":
    main()

