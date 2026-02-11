"""
High-level parser for IMU sessions in the data folder.

This file orchestrates the conversion of all raw IMU log files in a given session folder
(under data/raw/<session_name>/) into normalized CSV files (under data/processed/<session_name>/).
It detects each log file type, selects the appropriate parser, and writes standardized CSVs
for further analysis.

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from common import write_csv, raw_session_dir, processed_session_dir
from .arduino import parse_arduino_log
from .sporsa import parse_sporsa_log


def _classify_file(path: Path) -> Optional[str]:
    """
    Return a sensor type string ("arduino" or "sporsa") based on file name,
    or None if the file should be ignored.
    """
    name = path.stem.lower()
    if "arduino" in name:
        return "arduino"
    if "sporsa" in name:
        return "sporsa"
    return None


def _process_file(sensor_type: str, src: Path, dst: Path) -> None:
    if sensor_type == "arduino":
        samples = parse_arduino_log(src)
    elif sensor_type == "sporsa":
        samples = parse_sporsa_log(src)
    else:
        return

    write_csv(samples, dst)


def process_session(session_name: str) -> None:
    """
    Parse all known sensor logs for a given session and write CSVs.
    """
    raw_dir = raw_session_dir(session_name)
    out_dir = processed_session_dir(session_name)

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
        _process_file(sensor_type, src, dst)


def main(argv: Optional[list[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        print("Usage: python3 -m parser.session <session_name>")
        return

    session_name = argv[0]
    process_session(session_name)


if __name__ == "__main__":
    main()


