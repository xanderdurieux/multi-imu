"""
High-level parser for IMU sessions in the ../data folder.

Directory layout (relative to this file):

    ../data/
        raw/
            session_1/
                arduino_imu.txt
                sporsa_imu.txt
                ...
            session_2/
                ...
        processed/
            session_1/
                arduino_imu.csv
                sporsa_imu.csv
                ...

Usage (from the repo root):

    python3 analysis/parse_session.py <session_name>

Example:

    python3 analysis/parse_session.py session_1

This will read from:
    data/raw/<session_name>/
and write CSVs to:
    data/processed/<session_name>/

File naming:
- All files in the raw session folder are inspected.
- Files whose name contains "arduino" are parsed with the Arduino parser.
- Files whose name contains "sporsa"  are parsed with the Sporsa parser.
- Output CSVs are named "<raw_stem>.csv" in the processed folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

try:
    # When run as a module: python3 -m analysis.parse_session
    from analysis.src.parse_arduino import parse_arduino_log
    from analysis.src.parse_sporsa import parse_sporsa_log
    from analysis.src.imu_common import IMUSample, write_csv
except ImportError:
    # When run directly as a script: python3 analysis/parse_session.py
    from src.parse_arduino import parse_arduino_log
    from src.parse_sporsa import parse_sporsa_log
    from src.imu_common import IMUSample, write_csv


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
    here = Path(__file__).resolve().parent
    data_root = here.parent / "data"

    raw_dir = data_root / "raw" / session_name
    out_dir = data_root / "processed" / session_name

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
    """
    CLI entry point.

    Usage:
        python3 analysis/parse_session.py <session_name>
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 1:
        print("Usage: python3 analysis/parse_session.py <session_name>")
        return

    session_name = argv[0]
    process_session(session_name)


if __name__ == "__main__":
    main()


