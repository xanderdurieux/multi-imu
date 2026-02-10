"""
Convert a raw Arduino BLE IMU log to a normalized CSV.

Output CSV columns (in order):
timestamp, ax, ay, az, gx, gy, gz, mx, my, mz

This script intentionally avoids an argument parser; edit the paths in `main()`.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path
from typing import Iterator, Optional

try:
    # Preferred when installed / run as module:
    #   python3 -m analysis.parser.src.parse_arduino
    from analysis.parser.src.parse_common import IMUSample, write_csv
except ImportError:
    # Fallback when run from within src/:
    from parse_common import IMUSample, write_csv

GRAVITY = 9.81


def _parse_arduino_line(line: str) -> Optional[IMUSample]:
    """
    Parse a single line from the Arduino log produced by the BLE central.

    Expected pattern (simplified):
        A<TAB><something><TAB><hex bytes ...>

    Only lines starting with "A" and containing hex payloads are considered.
    """
    if not line.startswith("A"):
        return None

    parts = line.split("\t")
    if len(parts) < 3:
        return None

    data_str = parts[2].replace("received", "")
    if "Notifications" in data_str:
        return None

    hex_bytes = re.findall(r"[0-9A-Fa-f]{2}", data_str)
    if not hex_bytes:
        return None

    raw = bytes(int(h, 16) for h in hex_bytes)

    # Data layout matches the SensorData struct on the Arduino:
    #   uint8_t sensorType;
    #   float x;
    #   float y;
    #   float z;
    #   uint32_t timestamp;
    try:
        sensor_type, x, y, z, sensor_time_ms = struct.unpack("<BfffI", raw[:17])
    except struct.error:
        # Fallback: older 20‑byte layout with 3 padding bytes after the type.
        try:
            sensor_type, x, y, z, sensor_time_ms = struct.unpack("<B3xfffI", raw[:20])
        except struct.error:
            return None

    if sensor_type == 1:  # accelerometer
        return IMUSample(
            timestamp=sensor_time_ms,
            ax=x * GRAVITY,
            ay=y * GRAVITY,
            az=z * GRAVITY,
        )
    if sensor_type == 2:  # gyroscope
        return IMUSample(
            timestamp=sensor_time_ms,
            gx=x,
            gy=y,
            gz=z,
        )
    if sensor_type == 4:  # magnetometer
        return IMUSample(
            timestamp=sensor_time_ms,
            mx=x,
            my=y,
            mz=z,
        )

    return None


def parse_arduino_log(txt_path: Path) -> Iterator[IMUSample]:
    """Yield IMUSample objects parsed from a raw Arduino log file."""
    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            sample = _parse_arduino_line(raw_line)
            if sample is not None:
                yield sample


def main(argv: Optional[list[str]] = None) -> None:
    """
    Usage (from the analysis/src/ directory):

        python3 parse_arduino.py <source_txt> <destination_csv>

    Or from the repo root:

        python3 analysis/src/parse_arduino.py <source_txt> <destination_csv>
        python3 -m analysis.src.parse_arduino <source_txt> <destination_csv>
    """

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("Usage: python3 parse_arduino.py <source_txt> <destination_csv>")
        return

    txt_path = Path(argv[0])
    csv_path = Path(argv[1])

    samples = parse_arduino_log(txt_path)
    write_csv(samples, csv_path)


if __name__ == "__main__":
    main()
