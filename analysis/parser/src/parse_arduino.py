"""
Convert a raw Arduino BLE IMU log to a normalized CSV.

Output CSV columns (in order):
timestamp, ax, ay, az, gx, gy, gz, mx, my, mz

"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path
from typing import Dict, Iterator, Optional

from .parse_common import IMUSample, write_csv

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
    sensor_type, x, y, z, sensor_time_ms = struct.unpack("<B3xfffI", raw)

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


def _merge_samples(a: IMUSample, b: IMUSample) -> IMUSample:
    """Merge two IMUSample objects with the same timestamp."""
    if a.timestamp != b.timestamp:
        # Should not happen in normal flow; prefer b's timestamp if it does.
        ts = b.timestamp
    else:
        ts = a.timestamp

    return IMUSample(
        timestamp=ts,
        ax=b.ax if b.ax is not None else a.ax,
        ay=b.ay if b.ay is not None else a.ay,
        az=b.az if b.az is not None else a.az,
        gx=b.gx if b.gx is not None else a.gx,
        gy=b.gy if b.gy is not None else a.gy,
        gz=b.gz if b.gz is not None else a.gz,
        mx=b.mx if b.mx is not None else a.mx,
        my=b.my if b.my is not None else a.my,
        mz=b.mz if b.mz is not None else a.mz,
    )


def parse_arduino_log(txt_path: Path) -> Iterator[IMUSample]:
    """
    Yield IMUSample objects parsed from a raw Arduino log file.

    Multiple packets with the same timestamp (acc/gyro/mag) are merged into
    a single row so that each timestamp appears only once in the CSV.
    """
    by_timestamp: Dict[int, IMUSample] = {}

    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            sample = _parse_arduino_line(raw_line)
            if sample is None:
                continue

            existing = by_timestamp.get(sample.timestamp)
            if existing is None:
                by_timestamp[sample.timestamp] = sample
            else:
                by_timestamp[sample.timestamp] = _merge_samples(existing, sample)

    for ts in sorted(by_timestamp.keys()):
        yield by_timestamp[ts]


def main(argv: Optional[list[str]] = None) -> None:
    """
    Usage (from the analysis/parser/src/ directory):

        python3 parse_arduino.py <source_txt> <destination_csv>

    Or from the repo root:

        python3 analysis/parser/src/parse_arduino.py <source_txt> <destination_csv>
        python3 -m analysis.parser.src.parse_arduino <source_txt> <destination_csv>
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
