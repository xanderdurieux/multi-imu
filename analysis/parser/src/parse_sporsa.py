"""
Convert a raw Sporsa IMU log to a normalized CSV.

Output CSV columns (in order):
timestamp, ax, ay, az, gx, gy, gz, mx, my, mz

"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Optional

from .parse_common import IMUSample, write_csv

ACCEL_SENS = {
    "4G": 0.122,
    "8G": 0.244,
    "16G": 0.488,
}

GYRO_SENS = {
    "125DPS": 4.375,
    "250DPS": 8.750,
    "500DPS": 17.500,
    "1000DPS": 35.000,
    "2000DPS": 70.000,
}


def _parse_sporsa_line(line: str) -> Optional[IMUSample]:
    """
    Parse a single line from a Sporsa log file.

    Expected format (txt):
        <timestamp_ms>,<acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<gyro_z>
    """
    parts = line.strip().split(",")
    if len(parts) != 7:
        return None

    ts_str = parts[0].replace("uart:~$ ", "")
    try:
        device_time_ms = int(ts_str)
    except ValueError:
        return None

    try:
        acc_x_raw = int(parts[1])
        acc_y_raw = int(parts[2])
        acc_z_raw = int(parts[3])

        gyro_x_raw = int(parts[4])
        gyro_y_raw = int(parts[5])
        gyro_z_raw = int(parts[6])
    except ValueError:
        return None

    acc_x = acc_x_raw * ACCEL_SENS["16G"] * 9.81 / 1000
    acc_y = acc_y_raw * ACCEL_SENS["16G"] * 9.81 / 1000
    acc_z = acc_z_raw * ACCEL_SENS["16G"] * 9.81 / 1000

    gyro_x = gyro_x_raw * GYRO_SENS["2000DPS"] / 1000
    gyro_y = gyro_y_raw * GYRO_SENS["2000DPS"] / 1000
    gyro_z = gyro_z_raw * GYRO_SENS["2000DPS"] / 1000

    return IMUSample(
        timestamp=device_time_ms,
        ax=acc_x,
        ay=acc_y,
        az=acc_z,
        gx=gyro_x,
        gy=gyro_y,
        gz=gyro_z,
    )


def parse_sporsa_log(txt_path: Path) -> Iterator[IMUSample]:
    """Yield IMUSample objects parsed from a raw Sporsa log file."""
    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            sample = _parse_sporsa_line(raw_line)
            if sample is not None:
                yield sample


def main(argv: Optional[list[str]] = None) -> None:
    """
    Usage (from the analysis/parser/src/ directory):

        python3 parse_sporsa.py <source_txt> <destination_csv>

    Or from the repo root:

        python3 analysis/parser/src/parse_sporsa.py <source_txt> <destination_csv>
        python3 -m analysis.parser.src.parse_sporsa <source_txt> <destination_csv>
    """

    if argv is None:
        argv = sys.argv[1:]

    if len(argv) != 2:
        print("Usage: python3 parse_sporsa.py <source_txt> <destination_csv>")
        return

    txt_path = Path(argv[0])
    csv_path = Path(argv[1])

    samples = parse_sporsa_log(txt_path)
    write_csv(samples, csv_path)


if __name__ == "__main__":
    main()

