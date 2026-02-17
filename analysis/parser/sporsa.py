"""Parser for raw Sporsa IMU log files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from common import CSV_COLUMNS, write_dataframe

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


def _parse_sporsa_line(line: str) -> Optional[dict]:
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

    row = {col: pd.NA for col in CSV_COLUMNS}
    row["timestamp"] = device_time_ms
    row["ax"] = acc_x
    row["ay"] = acc_y
    row["az"] = acc_z
    row["gx"] = gyro_x
    row["gy"] = gyro_y
    row["gz"] = gyro_z
    return row


def parse_sporsa_log(txt_path: Path) -> pd.DataFrame:
    """Parse a raw Sporsa log file into a standardized IMU DataFrame."""
    rows: list[dict] = []
    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            row = _parse_sporsa_line(raw_line)
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.DataFrame(rows)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[CSV_COLUMNS]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in CSV_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Sporsa log conversion."""
    parser = argparse.ArgumentParser(
        prog="python -m parser.sporsa",
        description="Convert a raw Sporsa log to processed IMU CSV format.",
    )
    parser.add_argument("source_txt", type=Path, help="Path to raw Sporsa text log.")
    parser.add_argument("destination_csv", type=Path, help="Output CSV path.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    df = parse_sporsa_log(args.source_txt)
    write_dataframe(df, args.destination_csv)


if __name__ == "__main__":
    main()
