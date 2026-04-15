"""Parser for raw Sporsa IMU log files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import re

from common.paths import CSV_COLUMNS, write_csv
from common.signals import add_imu_norms

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

# Magnetometer: datasheet sensitivity is 1.5 milligauss per LSB (16-bit counts).
# 1 Gauss = 100 µT  =>  1 milligauss = 0.1 µT.
MAG_UT_PER_LSB = 1.5 * 0.1  # 0.15 µT per raw count


_SPORSA_LINE_RE_FULL = re.compile(
    r"""^\s*
    (?:uart:~\$\s*)?              # optional REPL-style prefix
    (?P<ts>-?\d+)\s*,\s*          # device timestamp (ms)
    (?P<ax>-?\d+)\s*,\s*
    (?P<ay>-?\d+)\s*,\s*
    (?P<az>-?\d+)\s*,\s*
    (?P<gx>-?\d+)\s*,\s*
    (?P<gy>-?\d+)\s*,\s*
    (?P<gz>-?\d+)\s*,\s*
    (?P<mx>-?\d+)\s*,\s*
    (?P<my>-?\d+)\s*,\s*
    (?P<mz>-?\d+)\s*              # magnetometer z
    \s*$                          # allow trailing whitespace but no extra fields
    """,
    re.VERBOSE,
)

_SPORSA_LINE_RE_NO_MAG = re.compile(
    r"""^\s*
    (?:uart:~\$\s*)?              # optional REPL-style prefix
    (?P<ts>-?\d+)\s*,\s*          # device timestamp (ms)
    (?P<ax>-?\d+)\s*,\s*
    (?P<ay>-?\d+)\s*,\s*
    (?P<az>-?\d+)\s*,\s*
    (?P<gx>-?\d+)\s*,\s*
    (?P<gy>-?\d+)\s*,\s*
    (?P<gz>-?\d+)\s*              # no magnetometer fields in this format
    \s*$
    """,
    re.VERBOSE,
)


def _parse_sporsa_line(line: str) -> Optional[dict]:
    """
    Parse a single line from a Sporsa log file.

    Expected format (txt):
        <timestamp_ms>,<acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<gyro_z>,<mag_x>,<mag_y>,<mag_z>
    Optionally prefixed with "uart:~$ ".
    Any line that does not match this pattern is ignored.
    """
    if not line:
        return None

    m = _SPORSA_LINE_RE_FULL.match(line)
    has_mag = True
    if not m:
        m = _SPORSA_LINE_RE_NO_MAG.match(line)
        has_mag = False
        if not m:
            return None

    try:
        device_time_ms = int(m.group("ts"))
        acc_x_raw = int(m.group("ax"))
        acc_y_raw = int(m.group("ay"))
        acc_z_raw = int(m.group("az"))
        gyro_x_raw = int(m.group("gx"))
        gyro_y_raw = int(m.group("gy"))
        gyro_z_raw = int(m.group("gz"))
        mag_x_raw = int(m.group("mx")) if has_mag else None
        mag_y_raw = int(m.group("my")) if has_mag else None
        mag_z_raw = int(m.group("mz")) if has_mag else None
    except (ValueError, OverflowError):
        # If any numeric field is malformed, skip this line.
        return None

    acc_x = acc_x_raw * ACCEL_SENS["16G"] * 9.81 / 1000
    acc_y = acc_y_raw * ACCEL_SENS["16G"] * 9.81 / 1000
    acc_z = acc_z_raw * ACCEL_SENS["16G"] * 9.81 / 1000

    gyro_x = gyro_x_raw * GYRO_SENS["2000DPS"] / 1000
    gyro_y = gyro_y_raw * GYRO_SENS["2000DPS"] / 1000
    gyro_z = gyro_z_raw * GYRO_SENS["2000DPS"] / 1000

    mag_x = mag_x_raw * MAG_UT_PER_LSB if mag_x_raw is not None else pd.NA
    mag_y = mag_y_raw * MAG_UT_PER_LSB if mag_y_raw is not None else pd.NA
    mag_z = mag_z_raw * MAG_UT_PER_LSB if mag_z_raw is not None else pd.NA

    row = {col: pd.NA for col in CSV_COLUMNS}
    row["timestamp"] = device_time_ms
    row["ax"] = acc_x
    row["ay"] = acc_y
    row["az"] = acc_z
    row["gx"] = gyro_x
    row["gy"] = gyro_y
    row["gz"] = gyro_z
    row["mx"] = mag_x
    row["my"] = mag_y
    row["mz"] = mag_z
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

    # Heuristic: Drop all rows whose timestamps do not fall within the expected epoch
    # window for the specific recording, based on the folder name (date).
    # We expect timestamp (ms since 1970) to be within ±1 day of the folder date.
    import datetime

    # Extract the recording date from the folder (requires txt_path or a provided recording date).
    # Assume txt_path: <...>/<recording_name>/<stage>/sporsa.txt
    # And <recording_name>: YYYY-MM-DD_X
    try:
        recording_folder = txt_path.parent.parent.name
        date_str = recording_folder.split('_')[0]
        rec_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        # Epoch (ms) for midnight UTC of that recording day
        date_start_ms = int(rec_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        date_end_ms = date_start_ms + 24 * 60 * 60 * 1000
        ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
        # Accept timestamps within ±1 day (2 days window—sometimes logs deviate a bit)
        # (Strict matching: date_start_ms to date_end_ms, or lenient: ±1 day)
        window_start = date_start_ms - 24 * 60 * 60 * 1000
        window_end = date_end_ms + 24 * 60 * 60 * 1000
        if ((ts_numeric >= window_start) & (ts_numeric <= window_end)).any():
            df = df[(ts_numeric >= window_start) & (ts_numeric <= window_end)].copy()
    except Exception:
        # If any error, skip heuristic and do not filter
        pass

    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[CSV_COLUMNS]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in CSV_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return add_imu_norms(df)


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
    write_csv(df, args.destination_csv)


if __name__ == "__main__":
    main()
