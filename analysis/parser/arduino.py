"""Parser for raw Arduino BLE IMU log files."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import struct
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from common import CSV_COLUMNS, write_dataframe

GRAVITY = 9.81


def _parse_log_date(line: str) -> dt.date | None:
    """
    Parse the log date line written by nRF Connect
    """
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", line)
    if not m:
        return None
    year, month, day = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _parse_time_of_day_ms(time_str: str) -> int | None:
    """Parse 'HH:MM:SS.mmm' and return milliseconds since midnight."""
    try:
        t = dt.datetime.strptime(time_str, "%H:%M:%S.%f").time()
    except ValueError:
        return None
    return (t.hour * 3600 + t.minute * 60 + t.second) * 1000 + (t.microsecond // 1000)


def _parse_arduino_line(line: str) -> Optional[tuple[int, int, float, float, float]]:
    """
    Parse a single line from the Arduino log produced by the BLE central.

    Expected pattern (simplified):
        A<TAB><received timestamp><TAB><hex bytes ...>

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
    fmt = "<B3xfffI"
    if len(raw) != struct.calcsize(fmt):
        return None
    sensor_type, x, y, z, sensor_time_ms = struct.unpack(fmt, raw)

    return int(sensor_time_ms), int(sensor_type), float(x), float(y), float(z)


def parse_arduino_log(txt_path: Path) -> pd.DataFrame:
    """
    Parse a raw Arduino log file into a standardized IMU DataFrame.

    Multiple packets with the same timestamp (acc/gyro/mag) are merged into a
    single row so that each timestamp appears only once in the output.
    """
    # timestamp_ms -> row dict with standardized columns
    by_timestamp: Dict[int, dict] = {}

    log_date: dt.date | None = None
    day_offset = 0
    last_tod_ms: int | None = None

    with txt_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            if log_date is None:
                log_date = _parse_log_date(raw_line)

            parsed = _parse_arduino_line(raw_line)
            if parsed is None:
                continue
            ts_ms, sensor_type, x, y, z = parsed

            # Extract the host "received" time-of-day timestamp from the same line.
            received_epoch_ms: int | None = None
            parts = raw_line.split("\t")
            if log_date is not None and len(parts) >= 2:
                tod_ms = _parse_time_of_day_ms(parts[1].strip())
                if tod_ms is not None:
                    if last_tod_ms is not None and tod_ms + 1000 < last_tod_ms:
                        # Time-of-day went backwards significantly => crossed midnight.
                        day_offset += 1
                    last_tod_ms = tod_ms

                    d = log_date + dt.timedelta(days=day_offset)
                    # Build datetime in local time (mktime applies TZ + DST rules for that date).
                    # (Re-parse time properly to keep microseconds).
                    t = dt.datetime.strptime(parts[1].strip(), "%H:%M:%S.%f")
                    received_local = dt.datetime(
                        d.year, d.month, d.day, t.hour, t.minute, t.second, t.microsecond
                    )
                    received_epoch_ms = int(time.mktime(received_local.timetuple()) * 1000) + (
                        received_local.microsecond // 1000
                    )

            row = by_timestamp.get(ts_ms)
            if row is None:
                row = {col: pd.NA for col in CSV_COLUMNS}
                row["timestamp"] = ts_ms
                row["timestamp_received"] = pd.NA
                by_timestamp[ts_ms] = row

            if received_epoch_ms is not None:
                row["timestamp_received"] = received_epoch_ms

            if sensor_type == 1:  # accelerometer
                row["ax"] = x * GRAVITY
                row["ay"] = y * GRAVITY
                row["az"] = z * GRAVITY
            elif sensor_type == 2:  # gyroscope
                row["gx"] = x
                row["gy"] = y
                row["gz"] = z
            elif sensor_type == 4:  # magnetometer
                row["mx"] = x
                row["my"] = y
                row["mz"] = z

    if not by_timestamp:
        return pd.DataFrame(columns=CSV_COLUMNS)

    rows = [by_timestamp[ts] for ts in sorted(by_timestamp)]
    df = pd.DataFrame(rows)
    # Ensure column order and numeric coercion.
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[CSV_COLUMNS + [c for c in df.columns if c not in CSV_COLUMNS]]
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    for col in CSV_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp_received" in df.columns:
        df["timestamp_received"] = pd.to_numeric(df["timestamp_received"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Arduino log conversion."""
    parser = argparse.ArgumentParser(
        prog="python -m parser.arduino_legacy",
        description="Convert a raw Arduino BLE log to processed IMU CSV format.",
    )
    parser.add_argument("source_txt", type=Path, help="Path to raw Arduino text log.")
    parser.add_argument("destination_csv", type=Path, help="Output CSV path.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    df = parse_arduino_log(args.source_txt)
    write_dataframe(df, args.destination_csv)


if __name__ == "__main__":
    main()
