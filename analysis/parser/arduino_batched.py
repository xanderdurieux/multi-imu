"""Parser for raw Arduino BLE IMU log files (packed IMU + magnetometer)."""

from __future__ import annotations

import datetime as dt
import re
import struct
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from common import CSV_COLUMNS, write_dataframe

GRAVITY = 9.81

# UUID suffixes used by the packed IMU + magnetometer sketch
_IMU_PACKED_UUID_SUFFIX = "696d7534"
_MAG_UUID_SUFFIX = "696d7533"


def _parse_log_date(line: str) -> dt.date | None:
  """
  Parse the log date line written by nRF Connect.
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


def _parse_arduino_line(
  line: str,
) -> List[Tuple[int, int, float, float, float]]:
  """
  Parse one nRF Connect log line for the new packed IMU + magnetometer sketch.

  We only consume the informational 'I' lines, which contain both the
  characteristic UUID and the payload:

      I<TAB><time><TAB>Notification received from <UUID>, value: (0x) <hex bytes ...>

  For the packed IMU characteristic (UUID suffix 696d7534) each payload
  contains:

      struct ImuSamplePacked {
          int16_t ax, ay, az;
          int16_t gx, gy, gz;
          uint32_t timestamp_ms;
      };

  For the magnetometer characteristic (UUID suffix 696d7533):

      struct MagSample {
          float x, y, z;
          uint32_t timestamp_ms;
      };

  Returns a list of zero or more records:

      (sensor_timestamp_ms, sensor_type, x, y, z)

  with sensor_type:
      1 -> accelerometer (x, y, z in g)
      2 -> gyroscope     (x, y, z in deg/s)
      4 -> magnetometer  (x, y, z in sensor units)
  """
  if not line.startswith("I"):
    return []

  m = re.search(
    r"Notification received from ([0-9A-Fa-f-]+), value:\s*\(0x\)\s*([0-9A-Fa-f\- ]+)",
    line,
  )
  if not m:
    return []

  uuid = m.group(1).lower()
  hex_str = m.group(2)

  hex_bytes = re.findall(r"[0-9A-Fa-f]{2}", hex_str)
  if not hex_bytes:
    return []

  raw = bytes(int(h, 16) for h in hex_bytes)

  # Packed IMU (acc + gyro) characteristic.
  # Supports both single-sample (16 bytes) and batched payloads
  # containing N consecutive ImuSamplePacked structs.
  if uuid.endswith(_IMU_PACKED_UUID_SUFFIX):
    fmt_packed = "<hhhhhhI"  # ax, ay, az, gx, gy, gz, timestamp (uint32)
    sample_size = struct.calcsize(fmt_packed)
    if len(raw) % sample_size != 0:
      return []

    records: List[Tuple[int, int, float, float, float]] = []

    for offset in range(0, len(raw), sample_size):
      ax_i16, ay_i16, az_i16, gx_i16, gy_i16, gz_i16, sensor_time_ms = struct.unpack_from(
        fmt_packed, raw, offset
      )

      # Decode to physical units matching the legacy parser:
      # - accelerometer in g (later converted to m/s^2)
      # - gyroscope in deg/s
      acc_scale = 8192.0  # LSB per g, matches Arduino BMI270 driver
      gyr_scale = 16.384  # LSB per deg/s, matches Arduino BMI270 driver

      ax = ax_i16 / acc_scale
      ay = ay_i16 / acc_scale
      az = az_i16 / acc_scale

      gx = gx_i16 / gyr_scale
      gy = gy_i16 / gyr_scale
      gz = gz_i16 / gyr_scale

      ts = int(sensor_time_ms)
      records.append((ts, 1, float(ax), float(ay), float(az)))  # accelerometer
      records.append((ts, 2, float(gx), float(gy), float(gz)))  # gyroscope

    return records

  # Magnetometer characteristic
  if uuid.endswith(_MAG_UUID_SUFFIX):
    fmt_mag = "<fffI"  # x, y, z, timestamp (uint32)
    if len(raw) != struct.calcsize(fmt_mag):
      return []

    mx, my, mz, sensor_time_ms = struct.unpack(fmt_mag, raw)
    ts = int(sensor_time_ms)
    return [(ts, 4, float(mx), float(my), float(mz))]

  # Unknown characteristic
  return []


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

      records = _parse_arduino_line(raw_line)
      if not records:
        continue

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

      for ts_ms, sensor_type, x, y, z in records:
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
    df["timestamp_received"] = pd.to_numeric(
      df["timestamp_received"], errors="coerce"
    )
  df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
  return df

def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for Arduino log conversion."""
    parser = argparse.ArgumentParser(
        prog="python -m parser.arduino",
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
