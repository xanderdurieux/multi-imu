"""Convert raw device logs into standardized IMU CSVs.

This script demonstrates how to parse Arduino BLE logs and Sporsa UART logs
into the library's CSV schema. It preserves the device-recorded timestamps so
accelerometer and gyroscope samples that share the same device time align on a
common ``timestamp`` column in the output.

Expected input formats
----------------------
* **Arduino BLE log**: Lines beginning with ``A``, followed by the wall-clock
  time and a tab-separated hex payload. The payload layout matches the
  ``<B3xfffI`` struct used in :func:`multi_imu.io.parse_arduino_log`.
* **Sporsa UART log**: Comma-separated rows of the form
  ``<device_time_ms>,accX,accY,accZ,gyroX,gyroY,gyroZ`` where the acceleration
  and gyro values are integer counts scaled by the sensitivities defined in
  :mod:`multi_imu.io`.

Output layout
-------------
Standardized CSVs are written into separate output folders for each device and
include ``timestamp`` (seconds since the device session start) along with axis
columns (``ax/ay/az`` and ``gx/gy/gz``) and a magnitude column ``total`` for
acceleration and gyroscope data.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from multi_imu import (
    assemble_motion_sensor_data,
    export_raw_session,
    parse_arduino_log,
    parse_sporsa_log,
)


def _write_stream(df: pd.DataFrame, path: str, *, prefix: str) -> None:
    """Save a combined IMU dataframe to disk if it has samples."""

    if df.empty:
        return

    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, f"{prefix}_imu.csv"), index=False)


def _convert_arduino(log_path: str, output_dir: str, name: str) -> None:
    acc_df, gyro_df, mag_df = parse_arduino_log(log_path)

    export_raw_session({"acc": acc_df, "gyro": gyro_df, "mag": mag_df}, os.path.join(output_dir, "arduino"))

    imu = assemble_motion_sensor_data(acc_df, gyro_df, name=f"{name}_arduino")
    _write_stream(imu.data, os.path.join(output_dir, "arduino"), prefix=name)


def _convert_sporsa(log_path: str, output_dir: str, name: str) -> None:
    acc_df, gyro_df = parse_sporsa_log(log_path)

    export_raw_session({"acc": acc_df, "gyro": gyro_df}, os.path.join(output_dir, "sporsa"))

    imu = assemble_motion_sensor_data(acc_df, gyro_df, name=f"{name}_sporsa")
    _write_stream(imu.data, os.path.join(output_dir, "sporsa"), prefix=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arduino-log", help="Path to Arduino BLE log file")
    parser.add_argument("--sporsa-log", help="Path to Sporsa UART log file")
    parser.add_argument(
        "--output-dir", default="converted", help="Directory to write standardized CSVs"
    )
    parser.add_argument("--session-name", default="session", help="Prefix for output files")

    args = parser.parse_args()

    if not args.arduino_log and not args.sporsa_log:
        parser.error("Provide at least one of --arduino-log or --sporsa-log")

    if args.arduino_log:
        _convert_arduino(args.arduino_log, args.output_dir, args.session_name)

    if args.sporsa_log:
        _convert_sporsa(args.sporsa_log, args.output_dir, args.session_name)


if __name__ == "__main__":
    main()
