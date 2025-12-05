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
    parse_phone_log,
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
        

def _convert_raw_session(session_id: int, include_phone: bool = False):
	_convert_arduino(f"data/raw/arduino/log{session_id}.txt", f"data/out/", f"session{session_id}")
	_convert_sporsa(f"data/raw/sporsa/sporsa-session{session_id}.txt", f"data/out/", f"session{session_id}")
      

def main():
	for i in range(1, 9):
		_convert_raw_session(i)

if __name__ == "__main__":
    main()
