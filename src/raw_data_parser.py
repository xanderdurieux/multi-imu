import struct, re, os, csv
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta, UTC
from typing import Dict

ACCEL_SENS = {
    '4G': 0.122,
    '8G': 0.244,
    '16G': 0.488    
}
 
GYRO_SENS = {
    '125DPS': 4.375,
    '250DPS': 8.750,
    '500DPS': 17.500,
    '1000DPS': 35.000,
    '2000DPS': 70.000
}


@dataclass
class SensorData:
    ts: datetime
    x: float
    y: float
    z: float


def get_arduino_date(log_file):
    with open(log_file) as f:
            first_line = f.readline()
            date_str = first_line.split(",")[1]
            return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()


def parse_arduino_log(log_file):
	"""
	Parse the Arduino log file and return the accelerometer and gyroscope data.

	Args:
		log_file: The path to the Arduino log file.
	Returns:
		acc_df: A pandas DataFrame containing the accelerometer data.
		gyro_df: A pandas DataFrame containing the gyroscope data.
		mag_df: A pandas DataFrame containing the magnetometer data.
	"""

	acc_data = []
	gyro_data = []
	mag_data = []

	with open(log_file) as f:
		for line in f:
			if line[0] != "A":
				continue

			linelist = line.split("\t")

			ts_str = linelist[1]
			d = get_arduino_date(log_file)
			t = datetime.strptime(ts_str, "%H:%M:%S.%f").time()
			ts = datetime.combine(d, t)

			data_str = linelist[2]
			data_str = data_str.replace("received", "")

			if 'Notifications' in data_str:
				continue
			hex_bytes = re.findall(r'[0-9A-Fa-f]{2}', data_str)
			raw = bytes(int(h, 16) for h in hex_bytes)
			sensorType, x, y, z, _ = struct.unpack("<B3xfffI", raw)

			if sensorType == 1:
				x *= 9.81
				y *= 9.81
				z *= 9.81
				acc_data.append(SensorData(ts=ts, x=x, y=y, z=z))
			elif sensorType == 2:
				gyro_data.append(SensorData(ts=ts, x=x, y=y, z=z))
			elif sensorType == 4:
				mag_data.append(SensorData(ts=ts, x=x, y=y, z=z))
			else:
				continue

	return pd.DataFrame(acc_data), pd.DataFrame(gyro_data), pd.DataFrame(mag_data)

def parse_sporsa_log(log_file):
	"""
	Parse the Sporsa log file and return the accelerometer and gyroscope data.

	Args:
		log_file: The path to the Sporsa log file.
	Returns:
		acc_df: A pandas DataFrame containing the accelerometer data.
		gyro_df: A pandas DataFrame containing the gyroscope data.
	"""

	acc_data = []
	gyro_data = []

	with open(log_file) as f:
		for line in f:

			linelist = line.split(",")
			if len(linelist) != 7:
				continue

			ts_str = linelist[0]
			ts_str = ts_str.replace("uart:~$ ", "")
			ts = datetime.fromtimestamp(int(ts_str) / 1000.0, UTC) + timedelta(hours=1)

			# Convert raw accelerometer data to m/s^2 (assuming 16g range)
			acc_x = int(linelist[1]) * ACCEL_SENS['16G'] * 9.81 / 1000
			acc_y = int(linelist[2]) * ACCEL_SENS['16G'] * 9.81 / 1000
			acc_z = int(linelist[3]) * ACCEL_SENS['16G'] * 9.81 / 1000
			# Convert raw gyroscope data to dps (assuming 2000 dps range)
			gyro_x = int(linelist[4]) * GYRO_SENS['2000DPS'] / 1000
			gyro_y = int(linelist[5]) * GYRO_SENS['2000DPS'] / 1000
			gyro_z = int(linelist[6]) * GYRO_SENS['2000DPS'] / 1000

			acc_data.append(SensorData(ts=ts, x=acc_x, y=acc_y, z=acc_z))
			gyro_data.append(SensorData(ts=ts, x=gyro_x, y=gyro_y, z=gyro_z))

	return pd.DataFrame(acc_data), pd.DataFrame(gyro_data)


def parse_phone_log(log_file):
        """
        Parse the phone log file and return the data.
	
	Args:
		log_file: The path to the phone log file.
	Returns:
		df: A pandas DataFrame containing the data.
	"""

	data = []

	with open(log_file) as f:
		reader = csv.reader(f, delimiter=',')

		for line in reader:	
			if reader.line_num == 1:
				continue

			ts = datetime.fromtimestamp(int(line[0]) / 1000000000.0, UTC)
			x = float(line[2])
			y = float(line[3])
			z = float(line[4])
			data.append(SensorData(ts=ts, x=x, y=y, z=z))

        return pd.DataFrame(data)

def _ts_to_seconds(ts: datetime) -> float:
        """Convert datetime to seconds since epoch, assuming UTC when naive."""

        if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        return ts.timestamp()


def standardize_sensor_frame(df: pd.DataFrame, sensor_type: str, add_magnitude: bool = True) -> pd.DataFrame:
        """Convert parsed dataframes into the library's standard CSV schema.

        The multi-IMU analysis pipeline expects a ``timestamp`` column in seconds
        plus axis-specific columns (``ax``, ``ay``, ``az`` for accelerometers;
        ``gx``, ``gy``, ``gz`` for gyroscopes; and ``mx``, ``my``, ``mz`` for
        magnetometers). This helper performs that conversion and optionally adds
        a vector magnitude column for quick inspection.
        """

        if df.empty:
                return pd.DataFrame()

        axis_map = {
                "acc": ("ax", "ay", "az"),
                "gyro": ("gx", "gy", "gz"),
                "mag": ("mx", "my", "mz"),
        }

        if sensor_type not in axis_map:
                raise ValueError(f"Unsupported sensor type: {sensor_type}")

        axis_labels = axis_map[sensor_type]
        standardized = df.copy()
        standardized["timestamp"] = standardized["ts"].apply(_ts_to_seconds)
        standardized = standardized.rename(columns={
                "x": axis_labels[0],
                "y": axis_labels[1],
                "z": axis_labels[2],
        })
        standardized = standardized.drop(columns=["ts"])

        if add_magnitude:
                standardized["total"] = np.sqrt(sum(standardized[label] ** 2 for label in axis_labels))

        ordered_cols = ["timestamp", *axis_labels]
        if add_magnitude:
                ordered_cols.append("total")

        return standardized[ordered_cols]


def export_to_csv(sensor_frames: Dict[str, pd.DataFrame], output_file: str, add_magnitude: bool = True):
        """Write standardized sensor CSVs for downstream analysis.

        Args:
                sensor_frames: Mapping of sensor type (``acc``, ``gyro``, ``mag``)
                        to raw parsed DataFrames containing ``ts``, ``x``, ``y``, ``z``.
                output_file: Directory to place the CSV exports.
                add_magnitude: Whether to append the vector magnitude column.
        """

        os.makedirs(output_file, exist_ok=True)

        for sensor_type, df in sensor_frames.items():
                standardized = standardize_sensor_frame(df, sensor_type, add_magnitude)
                standardized.to_csv(os.path.join(output_file, f"{sensor_type}.csv"), index=False)


if __name__ == "__main__":

        for i in range(1, 9):
                acc_df, gyro_df, mag_df = parse_arduino_log("data/raw/arduino/log" + str(i) + ".txt")
                export_to_csv({"acc": acc_df, "gyro": gyro_df, "mag": mag_df}, "data/processed/session_" + str(i) + "/arduino")

                acc_df, gyro_df = parse_sporsa_log("data/raw/sporsa/sporsa-session" + str(i) + ".txt")
                export_to_csv({"acc": acc_df, "gyro": gyro_df}, "data/processed/session_" + str(i) + "/sporsa")

        acc_df = parse_phone_log("data/raw/smartphone/AccelerometerUncalibrated.csv")
        gyro_df = parse_phone_log("data/raw/smartphone/GyroscopeUncalibrated.csv")
        export_to_csv({"acc": acc_df, "gyro": gyro_df}, "data/processed/session_8/phone")
