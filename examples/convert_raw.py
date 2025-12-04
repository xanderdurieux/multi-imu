from multi_imu import (
    parse_arduino_log,
    parse_sporsa_log,
    parse_phone_log,
    export_raw_session,
)   


def convert_raw_session(session_id: int, include_phone: bool = False):
	acc, gyro, mag = parse_arduino_log(f"data/raw/arduino/log{session_id}.txt")
	export_raw_session({"acc": acc, "gyro": gyro, "mag": mag}, f"data/new/session_{session_id}/arduino", False)

	acc, gyro = parse_sporsa_log(f"data/raw/sporsa/sporsa-session{session_id}.txt")
	export_raw_session({"acc": acc, "gyro": gyro}, f"data/new/session_{session_id}/sporsa", False)

	if include_phone:
		acc_data = parse_phone_log("data/raw/smartphone/AccelerometerUncalibrated.csv")
		gyro_data = parse_phone_log("data/raw/smartphone/GyroscopeUncalibrated.csv")
		mag_data = parse_phone_log("data/raw/smartphone/MagnetometerUncalibrated.csv")
		export_raw_session({"acc": acc_data, "gyro": gyro_data, "mag": mag_data}, "data/new/smartphone", False)

def main():
	for i in range(1, 9):
		convert_raw_session(i, include_phone=(i==8))

if __name__ == "__main__":
    main()