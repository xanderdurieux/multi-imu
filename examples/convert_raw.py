"""Convert raw Arduino/Sporsa logs into joined IMU CSVs in ``data/processed``."""

from pathlib import Path
from typing import Iterable

from multi_imu import export_joined_imu_frame, export_sensor_frames, parse_arduino_log, parse_sporsa_log


def convert_session(session: int) -> None:
    arduino_log = Path(f"data/raw/arduino/log{session}.txt")
    sporsa_log = Path(f"data/raw/sporsa/sporsa-session{session}.txt")
    output_dir = Path(f"data/processed/session{session}")
    arduino_out = output_dir / "arduino"
    sporsa_out = output_dir / "sporsa"

    if not arduino_log.exists() or not sporsa_log.exists():
        print(f"Skipping session {session}: missing log files")
        return

    arduino_acc, arduino_gyro, arduino_mag = parse_arduino_log(str(arduino_log))
    sporsa_acc, sporsa_gyro = parse_sporsa_log(str(sporsa_log))
    
    export_sensor_frames({"acc": arduino_acc, "gyro": arduino_gyro, "mag": arduino_mag}, str(arduino_out))
    export_sensor_frames({"acc": sporsa_acc, "gyro": sporsa_gyro}, str(sporsa_out))


    export_joined_imu_frame({"acc": arduino_acc, "gyro": arduino_gyro, "mag": arduino_mag}, str(arduino_out))
    export_joined_imu_frame({"acc": sporsa_acc, "gyro": sporsa_gyro}, str(sporsa_out))

    print(f"Session {session} written to {arduino_out} and {sporsa_out}")
    

def main(sessions: Iterable[int]) -> None:
    for session in sessions:
        convert_session(session)

if __name__ == "__main__":
    main(range(1, 9))