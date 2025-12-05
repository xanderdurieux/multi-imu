"""Example: convert raw logs to joined IMU CSVs and read them back."""

from pathlib import Path

from multi_imu import load_imu_csv


def main():

    session = 6

    arduino_imu = load_imu_csv(Path(f"data/processed/session{session}/arduino/imu.csv"), name="arduino")
    sporsa_imu = load_imu_csv(Path(f"data/processed/session{session}/sporsa/imu.csv"), name="sporsa")

    
    
if __name__ == "__main__":
    main()
