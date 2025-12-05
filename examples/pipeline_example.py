"""Example: convert raw logs to joined IMU CSVs and read them back."""

from pathlib import Path

from multi_imu import (
	load_imu_csv,
	apply_synchronization,
	lida_synchronize,
)

def main():

    session = 6

    arduino_imu = load_imu_csv(Path(f"data/processed/session{session}/arduino/imu.csv"), name="arduino")
    sporsa_imu = load_imu_csv(Path(f"data/processed/session{session}/sporsa/imu.csv"), name="sporsa")

    offset, slope, info = lida_synchronize(sporsa_imu, arduino_imu, try_both_directions=True)
    synced_arduino = apply_synchronization(arduino_imu, offset, slope)
    
    plot_comparison(sporsa_imu, synced_arduino, columns=["ax", "ay", "az"])
    plot_comparison(sporsa_imu, synced_arduino, columns=["gx", "gy", "gz"])
    plot_comparison(sporsa_imu, synced_arduino, columns=["mx", "my", "mz"])
    

if __name__ == "__main__":
    main()
