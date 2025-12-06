"""Example: convert raw logs to joined IMU CSVs, synchronize, and process them."""

from pathlib import Path
import matplotlib.pyplot as plt

from multi_imu import (
	load_imu_csv,
	apply_synchronization,
	export_imu_csv,
	lida_synchronize,
	plot_comparison,
	plot_comparison_grid,
)

def main():
    session = 4
    base_path = Path(f"data/processed/session{session}")

    # Step 1: Load sensor data
    print(f"Loading sensors for session {session}...")
    arduino_acc = load_imu_csv(base_path / "arduino/acc.csv", name="arduino")
    sporsa_acc = load_imu_csv(base_path / "sporsa/acc.csv", name="sporsa")

    arduino_gyro = load_imu_csv(base_path / "arduino/gyro.csv", name="arduino")
    sporsa_gyro = load_imu_csv(base_path / "sporsa/gyro.csv", name="sporsa")

    print(f"Arduino acc: {len(arduino_acc.data)} samples, {arduino_acc.sample_rate_hz:.1f} Hz")
    print(f"Arduino gyro: {len(arduino_gyro.data)} samples, {arduino_gyro.sample_rate_hz:.1f} Hz")
    print(f"Sporsa acc: {len(sporsa_acc.data)} samples, {sporsa_acc.sample_rate_hz:.1f} Hz")
    print(f"Sporsa gyro: {len(sporsa_gyro.data)} samples, {sporsa_gyro.sample_rate_hz:.1f} Hz")

    # Step 2: Synchronize sensors (use sporsa as reference)
    print("\nSynchronizing sensors using LIDA...")
    offset, slope, info = lida_synchronize(sporsa_acc, arduino_acc, try_both_directions=True)
    
    print(f"Offset: {offset:.3f} s")
    print(f"Slope (drift): {slope:.6f}")
    if "crosscorr_peak" in info:
        print(f"Cross-correlation peak: {info['crosscorr_peak']:.4f}")
    if "drift_detected" in info:
        print(f"Clock drift detected: {info['drift_detected']}")

    # Step 3: Apply synchronization
    synced_arduino_acc = apply_synchronization(arduino_acc, offset, slope)
    synced_arduino_gyro = apply_synchronization(arduino_gyro, offset, slope)

    # Step 4: Visualize before/after
    print("\nCreating visualization...")
    plot_comparison_grid([sporsa_acc, sporsa_gyro], [synced_arduino_acc, synced_arduino_gyro], rows=["x", "y", "z"], types=["a", "g"])
    plt.suptitle(f"Session {session}: Synchronized Sensors", fontsize=14, fontweight="bold")
    
    # Step 5: Save synchronized data
    print("\nSaving synchronized data...")
    sync_output_dir = base_path / "synchronized"
    export_imu_csv(synced_arduino_acc, str(sync_output_dir / "arduino_synced.csv"))
    export_imu_csv(sporsa_acc, str(sync_output_dir / "sporsa_synced_acc.csv"))
    export_imu_csv(synced_arduino_gyro, str(sync_output_dir / "arduino_synced_gyro.csv"))
    export_imu_csv(sporsa_gyro, str(sync_output_dir / "sporsa_synced_gyro.csv"))
    print(f"Synchronized data saved to {sync_output_dir}")
        
    plt.show()

    


    
    print("\n✓ Processing complete!")

if __name__ == "__main__":
    main()
