"""Minimal end-to-end example using synthetic IMU data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multi_imu import (
    IMUSensorData,
    load_imu_csv,
    align_axes,
    compute_alignment_matrix,
    detect_braking_events,
    detect_falls,
    detect_turns,
    plot_comparison,
    plot_event_annotations,
    plot_magnitude,
    remove_gravity,
    resample_signal,
    synchronize_streams,
)


def _generate_synthetic_stream(name: str, rate: float, offset: float = 0.0, noise: float = 0.05) -> IMUSensorData:
    duration = 100.0
    timestamps = np.arange(0, duration, 1 / rate) + offset
    ax = np.sin(2 * np.pi * 0.5 * timestamps) + np.random.normal(scale=noise, size=len(timestamps))
    ay = np.cos(2 * np.pi * 0.2 * timestamps) + np.random.normal(scale=noise, size=len(timestamps))
    az = 9.81 + np.random.normal(scale=noise, size=len(timestamps))
    gz = np.gradient(np.sin(2 * np.pi * 0.1 * timestamps)) * rate
    data = pd.DataFrame({"timestamp": timestamps, "ax": ax, "ay": ay, "az": az, "gz": gz})
    return IMUSensorData(name=name, data=data, sample_rate_hz=rate)


def main():
    session_id = "session4"
    arduino = load_imu_csv(f"data/processed/{session_id}/arduino/acc.csv", name="arduino")
    sporsa = load_imu_csv(f"data/processed/{session_id}/sporsa/acc.csv", name="sporsa")

    arduino_resampled = resample_signal(arduino, target_rate_hz=sporsa.sample_rate_hz)
    synced = synchronize_streams(reference=sporsa, target=arduino_resampled)

    # alignment_matrix = compute_alignment_matrix(synced.reference, synced.target)
    # aligned_target = align_axes(synced.target, alignment_matrix)
    # synced = synced.__class__(reference=synced.reference, target=aligned_target, offset_seconds=synced.offset_seconds, alignment_matrix=alignment_matrix)

    # gravity_free_ref = remove_gravity(synced.reference)
    # gravity_free_target = remove_gravity(aligned_target)

    # fall_events = detect_falls(gravity_free_ref)
    # brake_events = detect_braking_events(gravity_free_ref)
    # turn_events = detect_turns(gravity_free_ref)

    plot_comparison(sporsa, arduino, columns=["ax", "ay", "az"])
    plot_comparison(synced.reference, synced.target, columns=["ax", "ay", "az"])

    plt.show()  # Display all plots

    print("Estimated offset (s):", synced.offset_seconds)
    print("Alignment matrix:\n", alignment_matrix)
    # print("Detected events:")
    # for evt in fall_events + brake_events + turn_events:
    #     print(evt)


if __name__ == "__main__":
    main()
