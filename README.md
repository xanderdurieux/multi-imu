# master-thesis

Research repository for a master's thesis on multi-IMU cycling instrumentation,
time synchronisation, and motion / incident analysis.

Two IMU sensors are used simultaneously:

- A **Sporsa** sensor mounted on the bicycle handlebar (reference device with a
  stable Unix-epoch clock).
- An **Arduino Nano 33 BLE** helmet-mounted sensor that streams IMU data over
  BLE (counter-based clock with no wall-clock reference).

The thesis pipeline ingests raw logs from both devices, synchronises their
clocks, calibrates them into a common world frame, estimates orientation, and
extracts features for incident analysis.

---

## Repository structure

| Folder | Contents |
|---|---|
| `arduino/` | Arduino firmware sketch for the Nano 33 BLE IMU peripheral. |
| `analysis/` | Python toolkit for offline processing of multi-IMU cycling data. |

See the README in each sub-folder for detailed usage instructions.

---

## Quick orientation

### Firmware (`arduino/`)

The Arduino sketch in `arduino/ble_imu_advertising/` turns the Nano 33 BLE into
a BLE IMU peripheral that streams accelerometer, gyroscope, and magnetometer
data to any connected central (e.g. the Sporsa receiver or a laptop).

See [`arduino/README.md`](arduino/README.md) for hardware requirements, UUIDs,
and upload instructions.

### Analysis toolkit (`analysis/`)

A Python package (requires Python ≥ 3.13, managed with `uv`) that implements a
six-stage offline pipeline:

1. **Parse** raw device logs into normalised per-recording IMU CSVs.
2. **Synchronise** Sporsa and Arduino clocks (four methods, automatic selection).
3. **Section** each recording into calibration-bounded sub-intervals.
4. **Calibrate** sensors into a world frame and correct biases.
5. **Estimate orientation** using complementary or Madgwick filters.
6. **Extract features** (duration, acceleration, gyroscope, jerk statistics)
   at recording and section level.

See [`analysis/README.md`](analysis/README.md) for full pipeline documentation,
data layout, CLI commands, and quality criteria.