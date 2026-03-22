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

A Python package (requires Python ≥ 3.13, managed with `uv`) that ships the
**stages implemented in this tree**:

1. **Parse** raw device logs into normalised per-recording IMU CSVs.
2. **Synchronise** Sporsa and Arduino clocks (four methods, automatic selection).
3. **Section** each recording into calibration-bounded sub-intervals.
4. **Static six-pose Arduino calibration** (`static_calibration/`) for helmet IMU
   bias and scale (separate from ride recordings).
5. **Visualise** parsed, synced, and—if you have produced those stages
   elsewhere—calibrated and orientation outputs.

World-frame calibration of ride recordings, continuous orientation filters, and
feature extraction are part of the broader thesis workflow but are **not**
included here. See [`analysis/README.md`](analysis/README.md) for data layout,
CLI commands for the available modules, and notes on the extended pipeline.