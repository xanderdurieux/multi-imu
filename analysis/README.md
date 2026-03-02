# master-thesis/analysis

Python toolkit for offline analysis and synchronization of multi-device IMU sensor data.

## Overview

This toolkit processes raw IMU logs from multiple devices (e.g., Sporsa sensor and Arduino sensor).

## Project layout

- **`common/`**
  - CSV schema utilities (`CSV_COLUMNS`) and helpers for loading/writing DataFrames.
  - Path helpers for locating data directories (`raw/`, `parsed/`, `synced/`, per-stage folders).
  - Convenience helpers such as `find_sensor_csv(session_name, stage, sensor_name)`.

- **`parser/`**
  - High-level session parsing (`parser.session`) from `raw/` → `parsed/`.
  - Device-specific parsers for Arduino (new + legacy) and Sporsa logs.
  - Session statistics (`parser.stats`) for timing/quality analysis.

- **`sync/`**
  - Implements SDA + LIDA synchronization pipeline for aligning two IMU streams.
  - Session-level synchronization (`sync.session`) from `parsed/` → `synced/`.
  - Low-level two-stream synchronization utilities (`sync.sync_streams`, `drift_estimator`, `align_df`, `common`).

- **`orientation/`**
  - Orientation estimation from IMU CSV streams: quaternion math, bias calibration, and lightweight fusion filters (complementary and Madgwick).
  - Single-file orientation pipeline (`orientation.pipeline`) and session-wide orientation over a stage (`orientation.session`).

- **`visualization/`**
  - Plotting tools for single-stream analysis and multi-stream comparison.
  - Sensor plots (`visualization.plot_sensor`) and stream comparison plots (`visualization.plot_comparison`).

## Setup

- Python `>= 3.13`
- Recommended: `uv` for dependencies and running tools.

From the `analysis/` directory:

```bash
cd master-thesis/analysis
uv sync
```

## Usage

### Parse a session

Converts raw device log files into standardized CSV format for analysis.

```bash
uv run -m parser.session <session_name>
```

- **Input**: Raw log files in `data/<session_name>/raw/`  
- **Output**: Parsed CSV files in `data/<session_name>/parsed/`

Detects and parses Arduino and Sporsa log files automatically, writing normalized CSV streams with standardized column names and units.


### Compute session statistics

Compute timing/quality statistics for all parsed CSVs in a session.

```bash
uv run -m parser.stats <session_name> [--out path/to/session_stats.json]
```

- **Input**: All CSVs in `data/<session_name>/parsed/`.
- **Output**: A compact JSON file (default `session_stats.json` in the parsed directory) summarizing sampling rate, jitter, gaps, and optional Arduino drift metrics per stream, plus an overall session quality score.


### Orientation estimation

Estimate body→world orientation from a parsed IMU CSV (columns `timestamp`, `ax`–`az`, `gx`–`gz`). The pipeline runs a complementary or Madgwick filter and appends quaternion and Euler columns so data can be expressed in a common world frame. **Units**: acceleration in m/s², gyroscope in rad/s (convert from deg/s with `× π/180` if needed).

```bash
uv run -m orientation.pipeline <input.csv> [output.csv]
```

- **Input**: Path to a parsed IMU CSV (e.g. `data/<session>/parsed/sporsa.csv`).
- **Flags**:
  - `--filter {complementary,madgwick}` (default: `complementary`)
  - `--calibration`: estimate constant accel+gyro bias from the first 5 seconds and apply before filtering
- **Output**: If omitted, writes to `<input_stem>_orientation.csv` in the same directory. The output CSV contains all original columns plus `qw`, `qx`, `qy`, `qz` and (in degrees) `yaw_deg`, `pitch_deg`, `roll_deg`.

By default no calibration is applied. If you pass `--calibration`, the pipeline estimates constant accel+gyro bias from the first 5 seconds and applies it before filtering.

### Session-wide orientation over a stage

Run orientation estimation for **all** CSVs in one session stage (e.g. `parsed`, `synced`) and collect basic quality metrics for each filter/variant.

```bash
uv run -m orientation.session <session_name>/<stage>
```

- **Input**: All CSV files in `data/<session_name>/<stage>/` (for example `data/session_7/synced/`).
- **Behavior**:
  - For each CSV, runs both filters (`complementary`, `madgwick`) with and without simple static calibration based on the first few seconds.
  - Writes one output CSV per (sensor, filter, calibration) variant into a new stage directory `data/<session_name>/<stage>_orientation/`.
  - Computes per-stream orientation quality metrics (e.g. gravity consistency and static tilt stability) and writes summary JSON `orientation_stats.json` into the same output directory.
- **Key options**:
  - `--static-window-ms`: duration (in ms) of the initial static window used to estimate accel/gyro bias for the calibrated variants (default: `5000`).
  - `--gravity`: assumed gravity magnitude in m/s² for the quality metrics (default: `9.81`).


### Synchronize IMU streams

Aligns a target stream to a reference stream using SDA + LIDA algorithms. The target stream's timestamps are transformed to match the reference stream's clock, accounting for both initial time offset and clock drift.

- **Reference stream**: Typically the Sporsa handlebar sensor (stable reference clock)  
- **Target stream**: Typically the Arduino rider sensor (needs alignment)

#### Session mode (automatic stream selection)

Automatically locates reference and target CSV files in the parsed session directory:

```bash
uv run -m sync.session <session_name>
```

Looks for `sporsa.csv` (reference) and `arduino.csv` (target) in `data/<session_name>/parsed/`.

#### Manual mode (explicit file paths)

Specify reference and target CSV files explicitly:

```bash
uv run -m sync.sync_streams \
  data/<session_name>/parsed/<reference>.csv \
  data/<session_name>/parsed/<target>.csv
```
#### Synchronization parameters

- `--sample-rate-hz <float>` (default: 50.0)
  - Common sampling rate for alignment signal computation. Higher rates improve precision but increase computation time.
  
- `--max-lag-seconds <float>` (default: 30.0)
  - Maximum absolute time lag to search during SDA coarse alignment. Should cover the expected initial time difference between streams.

- `--resample-rate-hz <float>` (optional)
  - Output sampling rate for uniformly resampled synchronized stream. Uses LIDA-style linear interpolation. Omit to skip resampled output.


### Visualize the sensor data

#### Plot a sensor stream

Visualize a single IMU stream with sensor components or vector magnitudes.

```bash
uv run -m visualization.plot_sensor <session_name>/<stage> <sensor_name> [--norm] [--split]
```

- **Input**: One CSV in `data/<session_name>/<stage>/` whose filename contains `sensor_name` (case-insensitive).
- **Flags**:
  - `--norm`: Plot vector norms instead of individual axes.
  - `--split`: Create one PNG per sensor type (`acc`, `gyro`, `mag`) instead of a combined figure.
- **Output**: PNG file(s) saved next to the CSV in the same stage directory.

#### Compare two streams

Overlay two IMU streams for visual evaluation of synchronization quality. Useful for comparing raw vs. synchronized streams or evaluating alignment accuracy.

```bash
uv run -m visualization.plot_comparison <session_name>/<stage> [sensor_name_a] [sensor_name_b] [--norm]
```

- **Input**: Two CSVs in `data/<session_name>/<stage>/` whose filenames contain `sensor_name_a` / `sensor_name_b`.
- **Flags**:
  - `sensor_name_a` (default: `sporsa`)
  - `sensor_name_b` (default: `arduino`)
  - `--norm`: Plot vector norms instead of individual axes; otherwise plot x/y/z in separate columns.
- **Output**: A PNG comparing the two streams, saved in the same stage directory.

#### Create plots of entire session

```bash
uv run -m visualization.plot_session <session_name> [session_name ...]
```
