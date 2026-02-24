# master-thesis/analysis

Python toolkit for offline analysis and synchronization of multi-device IMU sensor data.

## Overview

This toolkit processes raw IMU logs from multiple devices (e.g., Sporsa handlebar sensor and Arduino rider sensor), synchronizes their timestamps using SDA (Simple Data Alignment) and LIDA (Linear Interpolation Data Alignment) algorithms, and provides visualization tools for evaluating alignment quality.

## Modules

- **`common/`**
  - CSV schema utilities and path helpers for locating raw/processed data directories.

- **`parser/`**
  - Converts raw device-specific IMU log files into standardized CSV format.
  - Supports Arduino BLE and Sporsa sensor formats.
  - Outputs processed CSV files with normalized timestamps and sensor readings.

- **`sync/`**
  - Implements SDA + LIDA synchronization pipeline for aligning two IMU streams.
  - **SDA (Simple Data Alignment)**: Coarse discrete alignment using cross-correlation on orientation-invariant vector magnitudes (precision ~1 sample period).
  - **LIDA (Linear Interpolation Data Alignment)**: Refined sub-sample precision alignment using windowed correlation and linear drift estimation.
  - Provides functions for alignment signal construction, offset/drift estimation, model persistence, and stream resampling.

- **`orientation/`**
  - Orientation estimation from IMU CSV streams: quaternion math, sensor-level bias calibration, and lightweight fusion filters (complementary and Madgwick).
  - Produces body→world quaternions and Euler angles so all sensors can be expressed in a common world frame.

- **`plot/`**
  - Visualization tools for single-stream analysis and multi-stream comparison.
  - Supports plotting sensor components (x/y/z) or orientation-invariant vector magnitudes.
  - Useful for evaluating synchronization quality by comparing reference and target streams.

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

**Input**: Raw log files in `data/raw/<session_name>/`  
**Output**: Processed CSV files in `data/processed/<session_name>/`

```bash
uv run -m parser.session <session_name>
```

Detects and parses Arduino and Sporsa log files automatically, writing normalized CSV streams with standardized column names and units.

### Orientation estimation

Estimate body→world orientation from a parsed IMU CSV (columns `timestamp`, `ax`–`az`, `gx`–`gz`). The pipeline runs a complementary or Madgwick filter and appends quaternion and Euler columns so data can be expressed in a common world frame. **Units**: acceleration in m/s², gyroscope in rad/s (convert from deg/s with `× π/180` if needed).

**From the command line** (run from the `analysis/` directory):

```bash
uv run -m orientation.pipeline <input.csv> [output.csv]
```

- **Input**: Path to a parsed IMU CSV (e.g. `data/<session>/parsed/sporsa.csv`).
- **Output**: If omitted, writes to `<input_stem>_orientation.csv` in the same directory. The output CSV contains all original columns plus `qw`, `qx`, `qy`, `qz` and (in degrees) `yaw_deg`, `pitch_deg`, `roll_deg`.

**Optional arguments:**

- `--filter {complementary,madgwick}` (default: `complementary`)
- `--calibration`: estimate constant accel+gyro bias from the first 5 seconds and apply before filtering

By default no calibration is applied. If you pass `--calibration`, the pipeline estimates constant accel+gyro bias from the first 5 seconds and applies it before filtering. For full control (custom static window, saved calibration, etc.), use the Python API and pass a `BiasCalibration` (see below).

**From Python** (with optional calibration):

```python
from pathlib import Path
from common import load_dataframe, write_dataframe
from orientation import (
    run_complementary_on_dataframe,
    run_madgwick_on_dataframe,
    estimate_bias_from_dataframe_static_segment,
)

# Load parsed IMU CSV
df = load_dataframe(Path("data/<session>/parsed/sporsa.csv"))

# Optional: estimate bias from a static segment (sensor at rest)
calib = estimate_bias_from_dataframe_static_segment(
    df,
    start_time=df["timestamp"].iloc[0],
    end_time=df["timestamp"].iloc[0] + 5_000,  # first 5 seconds, same units as timestamp
    expected_gravity_body=[0, 0, -9.81],
)

# Run complementary or Madgwick filter
df_orient = run_complementary_on_dataframe(df, calibration=calib)
# or: df_orient = run_madgwick_on_dataframe(df, calibration=calib)

# Save result
write_dataframe(df_orient, Path("data/<session>/parsed/sporsa_orientation.csv"))
```

To choose the Madgwick filter from the command line, use the pipeline programmatically with `OrientationPipelineConfig(filter_type="madgwick")` and `run_orientation_pipeline_on_csv(..., config=cfg)`.

### Synchronize IMU streams

Aligns a target stream to a reference stream using SDA + LIDA algorithms. The target stream's timestamps are transformed to match the reference stream's clock, accounting for both initial time offset and clock drift.

**Reference stream**: Typically the Sporsa handlebar sensor (stable reference clock)  
**Target stream**: Typically the Arduino rider sensor (needs alignment)

#### Session mode (automatic stream selection)

Automatically locates reference and target CSV files in the processed session directory:

```bash
uv run -m sync.sync_streams session <session_name>
```

Looks for `sporsa.csv` (reference) and `arduino.csv` (target) in `data/processed/<session_name>/`.

#### Manual mode (explicit file paths)

Specify reference and target CSV files explicitly:

```bash
uv run -m sync.sync_streams pair \
  data/processed/<session_name>/<reference>.csv \
  data/processed/<session_name>/<target>.csv
```

#### Synchronization parameters

- `--sample-rate-hz <float>` (default: 50.0)
  - Common sampling rate for alignment signal computation. Higher rates improve precision but increase computation time.
  
- `--max-lag-seconds <float>` (default: 20.0)
  - Maximum absolute time lag to search during SDA coarse alignment. Should cover the expected initial time difference between streams.

- `--resample-rate-hz <float>` (optional)
  - Output sampling rate for uniformly resampled synchronized stream. Uses LIDA-style linear interpolation. Omit to skip resampled output.

#### Generated outputs

All outputs are written next to the target CSV file:

1. **`<target>_to_<reference>_sync.json`**
   - Complete synchronization model metadata including:
     - Time offset (seconds): Initial time difference between streams
     - Drift rate (seconds/second): Clock drift per second of target time
     - Scale factor: Derived from drift rate (1.0 + drift)
     - Quality metrics: Correlation scores, R² fit quality, number of windows used
     - Algorithm settings: Sampling rates, window sizes, search parameters

2. **`<target>_synced.csv`**
   - Target stream with timestamps transformed to reference clock.
   - Columns:
     - `timestamp`: Aligned timestamp (in reference clock, milliseconds)
     - `timestamp_orig`: Original target timestamp (preserved)
     - `timestamp_aligned`: Aligned timestamp (same as `timestamp` if replacement enabled)
     - All sensor columns (ax, ay, az, gx, gy, gz, mx, my, mz)

3. **`<target>_synced_resampled_<hz>hz.csv`** (optional)
   - Uniformly resampled synchronized stream at specified rate.
   - Uses linear interpolation (LIDA-style) for sub-sample precision.
   - Useful for downstream analysis requiring uniform sampling intervals.

### Plot a processed CSV

Visualize a single IMU stream with sensor components or vector magnitudes.

```bash
uv run -m plot.plot_device data/processed/<session_name>/<filename>
```

**Optional arguments:**
- `--magnitudes`: Plot orientation-invariant vector magnitudes (|acc|, |gyro|, |mag|) instead of x/y/z components
- `--output <path>`: Custom output PNG path (default: `<filename>.png`)
- `--title <string>`: Custom plot title (default: filename)

**Output**: PNG file showing three panels (acceleration, gyroscope, magnetometer) with time-series data.

### Compare two streams

Overlay two IMU streams for visual evaluation of synchronization quality. Useful for comparing raw vs. synchronized streams or evaluating alignment accuracy.

```bash
uv run -m plot.compare_streams \
  data/processed/<session_name>/sporsa.csv \
  data/processed/<session_name>/arduino_synced.csv
```

**Optional arguments:**
- `--split-axes`: Plot each x/y/z component in separate panels (9 panels total) to reduce visual clutter
- `--magnitudes`: Plot vector magnitudes instead of individual axis components (3 panels: |acc|, |gyro|, |mag|)
- `--relative-time`: Use time relative to each stream's start (x-axis starts at 0 for both streams)
- `--label-a <string>`: Custom label for first stream (default: filename stem)
- `--label-b <string>`: Custom label for second stream (default: filename stem)
- `--title <string>`: Custom figure title
- `--output <path>`: Custom output PNG path (default: `<stream_a>__vs__<stream_b>.png`)

**Output**: PNG file with overlaid time-series showing both streams. Well-synchronized streams should show aligned temporal patterns.

### Run full single-session pipeline

Execute complete workflow: synchronize streams and generate comparison plots for alignment evaluation.

```bash
uv run -m pipeline.session_pipeline <session_name>
```

**Pipeline steps:**
1. Synchronizes target stream (arduino) to reference stream (sporsa)
2. Generates single-stream overview plots for both raw and synchronized data
3. Creates comparison plots: raw vs. raw, synced vs. reference, synced magnitudes

**Optional arguments:**
- `--sample-rate-hz <float>` (default: 50.0): Alignment resampling rate
- `--max-lag-seconds <float>` (default: 20.0): Maximum lag search window
- `--resample-rate-hz <float>` (default: 100.0): Output resampling rate (set `<= 0` to skip resampled CSV)

**Generated artifacts** (in `data/processed/<session_name>/`):
- Synchronization JSON and CSV files (as described above)
- `sporsa_pipeline_overview.png`: Reference stream visualization
- `arduino_pipeline_overview.png`: Target stream (raw) visualization
- `arduino_synced_pipeline_overview.png`: Target stream (synchronized) visualization
- `sporsa_vs_arduino_raw.png`: Raw comparison (before synchronization)
- `sporsa_vs_arduino_synced.png`: Synchronized comparison (after alignment)
- `sporsa_vs_arduino_synced_magnitudes.png`: Synchronized comparison using vector magnitudes

## Synchronization Algorithm Details

The synchronization pipeline implements a two-stage approach:

1. **SDA (Simple Data Alignment)**: Coarse discrete alignment
   - Builds orientation-invariant alignment signals from accelerometer/gyroscope vector magnitudes
   - Uses cross-correlation to find integer sample lag
   - Precision limited to ~1 sample period (within 0.5 samples at best)

2. **LIDA (Linear Interpolation Data Alignment)**: Refined sub-sample alignment
   - Refines coarse lag using windowed correlation over time
   - Estimates linear clock drift (offset + drift_rate × time)
   - Enables sub-sample precision alignment via linear interpolation

**Reference**: Wang et al. (2023). Comparison between Two Time Synchronization and Data Alignment Methods for Multi-Channel Wearable Biosensor Systems Using BLE Protocol. *Sensors*, 23(5), 2465.
