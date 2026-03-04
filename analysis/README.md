# master-thesis/analysis

Python toolkit for offline analysis and synchronization of multi-device IMU sensor data.

## Overview

This toolkit processes raw IMU logs from multiple devices (e.g., Sporsa sensor and Arduino sensor).

## Project layout

- **`common/`**
  - CSV schema utilities (`CSV_COLUMNS`) and helpers for loading/writing DataFrames.
  - Path helpers for locating data directories (`raw/`, `parsed/`, `synced_lida/`, `synced_cal/`, per-stage folders).
  - Convenience helpers such as `find_sensor_csv(session_name, stage, sensor_name)`.

- **`parser/`**
  - High-level session parsing (`parser.session`) from `raw/` → `parsed/`.
  - Device-specific parsers for Arduino (new + legacy) and Sporsa logs.
  - Session statistics (`parser.stats`) for timing/quality analysis.

- **`sync/`**
  - Two independent synchronization methods for aligning two IMU streams:
    - **SDA + LIDA** (`sync.session`) from `parsed/` → `synced_lida/`.
    - **Calibration-sequence sync** (`sync.calibration_sync`) from `parsed/` → `synced_cal/`.
  - Comparison utility (`sync.compare_sync`) for evaluating both methods side-by-side.
  - Low-level utilities: `sync.sync_streams`, `drift_estimator`, `align_df`, `common`.

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

Two independent synchronization approaches are available, both operating on `parsed/` data and producing aligned `sporsa.csv` + `arduino.csv` pairs together with a `sync_info.json` model file.

- **Reference stream**: Typically the Sporsa handlebar sensor (stable reference clock)  
- **Target stream**: Typically the Arduino rider sensor (needs alignment)

#### Method 1 — SDA + LIDA (`synced_lida/`)

Spectral Density Alignment (SDA) finds a coarse time offset by cross-correlating the full spectral energy of the two streams. LIDA (Local Incremental Drift Alignment) then refines the per-sample offset by fitting a linear drift model through a sliding window of local cross-correlations.

**Session mode** (automatic stream selection from `parsed/`):

```bash
uv run -m sync.session <session_name>/<stage_in>
# e.g. uv run -m sync.session 2026-02-26_5/parsed
```

**Manual mode** (explicit file paths):

```bash
uv run -m sync.sync_streams \
  data/<session_name>/parsed/<reference>.csv \
  data/<session_name>/parsed/<target>.csv
```

**Key parameters:**

- `--sample-rate-hz <float>` (default: 100.0) — resampling rate for alignment signal.
- `--max-lag-seconds <float>` (default: 60.0) — coarse search range; must cover the expected initial time difference between devices.
- `--resample-rate-hz <float>` (optional) — write a uniformly resampled synced target CSV.

**Output:** `data/<session_name>/synced_lida/`

---

#### Method 2 — Calibration-sequence sync (`synced_cal/`)

Uses the known physical calibration events (a burst of acceleration peaks performed at the start and end of each recording) as time anchors. The method:

1. Detects calibration sequences in the reference sensor (Sporsa).
2. Identifies the matching opening calibration cluster in the target (Arduino) to establish a coarse offset — without relying on full-signal SDA.
3. Refines the offset at each calibration anchor independently by cross-correlation within a tight window.
4. Fits a linear drift model from the two refined anchor offsets.

This approach is more robust to Arduino's irregular sampling and drop-outs because it relies on two short, high-energy events rather than the full signal.

```bash
uv run -m sync.calibration_sync <recording_name>
# e.g. uv run -m sync.calibration_sync 2026-02-26_5
```

**Key parameters:**

- `--stage-in <stage>` (default: `parsed`) — input stage.
- `--sample-rate-hz <float>` (default: 100.0) — resampling rate for refinement.
- `--max-lag-seconds <float>` (default: 120.0) — search range for SDA fallback.

**Requirement:** The recording must have at least two calibration sequences (opening + closing) detectable in the reference sensor. Recordings without calibration data fall back with an error and no output is written.

**Output:** `data/<session_name>/synced_cal/`

---

#### Understanding `sync_info.json`

Both methods write a `sync_info.json` to their output directory:

```json
{
  "offset_seconds": 1772109177.458,
  "drift_seconds_per_second": 0.000290,
  "target_time_origin_seconds": 32.64,
  "correlation": {
    "offset_only": 0.11,
    "offset_and_drift": 0.30,
    "signal": "acc_norm"
  }
}
```

| Field | Meaning |
|---|---|
| `offset_seconds` | The constant part of the time mapping: `t_sporsa ≈ t_arduino_boot + offset`. For Unix-epoch Sporsa data this will be a large number (~1.77 × 10⁹). |
| `drift_seconds_per_second` | Clock drift rate between devices. Multiply by 10⁶ to get ppm. Typical Arduino drift is 0–2 000 ppm; values > 10 000 ppm indicate a failed refinement (calibration method falls back to 0). |
| `target_time_origin_seconds` | The Arduino timestamp (in seconds from boot) used as the model's time origin. |
| `correlation.offset_only` | Pearson r of `acc_norm` after offset-only correction, before drift removal. Low values (< 0.1) indicate a coarse offset error or poor signal overlap. |
| `correlation.offset_and_drift` | Pearson r after full offset + drift correction. Higher is better. Values > 0.2 indicate good alignment; < 0.05 suggests residual misalignment. |

The calibration method additionally stores:

```json
{
  "sync_method": "calibration_windows",
  "calibration": {
    "opening": { "offset_s": …, "t_tgt_s": …, "score": 1.4, "window_duration_s": 5.8 },
    "closing": { "offset_s": …, "t_tgt_s": …, "score": 1.1, "window_duration_s": 5.9 },
    "calibration_span_s": 421.0
  }
}
```

| Calibration field | Meaning |
|---|---|
| `score` | Cross-correlation peak height for the refinement window. Values > 1.0 indicate a clean, unambiguous match. Values < 0.5 suggest the cross-correlation locked onto a weak or spurious peak — treat the result with caution. |
| `calibration_span_s` | Time between opening and closing calibration in the target stream. Longer spans give more reliable drift estimates; spans < 60 s are too short to resolve drift reliably. |

---

#### Compare both methods

```bash
uv run -m sync.compare_sync <recording_name> [--plot]
uv run -m sync.compare_sync <session_prefix> --all [--plot]
# e.g. uv run -m sync.compare_sync 2026-02-26 --all --plot
```

Prints a side-by-side table for each recording and, with `--plot`, saves a `sync_method_comparison.png` overlaying the `acc_norm` from both synced stages.

**Interpreting the comparison output:**

| Metric | What to look for |
|---|---|
| **Δ offset (cal − lida)** | If both methods agree within ±1 s the offset is reliable. Differences of seconds indicate one method found a wrong coarse lag — the one with the higher final correlation is more trustworthy. Differences > 10 s almost always mean the SDA step failed for that recording. |
| **Drift (ppm)** | Both methods should give values in the same ballpark (< 2 000 ppm for typical Arduino modules). If one method shows 0 ppm it fell back to the no-drift default; the other method's estimate is likely better. If both are near 0 the recording is short (< 60 s calibration span) and drift cannot be resolved. |
| **Δ drift (cal − lida, ppm)** | Small absolute difference (< 500 ppm) = good agreement. Large difference with cal_drift = 0 means the calibration refinement failed for drift. Large difference with a plausible lida value may indicate the LIDA sliding-window was pulled by structural correlation artefacts rather than true drift. |
| **Corr offset + drift** | The final alignment quality. For recordings with clear shared motion events, values > 0.2 are expected. Values near 0 or negative despite apparent visual alignment often arise from Arduino drop-out artefacts diluting the Pearson computation, not from poor synchronization. |
| **Cal score (opening/closing)** | Scores > 1.0 mean the calibration window was cleanly detected. Scores < 0.5 warn of a poor anchor — drift from that recording should be treated as approximate. |

**When to prefer each method:**

- **Use `synced_cal`** when both calibration scores are ≥ 0.5 and the calibration span is ≥ 60 s. The anchor-based drift is physically grounded and not susceptible to spurious spectral correlations.
- **Use `synced_lida`** when no calibration sequences are present (recordings `_1`, `_6`) or when the calibration method fails. Also useful as a cross-check: if both methods agree within 1 s offset and 500 ppm drift the result is highly reliable.
- **Flag for manual inspection** any recording where `|Δ offset| > 5 s`, both final correlations are < 0.05, or any calibration score is < 0.3.


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
