# `parser/` — Raw Log Parsing, Timing Stats, and Section Splitting

`parser/` converts raw session logs into normalized IMU CSVs, computes basic stream-quality statistics, detects the opening/closing calibration routines, and splits recordings into section folders.

## What it does

- Parses raw `Sporsa` text logs into standardized IMU CSVs.
- Parses raw Arduino BLE logs and merges accel/gyro/mag packets by device timestamp.
- Writes per-recording parsed outputs under `data/recordings/<recording>/parsed/`.
- Computes timing and gap statistics in `session_stats.json`.
- Detects calibration sequences and stores them in `parsed/calibration_segments.json`.
- Splits synchronized or parsed recordings into `data/sections/<recording>s<section>/`.
- Transfers recording-level interval labels to section-level label files during splitting.
- Provides GPS parsers for GPX, CSV, and NMEA files.

## Package layout and file status

- `session.py` (FINAL): session-level parser/orchestrator and the main `python -m parser` entry point.
- `arduino.py` (FINAL): raw Arduino BLE log parser.
- `sporsa.py` (FINAL): raw Sporsa log parser.
- `stats.py` (FINAL): timing, rate, jitter, and drift summaries for parsed CSVs.
- `calibration_segments.py` (FINAL): calibration-sequence detection and JSON export/loading.
- `split_sections.py` (STALE): split recordings into sections and optionally re-sync each section.
- `gps.py` (UNUSED): GPS parsers for GPX, CSV, and NMEA text sources.

## CLI usage

From `analysis/`:

```bash
uv run -m parser 2026-02-26
uv run -m parser 2026-02-26 --no-plot
uv run -m parser.stats 2026-02-26_r5
uv run -m parser.split_sections 2026-02-26_r5/synced
```

## Session parsing outputs

For a session folder `data/_sessions/<date>/` containing `arduino/*.txt` and `sporsa/*.txt`, `process_session()`:

- matches files into recording pairs like `<date>_r1`, `<date>_r2`, ...
- writes `parsed/sporsa.csv` and `parsed/arduino.csv`
- writes parsed-stage plots when enabled
- writes `parsed/session_stats.json`
- writes `parsed/calibration_segments.json`

## Parsed sensor outputs

Both parsers emit the shared IMU schema with millisecond timestamps in the `timestamp` column:

```text
timestamp, ax, ay, az, acc_norm, gx, gy, gz, gyro_norm, mx, my, mz, mag_norm
```

Arduino parsing also keeps `timestamp_received` when the host-side receive time can be recovered from the BLE log. That extra column is used for drift estimation in `stats.py`.

## Sensor-specific behavior

- `sporsa.py` parses comma-separated integer samples and rescales them to SI-like units used by the pipeline.
- `arduino.py` decodes BLE payloads matching the Arduino `SensorData` struct and merges packets with the same device `millis()` timestamp into one row.
- Magnetometer values may be missing at the start of Arduino recordings and remain `NaN`.

## Timing stats

`stats.py` summarizes each CSV stream with:

- sample count
- start/end timestamps and duration
- median and standard deviation of inter-sample intervals
- estimated sampling rate
- simple gap and missing-sample heuristics
- Arduino device-to-received drift summary when `timestamp_received` is available

## Section splitting

`split_sections.py` detects calibration segments in a reference sensor, slices every requested sensor into matching section windows, and writes section folders like:

```text
data/sections/<recording_name>s<section_idx>/
    sporsa.csv
    arduino.csv
    labels/labels.csv         # when recording-level labels can be transferred
    sync_info.json            # when per-section sync runs
    *.png                     # when plots are enabled
```

If fewer than two calibration sequences are found, the full recording is written as section `s1`.
