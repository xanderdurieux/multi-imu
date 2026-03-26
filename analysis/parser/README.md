# `parser/` — Raw Log Parsing and Section Splitting

This package converts raw device logs from the Sporsa and Arduino IMUs into
standardized per-recording CSV files, computes timing quality statistics, and
splits synchronized recordings into calibration-bounded sections.

---

## Package layout

```
parser/
├── __init__.py          public API
├── arduino.py           Arduino BLE log parser
├── arduino_batched.py   Arduino batched-packet parser variant
├── sporsa.py            Sporsa log parser
├── stats.py             timing and quality statistics
├── split_sections.py    section splitting from calibration sequences
└── session.py           session-level orchestrator + CLI
```

---

## Modules

### `session.py` — session-level orchestrator

The top-level entry point. Discovers all raw log files for a session,
matches them into recording pairs (one Sporsa + one Arduino file per
recording index), parses each pair, writes normalized CSVs, generates
initial diagnostic plots, and records timing statistics.

```bash
uv run -m parser.session 2026-02-26
```

- **Input**: `data/sessions/2026-02-26/{arduino,sporsa}/*.txt`
- **Output** per recording `2026-02-26_r<k>`:
  - `session_stats.json` — stream timing stats (recording root)
  - `parsed/sporsa.csv`
  - `parsed/arduino.csv`
  - `parsed/*.png` — diagnostic plots (acc_norm, calibration segments, timing)
- **Output** for the whole session:
  - `data/sessions/2026-02-26/session_stats.json` — compact summary across recordings

**File matching:** recording numbers are extracted from filenames using patterns
like `session10`, `log3`, or the last integer found. Sporsa and Arduino files
with the same number are paired into one recording.

---

### `sporsa.py` — Sporsa log parser

Parses the Sporsa (bicycle) raw text log into a normalized DataFrame.

**Output columns:**

| Column | Units | Description |
|---|---|---|
| `timestamp` | ms | Unix epoch milliseconds |
| `ax, ay, az` | m/s² | Accelerometer (body frame) |
| `gx, gy, gz` | deg/s | Gyroscope (body frame) |
| `mx, my, mz` | µT | Magnetometer (body frame) |

---

### `arduino.py` / `arduino_batched.py` — Arduino log parsers

Parses the Arduino (helmet) BLE log. The Arduino transmits two formats:
- **Standard**: one IMU sample per BLE packet.
- **Batched**: multiple IMU samples bundled in one BLE packet to reduce
  transmission latency.

**Output columns:**

| Column | Units | Description |
|---|---|---|
| `timestamp` | ms | Host-received BLE timestamp (wall clock, epoch ms) |
| `ax, ay, az` | m/s² | Accelerometer (body frame) |
| `gx, gy, gz` | deg/s | Gyroscope (body frame) |
| `mx, my, mz` | µT | Magnetometer (body frame, NaN when BMM150 not ready) |

**Arduino-specific artefacts:**
- **Dropout packets**: occasional near-zero BLE packets (`‖acc‖ ≈ 0`) caused
  by failed transmissions. The calibration and sync packages detect and remove
  these.
- **Clock drift**: the Arduino uses `millis()` (device uptime), which drifts
  relative to the host wall clock by 200–500 ppm. Corrected during
  synchronisation.
- **BMM150 startup delay**: the magnetometer takes ≈ 1.5 s to stabilize after
  power-on, so the first few seconds of `mx/my/mz` may be NaN.

---

### `stats.py` — timing and quality statistics

```python
from parser.stats import compute_stream_timing_stats, write_recording_stats

write_recording_stats("2026-02-26_r5")  # reads parsed/*.csv, writes recording/session_stats.json
```

Computes per-sensor timing quality metrics and writes them to
`data/recordings/<recording>/session_stats.json`.

**Key metrics:**

| Metric | Description |
|---|---|
| `median_interval_ms` | Median inter-sample interval (nominal: 10 ms Sporsa, 17 ms Arduino) |
| `std_ms` | Interval standard deviation (jitter) |
| `gap_count` | Number of intervals > 1.5× median (packet loss events) |
| `missing_samples` | Estimated total missing samples from gap events |
| `arduino_drift_ppm` | Fitted clock drift from host vs device timestamp comparison |
| `arduino_drift_r2` | R² of the linear drift fit (quality indicator) |

**Interpreting `session_stats.json`:**
- `gap_count` > 10% of samples → moderate packet loss; inspect `plot_timing` plots.
- `std_ms` > 5 ms → significant jitter; may affect sync quality.
- `arduino_drift_r2` < 0.9 → poor drift fit; drift correction may be unreliable.

---

### `split_sections.py` — section splitting

Detects calibration sequences in a synchronized recording and splits all
sensor CSVs into calibration-bounded **sections**.

```bash
uv run -m parser.split_sections 2026-02-26_r5/synced_cal
uv run -m parser.split_sections 2026-02-26_r5/synced_cal --no-plot
```

```python
from parser.split_sections import split_recording_into_sections

section_dirs = split_recording_into_sections(
    recording_name="2026-02-26_r5",
    stage_in="synced_cal",
    sensors=["sporsa", "arduino"],
    reference_sensor="sporsa",
    plot=True,
)
```

**How it works:**
1. Detect calibration segments in the reference sensor (Sporsa) using
   `common.calibration_segments.find_calibration_segments`.
2. For a recording with N calibration segments, produce N−1 sections.
   Section `k` spans from the end of calibration segment `k` to the start
   of calibration segment `k+1`, including a small overlap with each flanking
   calibration for context.
3. Write `data/sections/<recording_name>s<section_k>/sporsa.csv` and
   `data/sections/<recording_name>s<section_k>/arduino.csv`
   for each section.
4. Optionally generate per-section sensor and comparison plots.

**Output layout:**

```
data/sections/<recording_name>s<section_idx>/
    sporsa.csv
    arduino.csv
    *.png (if plot=True)
```

**Section inclusion:** sections where only one calibration sequence is
detected (e.g. the recording was stopped early) produce a degenerate split.
The main README describes inclusion criteria for downstream analysis.

---

## Normalized CSV schema

All CSVs produced by the parser follow this schema (defined in
`common.csv_schema`):

```
timestamp, ax, ay, az, gx, gy, gz, mx, my, mz
```

| Column | Units | Notes |
|---|---|---|
| `timestamp` | ms | Unix epoch ms (both sensors after sync) |
| `ax, ay, az` | m/s² | Accelerometer — body frame |
| `gx, gy, gz` | deg/s | Gyroscope — body frame |
| `mx, my, mz` | µT | Magnetometer — body frame, NaN when unavailable |

Missing columns are filled with `NaN` to maintain a consistent layout across
all pipeline stages.
