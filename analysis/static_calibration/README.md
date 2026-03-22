# Static calibration

Estimate **accelerometer bias and per-axis scale** and **gyroscope bias** from **six short, stationary** Arduino IMU logs—one pose per axis direction so gravity aligns approximately with **+X, −X, +Y, −Y, +Z, −Z** in the sensor frame.

Raw logs are parsed with [`parser.arduino.parse_arduino_log`](../parser/arduino.py) (same BLE hex format as session recordings). Means are taken over the **central** part of each file (default **5%** trimmed from each end by time) to reduce edge transients.

## Data layout

All paths are under **`analysis/data/calibrations/`** (see `calibration_data_dir()` in `imu_static.py`).

| Location | Contents |
|----------|----------|
| `raw/*.txt` | Raw Arduino text logs (default input) |
| `parsed/*.csv` | Parsed IMU CSVs (`calib_<face>.csv` from inferred dominant axis) |
| `arduino_imu_calibration.json` | Estimated parameters + per-recording summaries |
| `plots/` | Diagnostic figures (see below) |

If `raw/` is empty, the pipeline falls back to `*.txt` directly under the calibrations root.

## Run the pipeline

From the **`analysis`** directory (with the project environment, e.g. `uv run`):

```bash
uv run python -m static_calibration
```

Optional arguments: `raw_log_paths`, `parsed_dir`, `output_json`, `plots_dir`, `trim_fraction` (default `0.05`), `write_plots` (default `True`).

## Outputs

### JSON (`arduino_imu_calibration.json`)

- **`accelerometer`**: `bias` and `scale` per axis (`x`, `y`, `z`)
- **`gyroscope`**: `bias_deg_s` per axis
- **`gravity_m_s2`**, **`face_counts`**, **`warnings`**
- **`source_logs`**, **`per_recording`** (stem, `dominant_face`, window times, means, sample counts)

### Plots (`plots/`)

- **`recordings_overview.png`** — per recording: accelerometer **one subplot per axis** (independent y-scale) plus gyro XYZ
- **`recordings/<dominant_face>/`** — one detailed figure per recording (same acc layout + gyro)
- **`calibration_parameters.png`** — fitted parameters, face counts, per-recording means, and boxplots of samples inside each analysis window

## Apply calibration elsewhere

```python
from pathlib import Path

import pandas as pd
from common import load_dataframe, write_dataframe
from static_calibration import calibration_data_dir, apply_calibration_to_dataframe, load_calibration

cal = load_calibration(calibration_data_dir() / "arduino_imu_calibration.json")
df = load_dataframe(Path("…/some_parsed.csv"))
out = apply_calibration_to_dataframe(df, cal)
write_dataframe(out, Path("…/some_calibrated.csv"))
```

## Package layout

- **`imu_static.py`** — trimming, summarization, estimation, CSV export, load/apply helpers
- **`run.py`** — `run_calibration_pipeline`, `main`
- **`plotting.py`** — overview, per-recording, parameter/boxplot figures
- **`__main__.py`** — entrypoint for `python -m static_calibration`