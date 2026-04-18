# `static_calibration/` — Arduino Hardware Calibration

This package estimates Arduino accelerometer bias/scale and gyroscope bias from stationary calibration logs. It is a hardware-level calibration step for the Arduino sensor only and writes a JSON file that the section-level `calibration/` stage can reuse.

## Expected input

Place raw Arduino BLE logs under:

- `analysis/data/_calibrations/raw/*.txt`

If `raw/` is empty, the code also falls back to `analysis/data/_calibrations/*.txt`.

Each log should be stationary with the board held in an approximately axis-aligned orientation so gravity mainly points along one of:

- `x+`, `x-`, `y+`, `y-`, `z+`, `z-`

The code infers the dominant face from the measured mean acceleration; the filenames do not have to encode the face.

## Run it

From `analysis/`:

```bash
uv run -m static_calibration
```

The default pipeline:

- parses each raw Arduino BLE log
- writes parsed CSVs under `data/_calibrations/parsed/`
- trims a small fraction from each recording edge before summarizing
- estimates accelerometer bias/scale and gyroscope bias
- writes `data/_calibrations/arduino_imu_calibration.json`
- writes overview, detail, and parameter plots under `data/_calibrations/plots/`

## Outputs

### JSON

`arduino_imu_calibration.json` includes:

- `gravity_m_s2`
- `accelerometer.bias`
- `accelerometer.scale`
- `gyroscope.bias_deg_s`
- `face_counts`
- `warnings`
- `source_logs`
- `per_recording`

### Parsed CSVs

The parsed files are written as `logN(<face>).csv` using the normal parser schema.

### Plots

- `plots/recordings_overview.png`
- `plots/recordings/<...>.png`
- `plots/calibration_parameters.png`

## Main modules

- `imu_static.py`: parsing summaries, face inference, parameter estimation, and apply/load helpers
- `plotting.py`: calibration plots
- `run.py`: default pipeline and CLI entry point

## Reuse in the main pipeline

The section-level `calibration/` stage automatically looks for:

```text
data/_calibrations/arduino_imu_calibration.json
```

When found, it uses those accelerometer and gyroscope calibration values for the Arduino intrinsics. Sporsa does not use this static accelerometer calibration path.
