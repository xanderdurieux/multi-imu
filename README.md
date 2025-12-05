# multi-imu

Lightweight helpers for turning raw IMU logs into clean CSVs and loading them back into Python.

## Features
- Parse Arduino BLE logs and Sporsa UART captures into accelerometer/gyroscope dataframes.
- Export standardized CSVs with consistent column names and timestamps.
- Combine accelerometer and gyroscope streams into a single IMU CSV.
- Load an IMU CSV into an `IMUSensorData` container for downstream work.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart
Run the conversion example to turn the bundled raw logs into standardized CSVs and load them
(ensure the repository root is on `PYTHONPATH`):
```bash
PYTHONPATH=src python3 examples/pipeline_example.py
```

Or load an existing standardized CSV directly:
```python
from multi_imu import load_imu_csv

arduino = load_imu_csv("data/processed/session1/arduino.csv", name="arduino", sample_rate_hz=50.0)
print(arduino.data.head())
```

### Converting raw logs
Use `examples/convert_raw.py` to translate raw Arduino or Sporsa captures into the
library's standardized CSV schema using device-recorded timestamps. Joined IMU CSVs are
written to `data/processed/sessionX/arduino.csv` and `data/processed/sessionX/sporsa.csv`.

```bash
PYTHONPATH=src python3 examples/convert_raw.py
```
