# master-thesis/analysis/parser

This folder contains a small command-line toolchain to convert raw IMU logs into
normalized CSV files for analysis.

## Goal

- Read raw sensor logs from `data/raw/<session_name>/`
- Parse each sensor file (e.g. Arduino Nano 33 BLE, Sporsa)
- Write processed CSVs with a fixed schema:
  - `timestamp, ax, ay, az, gx, gy, gz, mx, my, mz`
- Store the results in `data/processed/<session_name>/`

## Directory layout

Relative to this folder:

- **Raw data input**: `data/raw/<session_name>/`
  - Example:
    - `data/raw/session_1/arduino_imu.txt`
    - `data/raw/session_1/sporsa_imu.txt`
- **Processed output**: `data/processed/<session_name>/`
  - Example:
    - `data/processed/session_1/arduino_imu.csv`
    - `data/processed/session_1/sporsa_imu.csv`

`parse_session.py` inspects all files in the raw session folder:

- Files whose name contains **"arduino"** are parsed with the Arduino parser.
- Files whose name contains **"sporsa"** are parsed with the Sporsa parser.
- Output CSVs are named `<raw_stem>_imu.csv` in the processed folder.

## Components

- `src/parse_common.py` – shared `IMUSample` dataclass and CSV writer.
- `src/parse_arduino.py` – parser for Arduino Nano 33 BLE IMU logs.
- `src/parse_sporsa.py` – parser for Sporsa IMU logs.
- `parse_session.py` – orchestrates parsing of an entire session folder.

## Session parser (`parse_session.py`)

`parse_session.py` ties everything together for a given session:

- Computes:
  - `raw_dir = data/raw/<session_name>/`
  - `out_dir = data/processed/<session_name>/`
- Creates `out_dir` if needed.
- Iterates over files in `raw_dir`:
  - Classifies each by name (`arduino` / `sporsa` / ignore).
  - Chooses the correct parser.
  - Writes `<raw_stem>_imu.csv` in `out_dir`.

### CLI usage
From the `analysis/` folder, run the package as a module:

```bash
uv run -m parser.parse_session <session_name>
# or
python3 -m parser.parse_session <session_name>
```

This reads logs from `data/raw/<session_name>/` and writes processed CSVs to
`data/processed/<session_name>/`.

## Arduino parser (`src/parse_arduino.py`)

- Expects log lines produced by the BLE central, roughly in the form:
  - `A<TAB><something><TAB><hex bytes ...>`
- Only lines starting with `"A"` and containing a hex payload are parsed.
- The hex payload is decoded to match the `SensorData` struct on the Arduino:
  - `uint8_t sensorType`
  - `float x, y, z`
  - `uint32_t timestamp`
- The parser:
  - Converts accelerometer readings to m/s² (via multiplication by 9.81).
  - Leaves gyro and magnetometer values in their logged units.
  - Emits one `IMUSample` per decoded notification, filling only the relevant fields.

### CLI usage

```bash
python3 parser/src/parse_arduino.py <source_txt> <destination_csv>
```

## Sporsa parser (`src/parse_sporsa.py`)

- Expects each line to be:
  - `<timestamp_ms>,<acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<gyro_z>`
- The first field may be prefixed with `uart:~$ `, which is stripped.
- Raw accelerometer and gyroscope counts are converted using:
  - `ACCEL_SENS` (e.g. `"16G"`)
  - `GYRO_SENS` (e.g. `"2000DPS"`)
- The parser returns `IMUSample` objects with:
  - `timestamp` from the device time
  - `ax, ay, az` in m/s²
  - `gx, gy, gz` in deg/s

### CLI usage

```bash
python3 parser/src/parse_sporsa.py <source_txt> <destination_csv>
```