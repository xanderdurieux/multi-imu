# `calibration/` — Static Sensor Calibration

This package characterises each IMU sensor against a world frame (ENU:
East–North–Up) using the static windows that flank the tap-burst sequences
recorded at the start and end of every session. It is an offline, per-recording
process: run it once on `parsed/` data and the results persist in
`calibrated/`.

---

## Purpose

Raw IMU sensors have two main limitations that prevent direct cross-sensor
comparison:

1. **Sensor biases** — the gyroscope reads a non-zero angular velocity even
   when the sensor is stationary, and the accelerometer may report a gravity
   vector that is not aligned with the expected world-up axis.
2. **Unknown orientation** — each sensor is physically mounted at an arbitrary
   angle. Without knowing this angle, accelerations along "x/y/z" mean
   different things for the Sporsa (handlebar) and Arduino (helmet).

This package solves both problems:

- **Gyroscope bias**: estimated from the mean gyro readings during static
  windows, then subtracted from all samples.
- **Magnetometer hard-iron offset**: estimated from the mean magnetometer
  reading during static windows; subtracted from all magnetometer samples.
- **Sensor-to-world rotation**: the 3 × 3 matrix `R` that maps sensor-frame
  vectors into ENU world-frame vectors, computed via TRIAD (gravity + magnetic
  field references) or gravity-only as a fallback.

---

## Package layout

```
calibration/
├── __init__.py          public API
├── segments.py          calibration-sequence detector (shared across pipeline)
├── static_windows.py    extract static sub-DataFrames from detected sequences
├── per_sensor.py        estimate gyro bias, gravity vector, mag hard-iron
├── orientation.py       compute sensor-to-world rotation (TRIAD / gravity-only)
└── session.py           recording-level orchestrator + CLI
```

---

## Calibration sequence

Each recording starts and ends with a deliberate **calibration sequence**:

1. Hold the sensor perfectly still for ≈ 5 seconds (**pre-static window**).
2. Give the sensor ≥ 3 sharp taps or shakes (**peak burst**).
3. Hold still again for ≈ 5 seconds (**post-static window**).

The burst serves as both a timing anchor (for synchronisation) and a boundary
marker. The flanking static windows are used for bias and gravity estimation.

---

## Modules

### `segments.py` — calibration-sequence detector

The single shared definition for detecting calibration sequences across the
entire pipeline. Used by `static_windows.py`, `parser.split_sections`, and
`sync.calibration_sync`.

**Key types:**

```python
@dataclass
class CalibrationSegment:
    start_idx: int          # first static sample
    end_idx: int            # last static sample
    peak_indices: list[int] # indices of the tap peaks
```

**Key function:**

```python
segments = find_calibration_segments(
    df,
    sample_rate_hz=100.0,
    static_min_s=3.0,       # minimum flanking static duration
    static_threshold=1.5,   # |acc_norm - g| threshold for "static" (m/s²)
    peak_min_height=3.0,    # minimum peak height above g (m/s²)
    peak_min_count=3,       # minimum number of peaks to form a segment
    peak_max_gap_s=3.0,     # maximum gap between consecutive peaks
    static_gap_max_s=5.0,   # maximum gap between static and first peak
)
```

The detector also exposes `_acc_norm`, `_smooth`, and `_find_peaks` as
internal helpers — these are imported by `sync.calibration_sync` for peak
cluster matching.

---

### `static_windows.py` — extract static sub-DataFrames

```python
windows = extract_static_windows(df, sample_rate_hz=100.0, buffer_samples=10)
```

Returns a `StaticWindows` dataclass:

```python
@dataclass
class StaticWindows:
    segments: list[tuple[DataFrame, DataFrame]]  # (pre_static, post_static) per segment
    combined: DataFrame     # all static samples concatenated
    n_calibration_segments: int
```

**Notes:**
- Near-zero dropout packets (Arduino BLE artefacts) are removed before segment
  detection to prevent false peaks.
- The sample rate is auto-detected from the median timestamp interval, which
  is important for the Arduino whose effective rate drops from 100 Hz to ≈ 55 Hz
  after dropout removal.
- `mag_subset(min_samples)` returns only rows with valid magnetometer data,
  with a fallback to post-static windows only (handles the BMM150 ≈ 1.5 s
  startup delay that blanks the first pre-static window).

---

### `per_sensor.py` — bias and gravity estimation

```python
from calibration.per_sensor import calibrate_sensor, SensorCalibration

sensor_cal = calibrate_sensor(windows)
```

Returns a `SensorCalibration` dataclass:

| Field | Units | Description |
|---|---|---|
| `gyro_bias_deg_per_s` | deg/s | Mean gyro reading during static — subtract from raw gyro |
| `gravity_vector_m_per_s2` | m/s² | Mean accelerometer during static — direction + magnitude of gravity in sensor frame |
| `gravity_magnitude_m_per_s2` | m/s² | `‖gravity_vector‖` — should be close to 9.81 |
| `mag_hard_iron_uT` | µT | Mean magnetometer during static — the Earth field + hard-iron offset |
| `n_static_samples` | — | Number of acc/gyro samples used |
| `n_mag_samples` | — | Number of magnetometer samples used |
| `yaw_calibrated` | bool | `True` when magnetometer was available |

**Quality check:** `gravity_magnitude_m_per_s2` should be within 0.5 m/s² of
9.81. Large deviations indicate that the static windows were not truly static
(e.g. the sensor was moved during calibration).

---

### `orientation.py` — sensor-to-world rotation

```python
from calibration.orientation import compute_orientation_from_vectors, OrientationCalibration

orientation = compute_orientation_from_vectors(
    gravity_sensor=sensor_cal.gravity_vector_m_per_s2,
    mag_sensor_corrected=sensor_cal.mag_hard_iron_uT,  # None → gravity-only fallback
)
```

Returns an `OrientationCalibration` dataclass:

| Field | Description |
|---|---|
| `rotation_sensor_to_world` | 3 × 3 rotation matrix mapping sensor frame → ENU world frame |
| `gravity_residual_m_per_s2` | `‖R @ g_sensor - g_world‖` — quality metric (lower is better) |
| `yaw_calibrated` | `True` when TRIAD was used (magnetometer available) |

**TRIAD method** (when magnetometer is available):
- Primary reference pair: measured specific-force vector `acc_static` (the accelerometer output, pointing upward) and world +Z `[0, 0, +9.81]`.
- Secondary reference pair: measured magnetic field direction and expected horizontal magnetic field direction.
- Builds orthonormal triads from each pair and computes `R = M_world @ M_sensor.T`.

**Gravity-only fallback** (no magnetometer):
- Uses Rodrigues' minimum rotation to align the measured gravity direction to
  world -Z. Yaw is set to zero (arbitrary heading reference).

---

### `session.py` — recording-level orchestrator

```python
from calibration.session import calibrate_recording

json_path = calibrate_recording(
    "2026-02-26_5",
    stage_in="parsed",   # input stage
    apply=True,          # write calibrated CSVs
    plot=True,           # generate calibration plots
)
```

**CLI:**

```bash
uv run -m calibration.session 2026-02-26_5
uv run -m calibration.session 2026-02-26_5 --stage parsed --no-apply --no-plot
```

**Pipeline steps** (per sensor):

1. Detect calibration sequences and extract static windows.
2. Estimate gyro bias, gravity vector, and magnetometer hard-iron offset.
3. Compute sensor-to-world rotation (TRIAD or gravity-only).
4. Write `calibrated/calibration.json` with all parameters and quality metrics.
5. If `apply=True`: subtract gyro bias, subtract mag hard-iron offset, rotate
   all vectors (acc, gyro, mag) to the world frame. Write `calibrated/<sensor>.csv`.

---

## `calibration.json` schema

```jsonc
{
  "metadata": {
    "recording": "2026-02-26_5",
    "stage": "parsed",
    "created_at_utc": "2026-02-26T10:00:00+00:00"
  },
  "sporsa": {
    "gyro_bias_deg_per_s": [0.741, -0.374, -1.711],
    "gravity_vector_m_per_s2": [-2.108, 0.276, 9.679],
    "gravity_magnitude_m_per_s2": 9.9092,
    "mag_hard_iron_offset_uT": [263.75, 234.83, -374.10],
    "rotation_sensor_to_world": [[...], [...], [...]],
    "quality": {
      "gravity_residual_m_per_s2": 0.0992,
      "n_static_samples": 6496,
      "n_mag_samples": 6496,
      "yaw_calibrated": true
    }
  },
  "arduino": { /* same structure */ }
}
```

**Quality thresholds:**

| Metric | Good | Marginal | Poor |
|---|---|---|---|
| `gravity_residual_m_per_s2` | ≤ 0.2 | 0.2 – 0.5 | > 0.5 |
| `n_static_samples` | ≥ 500 | 100 – 500 | < 100 |
| `yaw_calibrated` | `true` | — | `false` (gravity-only) |

---

## Outputs

```
data/recordings/<recording>/calibrated/
    calibration.json        calibration parameters and quality metrics
    sporsa.csv              bias-corrected, ENU-rotated Sporsa data
    arduino.csv             bias-corrected, ENU-rotated Arduino data
    <sensor>_world.png      world-frame signal plots (if plot=True)
```

**Note on world-frame CSVs:** The `calibrated/` CSVs have all sensor vectors
(acc, gyro, mag) expressed in the ENU world frame. These are intended for
static analysis and cross-sensor comparison. They are **not** the correct input
for orientation filters — the `orientation/` package reads `parsed/`
(body-frame) data and loads calibration parameters separately from
`calibration.json`.

---

## Recordings without calibration sequences

Some recordings (e.g. short tests or recordings where the tap protocol was
skipped) do not contain detectable calibration sequences. In this case,
`session.py` skips the sensor with a warning and writes an empty
`calibration.json` (metadata only). These recordings cannot be calibrated or
used for orientation estimation.

The `calibration.session` CLI prints:

```
WARNING: skipping sporsa — No calibration segments found in sporsa.csv.
This recording may lack a calibration sequence (e.g. rec1 or rec6).
```
