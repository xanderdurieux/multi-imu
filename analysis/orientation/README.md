# `orientation/` — Dynamic Orientation Estimation

This package estimates continuous body→world orientation from IMU streams
using sensor fusion filters. Orientation is represented as a quaternion
`q_bw` that maps body-frame vectors into the world frame:

```
v_world = q_bw ⊗ v_body ⊗ q_bw*
```

---

## Relationship to `calibration/`

The two packages have distinct, complementary roles:

| Package | Input | Output | Purpose |
|---|---|---|---|
| `calibration/` | `parsed/` body-frame data | `calibrated/calibration.json` + world-frame CSVs | Static sensor characterisation — one-time, per-recording |
| `orientation/` | `parsed/` body-frame data + `calibration.json` | `orientation/` quaternion CSVs | Dynamic orientation tracking — continuous, per-sample |

**Why `orientation/` reads `parsed/` and not `calibrated/`:**
The orientation filters (complementary, Madgwick) integrate gyroscope data
and use the accelerometer as a gravitational reference. Both operations require
**body-frame** sensor readings. The `calibrated/` CSVs contain vectors already
rotated to the world frame, which would make gyro integration incorrect and
accelerometer-based correction meaningless.

The calibration contributes two things:
1. **Gyro bias** — subtracted from `gx/gy/gz` before integration to prevent
   constant angular velocity drift.
2. **Initial orientation** — the static sensor-to-world rotation from TRIAD,
   used to seed the filter at the correct starting pose rather than identity.

---

## Package layout

```
orientation/
├── __init__.py       public API
├── quaternion.py     pure quaternion math
├── calibration.py    bridge to calibration/: load params + apply gyro bias
├── complementary.py  complementary filter (gyro + accelerometer tilt)
├── madgwick.py       Madgwick filter (gyro + accelerometer, AHRS backend)
├── pipeline.py       run a filter on an in-memory DataFrame or CSV
└── session.py        recording-level orchestrator + CLI
```

---

## Modules

### `quaternion.py` — quaternion math

Convention: `[w, x, y, z]`, body → world frame.

| Function | Description |
|---|---|
| `quat_identity()` | `[1, 0, 0, 0]` |
| `quat_normalize(q)` | unit-norm quaternion |
| `quat_conjugate(q)` | inverse of a unit quaternion |
| `quat_multiply(q1, q2)` | Hamilton product `q1 ⊗ q2` |
| `quat_from_axis_angle(axis, angle_rad)` | rotation from axis + angle |
| `quat_from_gyro(omega, dt)` | small-angle quaternion from angular velocity |
| `quat_rotate(q_bw, v_body)` | rotate body-frame vector to world frame |
| `quat_slerp(q0, q1, t)` | spherical linear interpolation |
| `euler_from_quat(q)` | Z-Y-X yaw, pitch, roll in radians |
| `quat_from_euler(yaw, pitch, roll)` | construct from Euler angles |
| `quat_from_rotation_matrix(R)` | convert 3×3 rotation matrix (Shepperd's method) |
| `tilt_quat_from_acc(acc_body)` | roll + pitch from accelerometer; yaw = 0 |

---

### `calibration.py` — bridge to `calibration/`

```python
from orientation.calibration import load_calibration_params, apply_gyro_bias

gyro_bias, initial_q = load_calibration_params("2026-02-26_5", "sporsa")
df_debiased = apply_gyro_bias(df, gyro_bias)
```

**`load_calibration_params(recording_name, sensor_name)`**

Reads `calibrated/calibration.json` and returns:
- `gyro_bias` — shape `(3,)`, same units as the raw CSV `gx/gy/gz` columns.
- `initial_q` — shape `(4,)` unit quaternion `[w, x, y, z]` representing the
  static sensor-to-world rotation at calibration time.

Raises `FileNotFoundError` if `calibration.json` does not exist (run
`calibration.session` first).

**`apply_gyro_bias(df, gyro_bias)`**

Returns a copy of `df` with `gyro_bias` subtracted from `gx/gy/gz`.

---

### `complementary.py` — complementary filter

```python
from orientation.complementary import ComplementaryOrientationFilter, ComplementaryFilterConfig

config = ComplementaryFilterConfig(
    accel_correction_gain=0.02,   # fraction of tilt error corrected per step
    gravity=9.81,
    accel_norm_tolerance_g=0.2,   # gating threshold: accept accel if |a| ∈ [0.8g, 1.2g]
)
filt = ComplementaryOrientationFilter(config)
filt.reset(initial_quaternion=initial_q)  # seed with calibration rotation

for k in range(len(df)):
    q = filt.step(dt[k], gyro[k], acc[k])
```

**Algorithm:**
1. Initialize roll/pitch from accelerometer tilt on the first valid sample
   (unless `reset(initial_quaternion=...)` was called — then the filter starts
   immediately at the calibration orientation).
2. Propagate orientation by gyro integration (Euler step).
3. Low-pass correct roll/pitch toward the accelerometer tilt estimate:
   `angle = (1 - gain) * gyro_predicted + gain * accel_measured`.
   Yaw is not observable without a magnetometer — it follows gyro integration only.
4. Gate: if `‖acc‖` deviates from `g` by more than `accel_norm_tolerance_g`,
   the accelerometer correction is skipped for that step.

---

### `madgwick.py` — Madgwick filter

```python
from orientation.madgwick import MadgwickOrientationFilter, MadgwickConfig

config = MadgwickConfig(
    beta=0.1,                    # gradient-descent gain
    accel_norm_tolerance_g=0.2,
)
filt = MadgwickOrientationFilter(config)
filt.reset(initial_quaternion=initial_q)

for k in range(len(df)):
    q = filt.step(dt[k], gyro[k], acc[k])
```

**Algorithm:** delegates to `ahrs.filters.Madgwick.updateIMU` (gyro + accel
step) or `ahrs.filters.AngularRate.update` (gyro-only, when accel is gated).
The gradient-descent step minimises the angle between the rotated
accelerometer reading and the expected world gravity direction.

**Filter comparison:**

| Property | Complementary | Madgwick |
|---|---|---|
| Drift correction | Low-pass blend of gyro + accel tilt | Gradient descent toward gravity |
| Yaw observable? | No (gyro-only yaw) | No (without magnetometer) |
| Tuning | `accel_correction_gain` | `beta` |
| Computational cost | Very low | Low |

---

### `pipeline.py` — filter on DataFrame or CSV

```python
from orientation.pipeline import run_complementary_on_dataframe, run_madgwick_on_dataframe

df_orient = run_complementary_on_dataframe(
    df,
    gyro_bias=gyro_bias,    # optional: subtract before filtering
    initial_q=initial_q,    # optional: starting pose
    config=ComplementaryFilterConfig(),
)
# df_orient has added columns: qw, qx, qy, qz, yaw_deg, pitch_deg, roll_deg
```

Also available:
- `run_madgwick_on_dataframe(df, gyro_bias, initial_q, config)`
- `run_orientation_pipeline_on_csv(csv_path, gyro_bias, initial_q, config)`

**Timestamp handling:** if the median timestamp diff is > 1.0 (indicating
milliseconds), timestamps are divided by 1000 before computing `dt` in seconds.

**CLI (no calibration):**

```bash
uv run -m orientation.pipeline <input.csv>
uv run -m orientation.pipeline <input.csv> --filter madgwick
```

---

### `session.py` — recording-level orchestrator

```python
from orientation.session import run_orientation_for_recording

json_path = run_orientation_for_recording(
    "2026-02-26_5",
    stage_in="parsed",   # body-frame input stage
    gravity=9.81,
)
```

**CLI:**

```bash
uv run -m orientation.session 2026-02-26_5
uv run -m orientation.session 2026-02-26_5 --stage parsed
```

**Requires** `calibrated/calibration.json` (run `calibration.session` first).

**Pipeline** (per sensor):
1. Load `gyro_bias` and `initial_q` from `calibration.json`.
2. Run complementary filter → `<sensor>__complementary_orientation.csv`.
3. Run Madgwick filter → `<sensor>__madgwick_orientation.csv`.
4. Compute quality statistics and write `orientation_stats.json`.

---

## `orientation_stats.json` schema

```jsonc
[
  {
    "recording": "2026-02-26_5",
    "stage_in": "parsed",
    "sensor_name": "sporsa",
    "filter_type": "complementary",
    "output_csv": "...sporsa__complementary_orientation.csv",
    "g_err_mean": 0.04,          // mean(|acc_world| - g)
    "g_err_std": 0.12,
    "g_err_abs_mean": 0.09,      // primary quality metric
    "g_err_abs_p95": 0.31,
    "static_fraction": 0.28,     // fraction of samples at rest
    "pitch_static_std_deg": 0.8, // pitch variability during static
    "roll_static_std_deg": 0.6,
    "num_static_samples": 14230
  },
  // ... one entry per (sensor, filter) pair
]
```

**Quality thresholds:**

| Metric | Good | Marginal | Poor |
|---|---|---|---|
| `g_err_abs_mean` | ≤ 0.3 m/s² | 0.3 – 0.8 m/s² | > 0.8 m/s² |
| `pitch_static_std_deg` / `roll_static_std_deg` | ≤ 2° | 2° – 5° | > 5° |

---

## Outputs

```
data/recordings/<recording>/orientation/
    sporsa__complementary_orientation.csv
    sporsa__madgwick_orientation.csv
    arduino__complementary_orientation.csv
    arduino__madgwick_orientation.csv
    orientation_stats.json
```

Each orientation CSV contains all original columns plus:

| Column | Description |
|---|---|
| `qw, qx, qy, qz` | Body→world quaternion (unit, `[w, x, y, z]`) |
| `yaw_deg` | Heading (rotation around world Z), in degrees |
| `pitch_deg` | Forward lean (rotation around world Y), in degrees |
| `roll_deg` | Lateral tilt (rotation around world X), in degrees |
