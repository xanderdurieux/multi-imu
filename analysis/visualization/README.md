# Visualization

This package produces diagnostic and thesis-quality figures for the two-sensor
IMU pipeline. Each module targets one processing stage and can be run from the
CLI or via `plot_session`. Stages such as `calibrated/` and `orientation/` must
already exist on disk (this repository does not generate them).

---

## Module overview

| Module | Targets stage | Key outputs |
|---|---|---|
| `plot_timing` | `parsed/` | Packet interval histograms, interval timeline, Arduino clock drift |
| `plot_sensor` | any | Per-sensor raw signal time series |
| `plot_comparison` | any | Two-sensor overlay on shared axes |
| `plot_calibration_segments` | `parsed/` | Calibration segment detection diagnostics |
| `plot_sync` | `synced/` | Sync method comparison, pre/post alignment |
| `plot_calibration` | `calibrated/` | World-frame sensor signals, calibration quality |
| `plot_orientation` | `orientation/` | Euler angles, linear acceleration, relative head-bike orientation |
| `plot_session` | all | Orchestrator — runs all of the above for a full session |

---

## `plot_session` — run everything at once

The recommended entry point.  Iterates every recording in a session and
dispatches the right plotter for each stage.

```bash
# Plot all recordings from session 2026-02-26
python -m visualization.plot_session 2026-02-26

# Plot only the parsed stage across all recordings
python -m visualization.plot_session 2026-02-26 --stage parsed

# Plot a single recording
python -m visualization.plot_session 2026-02-26 --stage synced
```

Intermediate sync stages (`synced_lida`, `synced_cal`, `synced_sda`,
`synced_online`) are intentionally skipped — use `plot_sync` to compare those
methods instead of viewing them individually.

---

## `plot_timing` — packet timing quality

**Saved in:** `parsed/`

```bash
python -m visualization.plot_timing 2026-02-26_r5
python -m visualization.plot_timing 2026-02-26_r5 --no-drift
```

### Outputs

#### `parsed_timing_intervals.png`

Histograms of inter-packet intervals (ms) for Sporsa and Arduino.

| What to look for | What it means |
|---|---|
| Sharp peak near 10 ms (Sporsa) | Nominal 100 Hz delivery — expected |
| Sharp peak near 17 ms (Arduino) | Nominal ~58 Hz BLE delivery — expected |
| Long tail or secondary peak | Occasional packet doubling/merging |
| Intervals beyond the red dashed gap threshold | Packet loss events |

The annotation box shows the median interval, standard deviation, and the total
number of gaps exceeding the threshold.

#### `parsed_timing_timeline.png`

Inter-packet interval plotted over recording time with red vertical markers at
each gap event.

| What to look for | What it means |
|---|---|
| Baseline around the median | Steady delivery — healthy |
| Sparse red gap markers | Occasional BLE dropout — acceptable |
| Dense red bands | Sustained packet loss (e.g. phone put down, BLE range issue) |
| Sudden step change in baseline | Device reset or clock discontinuity |

This plot is the primary tool for deciding whether a recording is usable.

#### `parsed_clock_drift.png` *(Arduino only)*

Scatter of the host-received timestamp minus the device timestamp (seconds)
versus device time, with a linear fit.

The Arduino BLE device runs its own independent clock (milliseconds since
boot).  The host phone records the wall-clock time when each BLE packet
arrives.  Because these two clocks tick at slightly different rates, the
offset grows linearly — this is *clock drift*.

| Panel | What to look for |
|---|---|
| Top — offset vs time | Points should fall tightly on the fit line; a clear linear trend confirms drift is present and can be corrected |
| Bottom — residuals | Should be centred near zero with small spread (< 2 ms); large residuals indicate noise or BLE retransmission jitter |
| Annotation box | `drift (fit)` and `drift (stored)` should agree closely; R² near 1 confirms the linear model is a good fit |

A drift of ±300 ppm means the Arduino clock runs 0.3 ms fast (or slow) per
second relative to the host — equivalent to a ~150 ms offset over a 500-second
recording.  This is why drift correction is essential during synchronisation.

---

## `plot_sync` — synchronisation quality

**Saved in:** `synced/`

```bash
python -m visualization.plot_sync 2026-02-26_r5
python -m visualization.plot_sync 2026-02-26_r5 --no-alignment
```

### Outputs

#### `sync_method_comparison.png`

Horizontal bar chart with two panels:
- **Left** — Pearson r of the `‖acc‖` signal overlap (higher is better).
- **Right** — Estimated clock drift magnitude in ppm (lower is more conservative; 0 means drift was not modelled).

The selected (best) method is highlighted with a red border.

| What to look for | What it means |
|---|---|
| One bar clearly ahead of the others | The winning method captures the true alignment well |
| Several bars near zero | Short or featureless recording; correlation is unreliable |
| Calibration method selected | Drift was corrected using two calibration anchors — the most accurate approach |
| SDA or LIDA selected | No calibration windows were detected; offset-only or reduced-drift correction was used |

The four methods are:

| Method | Description |
|---|---|
| **Calibration windows** | Cross-correlation within two calibration shaking bursts (opening + closing) gives both offset and drift with the best anchor quality |
| **LIDA** | SDA offset + linear drift estimated from the full signal; depends on assuming sensor motion is similar over long windows |
| **SDA** | Offset-only cross-correlation at a fixed drift of 0; fast but ignores drift |
| **Online** | Opening-anchor offset + pre-characterised drift from the device model; useful as a fallback when no closing anchor is available |

#### `sync_alignment.png`

Side-by-side `‖acc‖` (accelerometer norm) overlay.

- **Before sync (left):** Each sensor is plotted from its own t = 0.  The Sporsa
  uses epoch milliseconds; the Arduino uses device uptime milliseconds.  Shared
  motion features (e.g. a bump or brake) appear at completely different x-positions,
  illustrating why a shared time reference is needed.

- **After sync (right):** Both streams share the same time axis.  Shared motion
  events should appear at the same x-position.

| What to look for | What it means |
|---|---|
| Features align vertically after sync | Good synchronisation |
| Residual horizontal offset of peaks | Sub-optimal sync method; consider checking if calibration windows were detected |
| One trace ends much earlier | Duration mismatch — one sensor was stopped earlier |

---

## `plot_sensor` — raw signal time series

**Saved in:** the target stage directory.

```bash
python -m visualization.plot_sensor 2026-02-26_r5/parsed sporsa
python -m visualization.plot_sensor 2026-02-26_r5/synced arduino --acc --norm
```

Plots the accelerometer, gyroscope, and magnetometer axes for a single sensor.
Near-zero dropout packets (where `‖acc‖ ≈ 0`) are masked to NaN so they appear
as gaps rather than artefact spikes.

`--norm` plots the vector magnitude instead of individual axes.  `--acc`,
`--gyro`, `--mag` restrict the output to one sensor type.

---

## `plot_comparison` — two-sensor overlay

**Saved in:** the target stage directory.

```bash
python -m visualization.plot_comparison 2026-02-26_r5/synced
python -m visualization.plot_comparison 2026-02-26_r5/synced --norm
```

Overlays Sporsa (handlebar) and Arduino (helmet) on the same axes.  When both
streams share an epoch-ms time reference (i.e. after sync), a common x-axis is
used automatically; for the `parsed` stage each stream is normalised to its own
start.

Use this plot to:
- Visually confirm sync quality by checking whether shared motion events coincide.
- Compare signal amplitudes between the two IMU placements.
- Identify periods where one sensor was stationary while the other was in motion
  (e.g. the rider stood still while the bike was moved).

---

## `plot_calibration_segments` — calibration detection diagnostics

**Saved in:** `parsed/`  (called automatically by the parser).

```python
from visualization.plot_calibration_segments import plot_calibration_segments_from_detection
segments, info_df, path = plot_calibration_segments_from_detection(df, out_path="diag.png")
```

Two-panel figure:
- **Top:** Dynamic acceleration magnitude `|‖acc‖ − g|` (raw in grey, smoothed in blue).
- **Bottom:** Detected calibration segments as coloured bands with peak markers.

A calibration segment consists of a static period (|signal| below threshold) immediately
followed by ≥ 3 sharp tapping peaks within a short window.  These segments serve as:
1. Synchronisation anchors for the calibration-window sync method.
2. Section boundaries (the recording is split between each pair of calibration events).
3. Static windows for bias and gravity estimation during sensor calibration.

| What to look for | What it means |
|---|---|
| At least two segments per recording | Minimum for calibration-window sync and section splitting |
| Segments at the start and end | Ideal — maximises the calibration time span used for drift estimation |
| No segments detected | Recording was too short, or tapping protocol was not performed |

---

## `plot_calibration` — world-frame sensor signals

**Saved in:** `calibrated/`

```bash
python -m visualization.plot_calibration 2026-02-26_r5
python -m visualization.plot_calibration 2026-02-26_r5 --no-comparison
```

### Outputs

#### `<sensor>_world.png`

3 × 3 grid of accelerometer, gyroscope, and magnetometer signals after rotation
into the ENU (East–North–Up) world frame.

| What to look for | What it means |
|---|---|
| `az` trace centred near +9.81 m/s² (dashed reference) | Specific force correctly aligned to world +Z (Up) |
| Gyroscope traces centred near 0 deg/s during static periods | Gyro bias successfully removed |
| Annotation box — `residual` near 0 | Small gravity residual → good static calibration quality |
| `yaw_cal = True` | Magnetometer was available and used to fix the yaw reference |

#### `sporsa_vs_arduino_world.png` / `_norm.png`

Overlay of both sensors in the same ENU world frame.  Because both are
expressed in the same coordinate system, differences now reflect true
differences in sensor placement (handlebar vs helmet) rather than clock
or orientation artefacts.

---

## `plot_orientation` — orientation and head motion

**Saved in:** `orientation/`

```python
from visualization.plot_orientation import plot_orientation_stage
plot_orientation_stage("2026-02-26_r5")
```

### Per-file outputs

#### `*_orientation_euler.png`

Yaw, pitch, and roll in degrees over time.  These come from the on-board
complementary or Madgwick filter.

| Angle | Cycling interpretation |
|---|---|
| **Yaw** | Heading — turns left/right |
| **Pitch** | Forward lean — braking, accelerating, hill gradient |
| **Roll** | Lateral tilt — cornering |

#### `*_orientation_linacc_world.png`

World-frame linear acceleration (body-frame acceleration rotated to ENU,
gravity removed).  Residual near zero during truly static periods confirms
good orientation tracking.  Peaks correspond to actual linear accelerations
— braking, acceleration, or impacts.

### Multi-sensor outputs

#### `orientation_comparison.png`

Sporsa (handlebar) and Arduino (helmet) Euler angles on the same axes.
Shows how closely the two sensors track each other during periods of shared
motion (both tilt together when cornering) and how they diverge when the
rider moves their head independently.

#### `orientation_relative.png`

Difference angles (Δyaw, Δpitch, Δroll) = helmet − handlebar, interpolated
to a common time grid.

| Angle | What it shows |
|---|---|
| **ΔYaw** | Head turns — looking left/right while riding |
| **ΔPitch** | Head nod — looking down at phone/computer or up at an obstacle |
| **ΔRoll** | Head tilt — lateral head lean independent of the bicycle |

A flat line near 0° means the rider's head is aligned with the handlebar.
Sustained deviations indicate intentional head movements — checking a mirror,
looking at a hazard, or a moment of inattention.

---

## Internal utilities

### `_utils.py`

Shared helpers used by all plot modules.

| Function | Description |
|---|---|
| `mask_dropout_packets(df)` | Sets sensor columns to NaN for near-zero packets so gaps render as breaks rather than spikes |
| `time_axis_seconds(timestamps_ms)` | Converts millisecond timestamps to seconds from the first sample |
| `acc_norm_series(df)` | Computes `‖acc‖` row-wise |

### `labels.py`

Canonical column name groups (`SENSOR_COMPONENTS`) and display labels
(`SENSOR_LABELS`) shared across all plot modules.

---

## Output file naming convention

| Pattern | Description |
|---|---|
| `<sensor>.png` | Full three-axis sensor plot |
| `<sensor>_acc_norm.png` | Accelerometer norm for one sensor |
| `<sensor_a>_vs_<sensor_b>.png` | Two-sensor axis-by-axis comparison |
| `<sensor_a>_vs_<sensor_b>_norm.png` | Two-sensor norm comparison |
| `parsed_timing_intervals.png` | Interval histograms |
| `parsed_timing_timeline.png` | Interval timeline |
| `parsed_clock_drift.png` | Arduino clock drift |
| `sync_method_comparison.png` | Sync method bar chart |
| `sync_alignment.png` | Pre/post-sync acc_norm overlay |
| `<sensor>_world.png` | Calibrated ENU-frame sensor plot |
| `sporsa_vs_arduino_world.png` | Calibrated world-frame comparison |
| `*_orientation_euler.png` | Euler angles over time |
| `*_orientation_linacc_world.png` | World-frame linear acceleration |
| `orientation_comparison.png` | Both-sensor Euler overlay |
| `orientation_relative.png` | Relative head-bike orientation |
