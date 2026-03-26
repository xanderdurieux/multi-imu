# `derived/` — Physically interpretable dual-IMU derived signals

This stage runs **after calibration + orientation** and writes thesis-friendly motion
signals to per-section `derived/` folders.

## Why this exists
Raw IMU summaries are often hard to interpret physically. This module exports direct
motion signals in units and axes that map to cycling dynamics:

- gravity-compensated linear acceleration,
- longitudinal/lateral/vertical acceleration,
- interpretable angular rates,
- tilt (roll/pitch) rates,
- rider-minus-bicycle residual motion,
- bike→rider shock transmission,
- optional robust-normalized versions for cross-session comparison.

## Inputs and dependencies
- `calibrated/{sporsa,arduino}.csv` and `calibrated/calibration.json`
- `orientation/{sporsa,arduino}__<variant>.csv` (for tilt-rate derivatives)

## Alignment dependency (explicit)
- Works with **partial alignment** (`gravity_only`):
  - `lin_acc_world_*`, `acc_vertical_m_s2`, vertical residuals, shock transmission.
- Requires **full horizontal alignment** (`gravity_plus_forward` or `section_horizontal_frame`):
  - `acc_longitudinal_m_s2`, `acc_lateral_m_s2`,
  - `omega_roll_axis_rad_s`, `omega_pitch_axis_rad_s`, `omega_yaw_axis_rad_s`,
  - longitudinal/lateral rider-minus-bike residuals.

Signals that require full alignment are output as `NaN` when not trustworthy, and
quality flags are added to the CSV/metadata.

## Outputs
For each section:
- `derived/sporsa_signals.csv` (bike)
- `derived/arduino_signals.csv` (rider)
- `derived/cross_sensor_signals.csv` (residual + transmission)
- `derived/derived_signals_meta.json` (trust + dependency map)
- `derived/derived_validation_overview.png` (when validation script is run)
- `derived/derived_validation_metrics.json` (when validation script is run)

## Quality flags
Time-series include:
- `quality_full_alignment_ok`
- `quality_orientation_ok`
- `quality_calibration`
- cross-sensor flags like `quality_residual_full_alignment_ok`.

Section-level trust rules are exported in `derived_signals_meta.json`.

## Constants and assumptions (no unexplained magic numbers)
- Gravity constant: `9.81 m/s²`.
- Shock transmission uses 0.25 s rolling RMS (captures short transient bumps while
  suppressing single-sample spikes).
- Robust normalization uses median/MAD with scale factor `1.4826` (standard robust
  estimate of normal-distribution sigma).

## Suggested use by event type
- **Braking**: longitudinal deceleration (`acc_longitudinal_m_s2`), pitch-rate,
  rider-minus-bike longitudinal residual.
- **Bumps**: vertical acceleration peaks + `shock_transmission_gain` + vertical residual.
- **Swerving**: lateral acceleration + yaw-axis angular velocity + roll-rate.
- **Sprinting**: high-frequency longitudinal residual + roll-rate variability + shock gain changes.
- **Rider destabilization**: large residuals across vertical/lateral + elevated tilt rates
  not mirrored by bike sensor.

## Edge cases
- If orientation file is missing, tilt-rate columns are `NaN` and `quality_orientation_ok=0`.
- If only gravity alignment exists, horizontal-axis interpretability is disabled.
- If one sensor is missing, cross-sensor output is skipped.

## CLI
From `analysis/`:

```bash
uv run python -m derived <section_path>
uv run python -m derived.validate <section_path>
```
