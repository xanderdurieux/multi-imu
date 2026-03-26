# `calibration/` — Ride-level world-frame calibration

Ride-level calibration turns each section’s Sporsa/Arduino IMU CSVs into
world-frame signals by estimating gravity direction from a static window and
rotating the sensor frame so that +Z matches gravity (+Z-up convention).

For the Arduino (helmet) sensor, an optional standalone six-pose static
calibration (`static_calibration/`) is applied when available.

## Main module
- `calibrate.py`
  - Public entry point: `calibrate_section(section_path, ...)`
  - CLI entry point: `python -m calibration` (dispatches to
    `calibrate_sections_from_args`)

## CLI usage
From the `analysis/` directory:

```bash
uv run python -m calibration <section_path>
uv run python -m calibration <recording_name> --all-sections
```

Options exposed by this CLI:
- `--static-window` (seconds, default `3.0`)
- `--variance-threshold` (default `0.5`)

Horizontal frame alignment modes (yaw about +Z):
- `gravity_only` (default): vertical alignment only.
- `gravity_plus_forward`: per-sensor mean horizontal specific-force axis.
- `section_horizontal_frame` (recommended): estimate a section-local horizontal frame
  from the bike/reference sensor (Sporsa) using horizontal PCA + mean-force fallback,
  then transfer the same yaw to both sensors for consistent longitudinal/lateral axes.

Configure via pipeline:

```bash
uv run python -m pipeline --frame-alignment section_horizontal_frame ...
```

## Section-horizontal-frame workflow (recommended)
1. Gravity-align each sensor with a static window (`+Z` up).
2. Use Sporsa (or first available sensor) as reference to estimate horizontal forward axis.
3. Candidate methods evaluated internally:
   - mean horizontal specific force,
   - dominant horizontal PCA axis,
   - reference-frame transfer across sensors.
4. Choose robust axis and compute confidence metrics:
   - `straight_motion_confidence`,
   - `heading_stability`,
   - `horizontal_axis_reliability`,
   - `confidence_score`.
5. If reliability is too low, fall back to gravity-only yaw (`identity` yaw rotation).

Assumptions and limitations:
- No strong claim of true geodetic heading.
- Axis is *section-local motion interpretability* frame, not an absolute compass frame.
- Reliability drops for very short/static sections or highly symmetric turning patterns.

## Validation / QC helper
- `python -m calibration.validate <section_path>` prints a per-sensor status
  based on gravity residual and static sample count.

## Outputs (per section)
- `calibrated/sporsa.csv`, `calibrated/arduino.csv` (rotated signals)
- `calibrated/calibration.json` (per-sensor gravity residual, static window
  info, rotation matrix, calibration quality tier, frame confidence metadata)
- `calibrated/section_frame.json` (section-level frame metadata when using
  `section_horizontal_frame`)
- `calibrated/*.png` diagnostic figures (when enabled by the calling code)

## Thesis-ready plot suggestions
- `*_section_frame_diagnostics.png`: transformed long/lat/vert traces + confidence bars.
- `*_z_acc_before_after.png`: gravity alignment sanity check.
- `*_acc_norm_timeline.png`: static-window detection quality.
