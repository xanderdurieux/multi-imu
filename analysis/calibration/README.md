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

Forward-frame (yaw) alignment:
- `frame_alignment="gravity_only"` (default) or `frame_alignment="gravity_plus_forward"`
  can be set via `pipeline --frame-alignment ...` (the minimal
  `python -m calibration` CLI keeps the default).

## Validation / QC helper
- `python -m calibration.validate <section_path>` prints a per-sensor status
  based on gravity residual and static sample count.

## Outputs (per section)
- `calibrated/sporsa.csv`, `calibrated/arduino.csv` (rotated signals)
- `calibrated/calibration.json` (per-sensor gravity residual, static window
  info, rotation matrix, calibration quality tier)
- `calibrated/*.png` diagnostic figures (when enabled by the calling code)

