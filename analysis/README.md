# `analysis/` — Dual-IMU Cycling Pipeline

This package contains the thesis analysis pipeline for window-level cycling activity recognition from two IMUs:

- `Sporsa` on the bike frame
- `Arduino Nano 33 BLE` on the rider helmet

The processing flow is recording-first, then section-first:

`parse -> sync -> split -> calibration -> orientation -> derived -> events -> features -> exports -> evaluation`

Most downstream work happens per section under `data/sections/<recording>s<section_idx>/`.

## Main entry point

Run the end-to-end workflow from the `analysis/` directory:

```bash
uv run -m workflow data/_configs/workflow.default.json
uv run -m workflow --list-stages
uv run -m workflow --generate-config local.workflow.json
```

The workflow loader merges any override JSON over `data/_configs/workflow.default.json`.

## Stage overview

- `parser/`: parses raw session logs into normalized per-recording CSVs, timing stats, and calibration-segment metadata.
- `sync/`: aligns Arduino time to Sporsa time, runs multiple strategies, and selects one synced recording output.
- `parser.split_sections`: cuts recordings into calibration-bounded sections and can re-sync per section.
- `calibration/`: estimates section-level sensor intrinsics and body-to-world alignment using the opening protocol.
- `orientation/`: runs multiple orientation filters per section, scores them, and flattens the selected result.
- `derived/`: computes per-sensor and cross-sensor derived signals from calibrated data.
- `events/`: detects bumps, brakes, swerves, falls, and sensor disagreement events from derived signals.
- `features/`: extracts sliding-window bike, rider, cross-sensor, and event-derived features.
- `exports/`: aggregates section features plus calibration and sync metadata into dataset tables.
- `evaluation/`: trains and evaluates classifiers on exported feature tables using grouped cross-validation.
- `reporting/`: builds thesis/report figures, tables, dataset summaries, and bundle outputs.
- `visualization/`: shared plotting helpers used by multiple stages.
- `common/`: shared path, CSV, quaternion, signal, and statistics helpers.

## Data layout

- `data/_sessions/<date>/`: raw session inputs grouped by sensor.
- `data/recordings/<recording>/parsed/`: parsed sensor CSVs and parser diagnostics.
- `data/recordings/<recording>/synced/`: selected recording-level synchronization result.
- `data/sections/<recording>s<section>/`: section-level raw slices plus downstream stage outputs.
- `data/exports/`: aggregated dataset tables such as `features_fused.csv`.
- `data/evaluation/`: model evaluation summaries and figures.
- `data/report/`: reporting-stage figures, tables, and thesis bundles.

## Shared IMU CSV convention

Raw IMU tables are standardized through `common.paths.CSV_COLUMNS`:

```text
timestamp, ax, ay, az, acc_norm, gx, gy, gz, gyro_norm, mx, my, mz, mag_norm
```

- `timestamp` is stored in milliseconds.
- `acc_norm`, `gyro_norm`, and `mag_norm` are auto-filled from the axis triplets.
- Missing samples remain `NaN`; the pipeline does not replace gaps with zeros.

Some stage-specific files add extra columns such as `timestamp_received`, orientation outputs, labels, quality fields, or feature columns.

## Useful commands

```bash
uv run -m parser 2026-02-26
uv run -m sync 2026-02-26_r5
uv run -m static_calibration
uv run -m features --recording 2026-02-26_r5
uv run -m evaluation --features data/exports/features_fused.csv
uv run -m reporting
```
