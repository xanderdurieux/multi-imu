# `pipeline/` — End-to-end thesis preprocessing

This module orchestrates the full dual-IMU preprocessing chain:

`parser.session` → `sync` → `parser.split_sections` → `calibration` →
`orientation` → `features` → QC (`qc_section.json`) → consolidated exports.

## Recommended entry point
Run from the `analysis/` directory:

```bash
uv run python -m pipeline --session <session_name> \
  --sync-method best \
  --orientation-filter complementary_orientation \
  [--labels path/to/labels.csv]
```

## When to use flags
- `--session <session_name>`: parse raw logs for that session before syncing.
- `--recording <recording_id>`: process only specific recording folders
  under `analysis/data/recordings/` (repeatable).
- `--sync-method {best,sda,lida,calibration,online}`: choose a sync strategy
  (default: `calibration`).
- `--split-stage <stage>`: the recording stage folder passed into
  `parser.split_sections` (default: `synced`).
- `--orientation-filter <variant>`: which orientation variant CSV is used for
  feature extraction (default: `complementary_orientation`).
- `--frame-alignment {gravity_only,gravity_plus_forward,section_horizontal_frame}`: calibration frame
  alignment mode passed to `calibration.calibrate_section`.
- `--no-plots`: disable diagnostic plots.
- `--force`: re-run steps even when outputs already exist.
- `--labels <path>`: optional labels CSV/JSON used to resolve
  `scenario_label` on feature windows.
- `--skip-exports`: skip writing consolidated export CSVs/manifest files.

## Outputs
Per section, the pipeline writes:
- `calibrated/` (world-frame sensor CSVs + `calibration.json`)
- `orientation/` (orientation variant CSVs + `orientation_stats.json`)
- `features/` (`features.csv`, `features_stats.json`, `feature_schema.json`)
- `qc_section.json` (tiered QC reasons via `validation.comprehensive`)

Run-level summaries:
- `analysis/data/pipeline_run_summary.json`
- `analysis/data/pipeline_run_summary_sections.csv`

If `--skip-exports` is not set, consolidated exports are written to:
- `analysis/data/exports/` (thesis-ready feature tables + QC summary CSVs)
