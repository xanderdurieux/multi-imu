# `features/` — Windowed feature extraction and thesis exports

This package computes thesis-ready features for each calibration-bounded
section by sliding a fixed-size window over time and summarizing signal
statistics (including cross-sensor features).

## Main extraction module
- `extract.py`
  - Public entry point: `extract_section(section_path, section_name, ...)`
  - CLI entry point: `python -m features.extract` (or `python -m features` which defaults to extract)

### Feature extraction CLI
From `analysis/`:

```bash
uv run python -m features.extract <section_or_recording> \
  --window 1.0 --hop 0.5 \
  --orientation complementary_orientation \
  [--labels path/to/labels.csv]
```

Run all sections for a recording:

```bash
uv run python -m features.extract <recording_name> \
  --all-sections --orientation complementary_orientation \
  [--labels path/to/labels.csv]
```

Run a whole session (all recordings under `data/recordings/`):

```bash
uv run python -m features.extract <session_name> \
  --all --orientation complementary_orientation \
  [--labels path/to/labels.csv]
```

### Inputs / required stage outputs
For each section, extraction expects:
- `calibrated/calibration.json` (calibration quality gates; missing calibration skips the section)
- `calibrated/<sensor>.csv`
- `orientation/<sensor>__<orientation_variant>.csv` (the `--orientation` argument)

### Outputs (per section)
- `features/features.csv` (one row per window)
- `features/features_stats.json` (simple numeric summaries)
- `features/feature_schema.json` (feature column schema)
- Optional feature timeline plots (when enabled by the caller)

## Validation
```bash
uv run -m features.validate <section_path>
```

## Session aggregation (features_all.csv)
- `aggregate.py`: `python -m features.aggregate <session_name>`

This concatenates all `sections/*/features/features.csv` belonging to the
session into:
- `analysis/data/sessions/<session_name>/features_all.csv`

## Thesis-ready export tables (features_bike/rider/fused)
Exports are typically produced by the end-to-end `pipeline` via:
- `features.exports.export_thesis_feature_tables()`
- `features.exports.export_qc_summaries()`

Consolidated outputs:
- `analysis/data/exports/features_bike.csv`
- `analysis/data/exports/features_rider.csv`
- `analysis/data/exports/features_fused.csv`
- `analysis/data/exports/export_manifest.json`
- `analysis/data/exports/qc_sections_summary.csv` (when QC JSON exists)

