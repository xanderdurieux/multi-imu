# `evaluation/` — Feature-table evaluation report

This package generates a thesis-oriented evaluation report from a consolidated
feature table (typically `features_fused.csv`).

## CLI
From the `analysis/` directory:

```bash
uv run python -m evaluation <features_fused.csv> [out_dir]
```

If `out_dir` is omitted, outputs go to:
- `<features_csv_parent>/evaluation_report/`

## What gets computed
`evaluation/experiments.py:run_evaluation_report()` loads the fused features
and computes:
- per-feature missingness (NaN fraction)
- class counts for `scenario_label`
- within/between variance summaries
- simple effect sizes for feature pairs across class labels (Cohen-d style)

When scikit-learn is available, it also runs simple baseline classifiers with
leave-one-recording-out (group) validation.

## Inputs expected in the CSV
The report assumes the feature table contains:
- `scenario_label` (label column)
- `recording_id` (group column for leave-one-group-out)
- `section_id`, `window_start_s`, `window_end_s` (metadata; ignored for modeling)

