# Workflow

Runs the dual-IMU cycling pipeline from JSON config files. The normal stage
order is:

```text
parse → sync → split → calibration → orientation → derived → features → exports → dataset_summary → evaluation → report → thesis_bundle
```

Run all configured stages from the default config:

```bash
uv run python -m workflow
```

Run with an override config merged over `data/_configs/workflow.default.json`:

```bash
uv run python -m workflow data/_configs/workflow.dev.json
```

Run only selected stages:

```bash
uv run python -m workflow --stage exports evaluation report
```

Generate a config template:

```bash
uv run python -m workflow --generate-config data/_configs/workflow.local.json
```

For evaluation method details see [evaluation/README.md](../evaluation/README.md).

---

## Config Parameters

Config files are JSON objects merged over `data/_configs/workflow.default.json`.
Only fields defined by `WorkflowConfig` are accepted.

### Dataset Scope

- `sessions`: session prefixes to process, e.g. `["2026-02-26"]`.
- `recordings`: explicit recording names. When non-empty, overrides session
  discovery.

### Stage Selection

- `stages`: ordered list of stages to run. Valid values: `parse`, `sync`,
  `split`, `calibration`, `orientation`, `derived`, `features`, `exports`,
  `dataset_summary`, `evaluation`, `report`, `thesis_bundle`.
- `force`: overwrite stage outputs where supported.
- `skip_exports`: skip the exports stage body when the stage is listed.
- `no_plots`: suppress plot generation.
- `log_to_file`: write a timestamped log under `data/logs/`.

### Sensor Processing

- `sync_method`: `multi_anchor`, `one_anchor_adaptive`, `one_anchor_prior`,
  `signal_only`, or `auto`.
- `split_stage`: source stage for splitting — `parsed` or `synced`.
- `orientation_filter`: `madgwick`, `madgwick_marg`, `complementary`, `ekf`,
  `ekf_marg`, or `auto`.
- `sample_rate_hz`: nominal sampling rate used by feature extraction and derived
  signal logic.

### Labels And Features

- `labels_path`: optional explicit labels path. Normally empty; labels are
  selected via `label_set`.
- `label_set`: label version under `data/_labels/`, e.g. `v2`.
- `window_s`: feature window duration in seconds.
- `hop_s`: sliding-window hop in seconds.
- `event_aligned`: add label-centered event windows during feature extraction.
- `lag_features_n_lags`: number of lagged window-context feature copies to add;
  `0` disables lag features.
- `min_quality_label`: primary quality threshold used by dataset summaries,
  label-grid evaluation, event contrasts, and two-stage events. Valid values:
  `poor`, `marginal`, `good`.

### Evaluation

- `evaluation_seed`: random seed for model construction and permutation
  importance.
- `evaluation_methods`: list of methods to run — `label_grid`,
  `event_contrasts`, `two_stage_events`, or any combination.
- `evaluation_sessions`: optional session prefixes to evaluate from the existing
  full `features_fused.csv`. Empty means include all sessions unless recordings
  are selected below.
- `evaluation_exclude_sessions`: optional session prefixes to exclude from
  evaluation.
- `evaluation_recordings`: optional recording names to evaluate from the
  existing full `features_fused.csv`. When combined with `evaluation_sessions`,
  rows matching either include list are evaluated.
- `evaluation_exclude_recordings`: optional recording names to exclude from
  evaluation. Exclusions are applied after inclusions.
  Scoped feature copies are written per evaluation method, e.g.
  `data/evaluation/label_grid/features_scoped.csv`, leaving
  `data/exports/features_fused.csv` unchanged.
- `evaluation_label_col`: target label scheme. Use `auto` to run all known
  schemes.
- `evaluation_quality_levels`: quality thresholds to include in the label grid.
  `min_quality_label` is automatically added as the primary quality if absent.
- `evaluation_exclude_non_riding`: when true, remove windows whose
  `scenario_label_binary` is `non_riding` before classification.
- `evaluation_models`: classifiers for label-grid CV — `random_forest`,
  `hist_gradient_boosting`, `logistic_regression`, or `auto` for all three.
- `evaluation_permutation_models`: subset of models used for permutation
  importance. Keep short for faster runs. Use `["none"]` to disable
  permutation importance.
- `event_contrast_models`: classifiers for the event-contrast evaluator. Same
  valid values as `evaluation_models`.
- `two_stage_event_models`: classifiers for the two-stage event evaluator. Same
  valid values as `evaluation_models`.
- `two_stage_event_tasks`: `core`, `all`, or explicit task names from `turning`,
  `deceleration`, `high_effort`, `posture`.
- `two_stage_target_recall`: training-split detector recall target for threshold
  selection.

---

## Common Recipes

Rebuild features and run all evaluation:

```bash
uv run python -m workflow --stage features exports evaluation
```

Run only evaluation after exports exist:

```bash
uv run python -m workflow --stage evaluation
```

Run evaluation on a scoped copy of the full exported feature set:

```bash
uv run python -m workflow --stage evaluation --evaluation-session 2026-04-29
uv run python -m workflow --stage evaluation --exclude-evaluation-session 2026-05-04
```

Quick label-grid check with one model:

```json
{
  "stages": ["evaluation"],
  "evaluation_methods": ["label_grid"],
  "evaluation_models": ["hist_gradient_boosting"],
  "evaluation_quality_levels": ["marginal"]
}
```

Event contrasts only:

```json
{
  "stages": ["evaluation"],
  "evaluation_methods": ["event_contrasts"],
  "event_contrast_models": ["logistic_regression"]
}
```
