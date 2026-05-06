# Evaluation

Reads `data/exports/features_fused.csv` and writes results under
`data/evaluation/`. Three evaluation methods are available, each targeting a
different research question.

Run all methods via the workflow (recommended):

```bash
uv run python -m workflow --stage evaluation
```

Or run the evaluation module directly:

```bash
uv run python -m evaluation --help
```

---

## Label Grid

The main activity-recognition evaluation. Derives target labels from the
persisted `scenario_labels` overlap set, trains the configured classifiers, and
compares four feature sets:

- `bike`: Sporsa/frame features only.
- `rider`: Arduino/helmet features only.
- `fused_no_cross`: bike + rider features, excluding cross-sensor features.
- `fused`: bike + rider + `cross_*` features.

Cross-validation uses `GroupKFold` with `section_id` as the grouping column.

The grid axes are:

- **label scheme**: `evaluation_label_col`, or all known schemes when set to `auto`.
- **quality filter**: `evaluation_quality_levels`, with `min_quality_label` always
  added as the primary quality level if it is missing.

### Label Schemes

**Multi-class** (classify what activity is happening):

- `scenario_label_activity`: activity taxonomy, e.g. steady, turning, head
  motion, longitudinal effort.
- `scenario_label_coarse`: compact coarse activity taxonomy.

**Binary detection** (detect presence/absence of a specific phenomenon):

- `scenario_label_binary`: riding vs non-riding.
- `scenario_label_riding`: riding detail.
- `scenario_label_cornering`: cornering vs non-cornering.
- `scenario_label_head_motion`: head-motion vs non-head-motion.

### Running

Via the workflow:

```json
{
  "stages": ["exports", "evaluation"],
  "evaluation_methods": ["label_grid"],
  "evaluation_label_col": "auto",
  "evaluation_quality_levels": ["marginal", "good"],
  "evaluation_models": ["random_forest", "hist_gradient_boosting", "logistic_regression"],
  "evaluation_permutation_models": ["hist_gradient_boosting"],
  "min_quality_label": "marginal"
}
```

Directly — full label grid:

```bash
uv run python -m evaluation \
  --label-grid \
  --label auto \
  --qualities marginal good \
  --primary-quality marginal \
  --evaluation-models hist_gradient_boosting \
  --permutation-models hist_gradient_boosting
```

Use `--permutation-models none` or `"evaluation_permutation_models": ["none"]`
in workflow config to skip permutation importance.

Directly — single label/quality run:

```bash
uv run python -m evaluation \
  --label scenario_label_activity \
  --quality marginal
```

### Outputs

Under `data/evaluation/label_grid/`:

- `label_grid_summary.json`: run grid, success/failure metadata, model choices,
  quality filters.
- `label_grid_metrics.csv`: aggregate metrics across label schemes, qualities,
  models, and feature sets.
- `label_grid_imu_contribution.csv`: aggregate paired sensor-ablation deltas.

Report-stage summary figures are rendered from `label_grid_metrics.csv` under
`data/report/figures/evaluation/label_grid/`:

- `label_grid_heatmap_macro_f1.png`
- `label_grid_heatmap_accuracy.png`
- `label_grid_quality_grid_macro_f1.png`

Per-run under `data/evaluation/label_grid/<label_col>__q-<quality>/`:

- `evaluation_summary.json`: run metadata, classes, and metric summary.
- `metrics_table.csv`: accuracy and macro-F1 for each model and feature set.
- `imu_contribution.csv`: paired fold deltas for `fused` vs single/fused variants.
- `imu_contribution_per_class_<pair>__<model>.csv`: per-class F1 deltas.
- `<model>/per_class_report_<feature_set>.json`: sklearn per-class report.
- `<model>/fold_scores.csv`: fold-wise accuracy and macro-F1, with
  `feature_set`.
- `<model>/confusion_matrix.csv`: long-form OOF confusion matrices, with
  `feature_set`.
- `<model>/confusion_per_class.csv`: per-class recall/precision/F1 and dominant
  confusions, with `feature_set`.
- `<model>/confusion_top_pairs.csv`: largest off-diagonal confusion pairs, with
  `feature_set`.
- `<model>/feature_importance.csv`: model-native importances, already
  consolidated by `feature_set`.
- `<model>/permutation_importance.csv`: grouped permutation importance,
  primary-quality runs only, for configured permutation models.
- `<model>/permutation_importance_by_group.csv`: permutation importance
  aggregated into bike/rider/cross groups.
- Binary targets only: `<model>/binary_metrics_<feature_set>.json`,
  `<model>/misclassified_<feature_set>.csv`, and optional per-section overlay
  figures.

Per-model CSVs that used to be split as one file per feature set, such as
`confusion_top_pairs_bike.csv` and `confusion_top_pairs_rider.csv`, are merged
into a single file with a `feature_set` column.

---

## Event Contrasts

Tests narrower safety-relevant distinctions with the same feature sets. Label-
driven: no detector output is required. Builds binary datasets from
`scenario_labels`:

- `cornering_vs_swerving`: `cornering` vs `swerving`.
- `slowing_vs_hard_braking`: `slowing` vs `hard_braking`.

Rows containing only the normal token enter the normal class; rows containing
only the critical token enter the critical class. Rows containing both tokens are
excluded and kept in the audit table as `ambiguous_both_tokens`. Unrelated and
unlabeled rows are ignored.

Uses recording-level `GroupKFold` (`group_col="recording_name"`) to match the
leakage-avoidance requirement for second-IMU value claims.

Permutation importance is computed for every contrast/model/feature-set
combination — this is the slowest part; reduce `event_contrast_models` for quick
checks.

### Running

Via the workflow:

```json
{
  "stages": ["exports", "evaluation"],
  "evaluation_methods": ["event_contrasts"],
  "event_contrast_models": ["hist_gradient_boosting", "logistic_regression"],
  "min_quality_label": "marginal"
}
```

Directly:

```bash
uv run python -m evaluation \
  --event-contrasts \
  --event-contrast-models hist_gradient_boosting logistic_regression \
  --quality marginal
```

Fast check:

```bash
uv run python -m evaluation \
  --event-contrasts \
  --event-contrast-models logistic_regression
```

### Outputs

Under `data/evaluation/event_contrasts/`:

- `event_contrast_summary.json`: contrasts, models, grouping column, quality
  filter, skipped reasons, artifact paths.
- `event_contrast_metrics.csv`: per contrast/feature set/model — accuracy,
  balanced accuracy, macro-F1, PR-AUC, ROC-AUC, support counts.
- `event_contrast_imu_contribution.csv`: paired fold deltas for `fused` vs
  `bike`, `rider`, `fused_no_cross`, and `rider` vs `bike`.
- `event_contrast_feature_importance_<contrast>_<model>_<feature_set>.csv`:
  permutation importance per feature.
- `event_contrast_feature_importance_by_group_<contrast>_<model>_<feature_set>.csv`:
  permutation importance aggregated by bike/rider/cross group.
- `event_contrast_feature_stability.csv`: feature rank mean/std and top-k
  frequency across folds.
- `event_contrast_windows.csv`: audit table with section/window ids, timestamps,
  raw `scenario_labels`, resolved contrast label, and usage status.

---

## Two-Stage Events

Models a deployable event pipeline:

1. **Detector**: classify whether a broad event family is present in a window.
2. **Contrast**: route detector-positive windows into a classifier that
   distinguishes normal from safety-critical behavior.

The detector threshold is selected on the training split to hit
`two_stage_target_recall`, trading precision for coverage before routing to the
contrast model. Outputs report both oracle-gated and predicted-gated contrast
performance so detector misses remain visible.

### Tasks

**Core** (`two_stage_event_tasks: ["core"]` or `"all"`):

- `turning`: detector positive = `cornering` or `swerving`; contrast =
  `cornering` vs `swerving`.
- `deceleration`: detector positive = `slowing` or `hard_braking`; contrast =
  `slowing` vs `hard_braking`.

**Optional** (`two_stage_event_tasks: ["all"]` or explicit names):

- `high_effort`: `accelerating` vs `sprinting`.
- `posture`: `riding` vs `riding_standing`.

Each fold uses recording-level `GroupKFold`.

### Running

Via the workflow:

```json
{
  "stages": ["exports", "evaluation"],
  "evaluation_methods": ["two_stage_events"],
  "two_stage_event_tasks": ["all"],
  "two_stage_event_models": ["hist_gradient_boosting", "logistic_regression"],
  "two_stage_target_recall": 0.90,
  "min_quality_label": "marginal"
}
```

Directly:

```bash
uv run python -m evaluation \
  --two-stage-events \
  --two-stage-tasks all \
  --two-stage-event-models hist_gradient_boosting logistic_regression \
  --two-stage-target-recall 0.90 \
  --two-stage-hop-s 0.25 \
  --quality marginal
```

Fast check (single task, fast model):

```bash
uv run python -m evaluation \
  --two-stage-events \
  --two-stage-event-models logistic_regression \
  --two-stage-tasks turning \
  --two-stage-hop-s 0.25
```

### Outputs

Under `data/evaluation/two_stage_events/`:

- `two_stage_event_summary.json`: task configs, model list, target recall,
  support by recording, skipped reasons, artifact paths.
- `two_stage_event_metrics.csv`: detector, oracle contrast, predicted-gated
  contrast, and end-to-end critical-event metrics per task/feature set/model.
- `two_stage_event_fold_scores.csv`: fold-level metrics, detector thresholds,
  routed counts, missed-event counts, train/validation recording lists.
- `two_stage_event_predictions.csv`: OOF window predictions with detector
  probability, decision, contrast probability, final label, routing status.
- `two_stage_event_candidates.csv`: detector-positive windows for inference
  review.
- `two_stage_event_intervals.csv`: candidate windows merged into intervals
  (`hop_s * 1.5` maximum gap).
- `two_stage_event_imu_contribution.csv`: paired fold deltas for `fused` vs
  `bike`, `rider`, `fused_no_cross`, and `rider` vs `bike`.
