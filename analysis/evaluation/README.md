# `evaluation/` — Thesis experiment layer for dual-IMU features

This package runs a lightweight, interpretable experiment workflow on a fused
feature table (typically `features_fused.csv`).

## CLI
From the `analysis/` directory:

```bash
uv run python -m evaluation <features_fused.csv> [out_dir] --config evaluation/configs/thesis_experiment_config.json --seed 42
```

If `out_dir` is omitted, outputs go to:
- `<features_csv_parent>/evaluation_report/`

## What gets computed
`evaluation/experiments.py:run_evaluation_report()` generates thesis-ready tables and plots:

1. **Feature source comparison**
   - bike-only (`sporsa__*`) vs rider-only (`arduino__*`) vs fused
   - models: logistic regression, random forest, gradient boosting
   - leave-one-recording-out validation (recording-aware)

2. **Sync method comparison**
   - evaluates fused features separately for each `sync_method`

3. **Orientation method comparison**
   - evaluates fused features separately for each `orientation_method`

4. **Optional feature family ablation**
   - "all fused" vs "minus family" feature groups (e.g., bumps/braking/cornering)

5. **When labels are limited / as supportive evidence**
   - effect-size table (max pairwise Cohen's d)
   - within-class vs between-class variance ratio
   - PCA class scatter and KMeans overlay for separability inspection

## Output artifacts (for thesis)
- `thesis_table_model_metrics.csv` (balanced accuracy, precision, recall, F1)
- `classification_summary.csv`
- confusion matrix images (`cm_*.png`)
- feature-importance tables (`feature_importance_*.csv`)
- optional precision-recall curves (`pr_*.png`) when imbalance is strong
- `separability_effect_size.csv`
- `separability_within_between_variance.csv`
- `THESIS_SUMMARY.md` and `evaluation_summary.json` with recommendation text

## Config-driven controls
Use `evaluation/configs/thesis_experiment_config.json` to adjust:
- feature subsets (`feature_sets`)
- ablation families (`feature_family_ablation`)
- minimum label counts and reporting limits
- label/group/sync/orientation column names
- optional `locked_split_manifest` for fixed recording-level train/validation/test splits
- optional `qc_policy` for explicit inclusion/exclusion gates using `qc_section.json` metadata

## Locked recording split protocol

To force reproducible train/validation/test evaluation at recording level:

1. Create a manifest (example: `evaluation/configs/locked_splits.example.json`) with disjoint `train`, `validation`, and `test` recording lists.
2. Point `locked_split_manifest` in your experiment config to that manifest.
3. Run evaluation normally.

When enabled, evaluation trains only on `train` recordings and reports metrics on validation/test recordings only, preventing cross-recording leakage by construction.

## QC-driven inclusion protocol

Enable `qc_policy.enabled=true` in evaluation config to:
- compute section-level include/exclude decisions from `qc_section.json`,
- apply calibration and orientation confidence thresholds,
- export `qc_inclusion_section_decisions.csv` and `qc_inclusion_summary.json`,
- retain explicit exclusion reasons (no silent dropping).


## Determinism

All stochastic components use a single seed (`evaluation_seed`):
- model initialisation,
- PCA/KMeans projections,
- permutation importance.

Seed priority is: `--seed` CLI > `MULTI_IMU_EVALUATION_SEED` > config `evaluation_seed` > default `42`.

The chosen seed is recorded in `evaluation_summary.json` and `evaluation_manifest.json`.
