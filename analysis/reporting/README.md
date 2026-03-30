# Thesis Reporting Module

This module is for **thesis-facing artifacts** only.
It now prefers explicit diagnostics over placeholder outputs.

## Primary thesis question (foregrounded)

> **Does fusing bike-mounted and rider-mounted IMU signals improve scenario-discrimination evidence over single-sensor baselines under recording-aware evaluation?**

Reporting artifacts are organized around evidence for that question.
Secondary analyses (sync and orientation variants) are retained as supporting context, not primary claims.

## Commands

From `analysis/`:

### 1) Full thesis report bundle

```bash
uv run python -m reporting --output-dir outputs/thesis_report_bundle
```

Generates figures/tables + manifest + caption metadata.

### 2) Core figures only (deterministic, thesis chapter ready)

```bash
uv run python -m reporting --core-figures-only --output-dir outputs/thesis_report_bundle
```

Generates only core figures into:
- `outputs/thesis_report_bundle/core_thesis_figures/`

with deterministic stems:
- `thesis_core_01_pipeline_overview.{pdf,png}`
- `thesis_core_02_orientation_filter_comparison.{pdf,png}`
- `thesis_core_03_event_centered_bike_vs_rider.{pdf,png}`
- `thesis_core_04_feature_separability.{pdf,png}`
- `thesis_core_05_success_failure_case_studies.{pdf,png}`

and writes `core_figures_manifest.json`.

## Artifact status semantics (honest by design)

Each artifact has one status in the manifest:
- `real_result`: generated from available upstream outputs.
- `skipped`: intentionally omitted in current mode.
- `missing_prerequisite`: required upstream inputs absent; no placeholder file emitted.
- `failed`: runtime failure occurred while attempting generation.

This allows thesis bundles to distinguish **evidence** from **missing evidence**.

## Success/failure case mining

Representative case studies are mined automatically from section-level signals:
- QC tier (`qc_section.json`),
- quality/confidence metrics (`quality_metadata.json`),
- downstream proxy separability from labeled feature tables.

Outputs:
- `report_tables/case_studies.csv` and `.md` (compact qualitative table),
- `report_figures/success_failure_case_studies.{pdf,png}` (compact plot).

## Assumptions and limitations (explicit)

- If labeled features are sparse or absent, downstream proxy signals are unavailable.
- Case mining uses a transparent composite signal for selection; it is an aid for discussion, not a causal proof.
- Figures are deterministic in naming and output location, but numerical values depend on upstream workflow outputs.

## Caption alignment

`caption_suggestions.md` now includes artifact status so captions can be matched only to generated evidence.
