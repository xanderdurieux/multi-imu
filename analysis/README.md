# Multi-IMU Thesis Analysis Workflow

This repository contains the offline processing stack for dual-IMU cycling data (bike-mounted Sporsa + rider-mounted Arduino).

## Thesis framing (read first)

### Primary thesis question

> **Does dual-sensor fusion (bike + rider IMU) provide stronger scenario-discrimination evidence than single-sensor alternatives when evaluated with recording-aware protocols?**

### Secondary analyses (supporting, not primary claims)

- Synchronization-method comparisons.
- Orientation-method comparisons.
- Feature-family ablations.

These are supporting analyses and should not be reported as stronger evidence than the primary feature-source comparison.

---

## 1) Reproducible thesis entry points (official)

Run the full workflow via a single config file:

```bash
cd analysis
uv sync
uv run python -m workflow --config configs/workflow.thesis.json
```

Then run the **primary thesis experiment bundle**:

```bash
uv run python -m evaluation data/exports/features_fused.csv data/exports/thesis_primary --primary --config evaluation/configs/thesis_primary_experiment_config.json --seed 42
```

Then run thesis reporting:

```bash
uv run python -m reporting --output-dir outputs/thesis_report_bundle
```

Core figures only (deterministic filenames):

```bash
uv run python -m reporting --core-figures-only --output-dir outputs/thesis_report_bundle
```

---

## 2) Workflow stages

| Stage | Responsibility | Main module(s) |
|---|---|---|
| Data loading | Parse raw session files to normalized per-recording CSVs | `parser/` |
| Preprocessing | Recording-level checks and section splitting | `parser/stats.py`, `parser/split_sections.py` |
| Synchronization | Align Sporsa/Arduino streams (SDA/LIDA/Calibration/Online) | `sync/` |
| Calibration / orientation | World-frame calibration and orientation estimation | `calibration/`, `orientation/` |
| Feature extraction | Window features + exports | `features/`, `derived/` |
| Event analysis | Candidate event extraction from derived/orientation streams | `events/` |
| Evaluation | Primary thesis bundle + secondary ablations | `evaluation/` |
| Reporting | Thesis figures/tables with explicit evidence status metadata | `reporting/` |

---

## 3) Limits and claim discipline (examiner-facing)

- Missing prerequisites now produce explicit `missing_prerequisite` diagnostics in reporting manifests; placeholder artifacts are avoided.
- Composite case-study scoring (QC + confidence + downstream proxy) is for qualitative selection only.
- Reproducibility is deterministic for seeds/config/filenames, but outcomes still depend on available labeled data and upstream processing quality.
- If labels, QC metadata, or orientation/sync outputs are incomplete, secondary analyses should be reported as unavailable rather than interpreted.

---

## 4) Configuration handling

Use `configs/workflow.thesis.json` to control:
- dataset root (`data_root`),
- method choices (`sync_method`, `orientation_filter`, `frame_alignment`),
- run behavior (`force`, `no_plots`, `skip_exports`),
- event thresholds (`event_config_path`, `min_event_confidence`),
- reproducible subset selection (`session`, `recordings`),
- deterministic evaluation seed (`evaluation_seed`).

---

## 5) Smoke test (fixture dataset)

```bash
uv run python -m unittest tests.test_thesis_workflow_smoke
```

The smoke test validates pipeline structure and deterministic evaluation behavior on fixture data.
