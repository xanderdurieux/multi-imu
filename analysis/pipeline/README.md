# `pipeline/` — Stage orchestrator used by `workflow`

`pipeline` orchestrates the full dual-IMU preprocessing chain and is called by the higher-level `workflow` config runner.

`parser.session` → `sync` → `parser.split_sections` → `calibration` →
`orientation` → `derived` → `events` → `features` → QC (`qc_section.json`) → consolidated exports.

## Recommended thesis entry point
For reproducible thesis runs, use:

```bash
uv run python -m workflow --config configs/workflow.thesis.json
```

Use `python -m pipeline` only for development/debugging direct stage-level control. It is a legacy execution path for thesis reruns.

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
- `--event-config <path>`: optional JSON thresholds for event candidate extraction.
- `--event-centered-features`: use event-centered windows in `features` instead of uniform sliding windows.
- `--min-event-confidence`: confidence cutoff when `--event-centered-features` is enabled.
- `--skip-exports`: skip writing consolidated export CSVs/manifest files.

## Outputs
Per section, the pipeline writes:
- `calibrated/` (world-frame sensor CSVs + `calibration.json`)
- `orientation/` (orientation variant CSVs + `orientation_stats.json`)
- `derived/` (physically interpretable time-series + dependency/quality metadata)
- `events/` (`event_candidates.csv`, threshold config, summary, diagnostics)
- `features/` (`features.csv`, `features_stats.json`, `feature_schema.json`)
- `qc_section.json` (tiered QC reasons via `validation.comprehensive`)

Run-level summaries:
- `analysis/data/pipeline_run_summary.json`
- `analysis/data/pipeline_run_summary_sections.csv`

If `--skip-exports` is not set, consolidated exports are written to:
- `analysis/data/exports/` (thesis-ready feature tables + QC summary CSVs)


## Compact section summary artifacts
Generate one scan-friendly summary per processed section (Markdown or HTML).

```bash
uv run python -m pipeline.section_summary <section_id_or_recording> \
  [--all-sections | --all] \
  [--format markdown|html] \
  [--output-root data/section_summaries] \
  [--write-examples]
```

Each summary includes:
- identifiers (recording + section), duration, quality category,
- sync/calibration/orientation confidence,
- trustworthy derived signal families,
- top event candidates,
- salient feature values,
- dual-IMU comparison metrics,
- warnings and an automatic narrative paragraph,
- small interpretable quick plots.

### Proposed summary folder structure
```text
data/section_summaries/
  <recording_id>/
    <section_id>/
      summary.json
      summary.md or summary.html
      plots/
        plot_01.png
        plot_02.png
        plot_03.png
  examples/
    example_good_section_summary.md
    example_marginal_section_summary.md
```

Markdown output is PDF-friendly via standard Markdown-to-PDF conversion tooling.
