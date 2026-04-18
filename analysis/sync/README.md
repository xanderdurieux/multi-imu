# `sync/` — Recording-Level Time Alignment

`sync/` aligns the Arduino stream to the Sporsa stream at the recording level. It estimates a linear clock transform, scores multiple synchronization strategies, selects one winner, and flattens the selected result into `data/recordings/<recording>/synced/`.

## Problem setup

- Reference sensor: `sporsa`
- Target sensor: `arduino`
- Parsed inputs live under `data/recordings/<recording>/parsed/`
- Timestamps are stored in milliseconds in the `timestamp` column

The fitted model is a linear offset-plus-drift correction applied to the Arduino time base.

## Implemented strategies

- `multi_anchor`: uses multiple detected calibration anchors and fits offset plus drift.
- `one_anchor_adaptive`: starts from one anchor, then refines drift causally over later windows.
- `one_anchor_prior`: uses one anchor plus a fixed drift prior.
- `signal_only`: pure signal alignment without calibration anchors.

All methods write method-specific outputs first, then the selected one is copied into the flat `synced/` directory.

## Main files and status

- `model.py`: `SyncModel` plus timestamp transform helpers
- `stream_io.py`: stream loading, resampling, dropout handling
- `signals.py` and `activity.py`: activity-signal construction for alignment
- `anchors.py` (FINAL): calibration-anchor extraction
- `xcorr.py`: lag search and drift fitting helpers
- `strategies.py`: the four sync estimators
- `quality.py`: correlation scoring for aligned streams
- `orchestrate.py`: method comparison and winner selection
- `pipeline.py`: recording-level I/O, flattening, and CLI
- `sync_info_format.py`: consistent JSON payload builder

## CLI usage

From `analysis/`:

```bash
uv run -m sync 2026-02-26_r5
uv run -m sync 2026-02-26 --all
uv run -m sync 2026-02-26_r5 --method signal_only
```

## Outputs

Before selection, each successful method writes its own stage directory with:

- `sporsa.csv`
- `arduino.csv`
- `sync_info.json`

After selection, `pipeline.py` writes:

```text
data/recordings/<recording>/synced/
    sporsa.csv
    arduino.csv
    sync_info.json
    all_methods.json
```

- `sync_info.json` contains the selected model, method metadata, and correlation scores.
- `all_methods.json` stores the comparison summary across methods for that recording.

Per-method temporary directories are pruned after flattening.

## Public API

The main orchestration helpers are:

- `synchronize_recording_all_methods(recording_name)`
- `synchronize_recording_chosen_method(recording_name, method)`
- `synchronize_from_calibration(reference_csv=..., target_csv=..., output_dir=...)`

`parser.split_sections` uses `synchronize_from_calibration()` for section-level re-alignment after splitting.
