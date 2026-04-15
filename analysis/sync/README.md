# `sync/` — IMU stream synchronization

This package aligns the **target** sensor (Arduino helmet, `millis()` time base) to the **reference** sensor (Sporsa bicycle, Unix-epoch clock). It estimates a linear clock model (offset + drift), scores each candidate with accelerometer correlation, runs four estimation strategies, and picks one winner per recording.

Run commands from the `analysis/` directory so the package imports as `sync` (see [Usage](#usage)).

---

## Table of contents

1. [Problem and model](#problem-and-model)
2. [Package layout](#package-layout)
3. [End-to-end flow](#end-to-end-flow)
4. [The four sync methods](#the-four-sync-methods)
5. [Selection policy](#selection-policy)
6. [Outputs](#outputs)
7. [Usage](#usage)
8. [Module reference](#module-reference)
9. [Maintenance and code review notes](#maintenance-and-code-review-notes)

---

## Problem and model

### Clocks

- **Reference:** stable wall time (seconds since epoch in CSV `timestamp`, ms).
- **Target:** monotonic counter from boot; a constant shift and small crystal drift vs reference are expected.

### Linear model

Let \(t_{\mathrm{tgt}}\) be target time (seconds) and \(t_0\) the chosen origin (`target_time_origin_seconds`, usually the first target sample). The mapping to reference time is:

\[
t_{\mathrm{ref}} = t_{\mathrm{tgt}} + b + a\,(t_{\mathrm{tgt}} - t_0)
\]

- **`offset_seconds`** (\(b\)): shift at the origin.
- **`drift_seconds_per_second`** (\(a\)): slope of the correction vs elapsed target time from \(t_0\) (see `apply_linear_time_transform` in `model.py` for the exact ms-scale implementation).

Implementation: [`SyncModel`][sync-model], [`apply_linear_time_transform`][apply-linear], [`apply_sync_model`][apply-sync] in `model.py` (timestamps in CSV are ms; helpers convert to seconds internally).

### Quality metric

Every method fills `sync_info.json → correlation` via [`compute_sync_correlations`][compute-sync-corr] in `quality.py`:

- **`offset_only`:** same model with drift forced to zero (Pearson \(r\) of `acc_norm` on a common resampled grid).
- **`offset_and_drift`:** full model.

Both use the **overlap** of reference and (transformed) target time ranges after resampling at `sample_rate_hz`.

[sync-model]: #modelpy
[apply-linear]: #modelpy
[apply-sync]: #modelpy
[compute-sync-corr]: #qualitypy

---

## Package layout

| File | Role |
|------|------|
| [`model.py`](#modelpy) | `SyncModel` dataclass; timestamp transform and `apply_sync_model`. |
| [`stream_io.py`](#stream_iopy) | Load CSVs, norms, resampling, low-pass filter, dropout removal. |
| [`signals.py`](#signalspy) | Z-scoring, norm helpers with configurable axes, `build_activity_signal`, signal mode resolution. |
| [`activity.py`](#activitypy) | `AlignmentSeries` + `build_alignment_series` (resample → optional LPF → activity signal). |
| [`xcorr.py`](#xcorrpy) | FFT lag search, masked NCC, drift fit, windowed / adaptive refinement. |
| [`anchors.py`](#anchorspy) | Calibration detection, coarse offset, per-segment refinement, `extract_calibration_anchors`. |
| [`strategies.py`](#strategiespy) | Four public estimators: `estimate_*` → `(SyncModel, meta_dict)`. |
| [`quality.py`](#qualitypy) | `acc_norm_correlation`, `compute_sync_correlations`. |
| [`orchestrate.py`](#orchestratepy) | Method registry, load/compare `sync_info.json`, selection, logging table. |
| [`pipeline.py`](#pipelinepy) | Per-recording I/O, run all methods, flatten to `synced/`, CLI. |
| [`__init__.py`](#package-exports) | Public re-exports. |
| [`__main__.py`](#package-exports) | `python -m sync` → `pipeline.main`. |

There is no `core.py` / `methods.py`; older docs referred to a previous layout.

---

## End-to-end flow

1. **Inputs:** `data/recordings/<recording>/parsed/sporsa.csv` and `arduino.csv` (or paths from `sensor_csv("<recording>/<stage>", "<sensor>")`).
2. **Per method:** load streams → run one `estimate_*` → apply model to target → write `synced/<stage>/{sporsa,arduino}.csv` + `sync_info.json`.
3. **Selection:** read each stage’s `sync_info.json`, apply [Selection policy](#selection-policy), copy winner into flat `data/recordings/<recording>/synced/`, write `all_methods.json`, delete per-method subfolders.

Stages (folders under `synced/` before flattening) are defined in `orchestrate.METHOD_STAGES`.

---

## The four sync methods

Order below matches **tier preference** in `orchestrate.SYNC_METHODS` (strongest listed first). Anchor-based methods share [`extract_calibration_anchors`][extract-anchors] in `anchors.py`.

[extract-anchors]: #anchorspy

### 1. `multi_anchor` (`estimate_multi_anchor`)

- **Needs:** at least two matched calibration windows after refinement.
- **Drift:** weighted linear fit of per-anchor offsets vs target time (`fit_offset_drift` in `xcorr.py`).
- **Typical use:** best when opening and closing tap sequences are detected and span enough of the ride.
- **Meta:** `sync_method`, `drift_source: "anchor_fit"`, `calibration` (anchors, span, optional `fit_r2`).

### 2. `one_anchor_adaptive` (`estimate_one_anchor_adaptive`)

- **Needs:** one refined anchor.
- **Drift:** after the opening anchor (in reference time), runs **causal** windowed lag refinement (`adaptive_windowed_refinement`): each window’s search is centred on the running offset/drift estimate from **past** windows only. Final model refits offset/drift from all accepted windows if \(R^2\) ≥ `DEFAULT_MIN_FIT_R2`, else falls back to opening anchor offset and zero drift.
- **Meta:** `adaptive` block (accepted/rejected window counts, `fit_r2`, local corr stats).

### 3. `one_anchor_prior` (`estimate_one_anchor_prior`)

- **Needs:** one refined anchor.
- **Drift:** fixed prior `DEFAULT_DRIFT_PPM` (300 ppm by default), converted to `drift_seconds_per_second`; offset at origin is back-solved from the opening anchor.
- **Meta:** `drift_ppm_prior`, `drift_source: "prior_ppm"`.

### 4. `signal_only` (`estimate_signal_only`)

- **Needs:** no calibration (pure signal).
- **Steps:** build alignment series on both streams → global FFT lag (`estimate_lag`) → non-causal `windowed_lag_refinement` → `fit_offset_drift`. If \(R^2\) is below `DEFAULT_MIN_FIT_R2`, drift is zeroed (offset-only from LIDA fit path).
- **Meta:** `sda_score`, `windowed` stats, `drift_source` either `"sda"` or `"sda_lida"`.

Shared **anchor path** (for methods 1–3): detect reference calibrations → `bootstrap_coarse_offset` (opening-cluster peaks on target, else low-rate SDA fallback) → filter segments that map into target range → `refine_offset_at_calibration` per segment (local FFT lag on short windows).

---

## Selection policy

Implemented in `orchestrate.select_best_sync_method`:

1. If **`multi_anchor`** passes `_multi_anchor_passes` (calibration span ≥ 60 s, opening/closing anchor scores ≥ 0.5, correlation ≥ 0.2, |drift| ≤ 5000 ppm), it wins.
2. Otherwise, among methods that produced `sync_info.json`, choose the highest `correlation.offset_and_drift`. If drift magnitude exceeds 5000 ppm, the effective score is penalised by 0.5. Tie-break is implicit in iteration order over `SYNC_METHODS`.

`print_comparison` logs offset, drift (ppm), and a single row for correlation across all methods.

---

## Outputs

### Per-method directory (before flattening)

Example: `synced/multi_anchor/`

| Artifact | Content |
|----------|---------|
| `sporsa.csv` | Copy of parsed reference. |
| `arduino.csv` | Target with `timestamp` replaced by aligned time (see `apply_sync_model`). |
| `sync_info.json` | `SyncModel` fields as dict + method `meta` + `correlation`. |

### Flat `synced/` after selection

| File | Content |
|------|---------|
| `sporsa.csv`, `arduino.csv` | Winner copies. |
| `sync_info.json` | Winner’s full payload (model + meta + correlation). |
| `all_methods.json` | `recording`, `selected_method`, `selected_stage`, and for each method key a block with fields from `SyncSelectionResult.metrics` (`orchestrate.SyncMethodQuality` serialised): availability, offsets, correlation, drift, calibration summaries, adaptive anchor snapshot, etc. |

Downstream consumers include `exports/aggregate.aggregate_sync_params`, `visualization/plot_sync.py`, and `orientation/pipeline.py` (copies or reads `all_methods.json`).

### `sync_info.json` (guaranteed and common extras)

Core fields from `SyncModel`: `reference_csv`, `target_csv`, `target_time_origin_seconds`, `offset_seconds`, `drift_seconds_per_second`, `sample_rate_hz`, `max_lag_seconds`, `created_at_utc`.

Always added by `pipeline._run_method`: `correlation` (`offset_only`, `offset_and_drift`, `signal`, `sample_rate_hz`).

Method-specific keys include `sync_method`, `drift_source`, `signal_mode`, `calibration`, `adaptive`, `sda_score`, `windowed`, `drift_ppm_prior` as applicable.

---

## Usage

### CLI

From `analysis/`:

```bash
uv run -m sync 2026-02-26_r5
uv run -m sync 2026-02-26 --all
uv run -m sync 2026-02-26_r5 --method signal_only
```

### Python API

```python
from sync.pipeline import (
    synchronize_recording_all_methods,
    synchronize_recording_chosen_method,
    synchronize_from_calibration,
)

# Full recording: all methods + selection + flatten
result = synchronize_recording_all_methods("2026-02-26_r5")
# result.method_results, result.selection, result.selection_error

# Single method + flatten
synchronize_recording_chosen_method("2026-02-26_r5", "multi_anchor")

# Section dirs (used by parser.split_sections): only multi-anchor, custom paths
synchronize_from_calibration(
    reference_csv=path_to_sporsa,
    target_csv=path_to_arduino,
    output_dir=tmp_dir,
    sample_rate_hz=100.0,
    coarse_max_lag_s=10.0,   # SDA fallback max lag when opening calibration fails
    cal_search_s=5.0,        # ±seconds for per-anchor lag search
)
```

### Package exports

`sync.__init__` re-exports: `SyncModel`, `make_sync_model`, `apply_sync_model`, `apply_linear_time_transform`, `SYNC_METHODS`, `SyncMethodQuality`, `SyncSelectionResult`, `main`, `synchronize_from_calibration`, `synchronize_recording_all_methods`, `synchronize_recording_chosen_method`.

---

## Module reference

### `model.py`

| Symbol | Description |
|--------|-------------|
| `SyncModel` | Frozen dataclass holding paths, origin, offset, drift, `sample_rate_hz`, `max_lag_seconds`, `created_at_utc`. |
| `make_sync_model` | Construct `SyncModel` with current UTC timestamp. |
| `apply_linear_time_transform` | Vectorised ms → ms mapping using offset + drift about `target_origin_seconds`. |
| `apply_sync_model` | Copy target `DataFrame`, set `timestamp_orig`, `timestamp_aligned`, optionally replace `timestamp`. |

### `stream_io.py`

| Symbol | Description |
|--------|-------------|
| `VECTOR_AXES` | Default axis groups for acc/gyro/mag norms. |
| `load_stream` | Read CSV, drop NaN timestamps, sort. |
| `add_vector_norms` | Add `acc_norm`, `gyro_norm`, `mag_norm` when axes exist. |
| `infer_numeric_columns` | Numeric columns except optional skip (used by resamplers). |
| `resample_stream` | Uniform grid linear interpolation; optional column subset and time bounds. |
| `resample_to_reference_timestamps` | Interpolate target onto reference timestamps (NaN outside hull). |
| `lowpass_filter` | Zero-phase Butterworth on IMU columns. |
| `remove_dropouts` | Drop rows with `acc_norm` below a fraction of median (BLE dropouts). |

### `signals.py`

| Symbol | Description |
|--------|-------------|
| `SIGNAL_MODE_*`, `SIGNAL_MODES` | Named modes for differentiated norm signals (`acc_norm_diff`, etc.). |
| `zscore` | Per-series z-score with finite mask. |
| `add_vector_norms` | Like stream_io but takes `vector_axes` dict (reusable for non-standard CSVs). |
| `resolve_signal_mode` | Map explicit `signal_mode` or legacy `use_acc` / `use_gyro` / `differentiate` flags. |
| `build_activity_signal` | 1D signal array + resolved mode label from a `DataFrame` and axis map. |

### `activity.py`

| Symbol | Description |
|--------|-------------|
| `AlignmentSeries` | `timestamps_seconds`, `signal`, `sample_rate_hz`, `signal_mode`. |
| `build_alignment_series` | Resample at `sample_rate_hz`, optional LPF, build activity signal via `signals.build_activity_signal` with `VECTOR_AXES`. |

### `xcorr.py`

| Symbol | Description |
|--------|-------------|
| `fft_correlate_full` | FFT “full” cross-correlation. |
| `estimate_lag` | Integer lag maximising overlap-normalised correlation; optional max lag. |
| `masked_ncc` | Pearson-like NCC over finite overlap; returns score and valid fraction. |
| `fit_offset_drift` | Weighted or OLS line `offset vs (t - t0)`; returns intercept, slope, \(R^2\). |
| `windowed_lag_refinement` | Non-causal sliding windows around a coarse lag (signal-only / LIDA). |
| `adaptive_windowed_refinement` | Causal sliding refinement updating offset/drift from past windows. |

Defaults: window length/step, local search half-width, minimum window score, minimum fit \(R^2\) (module constants).

### `anchors.py`

| Symbol | Description |
|--------|-------------|
| `CalibrationWindowResult` | One refined anchor: offset, target/ref peak times, score, duration. |
| `CalibrationAnchorExtraction` | List of anchors + coarse offset + coarse method + segment counts. |
| `_cluster_peaks` | Group tap peaks into calibration-like clusters (internal). |
| `coarse_offset_from_opening_calibration` | Match first target cluster to first reference calibration median time. |
| `refine_offset_at_calibration` | Crop ref/tgt around segment, build alignment series, FFT lag within ±search. |
| `detect_reference_calibrations` | Wrapper around `parser.calibration_segments.find_calibration_segments`. |
| `filter_segments_in_target_range` | Drop reference segments whose predicted target centre is off-range. |
| `bootstrap_coarse_offset` | Opening calibration coarse offset, or SDA fallback at low sample rate. |
| `extract_calibration_anchors` | Full pipeline: detect → coarse → filter → refine; optional `sda_fallback_max_lag_s` for fallback SDA. |
| `calibration_anchor_to_dict` | JSON-serialisable anchor metadata. |

### `strategies.py`

| Symbol | Description |
|--------|-------------|
| `estimate_multi_anchor` | Two+ anchors, weighted drift fit; kwargs `anchor_search_seconds`, `sda_fallback_max_lag_s`. |
| `estimate_one_anchor_adaptive` | One anchor + causal windows; same tuning kwargs. |
| `estimate_one_anchor_prior` | One anchor + ppm prior; same tuning kwargs. |
| `estimate_signal_only` | SDA + LIDA; uses `max_lag_seconds` for global lag search. |

Internal `_build_calibration_meta` / `_require_anchor_count` reduce duplication for `calibration` JSON.

### `quality.py`

| Symbol | Description |
|--------|-------------|
| `acc_norm_correlation` | Pearson \(r\) on overlapping resampled `acc_norm`. |
| `compute_sync_correlations` | Offset-only vs full model correlations; returns dict for `sync_info.json`. |

### `orchestrate.py`

| Symbol | Description |
|--------|-------------|
| `SYNC_METHODS`, `METHOD_STAGES`, `METHOD_LABELS` | Canonical IDs and folder names. |
| `method_stage`, `method_label` | Resolve method → path / label. |
| `SyncMethodQuality`, `SyncSelectionResult` | Typed summaries; `SyncSelectionResult.metrics` → `all_methods.json`. |
| `extract_quality` | Parse one `sync_info.json` into `SyncMethodQuality`. |
| `compare_sync_models` | Load all method JSON blobs for a recording. |
| `select_best_sync_method` | Selection policy. |
| `print_comparison` | Logged comparison table. |

### `pipeline.py`

| Symbol | Description |
|--------|-------------|
| `MethodResult`, `RecordingResult` | Per-method status and final selection handle. |
| `_run_method` | Load parsed CSVs, dispatch strategy, write stage outputs. |
| `synchronize_recording_all_methods` | Run all tiers, select, flatten. |
| `synchronize_recording_chosen_method` | One tier, synthesise `SyncSelectionResult`, flatten. |
| `synchronize_from_calibration` | Standalone multi-anchor sync for arbitrary paths (section splitter). |
| `main` | Argument parser: recording or session `--all`, optional `--method`. |

---

## Maintenance and code review notes

### Fixes applied while auditing

- **`synchronize_from_calibration`** previously passed unsupported kwargs into `estimate_multi_anchor`; `anchor_search_seconds` and `sda_fallback_max_lag_s` are now threaded from `cal_search_s` and `coarse_max_lag_s`. Section-level sync therefore honours the splitter’s search window and SDA fallback lag.
- **`print_comparison`** printed one correlation column per method in separate rows; it now prints a single **Corr offset+drift** row aligned with all methods.
- **Unused** `build_activity_signal` wrapper in `activity.py` was removed (only `build_alignment_series` is used internally).

### Duplication that is intentional (for now)

- **`add_vector_norms`** exists in both `stream_io.py` (fixed `VECTOR_AXES`) and `signals.py` (parameterised axes). Consolidating would touch `quality.py` and `activity.py`; keeping both avoids churn while `signals` stays generic.

### Possible future simplifications

- **`pipeline` argparse:** the user preference is to avoid CLI parsers in scripts; here the parser is the package entrypoint—acceptable, but a thinner `main` could move to a tiny `cli.py` if desired.
- **Selection policy:** `_multi_anchor_passes` and drift penalty thresholds are hard-coded; a single config object would make thesis experiments easier.
- **`aggregate_sync_params`** still reads optional keys (`calibration_usage_strategy`, `segment_aware_used`) that the current sync pipeline does not emit; harmless, but could be dropped from exports if unused.
- **Thesis plots** (`visualization/thesis_plots.py`) reimplement a private `_build_activity_signal` instead of importing `sync.activity.build_alignment_series`; importing would deduplicate behaviour at the cost of coupling plotting to sync resampling defaults.

### API / output stability

- Changing keys inside `sync_info.json` or `all_methods.json` will affect `plot_sync.py` and `aggregate.py`. The user indicated plots may be updated after pipeline changes; exports should be updated alongside any schema change.

---

## Licence / context

Part of the Multi-IMU analysis pipeline; depends on `common.paths`, `parser.calibration_segments`, and recording layout under `data/recordings/`.
