# `sync/` — IMU Stream Synchronization

This package aligns the Arduino (helmet, **target**) timestamps to the Sporsa
(bicycle, **reference**) clock. It provides four sync methods, shared signal
processing utilities, a comparison/selection pipeline, and a session-level
orchestrator for the broader Multi-IMU workflow.

---

## Concepts

### The sync problem

The Sporsa sensor uses a stable Unix-epoch clock. The Arduino sensor uses
`millis()` — a counter that starts from zero at boot with no wall-clock
reference. Two parameters need to be estimated:

- **Offset** (`offset_seconds`): constant shift, `t_ref = t_tgt + offset` at
  the target's time origin. Accounts for the gap between sensor boot times
  and absolute time epochs.
- **Drift** (`drift_seconds_per_second`): the Arduino crystal runs slightly
  fast or slow. A drift of 400 ppm means the Arduino gains/loses 400 µs per
  second of recording.

The corrected target timestamp is:

```
t_ref = t_tgt + offset + drift × (t_tgt − t_origin)
```

This linear model is captured in the `SyncModel` dataclass (`core.py`) and
serialised to `sync_info.json`.

### Quality metric

All four methods compute the Pearson r of `acc_norm` between the two
resampled streams over their overlap window — evaluated twice:

1. **Offset only** (`corr.offset_only`): drift correction is zeroed out.
2. **Offset + drift** (`corr.offset_and_drift`): full model applied.

Both values are written to `sync_info.json` by `core.compute_sync_correlations`
so they are directly comparable across methods.

---

## Package layout

```
sync/
├── core.py        # shared stream utilities, SDA/LIDA math, SyncModel, metrics
├── methods.py     # the four concrete sync methods + recording/file I/O
├── pipeline.py    # orchestration, comparison, selection, CLI main()
├── __main__.py    # delegates to pipeline.main()
└── __init__.py    # package exports
```

---

## Recommended workflow

Input is always ``data/recordings/<recording>/parsed/`` (Sporsa + Arduino CSVs).

```bash
# One recording: run all four methods, select best, flatten into synced/
uv run -m sync 2026-02-26_r5

# Whole session (every folder 2026-02-26_r*)
uv run -m sync 2026-02-26 --all
```

Per-method outputs live under ``synced/sda`` … only until selection finishes; then
they are removed and only the chosen streams remain in ``synced/`` together with
``all_methods.json``.

### Python API

```python
from sync.pipeline import synchronize_recording_all_methods, synchronize_session

result = synchronize_recording_all_methods("2026-02-26_r5")
results = synchronize_session("2026-02-26")
```

`RecordingResult` lists per-method success/failure and the selected method (if any).

### CLI

```text
python -m sync <recording_or_session_prefix> [--all]
```

---

## `pipeline.py`

High-level orchestration: run methods, compare them, apply the selected result,
and expose the CLI entrypoint used by ``python -m sync``.

---

## Sync methods

### Method 1 — SDA only (`methods.py`)

**When to use:** quick baseline; recordings too short for reliable drift
estimation; sanity-checking the offset independently of drift.

**Algorithm:**

1. Run `estimate_offset()` (SDA) on the full recording.
2. Build a `SyncModel` with `drift_seconds_per_second = 0.0`.
3. Apply the offset-only correction to the target timestamps.
4. Compute quality correlations via `compute_sync_correlations()`.

**Output (intermediate):** `synced/sda/`  
**`sync_info.json` extras:** `sync_method: "sda_offset_only"`, `sda_score`

---

### Method 2 — SDA + LIDA (`methods.py`)

**When to use:** standard offline post-processing for recordings without
reliable calibration sequences.

**Algorithm:**

1. Run SDA (coarse offset).
2. Slide a window over the recording to measure the local offset at each
   position (LIDA refinement).
3. Fit a linear model `offset(t) = offset_0 + drift × Δt` to the scatter.
4. Apply the full `offset + drift × Δt` model.

**Output (intermediate):** `synced/lida/`  
**`sync_info.json` extras:** `sync_method: "sda_lida"`

**Notable parameters** (passed to `synchronize_recording`):

| Parameter | Default | Meaning |
|---|---|---|
| `sample_rate_hz` | 100 Hz | Resampling rate for cross-correlation |
| `max_lag_seconds` | 60 s | Maximum SDA search window |
| `window_seconds` | 20 s | LIDA window length |
| `window_step_seconds` | 10 s | LIDA window stride |
| `local_search_seconds` | 2 s | LIDA local refinement range |

The module also exposes `synchronize(reference_csv, target_csv, ...)` for
file-to-file use without the recording-directory convention.

---

### Method 3 — Calibration-sequence sync (`methods.py`)

**When to use:** preferred method when both opening and closing calibration
tap-bursts are present and well-detected. Gives the most precise offset and
drift because it uses known, sharp events as timing anchors.

**Algorithm:**

1. Detect calibration segments in the **reference** sensor only. Requires ≥ 2
   segments (opening + closing).
2. Coarse offset: match the opening calibration cluster in the target by
   peak-cluster timing. Falls back to low-rate SDA if no cluster is found.
3. Fine offset at each in-range calibration window: narrow-window SDA
   cross-correlation centred on each calibration's peak burst.
4. If 3+ windows are successfully refined, fit a weighted linear model
   `offset(t) = offset_0 + drift * (t - t_origin)` across all windows
   (weights from per-window correlation scores). If fit quality is poor,
   fall back to the previous first/last-anchor estimate.
5. If the resulting drift exceeds 1 % (physically implausible), fall back to
   the recording-duration ratio.

**Output (intermediate):** `synced/cal/`

**`sync_info.json` extras:**

| Key | Meaning |
|---|---|
| `sync_method` | `"calibration_windows"` |
| `drift_source` | `"calibration_windows"` or `"duration_ratio"` |
| `calibration.opening.score` | Cross-correlation quality of the opening window |
| `calibration.closing.score` | Cross-correlation quality of the closing window |
| `calibration.calibration_span_s` | Time between opening and closing calibrations |

**Quality indicators:** both scores ≥ 0.5, span ≥ 60 s,
`drift_source == "calibration_windows"`.

---

### Method 4 — Online sync (`methods.py`)

**When to use:** real-time/causal context where the closing calibration has
not yet occurred. Also useful as a reference point when evaluating how much
drift matters for a given recording.

**Algorithm:**

1. Detect the opening calibration in the reference only.
2. Match the opening cluster in the target; refine with narrow-window SDA.
3. Load a pre-characterised median drift from
   `data/drift_characterisation.json` (falls back to 400 ppm).
4. Back-propagate the refined offset to the target time origin using the
   loaded drift.

**Output (intermediate):** `synced/online/`  
**`sync_info.json` extras:** `sync_method: "online_opening_anchor"`,
`drift_ppm_source`, `drift_ppm_applied`

**Comparison of all methods:**

| Property | SDA | LIDA | Calibration | Online |
|---|---|---|---|---|
| Requires full recording | Yes | Yes | Yes | No (causal) |
| Calibration tap required | No | No | Yes (both) | Yes (opening only) |
| Drift source | Signal | Signal | Two anchors | Pre-characterised |
| Typical precision | Low | Medium | High | Medium |

---

## Comparison and selection

After any methods have been run, `pipeline.py` compares them and picks the
best result.

### Selection heuristic

1. **Calibration first:** if calibration-sync passes all quality gates
   (span ≥ 60 s, both scores ≥ 0.5, drift ≤ 5 000 ppm, correlation ≥ 0.2)
   → choose `calibration`.
2. **Highest correlation otherwise:** pick whichever available method has the
   highest `corr.offset_and_drift`. Ties broken by preference order:
   `calibration > lida > sda > online`.

### `synced/` output

`apply_selection` (called automatically by the pipeline) copies the winner into
flat `synced/`, writes `all_methods.json`, and then removes the per-method
subfolders.

| File | Description |
|---|---|
| `sporsa.csv` | Reference sensor (copy from winner) |
| `arduino.csv` | Synchronised target sensor (copy from winner) |
| `sync_info.json` | The winning method's model |
| `all_methods.json` | Comparison metrics for all four methods |

---

## Core algorithm modules

### `core.py` (SDA section) — Signal-Density Alignment

Estimates a single coarse offset by cross-correlating a 1-D *activity signal*
derived from both IMU streams.

1. Resample both streams to a uniform grid.
2. Compute a z-scored, orientation-invariant activity signal from selected
   vector norms. By default uses `acc_norm` differentiated once to emphasise
   transient events.
3. FFT cross-correlation to find the integer lag that maximises normalised
   overlap score.
4. Convert lag to seconds and adjust for differing start times.

Result: `OffsetEstimate(offset_seconds, lag_seconds, score, ...)`.

**Limitation:** SDA gives a single global offset with no drift estimate. Clock
drift accumulates over time and degrades alignment quality for long recordings.

---

### `core.py` (LIDA section) — Local Instance-based Drift Analysis

Extends SDA by fitting a linear drift model on top of the coarse offset.

1. Run SDA to get a coarse lag.
2. Slide a window (default 20 s, step 10 s) over the recording and find the
   best local lag within ±`local_search_seconds` at each position → scatter
   of `(t_target, local_offset)` points.
3. Weighted linear regression: `offset(t) = offset_0 + drift × (t − t_origin)`.
   Slope = drift; intercept = offset at target origin.
4. Return a `SyncModel`.

**`SyncModel` fields:**

| Field | Meaning |
|---|---|
| `offset_seconds` | Offset at `target_time_origin_seconds` |
| `drift_seconds_per_second` | Clock drift (positive = Arduino runs slow) |
| `target_time_origin_seconds` | Reference point for drift extrapolation |
| `sample_rate_hz` | Resampling rate used during estimation |

**Helpers:** `apply_sync_model`, `save_sync_model` / `load_sync_model`,
`resample_aligned_stream`.

---

### `core.py` (metrics) — Synchronization quality metrics

Used by every method to populate `sync_info.json`:

- `acc_norm_correlation(ref_df, tgt_df, *, sample_rate_hz)` — Pearson r over
  the shared timestamp window.
- `compute_sync_correlations(ref_df, tgt_df, model, *, sample_rate_hz)` —
  evaluates both offset-only and offset+drift correlations, returns the
  standard `"correlation"` dict block.

---

### `core.py` (streams) — Shared stream utilities

| Function | Purpose |
|---|---|
| `load_stream(path)` | Load sensor CSV, coerce numerics, sort by timestamp |
| `add_vector_norms(df)` | Append `acc_norm`, `gyro_norm`, `mag_norm` |
| `resample_stream(df, hz)` | Linear-interpolate onto a uniform time grid |
| `resample_to_reference_timestamps(tgt, ref)` | Resample target at reference timestamps |
| `lowpass_filter(df, cutoff_hz, sr_hz)` | Zero-phase Butterworth filter |
| `remove_dropouts(df)` | Remove near-zero acceleration rows (BLE dropout packets) |
| `apply_linear_time_transform(ts, ...)` | Apply `offset + drift × Δt` to a timestamp array |

---

## `sync_info.json` schema

Every sync method writes `sync_info.json` with these guaranteed fields plus
method-specific extras:

```jsonc
{
  "reference_csv": "<path>",
  "target_csv": "<path>",
  "target_time_origin_seconds": 12345.678,
  "offset_seconds": 3601.234,
  "drift_seconds_per_second": 4.1e-4,
  "sample_rate_hz": 100.0,
  "max_lag_seconds": 60.0,
  "created_at_utc": "2026-02-26T10:00:00+00:00",
  "sync_method": "<method_id>",
  "correlation": {
    "offset_only": 0.312,
    "offset_and_drift": 0.487,
    "signal": "acc_norm",
    "sample_rate_hz": 100.0
  }
}
```

`all_methods.json` (written by `apply_selection`) contains one block per
method plus the selection decision — enough to reproduce any sync result or
switch methods in downstream analysis without re-running the sync.
