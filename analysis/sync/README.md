# `sync/` — Recording-Level Time Alignment

`sync/` aligns the Arduino stream (target) to the Sporsa stream (reference) at
the recording level. Four strategies fit an affine offset-plus-drift clock model
``t_ref = t_tgt + b + a*(t_tgt - t0)``; each writes its own stage directory and
the best one is flattened into `data/recordings/<recording>/synced/`.

## Module layout

Each file owns one concern in the strategy pipeline:

| File            | Role                                                                |
| --------------- | ------------------------------------------------------------------- |
| `model.py`      | `SyncModel` dataclass + ms↔s time-transform helpers                 |
| `config.py`     | `SyncConfig` loaded from `sync_args.json`; see field scopes below   |
| `signals.py`    | Stream I/O, resampling, activity-signal build, post-fit correlation |
| `anchors.py`    | **FINAL.** Calibration-anchor extraction (used by every anchor-based tier) |
| `xcorr.py`      | FFT xcorr, masked NCC, affine fit, and windowed refinement (non-causal + causal) |
| `strategies.py` | The four estimators: `estimate_<method>(ref_df, tgt_df, ...) -> (SyncModel, meta)` |
| `selection.py`  | Loads per-method `sync_info.json`, scores, picks the winning tier   |
| `pipeline.py`   | Per-recording orchestration, stage-dir I/O, and the `python -m sync` CLI |

Approach per method:

- **`multi_anchor`** — all calibration anchors (`anchors.py`) → weighted affine fit (`xcorr.fit_offset_drift`).
- **`one_anchor_adaptive`** — opening anchor (`anchors.py`) + causal windowed refinement (`xcorr.adaptive_windowed_refinement`).
- **`one_anchor_prior`** — opening anchor (`anchors.py`) + fixed drift prior (`config.one_anchor_prior_drift_ppm`).
- **`signal_only`** — coarse lag via `xcorr.estimate_lag` + non-causal windowed refinement (`xcorr.windowed_lag_refinement`).

## Configuration

Parameters live in `data/_configs/sync_args.json`. Each field maps to a single
module/function:

| Key                                                 | Influences                                   |
| --------------------------------------------------- | -------------------------------------------- |
| `signal_mode`, `resample_rate_hz`                   | `signals.build_resampled_activity_signal`    |
| `min_valid_fraction`                                | `xcorr.masked_ncc` overlap gate              |
| `anchor_refinement.resample_rate_hz`, `.search_seconds` | `anchors._refine_offset_xcorr`           |
| `window_refinement.*`                               | `xcorr.windowed_lag_refinement`, `xcorr.adaptive_windowed_refinement`, `xcorr.fit_windowed_offset_drift` |
| `signal_only.coarse_search_seconds`                 | `strategies.estimate_signal_only` coarse span |
| `one_anchor_prior.drift_ppm`                        | `strategies.estimate_one_anchor_prior`       |

## CLI

From `analysis/`:

```bash
uv run -m sync 2026-02-26_r5                    # single recording, pick best
uv run -m sync 2026-02-26 --all                 # every recording in a session
uv run -m sync 2026-02-26_r5 --method signal_only  # force one method
```

## Outputs

Each method first writes `data/recordings/<recording>/synced/<method>/`:

- `sporsa.csv`, `arduino.csv`
- `sync_info.json` — selected affine model + correlation + per-method summary
- `sync_metadata.json` — hyperparameters + debug stats

After selection, those files are copied into flat
`data/recordings/<recording>/synced/` and the per-method directories are pruned.

## Public API

From `sync` (re-exported in `__init__.py`):

- `synchronize_recording_all_methods(recording_name)` — run all four, pick best
- `synchronize_recording_chosen_method(recording_name, method)` — run one
- `SyncModel`, `apply_sync_model`, `apply_linear_time_transform`
- `SYNC_METHODS`, `SyncMethodQuality`, `SyncSelectionResult`
