# `sync/`

The sync package is organized around four independent synchronization methods plus a pipeline that runs them all and selects the best result.

## Modules

- `sync_sda.py` — full-recording SDA offset-only synchronization.
- `sync_lida.py` — SDA + LIDA offset-and-drift synchronization.
- `sync_cal.py` — calibration-anchor synchronization using opening/closing tap bursts.
- `sync_online.py` — single-anchor synchronization using a pre-characterized drift rate.
- `pipeline.py` — runs methods, compares their metrics, and copies the winner to `synced/`.
- `plotting.py` — sync comparison and alignment plots.
- `helpers.py` — shared filesystem/result helpers used by the method modules and pipeline.
- `align_df.py`, `drift_estimator.py`, `metrics.py`, `common.py` — reusable algorithmic building blocks.

## Output layout

Each method writes its own output stage:

- `synced/sda/`
- `synced/lida/`
- `synced/cal/`
- `synced/online/`

The pipeline copies the selected winner to `synced/` and writes `synced/all_methods.json`.

## Typical usage

```bash
uv run -m sync.sync_sda 2026-02-26_5/parsed
uv run -m sync.sync_lida 2026-02-26_5/parsed
uv run -m sync.sync_cal 2026-02-26_5/parsed
uv run -m sync.sync_online 2026-02-26_5/parsed
uv run -m sync.pipeline 2026-02-26_5 --plot
```

For compatibility, the old module names (`sda_sync.py`, `lida_sync.py`, `calibration_sync.py`, `online_sync.py`) remain as thin wrappers around the new implementation modules.
