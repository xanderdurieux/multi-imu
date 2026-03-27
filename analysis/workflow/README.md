# `workflow/` — Reproducible thesis orchestrator

`workflow` is the **single entry point** for end-to-end runs.

```bash
uv run python -m workflow --config configs/workflow.thesis.json
```

It wraps `pipeline.run_pipeline` but adds:
- one-file run configuration,
- explicit dataset root handling,
- easy CLI overrides for quick experiments.

## Config schema

The default config (`configs/workflow.thesis.json`) defines:
- dataset location (`data_root`),
- selected sync/orientation/calibration methods,
- event thresholds config path,
- whether to generate plots/exports,
- optional session and recording filters.

Relative paths are resolved relative to the config file location.

## Typical usage

### 1) Full reproducible run

```bash
uv run python -m workflow --config configs/workflow.thesis.json
```

### 2) Re-run one recording with no plots

```bash
uv run python -m workflow --config configs/workflow.thesis.json \
  --recording 2026-02-26_r5 --force --no-plots
```

### 3) Use an external dataset root

```bash
uv run python -m workflow --config configs/workflow.thesis.json \
  --data-root /mnt/datasets/multi-imu/data
```
