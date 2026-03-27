# `orientation/` — Section-level attitude estimation

Orientation estimation runs filter variants over calibrated section data to
produce Euler angles (yaw/pitch/roll) and per-variant quality metrics.

Filters implemented in `estimate.py`:
- `madgwick_acc_only`
- `madgwick_9dof`
- `complementary_orientation`
- `ekf_orientation`

## Main module
- `estimate.py`
  - Public entry point: `estimate_section(section_path, variants=...)`
  - CLI entry points:
    - `python -m orientation` (runs `estimate_sections_from_args` with default variants)
    - `python -m orientation.estimate` (supports `--variant` to select filters)
- `validate.py`: simple QC status for `orientation_stats.json`

## CLI usage
From `analysis/`:

```bash
uv run python -m orientation <section_path>
uv run python -m orientation <recording_name> --all-sections
```

Select explicit filter variants (repeatable):

```bash
uv run python -m orientation.estimate <section_or_recording> \
  --all-sections --variant complementary_orientation --variant ekf_orientation
```

## Outputs (per section)
- `orientation/<sensor>__<variant>.csv` (timestamp + quaternion + Euler angles)
- `orientation/orientation_stats.json` (quality metrics per variant)
- Optional diagnostic plots (written when enabled by the caller)

## Validation
```bash
uv run -m orientation.validate <section_path>
```


## Dynamic cycling filter comparison

Use `compare_dynamic.py` to evaluate **all existing orientation outputs** on
static and dynamic cycling sections with overlay plots and summary tables.

```bash
uv run python -m orientation.compare_dynamic <section_path>
uv run python -m orientation.compare_dynamic <recording_name> --all-sections
```

Artifacts written to `orientation/comparison_dynamic/`:
- `summary_per_filter.csv`: smoothness/responsiveness, drift, roll/pitch dynamic consistency, event separability proxy
- `summary_pairwise_agreement.csv`: inter-filter quaternion-angle agreement stats
- `overlay_static.png` and `overlay_dynamic.png`: all filters overlaid on same section windows
- `comparison_summary.json`: per-sensor recommendation + magnetometer unreliability fractions
- (recording mode) `comparison_overall_summary.csv` and `THESIS_INTERPRETATION.md`

Magnetometer-based heading unreliability is flagged when calibrated magnetic
norm deviates strongly from section median or when heading jumps abruptly.
