# `labels/` — Manual scenario labeling utilities

These tools help you attach human scenario labels (scenario labels / fall or
incident events, etc.) to the automatically created feature windows.

## Label file format (CSV or JSON)
Use either:
- `*.csv` with a required header
- `*.json` as a list of objects with the same fields

Required CSV fields:
- `scope`: `recording` | `section` | `interval`
- `recording_id`: e.g. `2026-02-26_r5`
- `section_id`: e.g. `2026-02-26_r5s1` (required for `section` and `interval`)
- `window_start_s`, `window_end_s`: section-relative seconds (required for `interval`)
- `scenario_label`: the class / scenario name
- `label_source`: optional free text (e.g. `manual_v1`)

Interval matching uses an overlap rule consistent with feature windows:
windows are treated as half-open `[start, end)` in section-relative seconds.

Use `labels/labels_template.csv` as the starting point for your own sheet.

## Generate a scaffold CSV
Typical workflow:

1. Parse raw logs so each recording has `parsed/<sensor>.csv`:
   `uv run python -m parser.session <session_name>`
2. Create scaffold labels:
   `uv run python -m labels.scaffold labels/generated_scaffold.csv`
3. Fill `scenario_label`, then run the pipeline with:
   `uv run python -m pipeline --session <session_name> --labels labels/generated_scaffold.csv ...`

Scaffold commands:

```bash
uv run python -m labels.scaffold overview
uv run python -m labels.scaffold overview sections
uv run python -m labels.scaffold labels/generated_scaffold.csv
uv run python -m labels.scaffold labels/generated_scaffold.csv --recording-only
```

## Interactive event labeler (Plotly HTML)
Generate an HTML browser with synced IMU plots (+ optional GPS track) so you
can click intervals/peaks and download a label CSV:

```bash
uv run python -m labels.event_labeler 2026-02-26_r2/synced
uv run python -m labels.event_labeler sections/2026-02-26_r2s1 out.html
```

If `out.html` is omitted, it writes `event_labeler.html` next to the provided
folder.

