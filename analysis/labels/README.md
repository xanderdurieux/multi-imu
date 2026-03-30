# `labels/` — Manual + semi-automatic scenario labeling workflow

This package adds a lightweight semantic layer that can be joined to both:
- `features/features.csv` (window rows), and
- `events/event_candidates.csv` (event rows).

The workflow intentionally stays simple:
1. Start with section-level manual labels.
2. Add interval/event-level manual refinements where needed.
3. Optionally generate event suggestions from detected candidates, then confirm/edit.

## Supported label scopes
- `recording`: one label rule for all sections/windows in a recording.
- `section`: one manual label per section.
- `interval`: manual label over a section-relative time span `[window_start_s, window_end_s)`.
- `event`: label tied to event candidates (`event_id` preferred; fallback `event_type + event_time_s`).

Window resolution precedence is:
`interval > section > recording`.

## CSV/JSON format
Use either CSV (header required) or JSON list-of-objects.

Required fields:
- `scope`
- `recording_id`
- `scenario_label`

Conditionally required:
- `section_id` for `section`, `interval`, `event`
- `window_start_s`, `window_end_s` for `interval`
- `event_id` OR (`event_type` + `event_time_s`) for `event`

Recommended provenance metadata:
- `label_source`
- `labeler`
- `labeler_role`
- `labeled_at_utc`
- `label_confidence`
- `label_notes`
- `label_status`
- `ambiguity_label`
- `ambiguity_notes`
- `provenance_source`
- `source_artifact`
- `annotation_template_id`
- `annotation_example_id`
- `annotation_id`
- `annotation_batch_id`
- `double_label_group`
- `adjudication_status`
- `label_schema_version`
- `suggestion_source`
- `suggestion_rank`

Special semantic values are supported directly in `scenario_label` or `label_status`:
- `unknown`
- `ambiguous`
- `mixed`

No fixed enum is hardcoded in parser logic, so taxonomy can evolve over time.

Use `labels/labels_template.csv` as a starting point.

## Join behavior
### Join to feature windows
`features.extract` now attaches:
- `scenario_label`, `label_source`, `label_scope`
- all provenance fields above

for each window row by resolving label rules.

### Join to event tables
Use:
```bash
uv run python -m labels.workflow apply-event-labels data/sections/<section_id> --labels labels/event_labels.csv

# 4) Assign a defendable second-rater subset (e.g. 20%)
uv run python -m labels.workflow double-label-subset --annotations labels/section_labels.csv \
  --out labels/section_labels_with_double_subset.csv --reviewer-b reviewer_b --fraction 0.2 --seed 42

# 5) Compute inter-rater agreement for the doubly-labeled subset
uv run python -m labels.agreement --labels-a labels/reviewer_a.csv --labels-b labels/reviewer_b.csv \
  --out-dir outputs/inter_rater
```
This writes:
- `events/event_candidates_labeled.csv`

Matching precedence for `scope=event` rules:
1. exact `event_id`
2. same `event_type` + `event_time_s` within tolerance

## Lightweight commands
From `analysis/`:

```bash
# 1) Section-level scaffold (manual baseline)
uv run python -m labels.workflow scaffold-sections 2026-02-26_r1 --out labels/section_labels.csv --labeler reviewer_a

# 2) Suggest event labels from event candidates (semi-automatic)
uv run python -m labels.workflow suggest-events 2026-02-26_r1 --out labels/event_suggestions.csv --min-confidence 0.35

# 3) Human edits suggestions -> final labels, then apply to event table
uv run python -m labels.workflow apply-event-labels data/sections/2026-02-26_r1s1 --labels labels/event_labels_final.csv
```

## Proposed taxonomy for thesis scenarios
See:
- `labels/taxonomy_thesis_scenarios.json`

Suggested top-level labels:
- `steady_state`
- `surface_disturbance`
- `hard_braking`
- `rapid_swerve`
- `rider_bicycle_mismatch`
- `fall_or_drop`
- plus `unknown`, `ambiguous`, `mixed`

## Example labeled files
- `labels/examples/section_labels_example.csv`
- `labels/examples/event_labels_example.csv`
- `labels/examples/event_suggestions_example.csv`

## Small evaluation dataset annotation protocol (consistent + low effort)
Use this process for ~5–10 sections first:

1. **Freeze taxonomy version**
   - Set `label_schema_version` (for example `thesis_v1`) before annotation.
2. **Section pass (fast baseline)**
   - Fill one section-level row per section with `scenario_label` and `label_confidence`.
3. **Event pass (targeted refinement)**
   - Generate suggestions from event candidates, then confirm/edit each suggested row.
4. **Ambiguity policy**
   - Use `unknown` when evidence is insufficient.
   - Use `ambiguous` when multiple classes plausible and undecidable.
   - Use `mixed` when distinct classes co-occur clearly in a single interval/section.
5. **Provenance discipline**
   - Always populate `labeler`, `labeled_at_utc`, and `label_notes` for non-obvious cases.
6. **Inter-rater consistency check (recommended)**
   - Double-label ~20% of rows by a second reviewer, compare agreement, and refine guidelines before scaling.
   - Use `labels.agreement` to export `inter_rater_summary.json`, `inter_rater_by_scope.csv`, and explicit disagreement rows.
