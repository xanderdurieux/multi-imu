# Thesis smoke-test fixture

This fixture is intentionally tiny but representative. It includes:

- raw session logs for three recordings (`data/sessions/2026-01-15/{sporsa,arduino}`),
- a workflow config (`workflow.fixture.json`) that runs the canonical thesis path with plots disabled,
- a minimal labels file (`labels_fixture.csv`) so evaluation has class labels,
- an evaluation config (`evaluation.fixture.json`) with low sample thresholds for fixture scale.

Use this fixture via:

```bash
cd analysis
uv run python -m unittest tests.test_thesis_workflow_smoke
```
