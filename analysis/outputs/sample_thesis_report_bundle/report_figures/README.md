# Sample report figures

Binary figure artifacts (`.pdf`, `.png`) are intentionally not committed because this repository's PR workflow does not support binary files.

To regenerate the full publication-ready figures locally:

```bash
cd analysis
uv run python -m reporting --output-dir outputs/sample_thesis_report_bundle
```

Expected figure filenames:

- `pipeline_overview.pdf` / `.png`
- `orientation_filter_comparison.pdf` / `.png`
- `event_centered_bike_vs_rider.pdf` / `.png`
- `feature_separability.pdf` / `.png`
- `success_failure_case_studies.pdf` / `.png`
