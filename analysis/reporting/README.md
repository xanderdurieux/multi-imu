# Thesis Reporting Module

Generates publication-ready thesis figures/tables from processed section outputs.

## Run

```bash
cd analysis
uv run python -m reporting --output-dir outputs/thesis_report_bundle
```

## Scripts

- **Figure generation**: `reporting/figure_gen.py`
- **Table generation**: `reporting/table_gen.py`
- **Bundle orchestration + captions**: `reporting/bundle.py`

## Output structure

- `report_figures/`: final thesis/report figures (`.pdf` + `.png`)
- `report_tables/`: report tables (`.csv` + `.md`)
- `draft_debug/`: reserved for exploratory/debug visuals (kept separate from final plots)
- `caption_suggestions.md`: candidate captions + chapter placement suggestions
- `bundle_manifest.json`: machine-readable summary of generated artifacts

## Generated artifacts

- Pipeline overview figure.
- Quality summary table.
- Synchronization method comparison table.
- Orientation filter comparison figure.
- Event-centered bike vs rider figure.
- Feature separability figure.
- Single-sensor vs dual-sensor comparison table.
- Representative success/failure case-study figure + table.

## Repository note

The committed sample bundle keeps text artifacts (`manifest`, `captions`, `csv/md` tables) in git. Binary figures (`.pdf`, `.png`) are generated locally by the command above and intentionally not versioned to keep PR tooling compatible.
