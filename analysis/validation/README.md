# `validation/` — Section QC gates

This package contains quality heuristics used to decide whether a section is
usable for downstream motion/incident analysis.

## `validation/comprehensive.py`
`write_section_qc(section_path, ...)` computes a tier:
- `good`
- `marginal`
- `poor`

and writes it into:
- `qc_section.json` (inside the section folder)

The QC checks are intentionally simple baselines and consider:
- calibration outputs (`calibrated/calibration.json`)
- basic stream usability (timestamps duration + timing-gap proxy)
- feature presence (and missing/empty labels)
- orientation quality (via `orientation/orientation_stats.json`)

## Typical usage
The end-to-end `pipeline` calls this automatically after feature extraction.

If you need it manually (Python API):

```python
from pathlib import Path
from validation.comprehensive import write_section_qc

write_section_qc(Path(".../2026-02-26_r5s1/"), orientation_variant="complementary_orientation")
```

