# `common/` — Shared utilities

Small helpers shared across the thesis analysis pipeline.

## What’s inside
- `paths.py`: the repo’s directory conventions for sessions, recordings, and
  calibration-bounded sections.
- `csv_schema.py`: a standardized IMU CSV column set plus `load_dataframe()` /
  `write_dataframe()` to keep pipeline stages consistent.
- `calibration_segments.py`: calibration-sequence detection used by
  `parser.split_sections` and by sync methods that rely on calibration anchors.
- `quaternion.py`: quaternion math used by the orientation estimation stage.


## Data conventions (IMU CSVs)
All pipeline stages assume the same baseline columns:
`timestamp, ax, ay, az, gx, gy, gz, mx, my, mz`.

Missing columns are filled with missing values when writing/reading, so stages
can be combined even when some sensors or intermediate steps are unavailable.

