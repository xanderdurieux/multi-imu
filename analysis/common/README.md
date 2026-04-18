# `common/` — Shared Helpers

`common/` holds the shared building blocks used across the analysis pipeline.

## Modules

- `paths.py`: canonical data roots, recording/section path helpers, CSV/JSON I/O, label/config lookup helpers, and schema normalization.
- `signals.py`: vector norms, moving-average smoothing, and NaN-preserving Butterworth low-pass filtering.
- `statistics.py`: small NaN-safe scalar statistics used during feature extraction.
- `quaternion.py`: quaternion math used by the orientation stage.

## IMU CSV normalization

`read_csv()` and `write_csv()` normalize raw IMU tables to the project schema:

```text
timestamp, ax, ay, az, acc_norm, gx, gy, gz, gyro_norm, mx, my, mz, mag_norm
```

- `timestamp` is in milliseconds.
- `acc_norm`, `gyro_norm`, and `mag_norm` are recomputed when the raw axis triplets are present.
- Missing sensor values are kept as `NaN`.
- Extra non-schema columns are preserved after the standard columns.

This normalization is only applied to DataFrames that look like raw IMU streams. Derived tables keep their own columns.

## Path conventions

The most-used helpers are:

- `data_root()`, `sessions_root()`, `recordings_root()`, `sections_root()`, `exports_root()`, `evaluation_root()`
- `recording_dir()`, `recording_stage_dir()`, `section_dir()`, `section_stage_dir()`
- `sensor_csv("<recording>/<stage>", "<sensor>")`
- `iter_sections_for_recording(recording_name)`
- `recording_labels_csv()` and `section_labels_csv()`
- `default_workflow_config_path()`

`data_root()` can be overridden with `MULTI_IMU_DATA_ROOT`.

## I/O helpers

- `read_csv()` / `write_csv()` for schema-aware CSV I/O
- `read_json_file()` / `write_json_file()` for UTF-8 JSON
- `dataframe_to_json_records()` for converting pandas rows into JSON-native records

The rest of the codebase should prefer these helpers over ad hoc file handling so the path and schema assumptions stay consistent.
