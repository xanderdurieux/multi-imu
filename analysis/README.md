## Analysis: IMU Data Processing

Python toolkit for offline analysis of the IMU data.

### Modules

- **`common/`**
  - Shared types and helpers (`IMUSample`, CSV helpers, data paths).
- **`parser/`**
  - Parses raw IMU logs under `data/raw/<session_name>/` into CSV files under `data/processed/<session_name>/`.
- **`sync/`**
  - Synchronizes two processed IMU streams with correlation-based offset + linear drift estimation.
  - Provides reusable methods for loading, feature building, offset/drift estimation, model persistence, model application, and resampling.
- **`plot/`**
  - Plots time-series from processed CSV files.

### Setup

- Python `>= 3.13`
- Recommended: `uv` for dependencies and running tools.

From the `analysis/` directory:

```bash
cd master-thesis/analysis
uv sync
```

### Usage

- **Parse a session** (reads `data/raw/<session_name>/`, writes `data/processed/<session_name>/`):

```bash
uv run -m parser.session <session_name>
```

- **Synchronize IMU streams** (`sporsa` as reference, `arduino` as target):

Session mode (automatic stream selection in `data/processed/<session_name>/`):

```bash
uv run -m sync.sync_streams session <session_name>
```

Manual mode (explicit reference and target CSV paths):

```bash
uv run -m sync.sync_streams pair data/processed/<session_name>/<ref>.csv data/processed/<session_name>/<target>.csv
```

Optional CLI arguments:

- `--sample-rate-hz 60`
- `--max-lag-seconds 30`
- `--resample-rate-hz 100`

Generated outputs (next to target CSV):

- `<target>_to_<reference>_sync.json`
  - Contains estimated time offset, drift, scale, quality metrics and settings.
- `<target>_synced.csv`
  - Target stream in reference clock; includes `timestamp_orig` and `timestamp_aligned`.
- `<target>_synced_resampled_<hz>hz.csv` (only if `resample_hz` is provided)
  - Uniformly resampled synchronized target stream.

- **Plot a processed CSV** (writes a PNG next to the CSV):

```bash
uv run -m plot.plot_device data/processed/<session_name>/<filename>
```

- **Plot a processed CSV with vector magnitudes** (`|acc|`, `|gyro|`, `|mag|`):

```bash
uv run -m plot.plot_device data/processed/<session_name>/<filename> --magnitudes
```

- **Compare two streams in one plot** (for raw vs raw, synced vs reference, or other processed files):

```bash
uv run -m plot.compare_streams data/processed/<session_name>/sporsa.csv data/processed/<session_name>/arduino_synced.csv
```

Useful comparison options:

- `--label-a sporsa` (label for first stream)
- `--label-b arduino_synced` (label for second stream)
- `--output data/processed/<session_name>/comparison.png` (custom output path)
- `--relative-time` (x-axis starts at 0 for each stream)
- `--split-axes` (plots `x`, `y`, `z` in separate panels to avoid overlap)
- `--magnitudes` (plot magnitudes instead of x/y/z)
