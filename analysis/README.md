# master-thesis/analysis

Python toolkit for offline analysis of the IMU data.

## Modules

- **`common/`**
  - Shared types and helpers (`IMUSample`, CSV helpers, data paths).
- **`parser/`**
  - Parses raw IMU logs under `data/raw/<session_name>/` into CSV files under `data/processed/<session_name>/`.
- **`sync/`**
  - Synchronizes two processed IMU streams with correlation-based offset + linear drift estimation.
  - Provides reusable methods for loading, feature building, offset/drift estimation, model persistence, model application, and resampling.
- **`plot/`**
  - Plots time-series from processed CSV files.

## Setup

- Python `>= 3.13`
- Recommended: `uv` for dependencies and running tools.

From the `analysis/` directory:

```bash
cd master-thesis/analysis
uv sync
```

## Usage

### Parse a session
reads `data/raw/<session_name>/`, writes `data/processed/<session_name>/`

```bash
uv run -m parser.session <session_name>
```

### Synchronize IMU streams
`sporsa` as reference, `arduino` as target

- Session mode (automatic stream selection in `data/processed/<session_name>/`):

```bash
uv run -m sync.sync_streams session <session_name>
```

- Manual mode (explicit reference and target CSV paths):

```bash
uv run -m sync.sync_streams pair data/processed/<session_name>/<ref>.csv data/processed/<session_name>/<target>.csv
```

- Optional CLI arguments:
  - `--sample-rate-hz 60`
  - `--max-lag-seconds 30`
  - `--resample-rate-hz 100`

- Generated outputs (next to target CSV):
  - `<target>_to_<reference>_sync.json` (contains estimated time offset, drift, scale, quality metrics and settings)
  - `<target>_synced.csv` (target stream in reference clock; includes `timestamp_orig` and `timestamp_aligned`)
  - `<target>_synced_resampled_<hz>hz.csv` (only if `resample_hz` is provided: uniformly resampled synchronized target stream)

### Plot a processed CSV 

```bash
uv run -m plot.plot_device data/processed/<session_name>/<filename>
```

Optional CLI arguments

- `--magnitudes` (plot CSV with vector magnitudes)

### Compare two streams in one plot

```bash
uv run -m plot.compare_streams data/processed/<session_name>/sporsa.csv data/processed/<session_name>/arduino_synced.csv
```

Optional CLI arguments:

- `--split-axes` (separate x/y/z panels)
- `--magnitudes` (plot magnitudes instead of x/y/z)
- `--relative-time`

### Run full single-session pipeline
sync + useful plots in same processed folder

```bash
uv run -m pipeline.session_pipeline <session_name>
```

Optional pipeline arguments:

- `--sample-rate-hz 50`
- `--max-lag-seconds 20`
- `--resample-rate-hz 100` (set `<= 0` to skip resampled output CSV)
