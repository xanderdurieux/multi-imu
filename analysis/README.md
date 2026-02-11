## Analysis: IMU Data Processing

Python toolkit for offline analysis of the IMU data

### Modules

- **`common/`**
  - Shared types and helpers (`IMUSample`, CSV helpers, data paths).
- **`parser/`**
  - Parses raw IMU logs under `data/raw/<session_name>/` into CSV files under `data/processed/<session_name>/`.
- **`plot/`**
  - Plots time‑series from processed CSV files.

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

- **Plot a processed CSV** (writes a PNG next to the CSV):

```bash
uv run -m plot.plot_session data/processed/<session_name>/<filename>
```

Everything uses the shared `IMUSample` model and CSV schema from `common`, so new processing modules can plug into the same types later.


