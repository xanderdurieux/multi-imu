# master-thesis/analysis

Python toolkit for offline analysis, synchronization, calibration, and
visualization of multi-IMU cycling data used in the thesis.

The code turns raw logs from a bicycle-mounted Sporsa IMU and a rider-mounted
Arduino IMU into synchronized, calibrated, world-frame signals and diagnostic
plots that can be used for motion and fall / incident analysis.

**What lives in this directory:** `common/`, `parser/`, `sync/`, `visualization/`, and **`static_calibration/`** (standalone six-pose Arduino IMU calibration). Ride-level world calibration, orientation filtering, and feature extraction are **not** included here; the notes below describe how this tree fits into the broader thesis workflow when you add those tools separately.

## Terminology

- **Session**: one acquisition day containing multiple recordings, identified
  by a date string such as `2026-02-26`. Raw logs live under
  `analysis/data/sessions/<session_name>/`.
- **Recording**: one continuous multi-IMU file pair, stored under
  `analysis/data/recordings/<session_name>_<index>/`, for example
  `2026-02-26_5`. All intermediate processing stages are per-recording.
- **Section**: a contiguous sub-interval of a recording between two detected
  calibration sequences. Sections live under
  `analysis/data/recordings/<recording_name>/sections/section_N/` and are the
  main unit for detailed motion and incident analysis.

Wherever the code or README refers to these concepts, it uses the above
definitions.

---

## Project layout

- **`common/`**
  - CSV schema utilities and helpers for loading/writing DataFrames with a
    consistent column layout.
  - Path helpers (`paths.py`) for locating data directories
    (`sessions/`, `recordings/`, per-stage folders).
  - Convenience helpers such as `find_sensor_csv(recording_name, stage, sensor_name)`.

- **`parser/`**
  - High-level session parsing (`parser.session`) from raw logs to per-recording
    `parsed/` CSVs.
  - Device-specific parsers for Arduino (new + legacy) and Sporsa logs.
  - Recording-level statistics (`parser.stats`) for timing and quality analysis.
  - Calibration-based sectioning (`parser.split_sections`) that detects
    calibration sequences and splits recordings into sections.

- **`static_calibration/`**
  - Standalone **six static poses** Arduino calibration: parse raw logs, estimate
    accelerometer bias/scale and gyro bias, write JSON and diagnostic plots under
    `data/calibrations/`. See [`static_calibration/README.md`](static_calibration/README.md).
  - Run: `uv run python -m static_calibration` from `analysis/`.

- **`sync/`**
  - Four synchronization methods for aligning Sporsa and Arduino IMU streams:
    - **SDA only** (`sync.sda_sync` → `synced/sda/`): offset-only, no drift.
    - **SDA + LIDA** (`sync.lida_sync` → `synced/lida/`): offset + drift from windowed refinement.
    - **Calibration sync** (`sync.calibration_sync` → `synced/cal/`): offset + drift from tap-burst anchors.
    - **Online sync** (`sync.online_sync` → `synced/online/`): causal single-anchor + pre-characterised drift.
  - Full pipeline entry point: `python -m sync` runs all four methods, selects the best, copies to flat `synced/`, and writes comparison plots.
  - Shared primitives live in `sync.core` (streams, SDA alignment, LIDA `SyncModel`, correlation metrics).
  - See [`sync/README.md`](sync/README.md) for full algorithm and API documentation.

- **`visualization/`**
  - Plotting tools for single-stream analysis and multi-stream comparison:
    - Sensor plots (`visualization.plot_sensor`).
    - Sporsa–Arduino comparison plots (`visualization.plot_comparison`).
    - Calibration segment diagnostic plots (`visualization.plot_calibration_segments`).
    - Orientation and calibration plots (`visualization.plot_calibration`,
      `visualization.plot_orientation`).
  - Session-level orchestration (`visualization.plot_session`) that iterates
    all recordings and stages and regenerates plots.

---

## Setup

- **Python**: `>= 3.13`
- **Recommended**: `uv` for dependency management and running tools.

From the repository root:

```bash
cd master-thesis/analysis
uv sync
```

This creates an isolated environment with all required dependencies
(`pandas`, `numpy`, `matplotlib`, etc.).

---

## Data layout

The analysis code expects the following directory structure relative to
`analysis/`:

- **Raw input (per session)**  
  `analysis/data/sessions/<session_name>/`  
  Each session directory contains device-specific subfolders, typically:
  - `arduino/*.txt`
  - `sporsa/*.txt`

- **Static Arduino calibration (standalone)**  
  `analysis/data/calibrations/` — raw logs in `raw/`, parsed CSVs in `parsed/`,
  `arduino_imu_calibration.json`, and plots in `plots/`. Populated by
  `static_calibration` (see [`static_calibration/README.md`](static_calibration/README.md)).

- **Processed recordings**  
  `analysis/data/recordings/<recording_name>/` where
  `<recording_name> = "<session_name>_<index>"`, for example:
  - `analysis/data/recordings/2026-02-26_2/`
  - `analysis/data/recordings/2026-02-26_5/`

  Within each recording directory, a full thesis run can populate several *stages* (availability depends on which tools you run):

  - `parsed/` – normalized per-sensor CSVs and basic plots.
  - `synced/` – after `python -m sync`, the chosen alignment plus `all_methods.json` and comparison PNGs (per-method subfolders exist only briefly during the run).
  - `sections/section_N/` – per-section CSVs and plots, bounded by calibration
    events.
  - `calibrated/`, `orientation/`, `features/` – only appear if you run
    separate tooling (not bundled here); see “Extended pipeline” below.

- **Run logs** (optional)  
  `analysis/data/run_logs/<session_name>_pipeline_run.json` — if you use a
  session orchestrator that writes them.

---

## Thesis pipeline (reference)

The steps below describe the **intended** end-to-end chain (parse → sync →
sections → world calibration → orientation → features). **This checkout** ships
**parser**, **sync**, **visualization**, and **static_calibration**; run stages
manually with `uv run -m …` as documented in each section. A single-command
orchestrator and the downstream ride-calibration / orientation / feature tools
are **not** present here.

**Typical manual flow in this tree:**

1. `uv run -m parser.session <session_name>`
2. `uv run -m sync <recording_or_session> …` (see Stage 2)
3. Optional: `uv run -m parser.split_sections …` (Stage 3)
4. **Static Arduino IMU calibration** (six stationary logs): `uv run python -m static_calibration` (separate from ride recordings; see below)

---

## Stage 1 – Parse raw session logs (`parser.session`)

**Goal**: Convert device-specific logs into standardized per-recording IMU
CSVs with a consistent schema and units.

```bash
uv run -m parser.session <session_name>
# e.g. uv run -m parser.session 2026-02-26
```

- **Input**:  
  `data/sessions/<session_name>/{arduino,sporsa}/*.txt`

- **Output** (per recording `2026-02-26_k`):  
  - `data/recordings/2026-02-26_k/session_stats.json` (timing stats)  
  - `data/recordings/2026-02-26_k/parsed/` — `sporsa.csv`, `arduino.csv`, plots  
- **Output** (session): `data/sessions/<session_name>/session_stats.json`

---

## Static Arduino IMU calibration (`static_calibration/`)

Six **short, stationary** Arduino logs—one approximate pose per gravity direction
(±X, ±Y, ±Z)—are used to estimate **accelerometer bias and per-axis scale** and
**gyroscope bias**. This is **independent** of ride recordings; data lives under
`data/calibrations/`.

From `analysis/`:

```bash
uv run python -m static_calibration
```

This runs the full pipeline: parse each `raw/*.txt` → write `parsed/*.csv` →
estimate parameters → write `arduino_imu_calibration.json` → generate plots
(overview, per-face detail figures, calibration-parameter summary with boxplots).

Details, JSON fields, and applying the calibration to other CSVs:
[`static_calibration/README.md`](static_calibration/README.md).

**What to inspect**

- `arduino_imu_calibration.json` — fitted parameters, `face_counts`, `warnings`, `per_recording`.
- `plots/recordings_overview.png` — per-axis accelerometer panels (separate y-scales) plus gyro.
- `plots/recordings/<dominant_face>/` — one detail plot per recording.
- `plots/calibration_parameters.png` — fitted parameters vs per-recording means and within-window spread.

Exclude logs with empty parses, extreme dropouts, or poses that do not match the six-face model.

---

## Stage 2 – Synchronize IMU streams (`sync/`)

Four methods run internally (SDA, LIDA, calibration windows, online); the pipeline picks the best and flattens the result into `synced/`.

| Method | Module | Drift? |
|---|---|---|
| SDA only | `sync.sda_sync` | No |
| SDA + LIDA | `sync.lida_sync` | Yes |
| Calibration anchors | `sync.calibration_sync` | Yes |
| Online (single anchor) | `sync.online_sync` | Pre-characterised |

```bash
uv run -m sync 2026-02-26_5
uv run -m sync 2026-02-26 --all
```

The selected winner is written to `synced/` with `sync_info.json`,
`all_methods.json`, and a comparison plot.

See [`sync/README.md`](sync/README.md) for a full description of each method,
algorithm details, `sync_info.json` schema, and selection heuristics.

---

## Stage 3 – Split into calibration-bounded sections (`parser.split_sections`)

**Goal**: Turn each synchronized recording into multiple **sections**, each
bounded by two calibration sequences (opening and closing). These sections
are the atomic units for detailed motion and incident analysis.

```bash
uv run -m parser.split_sections <recording_name>/synced_cal
# e.g. uv run -m parser.split_sections 2026-02-26_5/synced_cal
```

- **Output**: `data/recordings/<recording_name>/sections/section_N/`
  - `sporsa.csv`, `arduino.csv`
  - Per-section sensor and comparison plots
  - Optional per-section `sync_info.json` when `sync=True`

Each section includes the closing calibration of the previous section and the
opening calibration of the next, making it suitable for:

- Local incident analysis.
- Comparing bicycle vs rider dynamics within a stable calibration window.

**Section inclusion hints**:

- Discard sections where:
  - Calibration detection fails or only one calibration sequence is present.
  - Plots show large gaps or obvious desynchronization.
- Prefer sections that:
  - Contain rich dynamic content (sprints, braking, bumps).
  - Have clean calibration motions at the boundaries.

---

## Extended pipeline *(not in this checkout)*

After `synced/` and `sections/`, the thesis workflow can continue with **ride-level
world-frame calibration** (producing `calibrated/` and `calibration.json`),
**orientation estimation** on body-frame CSVs (producing `orientation/` and
`orientation_stats.json`), and **feature extraction** into `features/`. Those
stages require separate code; this repository only documents the directory
names so you can interpret data trees and use `visualization.plot_calibration` /
`visualization.plot_orientation` when those folders exist.

---

## Stage 4 – Visualization (`visualization/`)

Plots are produced automatically by the pipeline, but can also be regenerated
manually. Modules such as `plot_calibration` and `plot_orientation` expect the
corresponding stage directories to exist (from external processing if not
present in this tree).

### Plot a single sensor stream

```bash
uv run -m visualization.plot_sensor <recording_name>/<stage> <sensor_name> [--norm] [--split]
```

- **Examples**:
  - `uv run -m visualization.plot_sensor 2026-02-26_5/parsed sporsa --norm`
  - `uv run -m visualization.plot_sensor 2026-02-26_5/calibrated arduino --norm`

### Compare two streams

```bash
uv run -m visualization.plot_comparison <recording_name>/<stage> [sensor_name_a] [sensor_name_b] [--norm]
```

Useful for visually validating synchronization (`parsed` vs `synced_*`) and
for comparing bicycle vs rider dynamics.

### Plot all recordings for a session

```bash
uv run -m visualization.plot_session <session_name> [--stage STAGE]
```

This iterates all recordings whose name starts with `<session_name>_` and
regenerates plots for each stage (including sections, calibrated, and
orientation).

---

## Data-quality criteria for motion and incident analysis

Use this section when your recording tree includes the corresponding JSON
artifacts (`sync_info.json`, `calibration.json`, `orientation_stats.json`).
Downstream motion and incident detection experiments should only use
recordings and sections that meet basic quality criteria derived from those
files when present.

### Recording-level flags (suggested)

For each recording:

- **`sync_quality`**:
  - **good**:
    - Calibration-based sync available (`synced_cal/`).
    - `correlation.offset_and_drift ≥ 0.2`.
    - `|drift_seconds_per_second| ≤ 2e-3` (≤ 2000 ppm).
  - **marginal**:
    - Correlation between 0.05 and 0.2, or drift up to 5000 ppm.
  - **poor**:
    - Correlation < 0.05 or drift > 5000 ppm.

- **`calibration_quality`**:
  - **good**:
    - `gravity_residual_m_per_s2 ≤ 0.2`.
    - `n_static_samples ≥ 500`.
  - **marginal**:
    - Residual between 0.2 and 0.5, or fewer static samples.
  - **poor**:
    - Residual > 0.5 or calibration fails.

- **`orientation_quality`**:
  - Based on `orientation_stats.json` for the `__complementary_orientation`
    variant:
    - `g_err_abs_mean ≤ 0.3` and static pitch/roll std ≤ 2° → **good**.
    - up to 5° std → **marginal**.
    - larger deviations → **poor**.

These thresholds are intentionally conservative and can be tuned as more data
are analysed. For thesis figures, focus on **good** and at most **marginal**
recordings; treat **poor** ones as diagnostic examples, not as training data.

### Section-level inclusion criteria

For each `sections/section_N/`:

- Both sensors present and non-empty.
- No long gaps or discontinuities in `timestamp`.
- If per-section sync is used, per-section `sync_info.json` should meet the
  same qualitative checks as at the recording level.
- Section contains sufficient dynamic content for the intended analysis
  (e.g. non-trivial energy in `acc_norm` for incident-like events).

Sections failing these checks are best excluded from incident / fall
modelling, but can still be used for qualitative inspection.

---

## Visual pipeline overview

The following diagram summarizes the main flow for a single recording:

```mermaid
flowchart TD
  rawSession["Raw session logs (sporsa + arduino)"] --> parseSession["parser.session"]
  parseSession --> parsedStage["parsed/"]
  parsedStage --> syncSDA["sync.sda_sync"]
  parsedStage --> syncLIDA["sync.lida_sync"]
  parsedStage --> syncCal["sync.calibration_sync"]
  parsedStage --> syncOnline["sync.online_sync"]
  syncSDA --> syncedSDA["synced/sda/"]
  syncLIDA --> syncedLida["synced/lida/"]
  syncCal --> syncedCal["synced/cal/"]
  syncOnline --> syncedOnline["synced/online/"]
  syncedSDA & syncedLida & syncedCal & syncedOnline --> selection["sync.selection"]
  selection --> syncedBest["synced/"]
  syncedBest --> splitSections["parser.split_sections"]
  splitSections --> sectionsStage["sections/section_*/"]
  sectionsStage --> optionalCal["calibrated/ (external tooling)"]
  parsedStage --> optionalOri["orientation/ (external tooling)"]
  optionalCal -.-> optionalOri
  optionalOri --> optionalFeat["features/ (external tooling)"]
  sectionsStage --> optionalFeat
  rawCalib["data/calibrations/raw/*.txt"] --> staticCal["python -m static_calibration"]
  staticCal --> calibOut["data/calibrations/*.json + plots"]
```

**In this repository:** `parser`, `sync`, `visualization`, and **`static_calibration`**
(six-pose Arduino calibration, separate from the ride pipeline) are available.
**Dotted / optional** stages require other packages or branches. The thesis
workflow typically selects the best sync (often `synced/cal/` when quality
allows), writes `synced/`, then continues with sections and—when those tools
exist—world calibration, orientation, and features.
