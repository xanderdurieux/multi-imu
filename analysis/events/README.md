# `events/` — Event candidate extraction over derived dual-IMU signals

This stage adds an **interpretable event layer** on top of the existing windowed pipeline.
Instead of only sliding windows, it detects physically meaningful candidate events and writes
an event table that can drive event-centered feature windows.

## Target event types
- `bump_shock_candidate`
- `braking_burst`
- `swerve_roll_rate_candidate`
- `rider_bicycle_divergence`
- `fall_or_bicycle_drop_candidate` (optional, conservative)

## Why these detectors make physical sense
- **Bump/shock**: road impacts produce sharp vertical acceleration changes and often stronger bike→rider transmission gain.
- **Braking burst**: hard deceleration appears as negative longitudinal acceleration, usually coupled with pitch-rate change.
- **Swerve / high roll-rate**: rapid steering corrections raise roll-rate and often lateral acceleration.
- **Rider–bicycle divergence**: instability or active body movement increases rider-minus-bike residual signals.
- **Fall/drop candidate**: a combined pattern of high roll-rate, strong vertical drop, and high residual mismatch is consistent with loss-of-balance scenarios.

## Inputs
Per section:
- `derived/sporsa_signals.csv`
- `derived/arduino_signals.csv`
- `derived/cross_sensor_signals.csv`

## Outputs
Per section in `events/`:
- `event_candidates.csv` (machine-readable event table)
- `event_config.json` (effective thresholds/config)
- `event_summary.json` (counts + detector notes)
- `plots/event_*.png` (diagnostic aligned bike/rider plots)

Each event row includes:
- `timestamp`
- `event_type`
- `confidence`
- `key_trigger_signals`
- `event_window_start_idx`, `event_window_end_idx`
- `section`, `recording_id`, `section_id`
- `ambiguous_flag`, `failure_flags`

## De-duplication and ambiguity handling
- Per-type non-max suppression with configurable minimum temporal separation.
- Ambiguous/weak patterns are retained with `ambiguous_flag=1` and explanatory `failure_flags`.

## Configuration
`EventConfig` in `extract.py` is threshold-driven and JSON-overridable from CLI (`--config`).

## CLI
From `analysis/`:

```bash
# One section
uv run python -m events.extract data/sections/2026-02-26_r2s1

# All sections in one recording
uv run python -m events.extract 2026-02-26_r2 --all-sections

# All sections in all recordings of a session
uv run python -m events.extract 2026-02-26 --all

# Override thresholds
uv run python -m events.extract 2026-02-26_r2 --all-sections --config analysis/events_config.json
```

## Using event-centered windows in the feature extractor
```bash
uv run python -m features.extract 2026-02-26_r2 --all-sections \
  --event-centered --min-event-confidence 0.4
```

## Minimal manual evaluation plan
1. Run event extraction on all sections for a recording/session.
2. Sort `event_candidates.csv` by confidence and inspect top events per type.
3. Review matching diagnostic plots for signal alignment around event timestamp.
4. Spot-check low-confidence and `ambiguous_flag=1` rows to tune thresholds.
5. Track per-type false positives/false negatives in a small reviewer sheet and revise `event_config.json`.
