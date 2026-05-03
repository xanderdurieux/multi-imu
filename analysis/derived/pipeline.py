"""Derived pipeline: compute physical signals from calibrated IMU data for one section or recording."""

from __future__ import annotations

import logging
from pathlib import Path

from common.paths import iter_sections_for_recording, read_csv, sections_root, write_csv
from .signals import compute_sensor_signals, compute_cross_sensor_signals

log = logging.getLogger(__name__)

_SENSORS = ("sporsa", "arduino")


def process_section_derived(
    section_dir: Path,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
) -> bool:
    """Process section derived."""
    cal_dir = section_dir / "calibrated"
    if not cal_dir.exists():
        log.warning("Calibrated directory not found for %s — skipping.", section_dir.name)
        return False

    derived_dir = section_dir / "derived"

    # Check if outputs already exist and force is not set.
    expected_outputs = [
        derived_dir / "sporsa_signals.csv",
        derived_dir / "arduino_signals.csv",
        derived_dir / "cross_sensor_signals.csv",
    ]
    if not force and all(p.exists() for p in expected_outputs):
        log.info("Derived signals already exist for %s — skipping (use force=True to overwrite).", section_dir.name)
        return True

    # Load calibrated sensor data.
    sensor_dfs: dict[str, object] = {}
    for sensor in _SENSORS:
        csv_path = cal_dir / f"{sensor}.csv"
        if not csv_path.exists():
            log.warning("Calibrated %s CSV not found in %s — skipping section.", sensor, section_dir.name)
            return False
        try:
            sensor_dfs[sensor] = read_csv(csv_path)
        except Exception as exc:
            log.error("Failed to load %s for %s: %s", csv_path, section_dir.name, exc)
            return False

    derived_dir.mkdir(parents=True, exist_ok=True)

    # Load orientation CSVs (optional — produced by the orientation stage).
    orient_dir = section_dir / "orientation"
    orient_dfs: dict[str, object] = {}
    for sensor in _SENSORS:
        orient_csv = orient_dir / f"{sensor}.csv"
        if orient_csv.exists():
            try:
                orient_dfs[sensor] = read_csv(orient_csv)
            except Exception as exc:
                log.warning("Could not load orientation CSV for %s/%s: %s", section_dir.name, sensor, exc)

    # Compute and write per-sensor signals.
    sensor_signal_dfs: dict[str, object] = {}
    for sensor in _SENSORS:
        try:
            sig_df = compute_sensor_signals(
                sensor_dfs[sensor],
                sample_rate_hz=sample_rate_hz,
                df_orient=orient_dfs.get(sensor),
            )
        except Exception as exc:
            log.error("Failed to compute derived signals for %s/%s: %s", section_dir.name, sensor, exc)
            return False

        out_path = derived_dir / f"{sensor}_signals.csv"
        try:
            write_csv(sig_df, out_path)
        except Exception as exc:
            log.error("Failed to write %s: %s", out_path, exc)
            return False

        log.info("Wrote %s → %s (%d rows)", sensor, out_path, len(sig_df))
        sensor_signal_dfs[sensor] = sig_df

    # Compute and write cross-sensor signals.
    try:
        cross_df = compute_cross_sensor_signals(
            sensor_signal_dfs["sporsa"],
            sensor_signal_dfs["arduino"],
            sample_rate_hz=sample_rate_hz,
        )
    except Exception as exc:
        log.error("Failed to compute cross-sensor signals for %s: %s", section_dir.name, exc)
        return False

    cross_path = derived_dir / "cross_sensor_signals.csv"
    try:
        write_csv(cross_df, cross_path)
    except Exception as exc:
        log.error("Failed to write %s: %s", cross_path, exc)
        return False

    log.info("Wrote cross_sensor_signals → %s (%d rows)", cross_path, len(cross_df))
    return True


def process_recording_derived(
    recording_name: str,
    *,
    sample_rate_hz: float = 100.0,
    force: bool = False,
) -> list[bool]:
    """Process recording derived."""
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'.", recording_name)
        return []

    results: list[bool] = []
    for sec_dir in section_dirs:
        log.info("Processing derived signals for section %s ...", sec_dir.name)
        ok = process_section_derived(sec_dir, sample_rate_hz=sample_rate_hz, force=force)
        results.append(ok)

    n_ok = sum(results)
    log.info("Derived signals: %d/%d sections succeeded for recording '%s'.", n_ok, len(results), recording_name)
    return results
