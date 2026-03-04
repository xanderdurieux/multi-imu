"""Detect calibration sequences and split IMU recordings into sections.

A calibration sequence has the pattern:
  ~5 s static  →  ~5 acceleration peaks  →  ~5 s static

This module finds those sequences in the reference sensor, then splits all
configured sensors into matching time windows saved under::

    data/recordings/<recording_name>/sections/
        section_1/sporsa.csv
        section_1/arduino.csv
        section_2/sporsa.csv
        …

CLI usage::

    python -m parser.split_sections 2026-02-26_5/synced_lida
    python -m parser.split_sections 2026-02-26_5/synced_cal --no-plot
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from common.csv_schema import load_dataframe, write_dataframe
from common.paths import find_sensor_csv, recording_dir

log = logging.getLogger(__name__)

_GRAVITY_MS2 = 9.81


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CalibrationSegment(NamedTuple):
    """Indices (inclusive) of one calibration sequence in the DataFrame."""

    start_idx: int
    end_idx: int
    peak_indices: list[int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _acc_norm(df: pd.DataFrame) -> np.ndarray:
    acc = df[["ax", "ay", "az"]].to_numpy(dtype=float)
    return np.sqrt(np.nansum(acc ** 2, axis=1))


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """Centered box-car moving average, same length as input."""
    if window <= 1:
        return signal.copy()
    return (
        pd.Series(signal)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _find_peaks(
    signal: np.ndarray,
    *,
    height: float = 0.0,
    distance: int = 1,
) -> np.ndarray:
    """Return indices of local maxima above *height* with minimum *distance*.

    A point is a local maximum if it is strictly greater than both neighbours.
    When two peaks are closer than *distance* samples the taller one wins.
    """
    x = np.asarray(signal, dtype=float)
    n = len(x)
    if n < 3:
        return np.empty(0, dtype=int)

    # Collect all local maxima above height threshold
    candidates = [
        i for i in range(1, n - 1)
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] >= height
    ]
    if not candidates:
        return np.empty(0, dtype=int)

    if distance <= 1:
        return np.array(candidates, dtype=int)

    # Enforce minimum distance: greedily keep tallest peaks first
    cands = np.array(candidates, dtype=int)
    order = np.argsort(x[cands])[::-1]
    sorted_cands = cands[order]

    keep = np.ones(len(sorted_cands), dtype=bool)
    for i in range(len(sorted_cands)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(sorted_cands)):
            if keep[j] and abs(int(sorted_cands[i]) - int(sorted_cands[j])) < distance:
                keep[j] = False

    return np.sort(sorted_cands[keep])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_calibration_segments(
    df: pd.DataFrame,
    *,
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
) -> list[CalibrationSegment]:
    """Find calibration sequences in an IMU DataFrame.

    A calibration sequence is detected as:
      1. A run of ``static_min_s`` seconds where each sample's instantaneous
         ``|acc_norm - g|`` is below ``static_threshold`` (m/s²).
      2. Followed by a cluster of at least ``peak_min_count`` peaks where the
         lightly smoothed ``|acc_norm - g|`` exceeds ``peak_min_height`` (m/s²),
         with consecutive peaks no farther apart than ``peak_max_gap_s`` seconds.
      3. Followed by another run of ``static_min_s`` seconds.

    The flanking static regions must also lie within ``static_gap_max_s``
    seconds of the first/last peak, which prevents exercise-activity clusters
    from being matched to a static run that is many minutes away.

    Parameters
    ----------
    df:
        IMU DataFrame with ``ax``, ``ay``, ``az`` columns, sorted by timestamp.
    sample_rate_hz:
        Approximate sampling rate used to convert durations to sample counts.
    static_min_s:
        Minimum duration (seconds) of the static region on each side.
    static_threshold:
        Maximum instantaneous ``|acc_norm - g|`` (m/s²) to classify a sample
        as *static*.  Applied directly to the raw unsmoothed signal so that
        peaks cannot bleed into the neighbouring static regions.
    peak_min_height:
        Minimum ``|acc_norm - g|`` (m/s²) in the lightly smoothed signal for a
        local maximum to count as a calibration peak.
    peak_min_count:
        Minimum number of peaks in a cluster to qualify as calibration.
    peak_max_gap_s:
        Maximum gap (seconds) between two consecutive peaks in the same cluster.
    static_gap_max_s:
        Maximum allowed gap (seconds) between the end of the pre-static run and
        the first peak, and between the last peak and the start of the
        post-static run.  Keeps exercise-activity clusters from being matched
        to a distant static region.
    """
    norm = _acc_norm(df)
    if len(norm) == 0:
        return []

    g = float(np.nanmedian(norm))

    # Static classification uses the raw, unsmoothed dynamic so that sharp
    # movement peaks cannot bleed into the surrounding still regions.
    # Samples with near-zero norm are sensor dropouts (e.g. arduino blank
    # packets); treat them as static so they don't fragment static runs.
    dynamic_raw = np.abs(norm - g)
    is_dropout = norm < 0.1 * g
    is_static = (dynamic_raw < static_threshold) | is_dropout

    # Peak detection uses a light centered smooth (0.1 s) to reduce sample
    # noise while keeping individual peaks clearly visible.
    smooth_win = max(3, int(sample_rate_hz * 0.1))
    dynamic_smooth = np.abs(_smooth(norm, smooth_win) - g)

    peaks = _find_peaks(
        dynamic_smooth,
        height=peak_min_height,
        distance=max(1, int(sample_rate_hz * 0.3)),
    )

    if len(peaks) < peak_min_count:
        return []

    # Group peaks into clusters (consecutive peaks ≤ peak_max_gap_s apart)
    max_gap_samples = int(sample_rate_hz * peak_max_gap_s)
    clusters: list[list[int]] = []
    current: list[int] = [int(peaks[0])]
    for p in peaks[1:]:
        if int(p) - current[-1] <= max_gap_samples:
            current.append(int(p))
        else:
            if len(current) >= peak_min_count:
                clusters.append(current)
            current = [int(p)]
    if len(current) >= peak_min_count:
        clusters.append(current)

    if not clusters:
        return []

    # Enumerate all contiguous static runs so we can match clusters to
    # the nearest qualifying region on each side, regardless of any
    # transition samples between the static period and the first/last peak.
    static_min_samples = int(sample_rate_hz * static_min_s)
    static_gap_max_samples = int(sample_rate_hz * static_gap_max_s)

    static_runs: list[tuple[int, int, int]] = []  # (start, end_inclusive, length)
    run_start: int | None = None
    for i, flag in enumerate(is_static):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            static_runs.append((run_start, i - 1, i - run_start))
            run_start = None
    if run_start is not None:
        static_runs.append((run_start, len(is_static) - 1, len(is_static) - run_start))

    segments: list[CalibrationSegment] = []

    for cluster in clusters:
        c_start = cluster[0]
        c_end = cluster[-1]

        # Nearest preceding static run that ends before the first peak
        # and whose end is within static_gap_max_s of the first peak.
        pre_run: tuple[int, int] | None = None
        for rs, re, rl in reversed(static_runs):
            if re >= c_start:
                continue
            if rl < static_min_samples:
                continue
            if (c_start - re) > static_gap_max_samples:
                break  # runs are ordered; any earlier run will be even farther
            pre_run = (rs, re)
            break

        if pre_run is None:
            log.debug(
                "Cluster at sample %d skipped: no qualifying pre-static run within %.1fs",
                c_start, static_gap_max_s,
            )
            continue

        # Nearest following static run that starts after the last peak
        # and whose start is within static_gap_max_s of the last peak.
        post_run: tuple[int, int] | None = None
        for rs, re, rl in static_runs:
            if rs <= c_end:
                continue
            if (rs - c_end) > static_gap_max_samples:
                break
            if rl < static_min_samples:
                continue
            post_run = (rs, re)
            break

        if post_run is None:
            log.debug(
                "Cluster at sample %d skipped: no qualifying post-static run within %.1fs",
                c_end, static_gap_max_s,
            )
            continue

        segments.append(CalibrationSegment(
            start_idx=pre_run[0],
            end_idx=post_run[1],
            peak_indices=cluster,
        ))

    return segments


def _plot_section(recording_name: str, section_stage: str, sensors: list[str]) -> None:
    """Generate sensor and comparison plots for one section directory."""
    from visualization import plot_comparison, plot_sensor

    stage_ref = f"{recording_name}/{section_stage}"
    for sensor_name in sensors:
        try:
            plot_sensor.main([stage_ref, sensor_name])
            plot_sensor.main([stage_ref, sensor_name, "--norm", "--acc"])
        except SystemExit:
            pass
    try:
        plot_comparison.main([stage_ref])
        plot_comparison.main([stage_ref, "--norm"])
    except SystemExit:
        pass


def split_recording(
    recording_name: str,
    stage: str = "synced_cal",
    sensors: list[str] | None = None,
    *,
    reference_sensor: str = "sporsa",
    sample_rate_hz: float = 100.0,
    static_min_s: float = 3.0,
    static_threshold: float = 1.5,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    static_gap_max_s: float = 5.0,
    plot: bool = True,
    sync: bool = True,
) -> list[Path]:
    """Split a recording into sections for all sensors, grouped per section.

    Calibration sequences are detected in ``reference_sensor``.  Each section
    spans from the start of one calibration to the end of the next, so both
    the opening and closing calibrations are included.

    Outputs are saved to::

        data/recordings/<recording_name>/sections/
            section_1/
                sporsa.csv
                arduino.csv
                sporsa.png, arduino.png, …  (if plot=True)
            section_2/
                …

    If fewer than two calibration sequences are found a warning is logged and
    the full streams are saved under ``sections/section_1/``.

    Parameters
    ----------
    sensors:
        Sensor names to split.  Defaults to ``["sporsa", "arduino"]``.
    reference_sensor:
        The sensor used to detect calibration segments.  If not already in
        ``sensors``, it is prepended automatically.
    plot:
        If ``True``, generate sensor and comparison plots for every section.
    sync:
        If ``True`` (default), run calibration-based sync on each section
        after splitting so that the per-section timestamps are precisely
        aligned.

    Returns a list of written CSV paths in chronological order.
    """
    if sensors is None:
        sensors = ["sporsa", "arduino"]
    if reference_sensor not in sensors:
        sensors = [reference_sensor] + sensors

    # Load reference sensor and detect calibration segments.
    ref_csv = find_sensor_csv(recording_name, stage, reference_sensor)
    ref_df = load_dataframe(ref_csv)
    ref_df = ref_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    calibrations = find_calibration_segments(
        ref_df,
        sample_rate_hz=sample_rate_hz,
        static_min_s=static_min_s,
        static_threshold=static_threshold,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
        static_gap_max_s=static_gap_max_s,
    )

    # Load all sensor DataFrames up-front (skip missing ones with a warning).
    sensor_dfs: dict[str, pd.DataFrame] = {}
    for sensor in sensors:
        try:
            csv_path = find_sensor_csv(recording_name, stage, sensor)
        except FileNotFoundError:
            log.warning("Sensor '%s' not found in %s/%s — skipping", sensor, recording_name, stage)
            continue
        df = load_dataframe(csv_path)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        sensor_dfs[sensor] = df

    sections_root = recording_dir(recording_name) / "sections"
    available_sensors = list(sensor_dfs)

    def _write_section(section_idx: int, sensor_slices: dict[str, pd.DataFrame]) -> list[Path]:
        section_stage = f"section_{section_idx}"
        section_dir = sections_root / section_stage
        section_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for sensor, df in sensor_slices.items():
            out_path = section_dir / f"{sensor}.csv"
            write_dataframe(df, out_path)
            paths.append(out_path)
            log.info("Wrote %s/%s (%d rows)", section_stage, out_path.name, len(df))
        if plot:
            _plot_section(recording_name, f"sections/{section_stage}", available_sensors)
        return paths

    if len(calibrations) < 2:
        log.warning(
            "Fewer than 2 calibration sequences found in %s/%s/%s "
            "(%d found) — saving full streams as section_1",
            recording_name, stage, reference_sensor, len(calibrations),
        )
        return _write_section(1, sensor_dfs)

    written: list[Path] = []
    for i, (c_open, c_close) in enumerate(zip(calibrations, calibrations[1:]), start=1):
        ts_start = float(ref_df.iloc[c_open.start_idx]["timestamp"])
        ts_end = float(ref_df.iloc[c_close.end_idx]["timestamp"])

        slices: dict[str, pd.DataFrame] = {}
        for sensor, df in sensor_dfs.items():
            mask = (df["timestamp"] >= ts_start) & (df["timestamp"] <= ts_end)
            slices[sensor] = df.loc[mask].reset_index(drop=True)

        written.extend(_write_section(i, slices))

    if sync and reference_sensor in sensor_dfs:
        target_sensors = [s for s in available_sensors if s != reference_sensor]
        for tgt in target_sensors:
            sync_sections(
                recording_name,
                reference_sensor=reference_sensor,
                target_sensor=tgt,
                sample_rate_hz=sample_rate_hz,
                plot=plot,
            )

    return written


def sync_sections(
    recording_name: str,
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = 100.0,
    coarse_max_lag_s: float = 10.0,
    cal_search_s: float = 3.0,
    plot: bool = True,
) -> None:
    """Run calibration-based sync on each section of a recording, in-place.

    Each section must contain ≥ 2 calibration sequences (opening + closing).
    The target sensor's timestamps are corrected and its CSV is replaced
    in-place.  A ``sync_info.json`` is also written to each section directory.

    Parameters
    ----------
    recording_name:
        Name of the recording (e.g. ``"2026-02-26_5"``).
    reference_sensor:
        Sensor used as the time reference.
    target_sensor:
        Sensor whose timestamps are corrected.
    sample_rate_hz:
        Sampling rate for peak detection and cross-correlation.
    coarse_max_lag_s:
        Max lag (seconds) for the coarse SDA search.  Sections are already
        roughly synced, so a small value (e.g. 10 s) is sufficient.
    cal_search_s:
        Search window (±s) for each calibration window alignment.
    plot:
        If ``True``, regenerate sensor and comparison plots after syncing.
    """
    from sync.calibration_sync import synchronize_from_calibration

    sections_root = recording_dir(recording_name) / "sections"
    if not sections_root.exists():
        log.warning("No sections directory found for %s", recording_name)
        return

    section_dirs = sorted(
        (d for d in sections_root.iterdir() if d.is_dir() and d.name.startswith("section_")),
        key=lambda p: p.name,
    )

    for section_dir in section_dirs:
        ref_csv = section_dir / f"{reference_sensor}.csv"
        tgt_csv = section_dir / f"{target_sensor}.csv"

        if not ref_csv.exists() or not tgt_csv.exists():
            log.warning(
                "Skipping %s/%s: missing %s or %s",
                recording_name, section_dir.name, ref_csv.name, tgt_csv.name,
            )
            continue

        log.info("Syncing %s/%s ...", recording_name, section_dir.name)
        tmp_dir = section_dir / "_tmp_sync"
        try:
            sync_json_raw, synced_csv_raw, _ = synchronize_from_calibration(
                reference_csv=ref_csv,
                target_csv=tgt_csv,
                output_dir=tmp_dir,
                sample_rate_hz=sample_rate_hz,
                coarse_max_lag_s=coarse_max_lag_s,
                cal_search_s=cal_search_s,
            )

            shutil.move(str(synced_csv_raw), tgt_csv)
            shutil.move(str(sync_json_raw), section_dir / "sync_info.json")
            log.info("  → wrote %s/sync_info.json", section_dir.name)

        except Exception as exc:
            log.error(
                "Failed to sync %s/%s: %s",
                recording_name, section_dir.name, exc,
            )
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

        if plot:
            _plot_section(
                recording_name,
                f"sections/{section_dir.name}",
                [reference_sensor, target_sensor],
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m parser.split_sections",
        description=(
            "Detect calibration sequences in a recording and split it into "
            "sections.  Each section spans from the start of one calibration "
            "to the end of the next and is saved under "
            "data/recordings/<recording_name>/sections/section_N/."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        help=(
            "Recording name and input stage as '<recording_name>/<stage>' "
            "(e.g. '2026-02-26_5/synced_cal')."
        ),
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=["sporsa", "arduino"],
        metavar="SENSOR",
        help="Sensor names to split (default: sporsa arduino).",
    )
    parser.add_argument(
        "--reference-sensor",
        default="sporsa",
        help="Sensor used to detect calibration sequences (default: sporsa).",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=100.0,
        help="Approximate sampling rate in Hz (default: 100).",
    )
    parser.add_argument(
        "--static-min-s",
        type=float,
        default=3.0,
        help="Minimum duration of flanking static regions in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--static-threshold",
        type=float,
        default=1.5,
        help="Max |acc_norm - g| (m/s²) to classify a sample as static (default: 1.5).",
    )
    parser.add_argument(
        "--peak-min-height",
        type=float,
        default=3.0,
        help="Min |acc_norm - g| (m/s²) for a local maximum to count as a peak (default: 3.0).",
    )
    parser.add_argument(
        "--peak-min-count",
        type=int,
        default=3,
        help="Minimum number of peaks in a cluster to qualify as calibration (default: 3).",
    )
    parser.add_argument(
        "--peak-max-gap-s",
        type=float,
        default=3.0,
        help="Max gap in seconds between consecutive peaks in the same cluster (default: 3.0).",
    )
    parser.add_argument(
        "--static-gap-max-s",
        type=float,
        default=5.0,
        help=(
            "Max gap in seconds between the flanking static run and the "
            "first/last peak (default: 5.0)."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots for each section.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip running the calibration sync on each section after splitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)

    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("recording_name_stage must be '<recording_name>/<stage>'")
    recording_name, stage = parts

    written = split_recording(
        recording_name=recording_name,
        stage=stage,
        sensors=args.sensors,
        reference_sensor=args.reference_sensor,
        sample_rate_hz=args.sample_rate_hz,
        static_min_s=args.static_min_s,
        static_threshold=args.static_threshold,
        peak_min_height=args.peak_min_height,
        peak_min_count=args.peak_min_count,
        peak_max_gap_s=args.peak_max_gap_s,
        static_gap_max_s=args.static_gap_max_s,
        plot=not args.no_plot,
        sync=not args.no_sync,
    )

    print(f"\n{len(written)} file(s) written:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
