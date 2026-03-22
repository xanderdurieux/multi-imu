"""Online (single-anchor) synchronisation for live recording.

This module implements a causal synchronisation strategy suitable for live
data collection, where the closing calibration sequence has not yet occurred
and the full recording is unavailable.

Strategy
--------
1. At session start, the user performs the opening calibration tap-burst.
2. :func:`estimate_sync_from_opening_anchor` detects the opening calibration
   in the reference stream and the corresponding cluster in the target stream,
   yielding a precise initial offset.
3. Clock drift is **not** measurable in real time (requires two anchors separated
   in time), so a *pre-characterised* drift rate is applied instead.  This rate
   is derived from offline analysis of historical recordings
   (:func:`load_characterised_drift`).
4. The result is a :class:`~sync.drift_estimator.SyncModel` that can be used
   to correct the target stream's timestamps as new samples arrive.

Comparison with offline methods
---------------------------------
+---------------------------+---------------------+-----------------------------+
| Property                  | SDA + LIDA          | Online (this module)        |
+===========================+=====================+=============================+
| Requires full recording   | Yes                 | No (causal)                 |
+---------------------------+---------------------+-----------------------------+
| Calibration tap needed    | No                  | Yes (opening only)          |
+---------------------------+---------------------+-----------------------------+
| Drift estimation          | From full signal    | From historical mean        |
+---------------------------+---------------------+-----------------------------+
| Latency                   | Post-hoc            | Near-zero after cal event   |
+---------------------------+---------------------+-----------------------------+

CLI::

    python -m sync.online_sync <recording_name>/<stage>
    python -m sync.online_sync 2026-02-26_5/parsed --drift-ppm 400
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import shutil

import pandas as pd

from common import write_dataframe
from common.paths import find_sensor_csv, recording_stage_dir

from .calibration_sync import _coarse_offset_from_opening_calibration, _refine_offset_at_calibration
from .common import load_stream, remove_dropouts
from .drift_estimator import SyncModel, apply_sync_model, save_sync_model
from .metrics import compute_sync_correlations
from common.calibration_segments import find_calibration_segments

log = logging.getLogger(__name__)

# Default drift rate used when no historical data is available (ppm).
# Derived from the cal-sync estimates across all 2026-02-26 recordings with
# calibration_span_s >= 60 s.  Value is set conservatively to the median
# (see drift_characterisation.json after running analyse_drift.py).
DEFAULT_DRIFT_PPM = 400.0

# Path relative to the analysis directory where drift characterisation lives.
DRIFT_CHAR_JSON = Path(__file__).parent.parent / "data" / "drift_characterisation.json"


def load_characterised_drift(json_path: Path = DRIFT_CHAR_JSON) -> float:
    """Load the median drift rate (ppm) from the offline drift characterisation.

    Falls back to :data:`DEFAULT_DRIFT_PPM` if the file is absent or invalid.
    """
    if not json_path.exists():
        log.info(
            "Drift characterisation file not found (%s). "
            "Using default %.0f ppm.",
            json_path, DEFAULT_DRIFT_PPM,
        )
        return DEFAULT_DRIFT_PPM
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # Prefer cal-sync median (span >= 60 s); fall back to LIDA median.
        for key in ("cal_span_ge_60s", "lida"):
            block = data.get(key)
            if block and block.get("median_ppm") is not None:
                ppm = float(block["median_ppm"])
                log.info("Loaded characterised drift: %.1f ppm (from '%s')", ppm, key)
                return ppm
    except Exception as exc:
        log.warning("Could not load drift characterisation: %s. Using default.", exc)
    return DEFAULT_DRIFT_PPM


def estimate_sync_from_opening_anchor(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    drift_ppm: float | None = None,
    sample_rate_hz: float = 100.0,
    peak_min_height: float = 3.0,
    peak_min_count: int = 3,
    peak_max_gap_s: float = 3.0,
    cal_search_s: float = 5.0,
    peak_buffer_s: float = 1.0,
    reference_name: str = "",
    target_name: str = "",
) -> SyncModel:
    """Estimate a sync model using only the opening calibration anchor.

    This is the online-safe version of :func:`sync.calibration_sync.estimate_sync_from_calibration`.
    It requires only the opening calibration tap to have occurred, making it
    suitable for real-time applications where the closing calibration has not
    yet been performed.

    The clock drift is supplied externally (``drift_ppm``).  If ``None``, the
    pre-characterised median drift is loaded from
    ``data/drift_characterisation.json`` (requires running ``analyse_drift.py``
    first).

    Parameters
    ----------
    ref_df:
        Reference sensor DataFrame (Sporsa, stable Unix-epoch clock).
    tgt_df:
        Target sensor DataFrame (Arduino, ``millis()`` from boot).
    drift_ppm:
        Clock drift to apply in parts-per-million.  Positive = Arduino runs
        fast (its ``millis()`` advances faster than real time).  If ``None``,
        loaded from the offline characterisation file.
    sample_rate_hz:
        Resampling rate for the refinement cross-correlation.
    peak_min_height, peak_min_count, peak_max_gap_s:
        Calibration-sequence detection parameters (passed to
        :func:`parser.split_sections.find_calibration_segments`).
    cal_search_s:
        Search window (±s) for each calibration cross-correlation.
    peak_buffer_s:
        Buffer added around the calibration peaks in the reference window.
    reference_name, target_name:
        Optional labels stored in the returned ``SyncModel``.

    Returns
    -------
    SyncModel
        A linear sync model with the refined opening offset and the
        pre-characterised drift.

    Raises
    ------
    ValueError
        If no calibration segment is found in the reference stream, or if
        the opening calibration cannot be located in the target stream.
    """
    if drift_ppm is None:
        drift_ppm = load_characterised_drift()
    drift_s_per_s = drift_ppm * 1e-6

    # Detect opening calibration in the reference sensor.
    ref_cals = find_calibration_segments(
        ref_df,
        sample_rate_hz=sample_rate_hz,
        peak_min_height=peak_min_height,
        peak_min_count=peak_min_count,
        peak_max_gap_s=peak_max_gap_s,
    )
    if len(ref_cals) == 0:
        raise ValueError(
            "No calibration sequence found in reference sensor. "
            "Ensure the opening calibration tap has been recorded."
        )

    opening_cal = ref_cals[0]
    log.info(
        "Opening calibration detected in reference at ~%.1f s "
        "(peaks: %d)",
        float(ref_df.iloc[opening_cal.peak_indices[0]]["timestamp"]) / 1000.0,
        len(opening_cal.peak_indices),
    )

    tgt_clean = remove_dropouts(tgt_df)
    try:
        coarse_offset_s = _coarse_offset_from_opening_calibration(
            ref_df,
            tgt_clean,
            opening_cal,
            peak_min_height=peak_min_height,
            peak_min_count=peak_min_count,
        )
    except ValueError:
        log.warning(
            "Opening-calibration coarse offset failed; falling back to SDA at 5 Hz."
        )
        from .align_df import estimate_offset as _sda
        coarse = _sda(
            ref_df, tgt_clean,
            sample_rate_hz=5.0,
            max_lag_seconds=120.0,
            use_acc=True,
            use_gyro=False,
            differentiate=False,
        )
        coarse_offset_s = float(coarse.offset_seconds)

    # Fine offset: cross-correlate the calibration peak window.
    cal_result = _refine_offset_at_calibration(
        ref_df, tgt_df, opening_cal,
        coarse_offset_s=coarse_offset_s,
        sample_rate_hz=sample_rate_hz,
        peak_buffer_s=peak_buffer_s,
        search_s=cal_search_s,
    )

    log.info(
        "Opening anchor: offset=%.6f s (score=%.3f), drift=%.0f ppm (pre-characterised)",
        cal_result.offset_seconds,
        cal_result.correlation_score,
        drift_ppm,
    )

    tgt_origin_s = float(tgt_df["timestamp"].iloc[0]) / 1000.0
    offset_at_origin_s = (
        cal_result.offset_seconds
        - drift_s_per_s * (cal_result.t_tgt_seconds - tgt_origin_s)
    )

    return SyncModel(
        reference_csv=reference_name,
        target_csv=target_name,
        target_time_origin_seconds=tgt_origin_s,
        offset_seconds=offset_at_origin_s,
        drift_seconds_per_second=drift_s_per_s,
        sample_rate_hz=float(sample_rate_hz),
        max_lag_seconds=float(cal_search_s + 1.0),
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def synchronize_recording_online(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    drift_ppm: float | None = None,
    sample_rate_hz: float = 100.0,
    plot: bool = False,
) -> tuple[Path, Path, Path]:
    """Synchronise a recording using only the opening calibration anchor.

    Writes outputs to ``synced/online/`` alongside the other sync stages.

    Returns ``(reference_csv, synced_target_csv, sync_info_json)``.
    """
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)
    out_dir = recording_stage_dir(recording_name, "synced/online")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{recording_name}/synced/online] {reference_sensor} (ref) <- {ref_csv.name}")
    print(f"[{recording_name}/synced/online] {target_sensor} (target) <- {tgt_csv.name}")

    ref_df = load_stream(ref_csv)
    tgt_df = load_stream(tgt_csv)

    if drift_ppm is None:
        drift_ppm = load_characterised_drift()

    model = estimate_sync_from_opening_anchor(
        ref_df, tgt_df,
        drift_ppm=drift_ppm,
        sample_rate_hz=sample_rate_hz,
        reference_name=str(ref_csv),
        target_name=str(tgt_csv),
    )

    sync_json_path = out_dir / "sync_info.json"
    save_sync_model(model, sync_json_path)

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

    correlations = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)

    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "online_opening_anchor"
    sync_data["drift_ppm_source"] = "pre_characterised"
    sync_data["drift_ppm_applied"] = drift_ppm
    sync_data["correlation"] = correlations
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in aligned_df.columns
    ]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    ref_out = out_dir / f"{reference_sensor}.csv"
    tgt_out = out_dir / f"{target_sensor}.csv"
    shutil.copy2(ref_csv, ref_out)
    write_dataframe(aligned_df, tgt_out)

    print(f"[{recording_name}/synced/online] {ref_out.name}")
    print(f"[{recording_name}/synced/online] {tgt_out.name}")
    print(f"[{recording_name}/synced/online] {sync_json_path.name}")

    if plot:
        from visualization import plot_comparison
        stage_ref = f"{recording_name}/synced/online"
        try:
            plot_comparison.main([stage_ref])
            plot_comparison.main([stage_ref, "--norm"])
        except SystemExit:
            pass

    return ref_out, tgt_out, sync_json_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.online_sync",
        description=(
            "Online (single-anchor) sync using only the opening calibration tap. "
            "Applies a pre-characterised drift rate rather than estimating drift "
            "from start and end calibrations. Writes to synced/online/."
        ),
    )
    parser.add_argument(
        "recording_name_stage",
        help="'<recording_name>/<stage>' e.g. '2026-02-26_5/parsed'.",
    )
    parser.add_argument(
        "--drift-ppm", type=float, default=None,
        help="Drift rate to apply (ppm). If omitted, loads from drift_characterisation.json.",
    )
    parser.add_argument(
        "--reference-sensor", default="sporsa",
    )
    parser.add_argument(
        "--target-sensor", default="arduino",
    )
    parser.add_argument(
        "--sample-rate-hz", type=float, default=100.0,
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate visualisation plots for synced/online stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    parts = args.recording_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("recording_name_stage must be '<recording_name>/<stage>'")
    recording_name, stage_in = parts

    ref_out, tgt_out, sync_json = synchronize_recording_online(
        recording_name=recording_name,
        stage_in=stage_in,
        reference_sensor=args.reference_sensor,
        target_sensor=args.target_sensor,
        drift_ppm=args.drift_ppm,
        sample_rate_hz=args.sample_rate_hz,
        plot=args.plot,
    )
    print(f"\nreference: {ref_out}")
    print(f"synced:    {tgt_out}")
    print(f"model:     {sync_json}")


if __name__ == "__main__":
    main()
