"""Feature extraction pipeline: sliding-window extraction for one section or recording.

Reads all required inputs from a section directory, generates sliding windows,
calls extraction.extract_window_features for each window, and writes:

- ``<section_dir>/features/features.csv``   — one row per window
- ``<section_dir>/features/features_stats.json``
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.paths import (
    iter_sections_for_recording,
    read_csv,
    section_labels_csv,
    write_csv,
)
from labels.parser import load_labels
from labels.section_transfer import transfer_labels_to_sections
from .extraction import extract_window_features

log = logging.getLogger(__name__)

_SAMPLE_RATE_HZ_DEFAULT = 100.0


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_optional_csv(path: Path) -> pd.DataFrame:
    """Load a CSV if it exists; return an empty DataFrame otherwise."""
    if not path.exists():
        log.debug("Optional CSV not found: %s", path)
        return pd.DataFrame()
    try:
        return read_csv(path)
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return pd.DataFrame()


def _load_calibration_json(path: Path) -> dict[str, Any]:
    """Load calibration.json and return the dict (empty dict on failure)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to load calibration.json at %s: %s", path, exc)
        return {}


def _slice_by_time(
    df: pd.DataFrame,
    t_start: float,
    t_end: float,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """Return rows whose ``time_col`` value is in [t_start, t_end]."""
    if df is None or df.empty or time_col not in df.columns:
        return pd.DataFrame()
    mask = (df[time_col] >= t_start) & (df[time_col] <= t_end)
    return df.loc[mask].copy()


def _load_labels_for_section(section_dir: Path) -> pd.DataFrame:
    """Load labels from the canonical section label path."""
    labels_path = section_labels_csv(section_dir)
    if not labels_path.exists():
        return pd.DataFrame()

    rows = load_labels(labels_path)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([row.to_dict() for row in rows])


# ---------------------------------------------------------------------------
# Main section-level function
# ---------------------------------------------------------------------------

def extract_features_for_section(
    section_dir: Path,
    *,
    window_s: float = 2.0,
    hop_s: float = 1.0,
    min_samples: int = 10,
    sample_rate_hz: float = _SAMPLE_RATE_HZ_DEFAULT,
    force: bool = False,
) -> pd.DataFrame:
    """Extract sliding-window features for one section.

    Parameters
    ----------
    section_dir:
        Path to the section directory (e.g. ``data/sections/2026-02-26_r1s1``).
    window_s:
        Window length in seconds.
    hop_s:
        Hop (stride) length in seconds.
    min_samples:
        Minimum number of sporsa samples required in a window; windows below
        this threshold are skipped.
    sample_rate_hz:
        Nominal sampling rate used for spectral feature computation.
    force:
        If True, overwrite existing outputs.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame (one row per accepted window).  Also written to
        ``<section_dir>/features/features.csv``.
    """
    features_dir = section_dir / "features"
    features_csv = features_dir / "features.csv"
    features_stats_json = features_dir / "features_stats.json"

    if features_csv.exists() and not force:
        log.info(
            "Features already exist for %s — skipping (use force=True to overwrite)",
            section_dir.name,
        )
        return read_csv(features_csv)

    section_id = section_dir.name

    # ------------------------------------------------------------------
    # Load calibrated sensor data (required).
    # ------------------------------------------------------------------
    cal_dir = section_dir / "calibrated"
    sporsa_cal_path = cal_dir / "sporsa.csv"
    arduino_cal_path = cal_dir / "arduino.csv"

    if not sporsa_cal_path.exists():
        log.warning("Calibrated sporsa.csv not found in %s — cannot extract features.", section_id)
        return pd.DataFrame()

    try:
        sporsa_df = read_csv(sporsa_cal_path)
    except Exception as exc:
        log.error("Failed to load sporsa.csv for %s: %s", section_id, exc)
        return pd.DataFrame()

    try:
        arduino_df = read_csv(arduino_cal_path) if arduino_cal_path.exists() else pd.DataFrame()
    except Exception as exc:
        log.warning("Failed to load arduino.csv for %s: %s", section_id, exc)
        arduino_df = pd.DataFrame()

    # Ensure timestamps are numeric.
    sporsa_df["timestamp"] = pd.to_numeric(sporsa_df["timestamp"], errors="coerce")
    sporsa_df = sporsa_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if not arduino_df.empty and "timestamp" in arduino_df.columns:
        arduino_df["timestamp"] = pd.to_numeric(arduino_df["timestamp"], errors="coerce")
        arduino_df = arduino_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if sporsa_df.empty:
        log.warning("Sporsa calibrated data is empty for %s.", section_id)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Load derived signals (optional).
    # ------------------------------------------------------------------
    derived_dir = section_dir / "derived"
    sporsa_signals_df = _load_optional_csv(derived_dir / "sporsa_signals.csv")
    arduino_signals_df = _load_optional_csv(derived_dir / "arduino_signals.csv")
    cross_df = _load_optional_csv(derived_dir / "cross_sensor_signals.csv")

    for df_ref in (sporsa_signals_df, arduino_signals_df, cross_df):
        if not df_ref.empty and "timestamp" in df_ref.columns:
            df_ref["timestamp"] = pd.to_numeric(df_ref["timestamp"], errors="coerce")

    # ------------------------------------------------------------------
    # Load orientation (optional, selected method with Madgwick fallback).
    # ------------------------------------------------------------------
    orient_dir = section_dir / "orientation"
    orient_sporsa_df = _load_optional_csv(orient_dir / "sporsa.csv")
    orient_arduino_df = _load_optional_csv(orient_dir / "arduino.csv")

    for df_ref in (orient_sporsa_df, orient_arduino_df):
        if not df_ref.empty and "timestamp" in df_ref.columns:
            df_ref["timestamp"] = pd.to_numeric(df_ref["timestamp"], errors="coerce")

    # ------------------------------------------------------------------
    # Load events (optional).
    # ------------------------------------------------------------------
    events_df = _load_optional_csv(section_dir / "events" / "event_candidates.csv")
    if not events_df.empty:
        for col in ("start_ms", "end_ms"):
            if col in events_df.columns:
                events_df[col] = pd.to_numeric(events_df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Load labels (optional).
    # ------------------------------------------------------------------
    labels_df = _load_labels_for_section(section_dir)
    if not labels_df.empty:
        for col in ("start_ms", "end_ms"):
            if col in labels_df.columns:
                labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Load calibration quality metadata.
    # ------------------------------------------------------------------
    cal_json = _load_calibration_json(cal_dir / "calibration.json")

    # Schema v2: quality nested under "quality.overall"
    calibration_quality = str(cal_json.get("quality", {}).get("overall", "good"))

    align = cal_json.get("alignment", {})
    align_sporsa = align.get("sporsa", {})
    align_arduino = align.get("arduino", {})

    # yaw_confidence: how reliably the sensor's heading can be determined
    # (0 = completely unreliable, 1 = perfectly determined).  Both sensors'
    # values are extracted independently; the quality scorer uses min() to
    # bound the cross-sensor feature quality by the weaker sensor.
    yaw_conf_sporsa = float(align_sporsa.get("yaw_confidence", 1.0))
    yaw_conf_arduino = float(align_arduino.get("yaw_confidence", 1.0))

    # sync_confidence kept for backward compatibility with existing CSV columns.
    # It was previously read as sporsa yaw_confidence from a different path;
    # the explicit extraction above replaces the old fragile fallback chain.
    sync_confidence = yaw_conf_sporsa

    # ------------------------------------------------------------------
    # Build sliding windows.
    # ------------------------------------------------------------------
    t_min = float(sporsa_df["timestamp"].iloc[0])
    t_max = float(sporsa_df["timestamp"].iloc[-1])

    window_ms = window_s * 1000.0
    hop_ms = hop_s * 1000.0

    window_starts = np.arange(t_min, t_max - window_ms + 1e-6, hop_ms)

    if len(window_starts) == 0:
        log.warning(
            "Section %s is too short for even one window (duration=%.1f s, window=%.1f s).",
            section_id,
            (t_max - t_min) / 1000.0,
            window_s,
        )
        return pd.DataFrame()

    log.info(
        "Extracting features for %s: %.1f s recording, %d windows "
        "(window=%.1f s, hop=%.1f s)",
        section_id,
        (t_max - t_min) / 1000.0,
        len(window_starts),
        window_s,
        hop_s,
    )

    rows: list[dict] = []
    skipped_min_samples = 0
    skipped_all_nan = 0

    for w_idx, t_start in enumerate(window_starts):
        t_end = t_start + window_ms

        # Slice all DataFrames.
        w_sporsa = _slice_by_time(sporsa_df, t_start, t_end)
        w_arduino = _slice_by_time(arduino_df, t_start, t_end)
        w_sporsa_sig = _slice_by_time(sporsa_signals_df, t_start, t_end)
        w_arduino_sig = _slice_by_time(arduino_signals_df, t_start, t_end)
        w_cross = _slice_by_time(cross_df, t_start, t_end)
        w_orient_sporsa = _slice_by_time(orient_sporsa_df, t_start, t_end) if not orient_sporsa_df.empty else None
        w_orient_arduino = _slice_by_time(orient_arduino_df, t_start, t_end) if not orient_arduino_df.empty else None

        # Skip conditions.
        if len(w_sporsa) < min_samples:
            skipped_min_samples += 1
            continue

        if not w_sporsa_sig.empty and "acc_norm" in w_sporsa_sig.columns:
            acc_norm_arr = w_sporsa_sig["acc_norm"].to_numpy(dtype=float)
            if len(acc_norm_arr) > 0 and np.all(np.isnan(acc_norm_arr)):
                skipped_all_nan += 1
                continue

        try:
            feat_dict = extract_window_features(
                window_sporsa=w_sporsa,
                window_arduino=w_arduino,
                window_sporsa_signals=w_sporsa_sig,
                window_arduino_signals=w_arduino_sig,
                window_cross=w_cross,
                window_orientation_sporsa=w_orient_sporsa,
                window_orientation_arduino=w_orient_arduino,
                section_id=section_id,
                window_start_ms=float(t_start),
                window_end_ms=float(t_end),
                window_idx=w_idx,
                sample_rate_hz=sample_rate_hz,
                calibration_quality=calibration_quality,
                sync_confidence=sync_confidence,
                yaw_conf_sporsa=yaw_conf_sporsa,
                yaw_conf_arduino=yaw_conf_arduino,
                events_df=events_df if not events_df.empty else None,
                labels_df=labels_df if not labels_df.empty else None,
            )
            rows.append(feat_dict)
        except Exception as exc:
            log.error(
                "Feature extraction failed for %s window %d [%.1f–%.1f ms]: %s",
                section_id,
                w_idx,
                t_start,
                t_end,
                exc,
            )

    if skipped_min_samples > 0:
        log.debug(
            "%s: skipped %d windows with fewer than %d sporsa samples.",
            section_id,
            skipped_min_samples,
            min_samples,
        )
    if skipped_all_nan > 0:
        log.debug(
            "%s: skipped %d windows with all-NaN acc_norm.",
            section_id,
            skipped_all_nan,
        )

    if not rows:
        log.warning("No valid windows extracted for %s.", section_id)
        return pd.DataFrame()

    features_dir.mkdir(parents=True, exist_ok=True)
    features_df = pd.DataFrame(rows)

    write_csv(features_df, features_csv)
    log.info("Wrote %d windows to %s", len(features_df), features_csv)

    # ------------------------------------------------------------------
    # Write stats JSON.
    # ------------------------------------------------------------------
    stats: dict[str, Any] = {
        "section_id": section_id,
        "n_windows": len(features_df),
        "n_windows_skipped_min_samples": skipped_min_samples,
        "n_windows_skipped_all_nan": skipped_all_nan,
        "window_s": window_s,
        "hop_s": hop_s,
        "min_samples": min_samples,
        "sample_rate_hz": sample_rate_hz,
        "recording_duration_s": round((t_max - t_min) / 1000.0, 3),
        "calibration_quality": calibration_quality,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }

    if not features_df.empty:
        quality_counts = features_df["quality_tier"].value_counts().to_dict() if "quality_tier" in features_df.columns else {}
        stats["quality_tier_counts"] = quality_counts

        scenario_counts = features_df["scenario_label"].value_counts().to_dict() if "scenario_label" in features_df.columns else {}
        stats["scenario_label_counts"] = scenario_counts

        if "overall_quality_score" in features_df.columns:
            stats["mean_quality_score"] = round(float(features_df["overall_quality_score"].mean()), 4)

    features_stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Wrote features_stats.json → %s", features_stats_json)

    return features_df


# ---------------------------------------------------------------------------
# Convenience aliases and recording-level function
# ---------------------------------------------------------------------------

def process_section_features(
    section_dir: Path,
    **kwargs,
) -> pd.DataFrame:
    """Alias for :func:`extract_features_for_section`."""
    return extract_features_for_section(section_dir, **kwargs)


def process_recording_features(
    recording_name: str,
    **kwargs,
) -> pd.DataFrame:
    """Extract features for all sections of a recording and return combined DataFrame.

    Parameters
    ----------
    recording_name:
        Recording name (e.g. ``2026-02-26_r1``).
    **kwargs:
        Forwarded to :func:`extract_features_for_section`.

    Returns
    -------
    pd.DataFrame
        Concatenation of all section feature DataFrames (empty if none).
    """
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'.", recording_name)
        return pd.DataFrame()

    transfer_labels_to_sections(recording_name)

    all_dfs: list[pd.DataFrame] = []
    for sec_dir in section_dirs:
        log.info("Extracting features for section %s ...", sec_dir.name)
        try:
            df = extract_features_for_section(sec_dir, **kwargs)
            if not df.empty:
                all_dfs.append(df)
        except Exception as exc:
            log.error("Failed to extract features for %s: %s", sec_dir.name, exc)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(
        "Recording '%s': %d total windows across %d section(s).",
        recording_name,
        len(combined),
        len(all_dfs),
    )
    return combined
