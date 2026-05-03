"""Pipeline helpers for extract labelled sliding-window features from section signals."""

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
    parse_section_folder_name,
    read_csv,
    recording_stage_dir,
    section_labels_csv,
    write_csv,
)
from labels.parser import load_labels
from labels.section_transfer import transfer_labels_to_sections
from .extraction import extract_window_features
from .label_config import default_label_config, load_label_config
from .lag_features import add_lag_features

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


def _load_sync_metadata(section_id: str) -> dict[str, Any]:
    """Load sync metadata."""
    try:
        rec_name, _ = parse_section_folder_name(section_id)
    except Exception:
        return {}
    return _load_calibration_json(recording_stage_dir(rec_name, "synced") / "sync_info.json")


def _sensor_saturation(window_df: pd.DataFrame, kind: str, full_scale: float = 2000.0, frac: float = 0.95) -> tuple[float, int]:
    """Return sensor saturation."""
    axes = ("ax", "ay", "az") if kind == "acc" else ("gx", "gy", "gz")
    if window_df is None or window_df.empty or any(c not in window_df.columns for c in axes):
        return float("nan"), 0
    arr = window_df[list(axes)].to_numpy(dtype=float)
    sat = np.any(np.abs(arr) >= full_scale * frac, axis=1)
    f = float(np.mean(sat)) if len(sat) > 0 else float("nan")
    return f, int(f >= 0.01) if np.isfinite(f) else 0


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

_BACKGROUND_LABELS: frozenset[str] = frozenset({"non_riding", "calibration", "riding", "unlabeled", ""})


def extract_features_for_section(
    section_dir: Path,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    min_samples: int = 10,
    sample_rate_hz: float = _SAMPLE_RATE_HZ_DEFAULT,
    label_config_path: Path | str | None = None,
    event_aligned: bool = True,
    n_lags: int = 0,
    force: bool = False,
) -> pd.DataFrame:
    """Return extract features for section."""
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
    t_min = float(sporsa_df["timestamp"].iloc[0])
    t_max = float(sporsa_df["timestamp"].iloc[-1])

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
    # Load labels (optional).
    # ------------------------------------------------------------------
    labels_df = _load_labels_for_section(section_dir)
    if not labels_df.empty:
        for col in ("start_ms", "end_ms"):
            if col in labels_df.columns:
                labels_df[col] = pd.to_numeric(labels_df[col], errors="coerce")
    label_config = (
        load_label_config(label_config_path)
        if label_config_path is not None
        else default_label_config()
    )

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
    sync_meta = _load_sync_metadata(section_id)
    orient_stats = _load_calibration_json((section_dir / "orientation" / "orientation_stats.json"))
    quality_meta_base = {
        "synchronization_method": sync_meta.get("sync_method"),
        "synchronization_correlation": (sync_meta.get("correlation") or {}).get("offset_and_drift"),
        "synchronization_drift_ppm": sync_meta.get("drift_ppm"),
        "orientation_quality_sporsa": ((orient_stats.get("sensors") or {}).get("sporsa") or {}).get("quality"),
        "orientation_quality_arduino": ((orient_stats.get("sensors") or {}).get("arduino") or {}).get("quality"),
        "gravity_residual_sporsa": ((orient_stats.get("sensors") or {}).get("sporsa") or {}).get("gravity_residual_ms2"),
        "gravity_residual_arduino": ((orient_stats.get("sensors") or {}).get("arduino") or {}).get("gravity_residual_ms2"),
        "magnetometer_available_sporsa": bool("mx" in sporsa_df.columns),
        "magnetometer_available_arduino": bool("mx" in arduino_df.columns),
        "dropout_rate_sporsa": float(max(0.0, 1.0 - len(sporsa_df) / max(1.0, (t_max - t_min) / 1000.0 * sample_rate_hz))),
        "dropout_rate_arduino": float(max(0.0, 1.0 - len(arduino_df) / max(1.0, (t_max - t_min) / 1000.0 * sample_rate_hz))) if not arduino_df.empty else float("nan"),
        "orientation_dependent_valid": bool(not orient_sporsa_df.empty and not orient_arduino_df.empty),
    }

    # ------------------------------------------------------------------
    # Build sliding windows.
    # ------------------------------------------------------------------

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

        # Historically, we skipped windows when the derived `acc_norm` was all-NaN.
        # However `derived/*_signals.csv` can be missing, misaligned, or partially
        # broken while the calibrated stream is still perfectly valid. Feature
        # extraction itself can fall back to calibrated `|acc|`, so only skip
        # when *both* derived and calibrated acceleration are unusable.
        if not w_sporsa_sig.empty and "acc_norm" in w_sporsa_sig.columns:
            acc_norm_arr = w_sporsa_sig["acc_norm"].to_numpy(dtype=float)
            if len(acc_norm_arr) > 0 and np.all(np.isnan(acc_norm_arr)):
                # Fallback check on calibrated sample quality.
                if "acc_norm" in w_sporsa.columns:
                    cal_acc = w_sporsa["acc_norm"].to_numpy(dtype=float)
                    if len(cal_acc) > 0 and not np.all(np.isnan(cal_acc)):
                        pass  # keep window: calibrated stream is usable
                    else:
                        skipped_all_nan += 1
                        continue
                else:
                    skipped_all_nan += 1
                    continue

        try:
            acc_sat_frac, acc_sat_flag = _sensor_saturation(w_arduino if not w_arduino.empty else w_sporsa, "acc", full_scale=16 * 9.81)
            gyro_sat_frac, gyro_sat_flag = _sensor_saturation(w_arduino if not w_arduino.empty else w_sporsa, "gyro", full_scale=2000.0)
            quality_meta = dict(quality_meta_base)
            quality_meta.update({
                "acc_saturation_fraction": acc_sat_frac,
                "gyro_saturation_fraction": gyro_sat_frac,
                "acc_saturation_flag": acc_sat_flag,
                "gyro_saturation_flag": gyro_sat_flag,
            })
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
                labels_df=labels_df if not labels_df.empty else None,
                label_config=label_config,
                quality_metadata=quality_meta,
            )
            feat_dict["window_type"] = "sliding"
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

    # ------------------------------------------------------------------
    # Event-aligned windows.
    #
    # For each annotated event span that is not background riding, create
    # an additional window centred on the event midpoint.  This gives the
    # classifier a view where the event is always in a consistent position
    # within the window — something sliding windows cannot guarantee when
    # the hop size is larger than the event duration.
    #
    # Event-aligned windows are marked window_type="event_aligned" so they
    # can be kept separate for training (where labels are known) or dropped
    # for inference (where only sliding windows are used).
    # ------------------------------------------------------------------
    n_event_aligned = 0
    if event_aligned and not labels_df.empty:
        window_ms = window_s * 1000.0
        ea_idx_start = len(rows)
        label_col = (
            "scenario_label"
            if "scenario_label" in labels_df.columns
            else "label"
            if "label" in labels_df.columns
            else None
        )

        if label_col is not None:
            for ea_offset, (_, lrow) in enumerate(labels_df.iterrows()):
                raw_label = str(lrow.get(label_col, "")).strip()
                for token in raw_label.split("|"):
                    token = token.strip()
                    if token and token not in _BACKGROUND_LABELS:
                        break
                else:
                    continue  # all tokens are background – skip

                ev_start = float(lrow.get("start_ms", float("nan")))
                ev_end = float(lrow.get("end_ms", float("nan")))
                if not (np.isfinite(ev_start) and np.isfinite(ev_end) and ev_end > ev_start):
                    continue

                mid_ms = (ev_start + ev_end) / 2.0
                t_start = mid_ms - window_ms / 2.0
                t_end = t_start + window_ms

                # Clamp to recording bounds.
                if t_start < t_min:
                    t_start = t_min
                    t_end = t_start + window_ms
                if t_end > t_max:
                    t_end = t_max
                    t_start = t_end - window_ms

                if t_end - t_start < window_ms * 0.75:
                    continue

                w_sporsa = _slice_by_time(sporsa_df, t_start, t_end)
                if len(w_sporsa) < min_samples:
                    continue

                w_arduino = _slice_by_time(arduino_df, t_start, t_end)
                w_sporsa_sig = _slice_by_time(sporsa_signals_df, t_start, t_end)
                w_arduino_sig = _slice_by_time(arduino_signals_df, t_start, t_end)
                w_cross = _slice_by_time(cross_df, t_start, t_end)
                w_orient_sporsa = (
                    _slice_by_time(orient_sporsa_df, t_start, t_end)
                    if not orient_sporsa_df.empty
                    else None
                )
                w_orient_arduino = (
                    _slice_by_time(orient_arduino_df, t_start, t_end)
                    if not orient_arduino_df.empty
                    else None
                )

                try:
                    acc_sat_frac, acc_sat_flag = _sensor_saturation(
                        w_arduino if not w_arduino.empty else w_sporsa, "acc", full_scale=16 * 9.81
                    )
                    gyro_sat_frac, gyro_sat_flag = _sensor_saturation(
                        w_arduino if not w_arduino.empty else w_sporsa, "gyro", full_scale=2000.0
                    )
                    quality_meta = dict(quality_meta_base)
                    quality_meta.update({
                        "acc_saturation_fraction": acc_sat_frac,
                        "gyro_saturation_fraction": gyro_sat_frac,
                        "acc_saturation_flag": acc_sat_flag,
                        "gyro_saturation_flag": gyro_sat_flag,
                    })
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
                        window_idx=ea_idx_start + ea_offset,
                        sample_rate_hz=sample_rate_hz,
                        calibration_quality=calibration_quality,
                        sync_confidence=sync_confidence,
                        yaw_conf_sporsa=yaw_conf_sporsa,
                        yaw_conf_arduino=yaw_conf_arduino,
                        labels_df=labels_df if not labels_df.empty else None,
                        label_config=label_config,
                        quality_metadata=quality_meta,
                    )
                    feat_dict["window_type"] = "event_aligned"
                    rows.append(feat_dict)
                    n_event_aligned += 1
                except Exception as exc:
                    log.error(
                        "Event-aligned extraction failed for %s [%.1f–%.1f ms]: %s",
                        section_id,
                        t_start,
                        t_end,
                        exc,
                    )

        if n_event_aligned > 0:
            log.info(
                "%s: added %d event-aligned windows.",
                section_id,
                n_event_aligned,
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

    if n_lags > 0:
        features_df = add_lag_features(features_df, n_lags=n_lags)
        log.info("%s: added lag features (n_lags=%d).", section_id, n_lags)

    write_csv(features_df, features_csv)
    log.info("Wrote %d windows to %s", len(features_df), features_csv)

    # ------------------------------------------------------------------
    # Write stats JSON.
    # ------------------------------------------------------------------
    stats: dict[str, Any] = {
        "section_id": section_id,
        "n_windows": len(features_df),
        "n_windows_sliding": int((features_df["window_type"] == "sliding").sum()) if "window_type" in features_df.columns else len(features_df),
        "n_windows_event_aligned": n_event_aligned,
        "n_windows_skipped_min_samples": skipped_min_samples,
        "n_windows_skipped_all_nan": skipped_all_nan,
        "window_s": window_s,
        "hop_s": hop_s,
        "event_aligned": event_aligned,
        "lag_features_n_lags": n_lags,
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
    *,
    label_set: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Process recording features."""
    section_dirs = iter_sections_for_recording(recording_name)
    if not section_dirs:
        log.warning("No sections found for recording '%s'.", recording_name)
        return pd.DataFrame()

    transfer_labels_to_sections(recording_name, label_set=label_set)

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
