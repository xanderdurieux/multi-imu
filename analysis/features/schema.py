"""Human-readable descriptions for feature columns (thesis / data dictionary)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .families import GROUP_EXPLANATIONS, FEATURE_DEFS

# Populated alongside extract.py; keys are column names after export prefixes where noted.
FEATURE_COLUMN_DESCRIPTIONS: dict[str, str] = {
    # Window geometry
    "section": "Legacy section path recording/section_N",
    "recording_id": "Recording folder name",
    "section_id": "Section folder name section_N",
    "window_start_s": "Window start relative to section first timestamp (s)",
    "window_end_s": "Window end relative to section first timestamp (s)",
    "window_center_s": "Window centre relative to section first timestamp (s)",
    "window_source": "Windowing strategy used for this row (sliding or event_centered)",
    "event_type": "Event type when window_source=event_centered; empty for sliding windows",
    "event_confidence": "Event detector confidence [0,1] when event_centered",
    "event_timestamp": "Original event timestamp in sensor epoch ms (if available)",
    # Metadata
    "sync_method": "Synchronization method key (e.g. sda, lida, calibration, online)",
    "orientation_method": "Orientation filter stem (e.g. complementary_orientation)",
    "calibration_quality": "Worst per-sensor ride calibration quality in section",
    "orientation_quality": "Worst orientation quality tag across bike+rider for selected orientation_method",
    "upstream_confidence_score": "Conservative confidence score propagated from sync, calibration, and orientation quality [0,1]",
    "upstream_quality_flags": "Pipe-separated upstream quality issues (sync/calibration/orientation), or ok",
    "label_source": "Origin of scenario_label (none / manual / rule name)",
    "label_scope": "Resolved label scope (none/recording/section/interval)",
    "labeler": "Annotator identifier (person/tool)",
    "labeled_at_utc": "Annotation timestamp in UTC (ISO 8601)",
    "label_confidence": "Annotator confidence score [0,1]",
    "label_notes": "Free-text annotation notes",
    "label_status": "Annotation status tag (e.g., confirmed/unknown/ambiguous/mixed)",
    "label_schema_version": "Version of label taxonomy/schema used while annotating",
    "suggestion_source": "Optional suggestion provenance (e.g., event_candidate_detector)",
    "suggestion_rank": "Optional integer ranking of suggestion priority",
    "scenario_label": "Scenario or disturbance class for this window",
    "feature_reliability": "JSON map per feature family with reliable/confidence/reasons/affected_features",
    "feature_reliability_summary": "Pipe-separated union of reliability reasons across families, or ok",
    "exclude_from_downstream": "Pipe-separated features recommended for exclusion in downstream evaluation for this window",
    "quality_schema_version": "Version tag for unified quality metadata schema",
    "section_quality_score": "Section-level overall quality score propagated to each window [0,1]",
    "section_usability_category": "Section-level usability category (usable/caution/exclude)",
    "window_quality_score": "Window-level reliability score combining upstream + feature-family confidence [0,1]",
    "window_quality_label": "Window-level quality tier label (high/medium/low/poor)",
    "window_usability_category": "Window-level usability category (usable/caution/exclude)",
    "section_sync_confidence": "Section-level sync confidence score derived from selected sync method metrics",
    "section_sync_residual_quality": "Section-level residual alignment quality score",
    "section_interpolation_burden": "Section-level interpolation burden proxy from timestamp gap frequency",
    "section_packet_loss_burden": "Section-level packet loss burden proxy from estimated missing samples",
    "section_frame_estimation_confidence": "Section-level confidence in estimated section frame alignment",
    "section_feature_reliability_score": "Section-level average of window family reliability indicators",
    # Window sanity
    "sporsa__window_sanity": "Sanity flag for bike window (ok or semicolon-separated degenerate conditions)",
    "arduino__window_sanity": "Sanity flag for rider window (ok or semicolon-separated degenerate conditions)",
    # Per-sensor (prefix sporsa__ or arduino__)
    "acc_norm_mean": "Mean ‖accelerometer‖ in world frame (m/s²)",
    "acc_norm_max": "Max ‖accelerometer‖",
    "acc_norm_energy": "Sum of squared ‖accelerometer‖",
    "jerk_norm_max": "Max norm jerk from ‖acc‖ differences",
    "gyro_norm_max": "Max ‖gyroscope‖ (rad/s)",
    "gyro_energy": "Sum of squared ‖gyro‖",
    "vertical_acc_mean": "Mean Z (vertical) acceleration (m/s²)",
    "vertical_acc_std": "Std Z acceleration",
    "acc_norm_rms": "RMS ‖accelerometer‖",
    "acc_norm_std": "Std ‖accelerometer‖",
    "acc_norm_ptp": "Peak-to-peak ‖accelerometer‖",
    "gyro_norm_rms": "RMS ‖gyro‖",
    "gyro_norm_std": "Std ‖gyro‖",
    "gyro_norm_ptp": "Peak-to-peak ‖gyro‖",
    "dom_freq_acc_norm_hz": "Dominant FFT frequency of centred ‖acc‖ (Hz)",
    "acc_band_low_energy": "FFT energy ‖acc‖ in low band (default 0.5–3 Hz)",
    "acc_band_high_energy": "FFT energy ‖acc‖ in high band (default 3–15 Hz)",
    "acc_band_high_fraction": "High-band energy / total AC energy",
    "crest_factor_acc_norm": "Peak ‖acc‖ / RMS ‖acc‖",
    "entropy_acc_norm": "Histogram entropy of ‖acc‖",
    "zcr_acc_norm": "Zero-crossing rate of centred ‖acc‖",
    "longitudinal_acc_std": "Std deviation of X (longitudinal) acceleration when frame allows",
    "lateral_acc_std": "Std deviation of Y (lateral) acceleration",
    "pitch_mean_deg": "Mean pitch from orientation CSV (deg)",
    "pitch_std_deg": "Std pitch (deg)",
    "roll_mean_deg": "Mean roll (deg)",
    "roll_std_deg": "Std roll (deg)",
    "pitch_rate_mean_deg_s": "Mean absolute pitch rate (deg/s)",
    "roll_rate_mean_deg_s": "Mean absolute roll rate (deg/s)",
    # Cross-sensor
    "acc_norm_corr": "Pearson correlation of ‖acc‖ (interpolated window)",
    "acc_norm_lag_ms": "Lag at max cross-correlation of ‖acc‖ (ms)",
    "acc_energy_ratio": "sporsa acc energy / arduino acc energy",
    "gyro_energy_ratio": "sporsa gyro energy / arduino gyro energy",
    "pitch_corr": "Pearson correlation of pitch (if orientation available)",
    "pitch_divergence_std": "Std of pitch difference",
    "acc_norm_coherence_mean": "Mean magnitude-squared coherence of ‖acc‖ in band",
    "gyro_norm_coherence_mean": "Mean coherence of ‖gyro‖ in band",
    "peak_time_diff_acc_norm_s": "Time difference between ‖acc‖ peaks (s)",
    "shock_peak_ratio_bike_to_rider": "max ‖acc‖ bike / max ‖acc‖ rider (sporsa/arduino)",
    "shock_peak_ratio_rider_to_bike": "Inverse ratio",
    "vec_disagreement_mean_ms2": "Mean Euclidean ‖acc_bike − acc_rider‖ after interpolation",
    "roll_corr": "Pearson correlation of roll",
    "roll_divergence_std": "Std roll difference",
    "energy_ratio_longitudinal": "sum(ax²)_bike / sum(ax²)_rider on interpolated grid",
    "energy_ratio_lateral": "sum(ay²) ratio bike/rider",
    "energy_ratio_vertical": "sum(az²) ratio bike/rider",
    "feature_confidence__cross_sensor": "Confidence score for acc correlation/lag/coherence family [0,1]",
    "feature_reliable__cross_sensor": "1 if cross-sensor acc timing family is reliable, else 0",
    "feature_confidence__energy_ratio": "Confidence score for energy-ratio family [0,1]",
    "feature_reliable__energy_ratio": "1 if energy ratio features are reliable, else 0",
    "feature_confidence__orientation_cross_sensor": "Confidence score for pitch/roll cross-sensor family [0,1]",
    "feature_reliable__orientation_cross_sensor": "1 if orientation cross-sensor features are reliable, else 0",
    "feature_confidence__grouped": "Confidence score for grouped physically interpreted orientation-linked features [0,1]",
    "feature_reliable__grouped": "1 if grouped orientation-linked features are reliable, else 0",
}

for feat in FEATURE_DEFS:
    FEATURE_COLUMN_DESCRIPTIONS[feat.name] = (
        f"[{feat.group}] {feat.description} Hypothesis: {feat.hypothesis}."
    )


def write_feature_schema_json(out_path: Path) -> None:
    """Write JSON data dictionary for all documented columns."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "description": "Feature column dictionary for dual-IMU cycling analysis",
        "group_explanations": GROUP_EXPLANATIONS,
        "columns": FEATURE_COLUMN_DESCRIPTIONS,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
