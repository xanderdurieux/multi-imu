"""Dispatch recording- and section-level diagnostic plots for pipeline stages."""

from __future__ import annotations

from pathlib import Path

from common.paths import recording_stage_dir


def plot_recording_pipeline_stage(recording_name: str, stage: str) -> None:
    """Plot outputs for a recording-level stage directory (parsed / synced)."""
    if stage == "synced":
        from visualization.plot_sync import plot_sync_stage

        plot_sync_stage(recording_stage_dir(recording_name, stage))
    elif stage == "parsed":
        from visualization.plot_comparison import plot_stage_data

        plot_stage_data(recording_stage_dir(recording_name, stage))


def plot_section_pipeline_stage(section_dir: Path, stage: str) -> None:
    """Plot outputs for one section (calibration, orientation, derived, features)."""
    if stage == "calibration":
        from visualization.plot_calibration import plot_calibration_stage

        plot_calibration_stage(section_dir)
    elif stage == "orientation":
        from visualization.plot_orientation import plot_orientation_stage

        plot_orientation_stage(section_dir)
    elif stage == "derived":
        from visualization.plot_derived import plot_derived_stage

        plot_derived_stage(section_dir)
    elif stage == "features":
        from visualization.plot_features import plot_features_stage
        from visualization.plot_labels import plot_labels

        plot_features_stage(section_dir)
        plot_labels(section_dir, stage="sensor")
        plot_labels(section_dir, stage="features")
