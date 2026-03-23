"""Shared helpers for writing synchronisation outputs."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from common import find_sensor_csv, recording_stage_dir, write_dataframe

from .core import SyncModel, apply_sync_model, compute_sync_correlations, load_stream, save_sync_model

DEFAULT_REFERENCE_SENSOR = "sporsa"
DEFAULT_TARGET_SENSOR = "arduino"


@dataclass(frozen=True)
class SyncArtifacts:
    """Files produced by a synchronisation run."""

    reference_csv: Path
    target_csv: Path
    sync_json: Path


@dataclass(frozen=True)
class RecordingInputs:
    """Resolved recording-level input files and dataframes."""

    reference_csv: Path
    target_csv: Path
    reference_df: pd.DataFrame
    target_df: pd.DataFrame


@dataclass(frozen=True)
class SyncOutputs:
    """Computed model plus additional metadata to persist in sync_info.json."""

    model: SyncModel
    metadata: dict[str, Any]


def load_recording_inputs(
    recording_name: str,
    stage_in: str,
    *,
    reference_sensor: str = DEFAULT_REFERENCE_SENSOR,
    target_sensor: str = DEFAULT_TARGET_SENSOR,
) -> RecordingInputs:
    """Load the reference and target CSVs for one recording stage."""
    reference_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    target_csv = find_sensor_csv(recording_name, stage_in, target_sensor)
    reference_df = load_stream(reference_csv)
    target_df = load_stream(target_csv)
    if reference_df.empty or target_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")
    return RecordingInputs(
        reference_csv=reference_csv,
        target_csv=target_csv,
        reference_df=reference_df,
        target_df=target_df,
    )


def write_sync_outputs(
    *,
    reference_csv: Path,
    target_csv: Path,
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    outputs: SyncOutputs,
    out_dir: Path,
    correlation_rate_hz: float,
) -> SyncArtifacts:
    """Persist sync_info.json plus aligned target and copied reference CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sync_json = out_dir / "sync_info.json"
    save_sync_model(outputs.model, sync_json)

    payload = asdict(outputs.model)
    payload.update(outputs.metadata)
    payload["correlation"] = compute_sync_correlations(
        reference_df,
        target_df,
        outputs.model,
        sample_rate_hz=correlation_rate_hz,
    )
    sync_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    aligned = apply_sync_model(target_df, outputs.model, replace_timestamp=True)
    drop_cols = [
        col
        for col in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if col in aligned.columns
    ]
    if drop_cols:
        aligned = aligned.drop(columns=drop_cols)

    reference_out = out_dir / reference_csv.name
    target_out = out_dir / target_csv.name
    shutil.copy2(reference_csv, reference_out)
    write_dataframe(aligned, target_out)
    return SyncArtifacts(reference_csv=reference_out, target_csv=target_out, sync_json=sync_json)


def stage_output_dir(recording_name: str, method_name: str) -> Path:
    """Return the method-specific synced output directory for a recording."""
    return recording_stage_dir(recording_name, f"synced/{method_name}")
