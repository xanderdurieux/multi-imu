"""Shared helpers for synchronisation method modules and the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from common import find_sensor_csv, recording_stage_dir, write_dataframe

from .drift_estimator import SyncModel, apply_sync_model, save_sync_model
from .metrics import compute_sync_correlations

REFERENCE_SENSOR = "sporsa"
TARGET_SENSOR = "arduino"
METHOD_ORDER = ["cal", "lida", "sda", "online"]
METHOD_LABELS = {
    "sda": "SDA",
    "lida": "LIDA",
    "cal": "Calibration",
    "online": "Online",
}
METHOD_STAGES = {method: f"synced/{method}" for method in METHOD_ORDER}


@dataclass(frozen=True)
class SyncArtifacts:
    """Files produced by a synchronisation run."""

    method: str
    output_dir: Path
    reference_csv: Path
    target_csv: Path
    sync_info_json: Path


@dataclass(frozen=True)
class MethodSummary:
    """Pipeline-facing summary for one synchronisation method."""

    method: str
    stage: str
    label: str
    available: bool
    offset_seconds: float | None
    drift_ppm: float | None
    corr_offset_only: float | None
    corr_offset_and_drift: float | None
    sync_info_path: Path | None = None


@dataclass(frozen=True)
class MethodRunResult:
    """Outcome of executing one method inside the pipeline."""

    method: str
    ok: bool
    artifacts: SyncArtifacts | None = None
    error: str | None = None


@dataclass(frozen=True)
class SelectionResult:
    """Best method chosen by the pipeline."""

    method: str
    stage: str
    summary: MethodSummary


def stage_for_method(method: str) -> str:
    if method not in METHOD_STAGES:
        raise KeyError(f"Unknown synchronisation method: {method}")
    return METHOD_STAGES[method]


def load_recording_streams(
    recording_name: str,
    stage_in: str,
    *,
    reference_sensor: str = REFERENCE_SENSOR,
    target_sensor: str = TARGET_SENSOR,
) -> tuple[Path, Path]:
    return (
        find_sensor_csv(recording_name, stage_in, reference_sensor),
        find_sensor_csv(recording_name, stage_in, target_sensor),
    )


def build_sync_info(
    model: SyncModel,
    *,
    method: str,
    reference_csv: Path,
    target_csv: Path,
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    sample_rate_hz: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = asdict(model)
    info["created_at_utc"] = info.get("created_at_utc") or datetime.now(UTC).isoformat()
    info["method"] = method
    info["stage"] = stage_for_method(method)
    info["reference_csv"] = str(reference_csv)
    info["target_csv"] = str(target_csv)
    info["correlation"] = compute_sync_correlations(
        reference_df,
        target_df,
        model,
        sample_rate_hz=sample_rate_hz,
    )
    if extra:
        info.update(extra)
    return info


def write_sync_outputs(
    *,
    method: str,
    reference_csv: Path,
    reference_df: pd.DataFrame,
    target_csv: Path,
    target_df: pd.DataFrame,
    model: SyncModel,
    output_dir: Path,
    sample_rate_hz: float,
    extra_info: dict[str, Any] | None = None,
) -> SyncArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_out = output_dir / reference_csv.name
    target_out = output_dir / target_csv.name
    sync_info_out = output_dir / 'sync_info.json'

    shutil.copy2(reference_csv, reference_out)
    aligned_target = apply_sync_model(target_df, model, replace_timestamp=True)
    drop_columns = [
        column
        for column in ('timestamp_orig', 'timestamp_aligned', 'timestamp_received')
        if column in aligned_target.columns
    ]
    if drop_columns:
        aligned_target = aligned_target.drop(columns=drop_columns)
    write_dataframe(aligned_target, target_out)

    save_sync_model(model, sync_info_out)
    sync_info = build_sync_info(
        model,
        method=method,
        reference_csv=reference_csv,
        target_csv=target_csv,
        reference_df=reference_df,
        target_df=target_df,
        sample_rate_hz=sample_rate_hz,
        extra=extra_info,
    )
    sync_info_out.write_text(json.dumps(sync_info, indent=2), encoding='utf-8')

    return SyncArtifacts(
        method=method,
        output_dir=output_dir,
        reference_csv=reference_out,
        target_csv=target_out,
        sync_info_json=sync_info_out,
    )


def load_method_summary(recording_name: str, method: str) -> MethodSummary:
    stage = stage_for_method(method)
    sync_info_path = recording_stage_dir(recording_name, stage) / 'sync_info.json'
    if not sync_info_path.exists():
        return MethodSummary(
            method=method,
            stage=stage,
            label=METHOD_LABELS[method],
            available=False,
            offset_seconds=None,
            drift_ppm=None,
            corr_offset_only=None,
            corr_offset_and_drift=None,
            sync_info_path=None,
        )

    data = json.loads(sync_info_path.read_text(encoding='utf-8'))
    correlation = data.get('correlation', {}) or {}
    drift = data.get('drift_seconds_per_second')
    return MethodSummary(
        method=method,
        stage=stage,
        label=METHOD_LABELS[method],
        available=True,
        offset_seconds=data.get('offset_seconds'),
        drift_ppm=(float(drift) * 1e6 if drift is not None else None),
        corr_offset_only=correlation.get('offset_only'),
        corr_offset_and_drift=correlation.get('offset_and_drift'),
        sync_info_path=sync_info_path,
    )
