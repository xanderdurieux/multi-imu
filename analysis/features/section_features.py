"""Recording- and section-level feature extraction orchestration.

This module turns IMU CSVs under
``data/recordings/<recording_name>/`` into consolidated feature tables
under::

    data/recordings/<recording_name>/features/
        recording_features.csv
        section_features.csv

It is deliberately lightweight and can be called either directly from
``run_full_pipeline_for_session`` or via the standalone
``run_features_pipeline.py`` script.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from common import load_dataframe, recording_dir, recording_stage_dir, write_dataframe
from .window_features import compute_time_series_features


@dataclass
class FeatureRow:
    """Container for one feature row before serialisation."""

    recording: str
    sensor: str
    kind: str  # "recording" or "section"
    stage: str
    section_id: str | None
    features: dict[str, Any]

    def as_flat_dict(self) -> dict[str, Any]:
        flat = {
            "recording": self.recording,
            "sensor": self.sensor,
            "kind": self.kind,
            "stage": self.stage,
        }
        if self.section_id is not None:
            flat["section_id"] = self.section_id
        flat.update(self.features)
        return flat


def _features_dir(recording_name: str) -> Path:
    root = recording_dir(recording_name)
    out_dir = root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def extract_recording_features(
    recording_name: str,
    *,
    stage: str = "calibrated",
    sensors: Iterable[str] | None = None,
) -> Path:
    """Compute per-recording features for each sensor CSV in *stage*.

    Parameters
    ----------
    recording_name:
        Recording identifier (e.g. ``"2026-02-26_5"``).
    stage:
        Recording stage to use, typically ``"calibrated"`` or ``"orientation"``.
    sensors:
        Optional iterable of sensor name tokens to filter on (matched against
        CSV filenames). When ``None``, all CSVs in the stage are used.
    """
    stage_dir = recording_stage_dir(recording_name, stage)
    if not stage_dir.is_dir():
        raise FileNotFoundError(f"Stage directory not found: {stage_dir}")

    csv_files = sorted(stage_dir.glob("*.csv"))
    if sensors is not None:
        wanted = {s.lower() for s in sensors}
        csv_files = [p for p in csv_files if any(tok in p.stem.lower() for tok in wanted)]

    rows: list[FeatureRow] = []
    for csv_path in csv_files:
        df = load_dataframe(csv_path)
        sensor_name = csv_path.stem
        feats = compute_time_series_features(
            df,
            sensor=sensor_name,
            recording_name=recording_name,
            context={"kind": "recording", "stage": stage},
        )
        rows.append(
            FeatureRow(
                recording=recording_name,
                sensor=sensor_name,
                kind="recording",
                stage=stage,
                section_id=None,
                features=feats,
            )
        )

    out_dir = _features_dir(recording_name)
    out_csv = out_dir / "recording_features.csv"

    if not rows:
        # Write an empty but well-formed CSV to document that this step ran.
        pd.DataFrame(columns=["recording", "sensor", "kind", "stage"]).to_csv(
            out_csv,
            index=False,
        )
        return out_csv

    df_rows = pd.DataFrame([r.as_flat_dict() for r in rows])
    df_rows.to_csv(out_csv, index=False)
    return out_csv


def extract_section_features(
    recording_name: str,
    *,
    sensors: Iterable[str] | None = None,
) -> Path:
    """Compute per-section features for all sections of *recording_name*.

    This function looks under::

        data/recordings/<recording_name>/sections/section_*/

    and expects per-sensor CSVs (e.g. ``sporsa.csv``, ``arduino.csv``).
    """
    rec_root = recording_dir(recording_name)
    sections_root = rec_root / "sections"
    if not sections_root.is_dir():
        raise FileNotFoundError(f"No sections/ directory for recording {recording_name}")

    section_dirs = sorted(
        d for d in sections_root.iterdir() if d.is_dir() and d.name.startswith("section_")
    )

    rows: list[FeatureRow] = []
    for section_dir in section_dirs:
        section_id = section_dir.name  # e.g. "section_1"
        csv_files = sorted(section_dir.glob("*.csv"))
        if sensors is not None:
            wanted = {s.lower() for s in sensors}
            csv_files = [p for p in csv_files if any(tok in p.stem.lower() for tok in wanted)]

        for csv_path in csv_files:
            df = load_dataframe(csv_path)
            sensor_name = csv_path.stem
            feats = compute_time_series_features(
                df,
                sensor=sensor_name,
                recording_name=recording_name,
                context={
                    "kind": "section",
                    "stage": "sections",
                    "section_id": section_id,
                },
            )
            rows.append(
                FeatureRow(
                    recording=recording_name,
                    sensor=sensor_name,
                    kind="section",
                    stage="sections",
                    section_id=section_id,
                    features=feats,
                )
            )

    out_dir = _features_dir(recording_name)
    out_csv = out_dir / "section_features.csv"

    if not rows:
        pd.DataFrame(columns=["recording", "sensor", "kind", "stage", "section_id"]).to_csv(
            out_csv,
            index=False,
        )
        return out_csv

    df_rows = pd.DataFrame([r.as_flat_dict() for r in rows])
    df_rows.to_csv(out_csv, index=False)
    return out_csv

