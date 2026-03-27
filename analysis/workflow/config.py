"""Configuration helpers for reproducible thesis workflow runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowConfig:
    """Single-file configuration for running the thesis pipeline."""

    session: str | None = None
    recordings: list[str] | None = None
    data_root: str = "data"

    sync_method: str = "best"
    split_stage: str = "synced"
    orientation_filter: str = "complementary_orientation"
    frame_alignment: str = "gravity_only"

    no_plots: bool = False
    force: bool = False
    skip_exports: bool = False

    labels_path: str | None = None
    event_config_path: str | None = None
    event_centered_features: bool = False
    min_event_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> WorkflowConfig:
    """Load a JSON config file into :class:`WorkflowConfig`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")

    allowed = {f.name for f in WorkflowConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown workflow config key(s): {', '.join(unknown)}")

    return WorkflowConfig(**payload)


def merge_cli_overrides(base: WorkflowConfig, **overrides: Any) -> WorkflowConfig:
    """Return a config with non-None CLI values applied."""
    current = base.to_dict()
    for key, value in overrides.items():
        if value is not None:
            current[key] = value
    return WorkflowConfig(**current)
