"""Workflow configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_KNOWN_STAGES = [
    "parse",
    "sync",
    "split",
    "calibration",
    "orientation",
    "derived",
    "events",
    "features",
    "exports",
    "evaluation",
    "reporting",
]


@dataclass
class WorkflowConfig:
    """Validated workflow configuration."""

    # Dataset scope
    data_root: str = ""
    sessions: list[str] = field(default_factory=list)
    recordings: list[str] = field(default_factory=list)

    # Stage options
    sync_method: str = "auto"                        # "sda"|"lida"|"calibration"|"online"|"auto"
    split_stage: str = "synced"                      # "parsed"|"synced"
    frame_alignment: str = "gravity_only"            # "gravity_only"|"gravity_plus_forward"|"all"
    orientation_filter: str = "madgwick"             # "madgwick"|"complementary"|"all"
    sample_rate_hz: float = 100.0

    # Feature / event options
    event_config_path: str = ""
    event_centered_features: bool = False
    min_event_confidence: float = 0.3
    event_types: list[str] = field(default_factory=lambda: ["bump", "brake", "swerve", "disagree", "fall"])
    labels_path: str = ""
    window_s: float = 2.0
    hop_s: float = 1.0

    # Orchestration toggles
    no_plots: bool = False
    force: bool = False
    skip_exports: bool = False
    evaluation_seed: int = 42
    thesis_protocol_path: str = ""
    min_quality_label: str = "marginal"

    # Stages to run (empty = all)
    stages: list[str] = field(default_factory=list)
    from_stage: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowConfig":
        allowed = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    def validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors: list[str] = []
        valid_sync = {"sda", "lida", "calibration", "online", "auto"}
        if self.sync_method not in valid_sync:
            errors.append(f"sync_method must be one of {valid_sync}, got {self.sync_method!r}")
        valid_split = {"parsed", "synced"}
        if self.split_stage not in valid_split:
            errors.append(f"split_stage must be one of {valid_split}, got {self.split_stage!r}")
        valid_frame = {"gravity_only", "gravity_plus_forward", "all"}
        if self.frame_alignment not in valid_frame:
            errors.append(f"frame_alignment must be one of {valid_frame}")
        valid_orient = {"madgwick", "complementary", "all"}
        if self.orientation_filter not in valid_orient:
            errors.append(f"orientation_filter must be one of {valid_orient}")
        if self.from_stage:
            if self.from_stage not in _KNOWN_STAGES:
                errors.append(
                    f"from_stage must be one of {_KNOWN_STAGES}, got {self.from_stage!r}"
                )
            if self.stages:
                errors.append("Specify either 'stages' or 'from_stage' in config, not both.")
        for stage in self.stages:
            if stage not in _KNOWN_STAGES:
                errors.append(f"Unknown stage in stages: {stage!r}. Valid: {_KNOWN_STAGES}")
        if self.window_s <= 0:
            errors.append("window_s must be > 0")
        if self.hop_s <= 0:
            errors.append("hop_s must be > 0")
        return errors


def load_workflow_config(path: Path | str) -> WorkflowConfig:
    """Load and validate a workflow config JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cfg = WorkflowConfig.from_dict(data)
    errors = cfg.validate()
    if errors:
        raise ValueError("Invalid workflow config:\n" + "\n".join(f"  - {e}" for e in errors))
    return cfg
