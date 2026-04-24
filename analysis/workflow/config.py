"""Workflow configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.paths import load_workflow_config_data

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
    "dataset_summary",   # compact thesis-ready dataset audit (reads features_fused.csv)
    "evaluation",
    "report",
    "thesis_bundle",   # assemble final thesis figures/tables into report/thesis_bundle/
]


@dataclass
class WorkflowConfig:
    """Validated workflow configuration."""

    # Dataset scope
    data_root: str = ""
    sessions: list[str] = field(default_factory=list)
    recordings: list[str] = field(default_factory=list)

    # Stage options
    sync_method: str = "auto"                        # "multi_anchor"|"one_anchor_adaptive"|"one_anchor_prior"|"signal_only"|"auto"
    split_stage: str = "synced"                      # "parsed"|"synced"
    orientation_filter: str = "auto"                 # madgwick[_marg]|complementary|ekf[_marg]|auto
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
    log_to_file: bool = True
    force: bool = False
    skip_exports: bool = False
    evaluation_seed: int = 42
    evaluation_label_col: str = "scenario_label_coarse"
    thesis_protocol_path: str = ""
    min_quality_label: str = "marginal"

    # Stages to run
    stages: list[str] = field(default_factory=list)

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
        valid_sync = {"multi_anchor", "one_anchor_adaptive", "one_anchor_prior", "signal_only", "auto"}
        if self.sync_method not in valid_sync:
            errors.append(f"sync_method must be one of {valid_sync}, got {self.sync_method!r}")
        valid_split = {"parsed", "synced"}
        if self.split_stage not in valid_split:
            errors.append(f"split_stage must be one of {valid_split}, got {self.split_stage!r}")
        valid_orient = {
            "madgwick",
            "madgwick_marg",
            "complementary",
            "ekf",
            "ekf_marg",
            "auto",
        }
        if self.orientation_filter not in valid_orient:
            errors.append(f"orientation_filter must be one of {valid_orient}")
        if not self.stages:
            errors.append("stages must contain at least one pipeline stage.")
        for stage in self.stages:
            if stage not in _KNOWN_STAGES:
                errors.append(f"Unknown stage in stages: {stage!r}. Valid: {_KNOWN_STAGES}")
        if self.window_s <= 0:
            errors.append("window_s must be > 0")
        if self.hop_s <= 0:
            errors.append("hop_s must be > 0")
        valid_eval_label_cols = {
            "auto",
            "scenario_label",
            "scenario_label_coarse",
            "scenario_label_binary",
        }
        if self.evaluation_label_col not in valid_eval_label_cols:
            errors.append(
                "evaluation_label_col must be one of "
                f"{valid_eval_label_cols}, got {self.evaluation_label_col!r}"
            )
        return errors


def known_stages() -> list[str]:
    """Return the known stage names in canonical order."""
    return list(_KNOWN_STAGES)


def load_workflow_config(path: Path | str) -> WorkflowConfig:
    """Load and validate a workflow config JSON file.

    The default workflow config is always used as a base. Values from the
    provided config overwrite matching keys, allowing partial override files.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow config not found: {p}")
    merged_data = load_workflow_config_data(p)
    cfg = WorkflowConfig.from_dict(merged_data)
    errors = cfg.validate()
    if errors:
        raise ValueError("Invalid workflow config:\n" + "\n".join(f"  - {e}" for e in errors))
    return cfg