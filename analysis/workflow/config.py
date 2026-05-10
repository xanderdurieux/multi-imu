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
    "features",
    "exports",
    "dataset_summary",
    "evaluation",
    "inference",
    "report",
    "thesis_bundle",
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
    orientation_filter: str = "auto"                 # mahony
    sample_rate_hz: float = 100.0

    # Feature options
    labels_path: str = ""
    label_set: str = "v1"
    window_s: float = 2.0
    hop_s: float = 1.0
    event_aligned: bool = True       # add event-centred windows for labelled events
    lag_features_n_lags: int = 0     # 0 = disabled; 2 = add lag-1 and lag-2 context columns

    # Orchestration toggles
    no_plots: bool = False
    log_to_file: bool = True
    force: bool = False
    skip_exports: bool = False
    evaluation_seed: int = 42
    evaluation_methods: list[str] = field(
        default_factory=lambda: ["label_grid"]
    )
    evaluation_sessions: list[str] = field(default_factory=list)
    evaluation_exclude_sessions: list[str] = field(default_factory=list)
    evaluation_recordings: list[str] = field(default_factory=list)
    evaluation_exclude_recordings: list[str] = field(default_factory=list)
    # Label-grid evaluation settings
    label_grid_label_cols: list[str] = field(default_factory=lambda: ["auto"])
    label_grid_exclude_non_riding: bool = False
    # Sweep over the cross-product of {label_grid_label_cols} × {label_grid_quality_sweep}.
    label_grid_quality_sweep: list[str] = field(
        default_factory=lambda: ["marginal"]
    )
    # Classifiers to train in CV ("auto" = all three registered models).
    label_grid_models: list[str] = field(default_factory=lambda: ["auto"])
    # Models to compute permutation importance for (per-feature-set).
    # "auto" = all three; limit to one tree-ensemble to save runtime.
    label_grid_permutation_models: list[str] = field(
        default_factory=lambda: ["hist_gradient_boosting"]
    )
    label_grid_save_trained_models: bool = False

    event_contrast_models: list[str] = field(
        default_factory=lambda: ["hist_gradient_boosting", "logistic_regression"]
    )
    two_stage_event_models: list[str] = field(
        default_factory=lambda: ["hist_gradient_boosting", "logistic_regression"]
    )
    two_stage_event_tasks: list[str] = field(default_factory=lambda: ["core"])
    two_stage_target_recall: float = 0.90

    inference_model_paths: list[str] = field(default_factory=list)

    thesis_protocol_path: str = ""
    # Minimum quality filter applied to all evaluation methods.
    evaluation_min_quality: str = "marginal"

    # Stages to run
    stages: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkflowConfig":
        """Create an instance from a dictionary."""
        allowed = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in allowed}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""
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
        if not isinstance(self.label_set, str) or not self.label_set.strip():
            errors.append("label_set must be a non-empty string")
        valid_label_grid_spec_keywords = {"auto", "multiclass", "binary"}
        if not isinstance(self.label_grid_label_cols, list) or not self.label_grid_label_cols:
            errors.append("label_grid_label_cols must be a non-empty list of label-column specs")
        else:
            for spec in self.label_grid_label_cols:
                if not isinstance(spec, str) or not spec.strip():
                    errors.append(
                        f"label_grid_label_cols: each entry must be a non-empty string, got {spec!r}"
                    )
        if not isinstance(self.label_grid_exclude_non_riding, bool):
            errors.append("label_grid_exclude_non_riding must be a boolean")

        for field_name in (
            "evaluation_sessions",
            "evaluation_exclude_sessions",
            "evaluation_recordings",
            "evaluation_exclude_recordings",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"{field_name} must be a list of strings")

        if not isinstance(self.inference_model_paths, list) or not all(
            isinstance(p, str) for p in self.inference_model_paths
        ):
            errors.append("inference_model_paths must be a list of strings")

        valid_eval_methods = {"label_grid", "event_contrasts", "two_stage_events"}
        if not isinstance(self.evaluation_methods, list):
            errors.append("evaluation_methods must be a list")
        else:
            bad_methods = [m for m in self.evaluation_methods if m not in valid_eval_methods]
            if bad_methods:
                errors.append(
                    f"evaluation_methods contains invalid entries {bad_methods}; "
                    f"valid: {sorted(valid_eval_methods)}"
                )
            if not self.evaluation_methods:
                errors.append("evaluation_methods must contain at least one method")
        valid_qualities = {"poor", "marginal", "good"}
        if self.evaluation_min_quality not in valid_qualities:
            errors.append(
                f"evaluation_min_quality must be one of {sorted(valid_qualities)}, "
                f"got {self.evaluation_min_quality!r}"
            )
        if not isinstance(self.label_grid_quality_sweep, list):
            errors.append("label_grid_quality_sweep must be a list of quality labels")
        else:
            bad = [q for q in self.label_grid_quality_sweep if q not in valid_qualities]
            if bad:
                errors.append(
                    f"label_grid_quality_sweep contains invalid entries {bad}; "
                    f"valid: {sorted(valid_qualities)}"
                )
        valid_models = {"random_forest", "hist_gradient_boosting", "logistic_regression"}
        valid_or_auto = valid_models | {"auto"}
        valid_or_auto_or_none = valid_or_auto | {"none"}

        def _check_model_list(name: str, lst: Any, *, allow_none: bool = False) -> None:
            if not isinstance(lst, list):
                errors.append(f"{name} must be a list of model names")
                return
            if not lst:
                errors.append(f"{name} must be a non-empty list")
                return
            valid = valid_or_auto_or_none if allow_none else valid_or_auto
            bad = [m for m in lst if m not in valid]
            if bad:
                errors.append(
                    f"{name} contains invalid entries {bad}; "
                    f"valid: {sorted(valid)}"
                )
                return
            if "auto" in lst and len(lst) != 1:
                errors.append(
                    f'{name}: when using "auto", it must be the only list entry '
                    "(use explicit model names for subsets)"
                )
            if "none" in lst and len(lst) != 1:
                errors.append(
                    f'{name}: when using "none", it must be the only list entry'
                )

        _check_model_list("label_grid_models", self.label_grid_models)
        _check_model_list(
            "label_grid_permutation_models",
            self.label_grid_permutation_models,
            allow_none=True,
        )
        _check_model_list("event_contrast_models", self.event_contrast_models)
        _check_model_list("two_stage_event_models", self.two_stage_event_models)
        valid_two_stage_tasks = {
            "core",
            "all",
            "turning",
            "deceleration",
            "high_effort",
            "posture",
        }
        if not isinstance(self.two_stage_event_tasks, list):
            errors.append("two_stage_event_tasks must be a list")
        else:
            bad_tasks = [
                task for task in self.two_stage_event_tasks if task not in valid_two_stage_tasks
            ]
            if bad_tasks:
                errors.append(
                    f"two_stage_event_tasks contains invalid entries {bad_tasks}; "
                    f"valid: {sorted(valid_two_stage_tasks)}"
                )
            if {"core", "all"} & set(self.two_stage_event_tasks) and len(self.two_stage_event_tasks) != 1:
                errors.append("'core' or 'all' must be the only two_stage_event_tasks entry")
        try:
            target_recall = float(self.two_stage_target_recall)
        except (TypeError, ValueError):
            errors.append("two_stage_target_recall must be numeric")
        else:
            if not 0.0 < target_recall <= 1.0:
                errors.append("two_stage_target_recall must be in (0, 1]")
        return errors


def known_stages() -> list[str]:
    """Return the known stage names in canonical order."""
    return list(_KNOWN_STAGES)


def load_workflow_config(path: Path | str) -> WorkflowConfig:
    """Load workflow config."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Workflow config not found: {p}")
    merged_data = load_workflow_config_data(p)
    cfg = WorkflowConfig.from_dict(merged_data)
    errors = cfg.validate()
    if errors:
        raise ValueError("Invalid workflow config:\n" + "\n".join(f"  - {e}" for e in errors))
    return cfg
