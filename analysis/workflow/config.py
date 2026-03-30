"""Configuration helpers for reproducible thesis workflow runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SYNC_METHODS = {"best", "sda", "lida", "calibration", "online"}
FRAME_ALIGNMENTS = {"gravity_only", "gravity_plus_forward", "section_horizontal_frame"}
ORIENTATION_FILTERS = {
    "madgwick_acc_only",
    "madgwick_9dof",
    "complementary_orientation",
    "ekf_orientation",
}
SPLIT_STAGES = {"synced", "parsed"}


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
    evaluation_seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_type(payload: dict[str, Any], key: str, expected: type[Any] | tuple[type[Any], ...]) -> None:
    if key in payload and not isinstance(payload[key], expected):
        exp = (
            ", ".join(t.__name__ for t in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise ValueError(f"Config key '{key}' must be {exp}; got {type(payload[key]).__name__}")


def _validate_config_payload(payload: dict[str, Any], *, source: Path) -> None:
    allowed = {f.name for f in WorkflowConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown workflow config key(s): {', '.join(unknown)}")

    _require_type(payload, "session", (str, type(None)))
    _require_type(payload, "recordings", (list, type(None)))
    _require_type(payload, "data_root", str)
    _require_type(payload, "sync_method", str)
    _require_type(payload, "split_stage", str)
    _require_type(payload, "orientation_filter", str)
    _require_type(payload, "frame_alignment", str)
    _require_type(payload, "no_plots", bool)
    _require_type(payload, "force", bool)
    _require_type(payload, "skip_exports", bool)
    _require_type(payload, "labels_path", (str, type(None)))
    _require_type(payload, "event_config_path", (str, type(None)))
    _require_type(payload, "event_centered_features", bool)
    _require_type(payload, "min_event_confidence", (int, float))
    _require_type(payload, "evaluation_seed", int)

    recordings = payload.get("recordings")
    if isinstance(recordings, list):
        for i, value in enumerate(recordings):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Config key 'recordings[{i}]' must be a non-empty string")

    sync_method = payload.get("sync_method")
    if isinstance(sync_method, str) and sync_method not in SYNC_METHODS:
        raise ValueError(
            f"Invalid sync_method '{sync_method}' in {source}; expected one of {sorted(SYNC_METHODS)}"
        )
    split_stage = payload.get("split_stage")
    if isinstance(split_stage, str) and split_stage not in SPLIT_STAGES:
        raise ValueError(
            f"Invalid split_stage '{split_stage}' in {source}; expected one of {sorted(SPLIT_STAGES)}"
        )
    orientation = payload.get("orientation_filter")
    if isinstance(orientation, str) and orientation not in ORIENTATION_FILTERS:
        raise ValueError(
            "Invalid orientation_filter "
            f"'{orientation}' in {source}; expected one of {sorted(ORIENTATION_FILTERS)}"
        )
    frame = payload.get("frame_alignment")
    if isinstance(frame, str) and frame not in FRAME_ALIGNMENTS:
        raise ValueError(
            f"Invalid frame_alignment '{frame}' in {source}; expected one of {sorted(FRAME_ALIGNMENTS)}"
        )

    confidence = payload.get("min_event_confidence")
    if isinstance(confidence, (int, float)) and not (0.0 <= float(confidence) <= 1.0):
        raise ValueError("Config key 'min_event_confidence' must be in [0.0, 1.0]")

    eval_seed = payload.get("evaluation_seed")
    if isinstance(eval_seed, int) and eval_seed < 0:
        raise ValueError("Config key 'evaluation_seed' must be >= 0")

    data_root = payload.get("data_root")
    if isinstance(data_root, str) and not data_root.strip():
        raise ValueError("Config key 'data_root' must be a non-empty string")


def load_config(path: Path) -> WorkflowConfig:
    """Load a JSON config file into :class:`WorkflowConfig`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    _validate_config_payload(payload, source=path)

    return WorkflowConfig(**payload)


def merge_cli_overrides(base: WorkflowConfig, **overrides: Any) -> WorkflowConfig:
    """Return a config with non-None CLI values applied."""
    current = base.to_dict()
    for key, value in overrides.items():
        if value is not None:
            current[key] = value
    _validate_config_payload(current, source=Path("<cli-overrides>"))
    return WorkflowConfig(**current)
