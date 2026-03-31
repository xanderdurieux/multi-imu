"""Event detection configuration with per-section override support."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class EventConfig:
    # Bump/shock thresholds
    bump_acc_threshold_ms2: float = 4.0    # acc_deviation or acc_hf spike threshold
    bump_min_duration_s: float = 0.05
    bump_max_duration_s: float = 0.8
    bump_min_confidence: float = 0.3

    # Braking thresholds
    brake_acc_threshold_ms2: float = 3.0   # negative longitudinal acc
    brake_min_duration_s: float = 0.3
    brake_max_duration_s: float = 5.0

    # Swerve/corner thresholds
    swerve_gyro_threshold_dps: float = 30.0  # gyro_norm spike (degrees/s)
    swerve_min_duration_s: float = 0.2
    swerve_max_duration_s: float = 4.0

    # Disagreement thresholds
    disagree_threshold: float = 4.0        # cross-sensor disagree_score
    disagree_min_duration_s: float = 0.5

    # Fall/drop detection
    fall_acc_threshold_ms2: float = 15.0   # very large spike
    fall_gyro_threshold_dps: float = 150.0

    # NMS merging
    merge_gap_s: float = 0.1               # merge events within this gap

    @classmethod
    def load(cls, path: Path) -> "EventConfig":
        """Load config from JSON, falling back to defaults for missing keys."""
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)
