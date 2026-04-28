"""Scenario-label taxonomy configuration.

The mappings in ``data/_configs/labels.default.json`` are treated as a working
interpretation of the current annotations, not as a permanent ontology.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.paths import load_label_config_data


@dataclass(frozen=True)
class LabelConfig:
    """Resolved scenario-label priority and derived-label mappings."""

    label_priority: tuple[str, ...]
    derived_maps: dict[str, dict[str, str]]
    unlabeled_values: dict[str, str]
    unknown_values: dict[str, str]
    binary_incident_labels: frozenset[str]
    binary_non_riding_labels: frozenset[str]
    binary_normal_value: str = "normal"
    binary_incident_value: str = "incident"
    binary_non_riding_value: str = "non_riding"

    @property
    def priority_rank(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.label_priority)}

    def map_label(self, scheme: str, fine_label: str) -> str:
        """Map a fine scenario label to *scheme* using the configured rules."""
        label = str(fine_label or "").strip()
        unlabeled = self.unlabeled_values.get(scheme, "unlabeled")
        if not label or label.lower() in {"nan", "none"} or label == "unlabeled":
            return unlabeled

        if scheme == "scenario_label_binary":
            if label in self.binary_non_riding_labels:
                return self.binary_non_riding_value
            if label in self.binary_incident_labels:
                return self.binary_incident_value
            return self.binary_normal_value

        return self.derived_maps.get(scheme, {}).get(
            label,
            self.unknown_values.get(scheme, "unknown"),
        )


def _string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"label config: {field_name!r} must be a list.")
    out = tuple(str(item).strip() for item in value if str(item).strip())
    if not out:
        raise ValueError(f"label config: {field_name!r} must not be empty.")
    return out


def _string_map(value: Any, *, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"label config: {field_name!r} must be an object.")
    return {
        str(key).strip(): str(mapped).strip()
        for key, mapped in value.items()
        if str(key).strip() and str(mapped).strip()
    }


def _build(payload: dict[str, Any]) -> LabelConfig:
    priority = _string_list(payload.get("label_priority"), field_name="label_priority")

    derived = payload.get("derived_labels")
    if not isinstance(derived, dict):
        raise ValueError("label config: missing 'derived_labels' object.")

    derived_maps: dict[str, dict[str, str]] = {}
    unlabeled_values: dict[str, str] = {}
    unknown_values: dict[str, str] = {}

    for scheme in ("scenario_label_activity", "scenario_label_coarse"):
        block = derived.get(scheme)
        if not isinstance(block, dict):
            raise ValueError(f"label config: missing {scheme!r} object.")
        derived_maps[scheme] = _string_map(block.get("map"), field_name=f"{scheme}.map")
        unlabeled_values[scheme] = str(block.get("unlabeled_value", "unlabeled")).strip() or "unlabeled"
        unknown_values[scheme] = str(block.get("unknown_value", "unknown")).strip() or "unknown"

    binary = derived.get("scenario_label_binary")
    if not isinstance(binary, dict):
        raise ValueError("label config: missing 'scenario_label_binary' object.")
    unlabeled_values["scenario_label_binary"] = (
        str(binary.get("unlabeled_value", "unlabeled")).strip() or "unlabeled"
    )

    incident_labels = frozenset(
        _string_list(binary.get("incident_labels"), field_name="scenario_label_binary.incident_labels")
    )
    non_riding_labels = frozenset(
        _string_list(binary.get("non_riding_labels"), field_name="scenario_label_binary.non_riding_labels")
    )

    overlap = incident_labels & non_riding_labels
    if overlap:
        raise ValueError(
            "label config: binary incident/non-riding label sets overlap: "
            + ", ".join(sorted(overlap))
        )

    return LabelConfig(
        label_priority=priority,
        derived_maps=derived_maps,
        unlabeled_values=unlabeled_values,
        unknown_values=unknown_values,
        binary_incident_labels=incident_labels,
        binary_non_riding_labels=non_riding_labels,
        binary_normal_value=str(binary.get("normal_value", "normal")).strip() or "normal",
        binary_incident_value=str(binary.get("incident_value", "incident")).strip() or "incident",
        binary_non_riding_value=str(binary.get("non_riding_value", "non_riding")).strip() or "non_riding",
    )


def load_label_config(path: Path | str | None = None) -> LabelConfig:
    """Load and validate label config, merging *path* over the default JSON."""
    return _build(load_label_config_data(path))


@lru_cache(maxsize=1)
def default_label_config() -> LabelConfig:
    """Return the default label config, cached from ``labels.default.json``."""
    return load_label_config()
