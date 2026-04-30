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
    riding_value: str = "riding"
    # scenario_label_riding is annotation-driven, not derived from the
    # priority-collapsed dominant label. Only literal labels listed in these
    # sets contribute; every other label is ignored for this objective.
    riding_riding_labels: frozenset[str] = frozenset({"riding"})
    riding_non_riding_labels: frozenset[str] = frozenset({"non_riding"})

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

        if scheme == "scenario_label_riding":
            # Strict, annotation-driven: only the configured literal sets
            # contribute. Anything else (cornering, accelerating, …) is
            # unlabeled for this objective. Multi-label resolution from the
            # full overlap set lives in :meth:`map_label_set`.
            if label in self.riding_non_riding_labels:
                return self.binary_non_riding_value
            if label in self.riding_riding_labels:
                return self.riding_value
            return self.unlabeled_values.get("scenario_label_riding", "unlabeled")

        return self.derived_maps.get(scheme, {}).get(
            label,
            self.unknown_values.get(scheme, "unknown"),
        )

    def map_label_set(self, scheme: str, fine_labels) -> str:
        """Resolve *scheme* from the full set of labels overlapping a window.

        This is the multi-label-friendly entry point — the window's annotation
        is treated as a *set* of independent overlapping labels, not collapsed
        to a single "dominant" label. The per-objective rules decide which
        members of the set count.

        Currently implemented for ``scenario_label_riding`` only. Other schemes
        fall back to the configured priority order via :meth:`map_label`,
        applied to the highest-priority token in the set.
        """
        tokens = {
            str(t).strip()
            for t in (fine_labels or ())
            if str(t).strip() and str(t).strip().lower() not in {"nan", "none", "unlabeled"}
        }
        if not tokens:
            return self.unlabeled_values.get(scheme, "unlabeled")

        if scheme == "scenario_label_riding":
            # Both classes present in the same window → return non_riding so
            # we never silently call a window "riding" while it overlaps an
            # explicit non_riding span. True multi-label support (returning
            # both) belongs in a future labels[] column.
            if tokens & self.riding_non_riding_labels:
                return self.binary_non_riding_value
            if tokens & self.riding_riding_labels:
                return self.riding_value
            return self.unlabeled_values.get("scenario_label_riding", "unlabeled")

        # Fall back to single-label semantics on the priority-max token.
        ranked = sorted(tokens, key=lambda t: self.priority_rank.get(t, -1), reverse=True)
        return self.map_label(scheme, ranked[0])


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

    riding_block = derived.get("scenario_label_riding")
    if isinstance(riding_block, dict):
        unlabeled_values["scenario_label_riding"] = (
            str(riding_block.get("unlabeled_value", "unlabeled")).strip() or "unlabeled"
        )
        riding_value = str(riding_block.get("riding_value", "riding")).strip() or "riding"

        riding_riding_labels = frozenset(
            _string_list(
                riding_block.get("riding_labels", ["riding"]),
                field_name="scenario_label_riding.riding_labels",
            )
        )
        riding_non_riding_labels = frozenset(
            _string_list(
                riding_block.get("non_riding_labels", ["non_riding"]),
                field_name="scenario_label_riding.non_riding_labels",
            )
        )
        riding_overlap = riding_riding_labels & riding_non_riding_labels
        if riding_overlap:
            raise ValueError(
                "label config: scenario_label_riding riding/non_riding label sets overlap: "
                + ", ".join(sorted(riding_overlap))
            )
    else:
        unlabeled_values["scenario_label_riding"] = "unlabeled"
        riding_value = "riding"
        riding_riding_labels = frozenset({"riding"})
        riding_non_riding_labels = frozenset({"non_riding"})

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
        riding_value=riding_value,
        riding_riding_labels=riding_riding_labels,
        riding_non_riding_labels=riding_non_riding_labels,
    )


def load_label_config(path: Path | str | None = None) -> LabelConfig:
    """Load and validate label config, merging *path* over the default JSON."""
    return _build(load_label_config_data(path))


@lru_cache(maxsize=1)
def default_label_config() -> LabelConfig:
    """Return the default label config, cached from ``labels.default.json``."""
    return load_label_config()
