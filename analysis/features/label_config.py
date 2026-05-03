"""Label config helpers for extract labelled sliding-window features from section signals."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.paths import load_label_config_data


@dataclass(frozen=True)
class SetBasedBinaryScheme:
    """Data container for set based binary scheme."""

    name: str
    positive_value: str
    negative_value: str
    unlabeled_value: str
    positive_labels: frozenset[str]
    negative_labels: frozenset[str] = frozenset()
    qualifier_labels: frozenset[str] = frozenset()
    on_overlap: str = "positive"  # "positive" | "negative"

    def resolve(self, tokens: set[str]) -> str:
        """Resolve."""
        pos_hit = bool(tokens & self.positive_labels)
        neg_hit = bool(tokens & self.negative_labels)

        # Literal mode: both class lists are explicit.
        if self.negative_labels:
            if pos_hit and neg_hit:
                return self.negative_value if self.on_overlap == "negative" else self.positive_value
            if neg_hit:
                return self.negative_value
            if pos_hit:
                return self.positive_value
            return self.unlabeled_value

        # Qualified-positive mode: positive vs (qualifier without positive).
        if pos_hit:
            return self.positive_value
        if self.qualifier_labels and (tokens & self.qualifier_labels):
            return self.negative_value
        return self.unlabeled_value


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
    set_based_schemes: tuple[SetBasedBinaryScheme, ...] = ()

    @property
    def priority_rank(self) -> dict[str, int]:
        """Return priority rank."""
        return {label: idx for idx, label in enumerate(self.label_priority)}

    def set_based_scheme(self, name: str) -> SetBasedBinaryScheme | None:
        """Return set based scheme."""
        return next((s for s in self.set_based_schemes if s.name == name), None)

    @property
    def set_based_scheme_names(self) -> tuple[str, ...]:
        """Return set based scheme names."""
        return tuple(s.name for s in self.set_based_schemes)

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

        sb = self.set_based_scheme(scheme)
        if sb is not None:
            # Single-label fallback for set-based objectives: pretend the
            # one fine label is the entire overlap set.  Matches the
            # legacy behaviour of `to_riding_label` etc.
            return sb.resolve({label})

        return self.derived_maps.get(scheme, {}).get(
            label,
            self.unknown_values.get(scheme, "unknown"),
        )

    def map_label_set(self, scheme: str, fine_labels) -> str:
        """Map label set."""
        tokens = {
            str(t).strip()
            for t in (fine_labels or ())
            if str(t).strip() and str(t).strip().lower() not in {"nan", "none", "unlabeled"}
        }

        sb = self.set_based_scheme(scheme)
        if sb is not None:
            if not tokens:
                return sb.unlabeled_value
            return sb.resolve(tokens)

        if not tokens:
            return self.unlabeled_values.get(scheme, "unlabeled")

        ranked = sorted(tokens, key=lambda t: self.priority_rank.get(t, -1), reverse=True)
        return self.map_label(scheme, ranked[0])


def _string_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    """Return string list."""
    if not isinstance(value, list):
        raise ValueError(f"label config: {field_name!r} must be a list.")
    out = tuple(str(item).strip() for item in value if str(item).strip())
    if not out:
        raise ValueError(f"label config: {field_name!r} must not be empty.")
    return out


def _string_map(value: Any, *, field_name: str) -> dict[str, str]:
    """Return string map."""
    if not isinstance(value, dict):
        raise ValueError(f"label config: {field_name!r} must be an object.")
    return {
        str(key).strip(): str(mapped).strip()
        for key, mapped in value.items()
        if str(key).strip() and str(mapped).strip()
    }


def _build_set_based_scheme(name: str, block: dict[str, Any]) -> SetBasedBinaryScheme:
    """Build set based scheme."""
    if not isinstance(block, dict):
        raise ValueError(f"label config: set_based_binary_schemes.{name!r} must be an object.")

    pos_labels = frozenset(
        _string_list(block.get("positive_labels"), field_name=f"{name}.positive_labels")
    )
    neg_labels = frozenset(
        str(s).strip() for s in (block.get("negative_labels") or []) if str(s).strip()
    )
    qual_labels = frozenset(
        str(s).strip() for s in (block.get("qualifier_labels") or []) if str(s).strip()
    )

    if not neg_labels and not qual_labels:
        raise ValueError(
            f"label config: {name!r} needs either 'negative_labels' (literal mode) "
            "or 'qualifier_labels' (qualified-positive mode)."
        )
    overlap = pos_labels & neg_labels
    if overlap:
        raise ValueError(
            f"label config: {name!r} positive/negative label sets overlap: "
            + ", ".join(sorted(overlap))
        )

    on_overlap = str(block.get("on_overlap", "positive")).strip().lower()
    if on_overlap not in {"positive", "negative"}:
        raise ValueError(
            f"label config: {name!r} on_overlap must be 'positive' or 'negative'."
        )

    return SetBasedBinaryScheme(
        name=name,
        positive_value=str(block.get("positive_value", "positive")).strip() or "positive",
        negative_value=str(block.get("negative_value", "negative")).strip() or "negative",
        unlabeled_value=str(block.get("unlabeled_value", "unlabeled")).strip() or "unlabeled",
        positive_labels=pos_labels,
        negative_labels=neg_labels,
        qualifier_labels=qual_labels,
        on_overlap=on_overlap,
    )


def _build(payload: dict[str, Any]) -> LabelConfig:
    """Build."""
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

    sb_block = payload.get("set_based_binary_schemes") or {}
    if not isinstance(sb_block, dict):
        raise ValueError("label config: 'set_based_binary_schemes' must be an object if present.")
    set_based: list[SetBasedBinaryScheme] = []
    for name, block in sb_block.items():
        scheme = _build_set_based_scheme(str(name).strip(), block)
        unlabeled_values[scheme.name] = scheme.unlabeled_value
        set_based.append(scheme)

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
        set_based_schemes=tuple(set_based),
    )


def load_label_config(path: Path | str | None = None) -> LabelConfig:
    """Load and validate label config, merging *path* over the default JSON."""
    return _build(load_label_config_data(path))


@lru_cache(maxsize=1)
def default_label_config() -> LabelConfig:
    """Return the default label config, cached from ``labels.default.json``."""
    return load_label_config()
