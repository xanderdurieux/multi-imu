"""Window label resolution and configured label taxonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .label_config import LabelConfig, default_label_config


def _raw_labels_and_overlaps(
    overlapping: pd.DataFrame,
    overlap_ms: np.ndarray,
) -> list[tuple[str, float]]:
    """Vectorised pipe-split of compound labels with inherited overlap.

    Prefers ``scenario_label`` when present and non-empty, falling back to
    ``label``; rows lacking both are dropped. Each pipe-separated token
    inherits its row's overlap so priority ranking is not distorted.
    """
    scenario = overlapping.get("scenario_label")
    fallback = overlapping.get("label")

    if scenario is not None:
        raw = scenario.astype("string")
        if fallback is not None:
            raw = raw.where(raw.str.strip().fillna("").ne(""), fallback.astype("string"))
    elif fallback is not None:
        raw = fallback.astype("string")
    else:
        return []

    raw = raw.fillna("").str.strip()
    valid = raw.ne("")
    if not valid.any():
        return []

    out: list[tuple[str, float]] = []
    for raw_str, ov in zip(raw[valid].to_numpy(), overlap_ms[valid.to_numpy()], strict=False):
        ov_f = float(ov)
        for token in str(raw_str).split("|"):
            token = token.strip()
            if token:
                out.append((token, ov_f))
    return out


def _well_contained_candidates(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float,
) -> list[tuple[str, float]]:
    """Return ``(token, overlap_ms)`` pairs for labels overlapping the window.

    Containment-aware: prefers rows whose overlap covers ≥
    *containment_threshold* of the window; falls back to all overlapping rows
    if none meet the threshold so very short annotations aren't lost.
    Compound ``"riding|cornering"`` strings are pipe-split; each token
    inherits its source row's overlap so callers can rank by either overlap
    or priority without distortion.
    """
    if labels_df is None or labels_df.empty:
        return []
    if not {"start_ms", "end_ms"}.issubset(labels_df.columns):
        return []

    starts = labels_df["start_ms"].to_numpy(dtype=float)
    ends = labels_df["end_ms"].to_numpy(dtype=float)
    mask = (starts < window_end_ms) & (ends > window_start_ms)
    if not mask.any():
        return []

    overlapping = labels_df.loc[mask]
    overlap_ms = (
        np.minimum(ends[mask], window_end_ms) - np.maximum(starts[mask], window_start_ms)
    )
    window_dur_ms = max(window_end_ms - window_start_ms, 1.0)
    containment = overlap_ms / window_dur_ms

    contained_mask = containment >= containment_threshold
    if contained_mask.any():
        well_contained = overlapping.iloc[contained_mask]
        well_overlap = overlap_ms[contained_mask]
    else:
        well_contained = overlapping
        well_overlap = overlap_ms

    return _raw_labels_and_overlaps(well_contained, well_overlap)


def label_feature(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float = 0.5,
    config: LabelConfig | None = None,
) -> str:
    """Return the priority-collapsed dominant fine-grained label for the window.

    Kept for inspection and for legacy single-label derived schemes
    (``scenario_label_activity`` / ``_coarse`` / ``_binary``).  For objectives
    that require *all* overlapping labels — currently
    ``scenario_label_riding`` — use :func:`label_feature_set` instead.
    """
    candidates = _well_contained_candidates(
        labels_df,
        window_start_ms,
        window_end_ms,
        containment_threshold=containment_threshold,
    )
    if not candidates:
        return "unlabeled"

    priority_rank = (config or default_label_config()).priority_rank
    return max(
        candidates,
        key=lambda t: (t[1], priority_rank.get(t[0], -1)),
    )[0]


def label_feature_set(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float = 0.0,
) -> list[str]:
    """Return the de-duplicated set of fine labels overlapping the window.

    Multi-label-friendly counterpart to :func:`label_feature` — no priority
    collapse, no scheme mapping.  Callers select which labels are relevant
    per objective (e.g. ``scenario_label_riding`` only inspects the literal
    ``riding`` / ``non_riding`` annotations).

    Default ``containment_threshold = 0.0`` includes *every* overlapping
    annotation, not just well-contained ones.  Containment-based filtering is
    a single-label artifact (needed when collapsing to one dominant token);
    for set extraction we want all overlaps so a window straddling a
    non_riding → riding boundary surfaces both labels.
    """
    candidates = _well_contained_candidates(
        labels_df,
        window_start_ms,
        window_end_ms,
        containment_threshold=containment_threshold,
    )
    if not candidates:
        return []

    seen: set[str] = set()
    tokens: list[str] = []
    for tok, _ov in candidates:
        if tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return tokens


def to_coarse_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Map a fine-grained label using the configured coarse taxonomy."""
    return (config or default_label_config()).map_label("scenario_label_coarse", fine_label)


def to_activity_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Map a fine-grained label using the configured activity taxonomy."""
    return (config or default_label_config()).map_label("scenario_label_activity", fine_label)


def to_binary_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Map a fine-grained label using the configured binary taxonomy."""
    return (config or default_label_config()).map_label("scenario_label_binary", fine_label)


def to_set_based_label(
    scheme: str,
    fine_labels,
    config: LabelConfig | None = None,
) -> str:
    """Resolve any registered set-based binary scheme from a label set.

    Accepts any iterable of label strings (e.g. the output of
    :func:`label_feature_set` or a pipe-split ``scenario_labels`` cell).
    Schemes are configured under ``set_based_binary_schemes`` in
    ``labels.default.json``.
    """
    return (config or default_label_config()).map_label_set(scheme, fine_labels)


def to_riding_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Single-label fallback for ``scenario_label_riding``.

    Prefer :func:`to_set_based_label` with the full overlap set when
    available — a window straddling both ``riding`` and ``non_riding`` only
    resolves correctly under the multi-label rules.
    """
    return (config or default_label_config()).map_label("scenario_label_riding", fine_label)


def non_riding_labels(config: LabelConfig | None = None) -> frozenset[str]:
    """Return the configured fine labels treated as non-riding."""
    return (config or default_label_config()).binary_non_riding_labels
