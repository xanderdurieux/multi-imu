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


def label_feature(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float = 0.5,
    config: LabelConfig | None = None,
) -> str:
    """Return a single fine-grained label for the window.

    Strategy:
    1. Find all annotation rows overlapping [window_start_ms, window_end_ms].
    2. Compute containment = overlap_ms / window_duration_ms; keep rows with
       containment >= *containment_threshold*, else fall back to all overlaps.
    3. Pipe-split compound strings (``"riding|cornering"``) into tokens;
       each token inherits its source row's overlap_ms.
    4. Choose the token with largest overlap_ms; break ties with the configured
       priority rank. Unknown tokens use rank -1.
    """
    if labels_df is None or labels_df.empty:
        return "unlabeled"
    if not {"start_ms", "end_ms"}.issubset(labels_df.columns):
        return "unlabeled"

    starts = labels_df["start_ms"].to_numpy(dtype=float)
    ends = labels_df["end_ms"].to_numpy(dtype=float)
    mask = (starts < window_end_ms) & (ends > window_start_ms)
    if not mask.any():
        return "unlabeled"

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

    candidates = _raw_labels_and_overlaps(well_contained, well_overlap)
    if not candidates:
        return "unlabeled"

    priority_rank = (config or default_label_config()).priority_rank
    return max(
        candidates,
        key=lambda t: (t[1], priority_rank.get(t[0], -1)),
    )[0]


def to_coarse_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Map a fine-grained label using the configured coarse taxonomy."""
    return (config or default_label_config()).map_label("scenario_label_coarse", fine_label)


def to_activity_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Map a fine-grained label using the configured activity taxonomy."""
    return (config or default_label_config()).map_label("scenario_label_activity", fine_label)


def to_binary_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Map a fine-grained label using the configured binary taxonomy."""
    return (config or default_label_config()).map_label("scenario_label_binary", fine_label)


def non_riding_labels(config: LabelConfig | None = None) -> frozenset[str]:
    """Return the configured fine labels treated as non-riding."""
    return (config or default_label_config()).binary_non_riding_labels
