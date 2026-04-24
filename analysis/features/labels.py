"""Window label resolution and coarse/binary label taxonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Priority order for resolving simultaneous overlapping fine-grained labels.
# Higher index = higher priority. Used *only* to break ties when two
# annotations cover the same window equally; it does NOT define the class
# hierarchy used in training (see COARSE_MAP for that).
LABEL_PRIORITY: list[str] = [
    "calibration_sequence",
    "helmet_move",
    "grounded",
    "riding",
    "riding_standing",
    "forest",
    "uneven_road",
    "head_movement",
    "shoulder_check",
    "accelerating",
    "cornering",
    "braking",
    "sprinting",
    "sprint_standing",
    "hard_braking",
    "swerving",
    "fall",
]

LABEL_PRIORITY_RANK: dict[str, int] = {lbl: i for i, lbl in enumerate(LABEL_PRIORITY)}

# Coarse 4-class taxonomy for the main thesis experiment.
#   non_riding    filtered before training
#   steady_riding low-demand, steady-state riding
#   active_riding moderate-demand maneuvers
#   incident      safety-critical events
COARSE_MAP: dict[str, str] = {
    "calibration_sequence": "non_riding",
    "helmet_move": "non_riding",
    "grounded": "non_riding",
    "riding": "steady_riding",
    "riding_standing": "steady_riding",
    "forest": "steady_riding",
    "uneven_road": "steady_riding",
    "head_movement": "active_riding",
    "shoulder_check": "active_riding",
    "accelerating": "active_riding",
    "cornering": "active_riding",
    "braking": "active_riding",
    "sprinting": "active_riding",
    "sprint_standing": "active_riding",
    "hard_braking": "incident",
    "swerving": "incident",
    "fall": "incident",
}

# Activity 6-class taxonomy — grouped so each class has a distinct dual-IMU
# signature (bike frame vs. helmet). Merges happen along two principles:
#   1. Fine distinctions that single-sensor features already encode well
#      (e.g. uneven_road is a spectral variant of riding) collapse to a
#      single class.
#   2. Classes that *require* cross-sensor information to separate from
#      seated riding (standing posture, head yaw, swerve-vs-lean) are kept
#      distinct so the evaluation actually rewards bimodal feature sets.
ACTIVITY_MAP: dict[str, str] = {
    "calibration_sequence": "non_riding",
    "grounded": "non_riding",
    "helmet_move": "head_motion",
    "riding": "steady_seated",
    "forest": "steady_seated",
    "uneven_road": "steady_seated",
    "riding_standing": "standing",
    "sprint_standing": "standing",
    "accelerating": "longitudinal_effort",
    "braking": "longitudinal_effort",
    "sprinting": "longitudinal_effort",
    "hard_braking": "longitudinal_effort",
    "fall": "longitudinal_effort",
    "cornering": "turning",
    "swerving": "turning",
    "head_movement": "head_motion",
    "shoulder_check": "head_motion",
}

INCIDENT_LABELS: frozenset[str] = frozenset({"hard_braking", "swerving", "fall"})
NON_RIDING_LABELS: frozenset[str] = frozenset({"calibration_sequence", "helmet_move", "grounded"})


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
) -> str:
    """Return a single fine-grained label for the window.

    Strategy:
    1. Find all annotation rows overlapping [window_start_ms, window_end_ms].
    2. Compute containment = overlap_ms / window_duration_ms; keep rows with
       containment >= *containment_threshold*, else fall back to all overlaps.
    3. Pipe-split compound strings (``"riding|cornering"``) into tokens;
       each token inherits its source row's overlap_ms.
    4. Choose the token with largest overlap_ms; break ties with
       LABEL_PRIORITY_RANK (higher index wins). Unknown tokens use rank -1.
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

    return max(
        candidates,
        key=lambda t: (t[1], LABEL_PRIORITY_RANK.get(t[0], -1)),
    )[0]


def to_coarse_label(fine_label: str) -> str:
    """Map a fine-grained label to one of four coarse thesis classes."""
    if fine_label == "unlabeled":
        return "unlabeled"
    return COARSE_MAP.get(fine_label, "unknown")


def to_activity_label(fine_label: str) -> str:
    """Map a fine-grained label to one of six dual-IMU activity classes."""
    if fine_label == "unlabeled":
        return "unlabeled"
    return ACTIVITY_MAP.get(fine_label, "unknown")


def to_binary_label(fine_label: str) -> str:
    """Map a fine-grained label to incident / normal / non_riding / unlabeled."""
    if fine_label == "unlabeled":
        return "unlabeled"
    if fine_label in NON_RIDING_LABELS:
        return "non_riding"
    if fine_label in INCIDENT_LABELS:
        return "incident"
    return "normal"
