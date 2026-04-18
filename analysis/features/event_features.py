"""Event-overlap features for a single window."""

from __future__ import annotations

from typing import Any

import pandas as pd


def event_features(
    events_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
) -> dict[str, Any]:
    """Count events that overlap the window, plus max confidence by type."""
    feats: dict[str, Any] = {
        "events_bump_count": 0,
        "events_brake_count": 0,
        "events_swerve_count": 0,
        "events_disagree_count": 0,
        "events_max_bump_confidence": 0.0,
        "events_max_swerve_confidence": 0.0,
        "events_any": 0,
    }
    if events_df is None or events_df.empty:
        return feats

    required = {"event_type", "start_ms", "end_ms"}
    if not required.issubset(events_df.columns):
        return feats

    mask = (
        (events_df["start_ms"] < window_end_ms)
        & (events_df["end_ms"] > window_start_ms)
    )
    overlapping = events_df[mask]
    if overlapping.empty:
        return feats

    types = overlapping["event_type"].str.lower()

    feats["events_bump_count"] = int((types == "bump").sum())
    feats["events_brake_count"] = int((types == "brake").sum())
    feats["events_swerve_count"] = int((types == "swerve").sum())
    feats["events_disagree_count"] = int((types == "disagree").sum())
    feats["events_any"] = 1

    if "confidence" in overlapping.columns:
        bump_mask = types == "bump"
        swerve_mask = types == "swerve"
        if bump_mask.any():
            feats["events_max_bump_confidence"] = float(
                overlapping.loc[bump_mask, "confidence"].max()
            )
        if swerve_mask.any():
            feats["events_max_swerve_confidence"] = float(
                overlapping.loc[swerve_mask, "confidence"].max()
            )

    return feats
