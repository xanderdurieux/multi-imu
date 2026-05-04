"""Window label resolution and configured label taxonomy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .label_config import LabelConfig, default_label_config


def _raw_labels_and_overlaps(
    overlapping: pd.DataFrame,
    overlap_ms: np.ndarray,
) -> list[tuple[str, float]]:
    """Return raw labels and overlaps."""
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


def _filter_tokens(tokens: list[str], config: LabelConfig | None = None) -> list[str]:
    """Return tokens after removing ignored labels and unlabeled markers."""
    cfg = config or default_label_config()
    filtered: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = str(token).strip()
        if not token or token.lower() in {"nan", "none", "unlabeled"}:
            continue
        if token in cfg.ignored_labels or token in seen:
            continue
        seen.add(token)
        filtered.append(token)
    return filtered


def _well_contained_candidates(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float,
    config: LabelConfig | None = None,
) -> list[tuple[str, float]]:
    """Return well contained candidates."""
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

    candidates = _raw_labels_and_overlaps(well_contained, well_overlap)
    cfg = config or default_label_config()
    return [candidate for candidate in candidates if candidate[0] not in cfg.ignored_labels]


def label_feature(
    labels_df: pd.DataFrame | None,
    window_start_ms: float,
    window_end_ms: float,
    *,
    containment_threshold: float = 0.5,
    config: LabelConfig | None = None,
) -> str:
    """Return label feature."""
    candidates = _well_contained_candidates(
        labels_df,
        window_start_ms,
        window_end_ms,
        containment_threshold=containment_threshold,
        config=config,
    )
    if not candidates:
        return "unlabeled"

    cfg = config or default_label_config()
    priority_rank = cfg.priority_rank
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
    """Return label feature set."""
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


def resolve_label_from_tokens(
    fine_labels,
    *,
    config: LabelConfig | None = None,
) -> str:
    """Resolve a dominant fine label from an overlap set."""
    cfg = config or default_label_config()
    tokens = _filter_tokens(list(fine_labels or ()), cfg)
    if not tokens:
        return "unlabeled"
    ranked = sorted(tokens, key=lambda t: cfg.priority_rank.get(t, -1), reverse=True)
    return ranked[0]


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
    """Convert set based label."""
    return (config or default_label_config()).map_label_set(scheme, fine_labels)


def to_riding_label(fine_label: str, config: LabelConfig | None = None) -> str:
    """Convert riding label."""
    return (config or default_label_config()).map_label("scenario_label_riding", fine_label)


def non_riding_labels(config: LabelConfig | None = None) -> frozenset[str]:
    """Return the configured fine labels treated as non-riding."""
    return (config or default_label_config()).binary_non_riding_labels


def ensure_resolved_labels(
    df: pd.DataFrame,
    *,
    config: LabelConfig | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Materialize resolved label columns from ``scenario_labels`` (the multi-label set).

    Feature CSVs only persist the raw pipe-delimited overlap set
    (``scenario_labels``) — every consumer that wants a single-label target
    (``scenario_label``, ``scenario_label_activity``/``_coarse``/``_binary``,
    or any configured set-based scheme) calls this helper to derive the
    columns it needs.

    Parameters
    ----------
    df:
        DataFrame produced by the features stage. Must contain the
        ``scenario_labels`` column; otherwise the frame is returned unchanged.
    config:
        Label config used for priority ranking and taxonomy mapping. Defaults
        to :func:`default_label_config`.
    overwrite:
        If False (default), columns already present in ``df`` are left alone.
        Set to True to recompute every derived column from ``scenario_labels``.

    Returns
    -------
    A copy of ``df`` with the resolved columns added.
    """
    if "scenario_labels" not in df.columns:
        return df

    cfg = config or default_label_config()
    out = df.copy()
    pipe = out["scenario_labels"].fillna("unlabeled").astype(str)
    token_lists = pipe.map(lambda s: [t for t in s.split("|") if t.strip()])

    fine_series = token_lists.map(lambda toks: resolve_label_from_tokens(toks, config=cfg))

    if overwrite or "scenario_label" not in out.columns:
        out["scenario_label"] = fine_series
    if overwrite or "scenario_label_activity" not in out.columns:
        out["scenario_label_activity"] = fine_series.map(
            lambda lbl: to_activity_label(lbl, config=cfg)
        )
    if overwrite or "scenario_label_coarse" not in out.columns:
        out["scenario_label_coarse"] = fine_series.map(
            lambda lbl: to_coarse_label(lbl, config=cfg)
        )
    if overwrite or "scenario_label_binary" not in out.columns:
        out["scenario_label_binary"] = fine_series.map(
            lambda lbl: to_binary_label(lbl, config=cfg)
        )

    for scheme in cfg.set_based_scheme_names:
        if overwrite or scheme not in out.columns:
            out[scheme] = token_lists.map(
                lambda toks, sch=scheme: to_set_based_label(sch, toks, config=cfg)
            )

    return out
