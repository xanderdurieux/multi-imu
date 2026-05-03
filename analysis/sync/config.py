"""Config helpers for align arduino timestamps to the sporsa reference clock."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from common.paths import load_sync_config_data

SIGNAL_MODES: tuple[str, ...] = (
    "acc_norm_diff",
    "gyro_norm_diff",
    "acc_gyro_fused_diff",
)


@dataclass(frozen=True)
class AnchorRefinementConfig:
    """Data container for anchor refinement config."""

    resample_rate_hz: float = 100.0  # rate for acc_norm xcorr inside each anchor window
    search_seconds: float = 1.0       # ± lag span searched around the coarse offset


@dataclass(frozen=True)
class WindowRefinementConfig:
    """Data container for window refinement config."""

    window_seconds: float = 20.0
    step_seconds: float = 10.0
    local_search_seconds: float = 0.5   # ± lag span per window
    min_window_score: float = 0.10      # masked-NCC gate per window
    min_fit_r2: float = 0.10            # gate on the final offset+drift line fit
    min_fit_ppm: float = -1000.0      # gate on the final drift
    max_fit_ppm: float = 1000.0       # gate on the final drift


@dataclass(frozen=True)
class SyncConfig:
    """Synchronization search settings."""

    signal_mode: str
    resample_rate_hz: float = 100.0
    min_valid_fraction: float = 0.5
    anchor_refinement: AnchorRefinementConfig = field(
        default_factory=AnchorRefinementConfig
    )
    window_refinement: WindowRefinementConfig = field(
        default_factory=WindowRefinementConfig
    )
    signal_only_coarse_search_seconds: float = 60.0
    one_anchor_prior_drift_ppm: float = 300.0


def _build(payload: dict) -> SyncConfig:
    """Build."""
    signal_mode = payload.get("signal_mode")
    if not isinstance(signal_mode, str) or signal_mode not in SIGNAL_MODES:
        raise ValueError(
            f"sync config: 'signal_mode' must be one of {SIGNAL_MODES}; got {signal_mode!r}."
        )

    anchor = payload.get("anchor_refinement")
    if not isinstance(anchor, dict):
        raise ValueError("sync config: missing 'anchor_refinement' object.")

    window = payload.get("window_refinement")
    if not isinstance(window, dict):
        raise ValueError("sync config: missing 'window_refinement' object.")

    signal_only = payload.get("signal_only")
    if not isinstance(signal_only, dict):
        raise ValueError("sync config: missing 'signal_only' object.")

    one_anchor_prior = payload.get("one_anchor_prior")
    if not isinstance(one_anchor_prior, dict):
        raise ValueError("sync config: missing 'one_anchor_prior' object.")

    config = SyncConfig(
        signal_mode=signal_mode,
        resample_rate_hz=float(payload.get("resample_rate_hz", 100.0)),
        min_valid_fraction=float(payload.get("min_valid_fraction", 0.5)),
        anchor_refinement=AnchorRefinementConfig(
            resample_rate_hz=float(anchor.get("resample_rate_hz", 100.0)),
            search_seconds=float(anchor.get("search_seconds", 1.0)),
        ),
        window_refinement=WindowRefinementConfig(
            window_seconds=float(window.get("window_seconds", 20.0)),
            step_seconds=float(window.get("step_seconds", 10.0)),
            local_search_seconds=float(window.get("local_search_seconds", 0.5)),
            min_window_score=float(window.get("min_window_score", 0.10)),
            min_fit_r2=float(window.get("min_fit_r2", 0.10)),
            min_fit_ppm=float(window.get("min_fit_ppm", -1000.0)),
            max_fit_ppm=float(window.get("max_fit_ppm", 1000.0)),
        ),
        signal_only_coarse_search_seconds=float(
            signal_only.get("coarse_search_seconds", 60.0)
        ),
        one_anchor_prior_drift_ppm=float(
            one_anchor_prior.get("drift_ppm", 300.0)
        ),
    )

    if config.resample_rate_hz <= 0:
        raise ValueError("sync config: 'resample_rate_hz' must be > 0.")
    if not (0 < config.min_valid_fraction <= 1):
        raise ValueError("sync config: 'min_valid_fraction' must be in (0, 1].")
    if config.anchor_refinement.resample_rate_hz <= 0:
        raise ValueError("anchor_refinement: 'resample_rate_hz' must be > 0.")
    if config.anchor_refinement.search_seconds <= 0:
        raise ValueError("anchor_refinement: 'search_seconds' must be > 0.")
    if config.window_refinement.window_seconds <= 0:
        raise ValueError("window_refinement: 'window_seconds' must be > 0.")
    if config.window_refinement.step_seconds <= 0:
        raise ValueError("window_refinement: 'step_seconds' must be > 0.")
    if config.window_refinement.local_search_seconds <= 0:
        raise ValueError("window_refinement: 'local_search_seconds' must be > 0.")
    if config.window_refinement.min_window_score < 0:
        raise ValueError("window_refinement: 'min_window_score' must be >= 0.")
    if config.window_refinement.min_fit_r2 < 0:
        raise ValueError("window_refinement: 'min_fit_r2' must be >= 0.")
    if config.signal_only_coarse_search_seconds <= 0:
        raise ValueError("signal_only: 'coarse_search_seconds' must be > 0.")
    if config.window_refinement.min_fit_ppm >= config.window_refinement.max_fit_ppm:
        raise ValueError(
            "window_refinement: 'min_fit_ppm' must be smaller than 'max_fit_ppm'."
        )
    return config


@lru_cache(maxsize=1)
def default_sync_config() -> SyncConfig:
    """Return the default sync config (cached from ``sync_args.json``)."""
    return _build(load_sync_config_data())
