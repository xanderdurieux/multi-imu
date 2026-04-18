"""Reusable IMU activity-signal construction for synchronization."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from common.signals import add_vector_norms, first_difference, zscore_finite

SIGNAL_MODE_ACC_NORM_DIFF = "acc_norm_diff"
SIGNAL_MODE_GYRO_NORM_DIFF = "gyro_norm_diff"
SIGNAL_MODE_ACC_GYRO_FUSED_DIFF = "acc_gyro_fused_diff"
SIGNAL_MODES: tuple[str, ...] = (
    SIGNAL_MODE_ACC_NORM_DIFF,
    SIGNAL_MODE_GYRO_NORM_DIFF,
    SIGNAL_MODE_ACC_GYRO_FUSED_DIFF,
)


def resolve_signal_mode(
    *,
    signal_mode: str | None,
    use_acc: bool,
    use_gyro: bool,
    use_mag: bool,
    differentiate: bool,
) -> str | None:
    """Resolve the effective signal mode from explicit or legacy options."""
    if signal_mode is not None:
        if signal_mode not in SIGNAL_MODES:
            raise ValueError(
                f"Unknown signal_mode {signal_mode!r}; expected one of {SIGNAL_MODES}."
            )
        return signal_mode

    if differentiate and use_acc and not use_gyro and not use_mag:
        return SIGNAL_MODE_ACC_NORM_DIFF
    if differentiate and use_gyro and not use_acc and not use_mag:
        return SIGNAL_MODE_GYRO_NORM_DIFF
    if differentiate and use_acc and use_gyro and not use_mag:
        return SIGNAL_MODE_ACC_GYRO_FUSED_DIFF
    return None


def _norm_diff(signal: np.ndarray) -> np.ndarray:
    return zscore_finite(first_difference(zscore_finite(signal)))


def build_activity_signal(
    df: pd.DataFrame,
    *,
    vector_axes: dict[str, Iterable[str]],
    signal_mode: str | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    differentiate: bool = True,
) -> tuple[np.ndarray, str]:
    """Build a 1D alignment signal and return it with the resolved mode label."""
    base = add_vector_norms(df, vector_axes)
    resolved_mode = resolve_signal_mode(
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )

    if resolved_mode == SIGNAL_MODE_ACC_NORM_DIFF:
        return _norm_diff(base["acc_norm"].to_numpy(dtype=float)), resolved_mode

    if resolved_mode == SIGNAL_MODE_GYRO_NORM_DIFF:
        return _norm_diff(base["gyro_norm"].to_numpy(dtype=float)), resolved_mode

    if resolved_mode == SIGNAL_MODE_ACC_GYRO_FUSED_DIFF:
        acc = _norm_diff(base["acc_norm"].to_numpy(dtype=float))
        gyro = _norm_diff(base["gyro_norm"].to_numpy(dtype=float))
        return zscore_finite(np.nanmean(np.vstack([acc, gyro]), axis=0)), resolved_mode

    components: list[np.ndarray] = []

    def _append_if_selected(flag: bool, name: str) -> None:
        if not flag:
            return
        col = f"{name}_norm"
        if col not in base.columns:
            return
        values = base[col].to_numpy(dtype=float)
        if np.isfinite(values).any():
            components.append(values)

    _append_if_selected(use_acc, "acc")
    _append_if_selected(use_gyro, "gyro")
    _append_if_selected(use_mag, "mag")

    if not components:
        raise ValueError("No valid activity channels selected for alignment.")

    signal = np.nanmean(np.vstack([zscore_finite(c) for c in components]), axis=0)
    if differentiate:
        signal = first_difference(signal)
    return zscore_finite(signal), "legacy"
