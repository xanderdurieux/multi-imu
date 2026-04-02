"""Reusable IMU activity-signal construction for synchronization."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SIGNAL_MODE_ACC_NORM_DIFF = "acc_norm_diff"
SIGNAL_MODE_GYRO_NORM_DIFF = "gyro_norm_diff"
SIGNAL_MODE_ACC_GYRO_FUSED_DIFF = "acc_gyro_fused_diff"
SIGNAL_MODES: tuple[str, ...] = (
    SIGNAL_MODE_ACC_NORM_DIFF,
    SIGNAL_MODE_GYRO_NORM_DIFF,
    SIGNAL_MODE_ACC_GYRO_FUSED_DIFF,
)


def zscore(signal: np.ndarray) -> np.ndarray:
    """Return a finite z-scored copy of *signal*."""
    x = np.asarray(signal, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x, dtype=float)
    mu = float(np.nanmean(x[finite]))
    sigma = float(np.nanstd(x[finite]))
    if sigma < 1e-9:
        out = np.zeros_like(x, dtype=float)
        out[~finite] = 0.0
        return out
    out = (x - mu) / sigma
    out[~finite] = 0.0
    return out


def add_vector_norms(
    df: pd.DataFrame,
    *,
    vector_axes: dict[str, Iterable[str]],
) -> pd.DataFrame:
    """Add vector norms for the provided sensor axes."""
    out = df.copy()
    for name, axes in vector_axes.items():
        axes = list(axes)
        if all(col in out.columns for col in axes):
            arr = out[axes].to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                out[f"{name}_norm"] = np.sqrt(np.nansum(arr * arr, axis=1))
        else:
            out[f"{name}_norm"] = np.nan
    return out


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
            raise ValueError(f"Unknown signal_mode {signal_mode!r}; expected one of {SIGNAL_MODES}.")
        return signal_mode

    if differentiate and use_acc and not use_gyro and not use_mag:
        return SIGNAL_MODE_ACC_NORM_DIFF
    if differentiate and use_gyro and not use_acc and not use_mag:
        return SIGNAL_MODE_GYRO_NORM_DIFF
    if differentiate and use_acc and use_gyro and not use_mag:
        return SIGNAL_MODE_ACC_GYRO_FUSED_DIFF
    return None


def _differentiate(signal: np.ndarray) -> np.ndarray:
    if signal.size <= 1:
        return signal.copy()
    return np.diff(signal, prepend=signal[0])


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
    base = add_vector_norms(df, vector_axes=vector_axes)
    resolved_mode = resolve_signal_mode(
        signal_mode=signal_mode,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        differentiate=differentiate,
    )

    if resolved_mode == SIGNAL_MODE_ACC_NORM_DIFF:
        signal = _differentiate(zscore(base["acc_norm"].to_numpy(dtype=float)))
        return zscore(signal), resolved_mode

    if resolved_mode == SIGNAL_MODE_GYRO_NORM_DIFF:
        signal = _differentiate(zscore(base["gyro_norm"].to_numpy(dtype=float)))
        return zscore(signal), resolved_mode

    if resolved_mode == SIGNAL_MODE_ACC_GYRO_FUSED_DIFF:
        acc = _differentiate(zscore(base["acc_norm"].to_numpy(dtype=float)))
        gyro = _differentiate(zscore(base["gyro_norm"].to_numpy(dtype=float)))
        stacked = np.vstack([zscore(acc), zscore(gyro)])
        return zscore(np.nanmean(stacked, axis=0)), resolved_mode

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

    stacked = np.vstack([zscore(c) for c in components])
    signal = np.nanmean(stacked, axis=0)
    if differentiate:
        signal = _differentiate(signal)
    return zscore(signal), "legacy"
