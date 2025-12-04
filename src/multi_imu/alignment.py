"""Axis alignment utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .data_models import IMUSensorData


CALIBRATION_COLUMNS = ["ax", "ay", "az"]


def compute_alignment_matrix(reference: IMUSensorData, target: IMUSensorData, columns=CALIBRATION_COLUMNS) -> pd.DataFrame:
    """Compute a best-fit rotation matrix to align target axes to reference."""
    ref_vectors = reference.data[columns].to_numpy()
    tgt_vectors = target.data[columns].to_numpy()

    if len(ref_vectors) != len(tgt_vectors):
        min_len = min(len(ref_vectors), len(tgt_vectors))
        ref_vectors = ref_vectors[:min_len]
        tgt_vectors = tgt_vectors[:min_len]

    ref_mean = ref_vectors.mean(axis=0)
    tgt_mean = tgt_vectors.mean(axis=0)
    ref_centered = ref_vectors - ref_mean
    tgt_centered = tgt_vectors - tgt_mean

    h = tgt_centered.T @ ref_centered
    u, _, vt = np.linalg.svd(h)
    r = u @ vt

    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt

    return pd.DataFrame(r, index=columns, columns=columns)


def align_axes(target: IMUSensorData, alignment_matrix: pd.DataFrame) -> IMUSensorData:
    """Apply an alignment matrix to IMU axes."""
    df = target.data.copy()
    cols = [c for c in alignment_matrix.columns if c in df.columns]
    aligned_vectors = df[cols].to_numpy() @ alignment_matrix.loc[cols, cols].to_numpy().T
    df[cols] = aligned_vectors
    return IMUSensorData(name=f"{target.name}_aligned", data=df, sample_rate_hz=target.sample_rate_hz)


__all__ = ["compute_alignment_matrix", "align_axes"]
