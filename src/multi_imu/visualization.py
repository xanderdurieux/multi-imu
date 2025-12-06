"""Matplotlib-based visualization helpers for IMU streams."""
from __future__ import annotations
from typing import Iterable, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

from .data_models import IMUSensorData


def _prepare_axes(num_series: int) -> List:
    fig, axes = plt.subplots(num_series, 1, figsize=(10, 3 * num_series), sharex=True)
    if num_series == 1:
        axes = [axes]
    return axes


def _prepare_axes_grid(num_rows: int, num_cols: int) -> List:
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6 * num_cols, 3 * num_rows), sharex='col')
    if num_rows == 1 and num_cols == 1:
        axes = [axes]
    return axes


def plot_timeseries(stream: IMUSensorData, columns: Iterable[str], title: Optional[str] = None):
    axes = _prepare_axes(len(list(columns)))
    for ax, col in zip(axes, columns):
        ax.plot(stream.data["timestamp"], stream.data[col], label=f"{stream.name} {col}")
        ax.set_ylabel(col)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    if title:
        axes[0].set_title(title)
    return axes


def plot_comparison(reference: IMUSensorData, target: IMUSensorData, rows: Iterable[str]):
    axes = _prepare_axes(len(list(rows)))
    for ax, row_name in zip(axes, rows):
        ax.plot(reference.data["timestamp"], reference.data[row_name], label=f"{reference.name} {row_name}")
        if row_name in target.data:
            ax.plot(target.data["timestamp"], target.data[row_name], label=f"{target.name} {row_name}")
        ax.set_ylabel(row_name)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Sensor comparison")
    return axes


def plot_comparison_grid(references: List[IMUSensorData], targets: List[IMUSensorData], rows: Iterable[str], types: Iterable[str]):
    axes = _prepare_axes_grid(len(list(rows)), len(list(types)))
    for col, (ref, tgt) in enumerate(zip(references, targets)):
        for row, r in enumerate(rows):
            ax = axes[row][col]
            row_name = f"{types[col]}{r}"
            ax.plot(ref.data["timestamp"], ref.data[row_name], label=f"{ref.name} {row_name}")
            if row_name in tgt.data:
                ax.plot(tgt.data["timestamp"], tgt.data[row_name], label=f"{tgt.name} {row_name}")
            ax.set_ylabel(row_name)
            ax.legend()
        axes[-1][col].set_xlabel("Time (s)")
    return axes


def plot_magnitude(stream: IMUSensorData, columns=("ax", "ay", "az"), label: str = "|a|"):
    axes = _prepare_axes(1)
    df = stream.data
    mag = (df[list(columns)] ** 2).sum(axis=1) ** 0.5
    axes[0].plot(df["timestamp"], mag, label=label)
    axes[0].set_ylabel(label)
    axes[0].legend()
    axes[0].set_xlabel("Time (s)")
    axes[0].set_title(f"{stream.name} magnitude")
    return axes


__all__ = ["plot_timeseries", "plot_comparison", "plot_magnitude"]
