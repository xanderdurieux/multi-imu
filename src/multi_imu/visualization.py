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


def plot_comparison(reference: IMUSensorData, target: IMUSensorData, columns: Iterable[str]):
    axes = _prepare_axes(len(list(columns)))
    for ax, col in zip(axes, columns):
        ax.plot(reference.data["timestamp"], reference.data[col], label=f"{reference.name} {col}")
        if col in target.data:
            ax.plot(target.data["timestamp"], target.data[col], label=f"{target.name} {col}")
        ax.set_ylabel(col)
        ax.legend()
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Sensor comparison")
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


def plot_event_annotations(stream: IMUSensorData, events: List[dict], column: str = "ax"):
    axes = _prepare_axes(1)
    axes[0].plot(stream.data["timestamp"], stream.data[column], label=f"{stream.name} {column}")
    for event in events:
        axes[0].axvline(event["timestamp"], color="r", linestyle="--", label=event["label"])
    axes[0].legend()
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(column)
    axes[0].set_title("Event annotations")
    return axes


__all__ = ["plot_timeseries", "plot_comparison", "plot_magnitude", "plot_event_annotations"]
