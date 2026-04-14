"""Visualization utilities for IMU pipeline stages."""

from . import (
    plot_calibration,
    plot_comparison,
    plot_features,
    plot_orientation,
    plot_sensor,
    plot_sync,
)

__all__ = [
    "plot_calibration",
    "plot_comparison",
    "plot_features",
    "plot_orientation",
    "plot_sensor",
    "plot_sync",
    "thesis_plots",
]


def __getattr__(name: str):
    if name == "thesis_plots":
        from . import thesis_plots as _thesis_plots

        return _thesis_plots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
