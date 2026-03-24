"""Ride-level world-frame calibration for multi-IMU sections."""

from __future__ import annotations


def __getattr__(name):  # noqa: D401
    if name == "calibrate_section":
        from .calibrate import calibrate_section
        return calibrate_section
    if name == "validate":
        from .validate import validate
        return validate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["calibrate_section", "validate"]
