"""Orientation estimation for calibrated IMU data."""

from __future__ import annotations


def __getattr__(name):  # noqa: D401
    if name == "estimate_section":
        from .estimate import estimate_section
        return estimate_section
    if name == "validate":
        from .validate import validate
        return validate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["estimate_section", "validate"]
