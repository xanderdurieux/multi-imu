"""Parsers for raw IMU log files.

This package avoids eager imports so single submodules can still be executed
even when optional higher-level pipeline modules are unavailable in the current
checkout.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "parse_arduino_log",
    "parse_phone_recording",
    "parse_sporsa_log",
    "load_gps",
    "process_session",
    "write_recording_stats",
    "compute_recording_quality_summary",
    "find_calibration_segments",
    "split_recording",
]


_ATTR_TO_MODULE = {
    "parse_arduino_log": (".arduino", "parse_arduino_log"),
    "parse_phone_recording": (".phone", "parse_phone_recording"),
    "parse_sporsa_log": (".sporsa", "parse_sporsa_log"),
    "load_gps": (".gps", "load_gps"),
    "process_session": (".session", "process_session"),
    "write_recording_stats": (".stats", "write_recording_stats"),
    "compute_recording_quality_summary": (".stats", "compute_recording_quality_summary"),
    "find_calibration_segments": (".split_sections", "find_calibration_segments"),
    "split_recording": (".split_sections", "split_recording"),
}


def __getattr__(name: str) -> Any:
    """Resolve parser exports lazily so lightweight commands stay usable."""

    try:
        module_name, attr_name = _ATTR_TO_MODULE[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
