"""Dual-IMU cycling event detection stage.

Detects bump, brake, swerve, disagreement, and fall events from derived
IMU signals produced by the ``derived/`` pipeline stage.

Usage
-----
From Python::

    from events import detect_events, process_section_events, process_recording_events
    from events import EventConfig, EventCandidate

CLI::

    python -m events <section_name>
    python -m events --recording <recording_name>
"""

from events.config import EventConfig
from events.detectors import EventCandidate, detect_events
from events.pipeline import process_section_events, process_recording_events

__all__ = [
    "EventConfig",
    "EventCandidate",
    "detect_events",
    "process_section_events",
    "process_recording_events",
]
