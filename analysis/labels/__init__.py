"""Scenario labeling utilities for recordings, sections, intervals, and events.

Interactive IMU + GPS labeling HTML:
``python -m labels.event_labeler <recording/stage>``.

Lightweight annotation workflow helpers:
``python -m labels.workflow --help``.
"""

from .parser import PROVENANCE_FIELDS, LabelIndex, load_labels_from_path, warn_unlabeled_windows

__all__ = [
    "PROVENANCE_FIELDS",
    "LabelIndex",
    "load_labels_from_path",
    "warn_unlabeled_windows",
]
