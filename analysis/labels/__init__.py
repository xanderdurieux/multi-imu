"""Scenario labels for recordings, sections, and time intervals.

For scaffold CSV generation and timing tables see ``labels.scaffold`` (run as
``python -m labels.scaffold``) to avoid import cycles with ``python -m labels.scaffold``.

Interactive IMU + GPS labeling HTML: ``python -m labels.event_labeler <recording/stage>``.
"""

from .parser import LabelIndex, load_labels_from_path, warn_unlabeled_windows

__all__ = [
    "LabelIndex",
    "load_labels_from_path",
    "warn_unlabeled_windows",
]
