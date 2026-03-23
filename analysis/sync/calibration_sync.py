"""Backward-compatible wrapper for :mod:`sync.sync_cal`."""

from .sync_cal import *  # noqa: F401,F403
from .sync_cal import synchronize_pair as synchronize_from_calibration
from .sync_cal import synchronize_recording as synchronize_recording_from_calibration
