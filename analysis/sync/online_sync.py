"""Backward-compatible wrapper for :mod:`sync.sync_online`."""

from .sync_online import *  # noqa: F401,F403
from .sync_online import synchronize_recording as synchronize_recording_online
