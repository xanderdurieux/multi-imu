"""Backward-compatible wrapper for :mod:`sync.sync_sda`."""

from .sync_sda import *  # noqa: F401,F403
from .sync_sda import synchronize_recording as synchronize_recording_sda
