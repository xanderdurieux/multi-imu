"""Backward-compatible wrapper for :mod:`sync.sync_lida`."""

from .sync_lida import *  # noqa: F401,F403
from .sync_lida import synchronize_pair as synchronize
