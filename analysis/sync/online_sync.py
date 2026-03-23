"""Compatibility wrapper for :mod:`sync.sync_online`."""

from .sync_online import *  # noqa: F401,F403
from .sync_online import main


if __name__ == '__main__':
    main()
