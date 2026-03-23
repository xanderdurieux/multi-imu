"""Compatibility wrapper for :mod:`sync.sync_cal`."""

from .sync_cal import *  # noqa: F401,F403
from .sync_cal import main


if __name__ == '__main__':
    main()
