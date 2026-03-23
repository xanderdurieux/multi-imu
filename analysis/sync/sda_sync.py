"""Compatibility wrapper for :mod:`sync.sync_sda`."""

from .sync_sda import *  # noqa: F401,F403
from .sync_sda import main


if __name__ == '__main__':
    main()
