"""Compatibility shim: use ``python -m labels.event_labeler`` instead."""

from __future__ import annotations

from labels.event_labeler import main, write_event_labeler_html

__all__ = ["main", "write_event_labeler_html"]

if __name__ == "__main__":
    main()
