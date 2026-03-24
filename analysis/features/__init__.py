"""Feature extraction for incident detection."""

from __future__ import annotations


def __getattr__(name):  # noqa: D401
    if name == "extract_section":
        from .extract import extract_section
        return extract_section
    if name == "aggregate_session":
        from .aggregate import aggregate_session
        return aggregate_session
    if name == "validate":
        from .validate import validate
        return validate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["extract_section", "aggregate_session", "validate"]
