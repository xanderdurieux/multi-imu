"""End-to-end pipelines built from reusable parsing, sync, and plotting modules."""

from .session_pipeline import run_session_pipeline

__all__ = [
    "run_session_pipeline",
]
