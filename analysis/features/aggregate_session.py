"""Session-level aggregation entry point: uv run -m features.aggregate_session."""
from .aggregate import aggregate_session

if __name__ == "__main__":
    import sys
    import logging
    if len(sys.argv) < 2:
        print("Usage: uv run -m features.aggregate_session <session_name>")
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    aggregate_session(sys.argv[1])
