"""Session-level aggregation of features."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common import recordings_root, sessions_root

log = logging.getLogger(__name__)


def aggregate_session(session_name: str) -> Path:
    """Concatenate all features.csv across recordings and sections.

    Writes data/sessions/<session_name>/features_all.csv
    """
    session_name = session_name.strip().rstrip("/")
    out_dir = sessions_root() / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_all.csv"

    rec_root = recordings_root()
    prefix = session_name + "_"
    all_dfs = []

    for rec_dir in sorted(rec_root.iterdir()):
        if not rec_dir.is_dir() or not rec_dir.name.startswith(prefix):
            continue
        sections_dir = rec_dir / "sections"
        if not sections_dir.exists():
            continue
        for sec_dir in sorted(sections_dir.iterdir()):
            if not sec_dir.is_dir() or not sec_dir.name.startswith("section_"):
                continue
            feat_csv = sec_dir / "features" / "features.csv"
            if not feat_csv.exists():
                continue
            try:
                df = pd.read_csv(feat_csv)
            except pd.errors.EmptyDataError:
                continue
            if df.empty:
                continue
            df["recording"] = rec_dir.name
            all_dfs.append(df)

    if not all_dfs:
        log.warning("No features found for session %s", session_name)
        empty = pd.DataFrame()
        empty.to_csv(out_path, index=False)
        return out_path

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(combined))
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: uv run -m features.aggregate <session_name>")
        sys.exit(1)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    aggregate_session(sys.argv[1])
