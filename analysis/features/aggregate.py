"""Session-level aggregation of features."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common import sessions_root
from common.paths import sections_root, parse_section_folder_name

log = logging.getLogger(__name__)


def aggregate_session(session_name: str) -> Path:
    """Concatenate all features.csv across recordings and sections.

    Writes data/sessions/<session_name>/features_all.csv
    """
    session_name = session_name.strip().rstrip("/")
    out_dir = sessions_root() / session_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features_all.csv"

    sec_root = sections_root()
    prefix = session_name + "_r"
    all_dfs = []

    if not sec_root.exists():
        log.warning("Sections root missing: %s", sec_root)
        empty = pd.DataFrame()
        empty.to_csv(out_path, index=False)
        return out_path

    for sec_dir in sorted(sec_root.iterdir()):
        if not sec_dir.is_dir():
            continue
        try:
            rec_name, _sec_idx = parse_section_folder_name(sec_dir.name)
        except ValueError:
            continue
        if not rec_name.startswith(prefix):
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
        df["recording"] = rec_name
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
