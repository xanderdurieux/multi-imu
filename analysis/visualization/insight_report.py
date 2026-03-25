"""Generate static + interactive insight figures from aggregated or section features.

Usage (from ``analysis/``):

  uv sync
  uv run python -m visualization.insight_report <session_date>

If the session has ``data/sessions/<date>/features_all.csv`` (from
``python -m features.aggregate``), outputs go to
``data/sessions/<date>/insight_report/static`` and ``.../interactive``.

You can also pass a path to any ``features.csv`` or ``features_all.csv``; outputs
are written next to that file under ``insight_report/``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from common import analysis_root, sessions_root

log = logging.getLogger(__name__)


def _resolve_features_csv(target: str) -> Path | None:
    t = target.strip()
    p = Path(t)
    if p.suffix.lower() == ".csv" and p.is_file():
        return p.resolve()
    session_csv = (sessions_root() / t / "features_all.csv").resolve()
    if session_csv.is_file():
        return session_csv
    return None


def _output_roots(features_csv: Path) -> tuple[Path, Path]:
    base = features_csv.parent / "insight_report"
    return base / "static", base / "interactive"


def run(target: str, *, section_dir: Path | None = None) -> None:
    csv_path = _resolve_features_csv(target)
    if csv_path is None:
        log.error(
            "No features CSV found for %r. "
            "Provide a path to features_all.csv / features.csv or a session name with aggregated features.",
            target,
        )
        return
    static_dir, interactive_dir = _output_roots(csv_path)
    df = pd.read_csv(csv_path)
    log.info("Loaded %d rows from %s", len(df), csv_path)

    from .insight_plots import write_static_insight_bundle
    from .interactive_insights import write_interactive_bundle, write_section_sensor_html

    write_static_insight_bundle(df, static_dir)
    log.info("Wrote static figures to %s", static_dir)
    write_interactive_bundle(df, interactive_dir)
    log.info("Wrote interactive HTML to %s", interactive_dir)

    if section_dir is not None and section_dir.is_dir():
        write_section_sensor_html(section_dir, interactive_dir / "sensors")
        log.info("Wrote section sensor HTML under %s", interactive_dir / "sensors")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    argv = [a for a in sys.argv[1:] if a]
    if not argv:
        root = analysis_root()
        sys.stderr.write(
            f"Usage: uv run python -m visualization.insight_report <session_or_csv_path>\n"
            f"Example: uv run python -m visualization.insight_report 2026-02-26\n"
            f"Working directory should allow imports (typically cd {root}).\n"
        )
        sys.exit(1)
    target = argv[0]
    section = None
    if len(argv) >= 3 and argv[1] == "--section":
        section = Path(argv[2])
    run(target, section_dir=section)


if __name__ == "__main__":
    main()
