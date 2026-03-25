"""Run every visualization entrypoint across all recordings under ``data/recordings``.

``plot_session.plot_recording`` already covers sensor, comparison, timing, sync,
calibration (recording-level), orientation, and section stages. This module additionally
runs protocol / verify / insight plots and optional session insight reports.

Usage (from ``analysis/``)::

    uv run python -m visualization.run_all_recordings
"""

from __future__ import annotations

import logging
import sys

from common import recordings_root, sessions_root

log = logging.getLogger(__name__)


def _all_recording_names() -> list[str]:
    root = recordings_root()
    if not root.is_dir():
        log.error("Recordings root missing: %s", root)
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def _session_prefixes(recording_names: list[str]) -> list[str]:
    """Derive session date strings like ``2026-02-26`` from ``2026-02-26_5``."""
    seen: set[str] = set()
    for n in recording_names:
        if "_" in n:
            head, tail = n.rsplit("_", 1)
            if tail.isdigit():
                seen.add(head)
            else:
                seen.add(n)
        else:
            seen.add(n)
    return sorted(seen)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    names = _all_recording_names()
    if not names:
        log.warning("No recording directories found.")
        return

    log.info("Found %d recording(s).", len(names))

    from visualization.plot_session import plot_recording

    for rec in names:
        try:
            log.info("plot_recording %s", rec)
            plot_recording(rec)
        except Exception:
            log.exception("plot_recording failed for %s", rec)

    from common import recording_stage_dir
    from visualization.plot_orientation_verify import plot_orientation_verify_stage

    log.info("plot_orientation_verify (%d recordings)", len(names))
    for rec in names:
        stage_dir = recording_stage_dir(rec, "orientation")
        if not stage_dir.is_dir():
            log.debug("Skip plot_orientation_verify %s (no orientation stage)", rec)
            continue
        try:
            plot_orientation_verify_stage(rec, "orientation")
        except Exception:
            log.exception("plot_orientation_verify failed for %s", rec)

    from visualization.plot_orientation_protocol import main as protocol_main

    try:
        log.info("plot_orientation_protocol (%d recordings)", len(names))
        protocol_main(names)
    except Exception:
        log.exception("plot_orientation_protocol failed")

    from visualization.plot_orientation_insight import plot_insight_stage

    _fall = frozenset({"9", "10"})
    _head = frozenset({"4", "5"})
    for rec in names:
        suffix = rec.split("_")[-1]
        try:
            log.info("plot_orientation_insight %s", rec)
            plot_insight_stage(
                rec,
                "orientation",
                fall=suffix in _fall,
                head_movements=suffix in _head,
            )
        except Exception:
            log.exception("plot_orientation_insight failed for %s", rec)

    from visualization.insight_report import run as insight_report_run

    for session in _session_prefixes(names):
        feat = (sessions_root() / session / "features_all.csv").resolve()
        if feat.is_file():
            log.info("insight_report %s", session)
            try:
                insight_report_run(session)
            except Exception:
                log.exception("insight_report failed for session %s", session)
        else:
            log.debug("Skip insight_report %s (no %s)", session, feat)


if __name__ == "__main__":
    main()
    sys.exit(0)
