"""Session-level synchronization: run all methods on every recording.

This module orchestrates the full sync pipeline for an entire session (or a
single recording):

1. Run each of the four sync methods (SDA, LIDA, calibration, online) in
   order, capturing errors individually so one failing method never aborts
   the others.
2. Run :func:`~sync.selection.select_best_sync_method` and optionally copy
   the winner to ``synced/``.
3. Print a per-recording summary table.

CLI::

    # Single recording — run all methods, select best, copy to synced/
    python -m sync.session 2026-02-26_5 --apply --plot

    # Whole session — run all recordings
    python -m sync.session 2026-02-26 --all --apply

    # Only run specific methods for a session
    python -m sync.session 2026-02-26 --all --methods calibration lida
"""

from __future__ import annotations

import argparse
import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional

from common import recordings_root

from .sda_sync import synchronize_recording_sda
from .lida_sync import synchronize_recording
from .calibration_sync import synchronize_recording_from_calibration
from .online_sync import synchronize_recording_online
from .selection import (
    METHOD_LABELS,
    SyncSelectionResult,
    apply_selection,
    select_best_sync_method,
    print_comparison,
    compare_sync_models,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Outcome of running one sync method on one recording."""
    method: str
    ok: bool
    error: Optional[str] = None


@dataclass
class RecordingResult:
    """Outcome of running all requested methods on one recording."""
    recording_name: str
    method_results: list[MethodResult] = field(default_factory=list)
    selection: Optional[SyncSelectionResult] = None
    selection_error: Optional[str] = None

    @property
    def succeeded(self) -> list[str]:
        return [r.method for r in self.method_results if r.ok]

    @property
    def failed(self) -> list[str]:
        return [r.method for r in self.method_results if not r.ok]


# ---------------------------------------------------------------------------
# Per-recording orchestration
# ---------------------------------------------------------------------------

_METHOD_RUNNERS = {
    "sda": lambda rec, stage, **kw: synchronize_recording_sda(rec, stage, plot=kw.get("plot", False)),
    "lida": lambda rec, stage, **kw: synchronize_recording(rec, stage, plot=kw.get("plot", False)),
    "calibration": lambda rec, stage, **kw: synchronize_recording_from_calibration(
        rec, stage, plot=kw.get("plot", False)
    ),
    "online": lambda rec, stage, **kw: synchronize_recording_online(rec, stage, plot=kw.get("plot", False)),
}


def synchronize_recording_all_methods(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    methods: list[str] | None = None,
    apply: bool = True,
    plot: bool = False,
) -> RecordingResult:
    """Run sync methods on one recording, then select the best result.

    Parameters
    ----------
    recording_name:
        Recording identifier, e.g. ``"2026-02-26_5"``.
    stage_in:
        Input stage directory (default: ``"parsed"``).
    methods:
        Subset of ``["sda", "lida", "calibration", "online"]`` to run.
        Defaults to all four.
    apply:
        If ``True``, copy the selected method's outputs to ``synced/`` via
        :func:`~sync.selection.apply_selection`.
    plot:
        If ``True``, generate plots for each method and the comparison.

    Returns
    -------
    RecordingResult
        Per-method success/failure flags and the selection outcome.
    """
    if methods is None:
        methods = ["sda", "lida", "calibration", "online"]

    result = RecordingResult(recording_name=recording_name)
    print(f"\n{'━' * 60}")
    print(f"  {recording_name}")
    print(f"{'━' * 60}")

    for method in methods:
        label = METHOD_LABELS.get(method, method)
        runner = _METHOD_RUNNERS.get(method)
        if runner is None:
            log.warning("Unknown sync method '%s' — skipping.", method)
            continue

        print(f"  [{label}] running …", end="", flush=True)
        try:
            runner(recording_name, stage_in, plot=plot)
            result.method_results.append(MethodResult(method=method, ok=True))
            print("  done")
        except Exception as exc:
            err = str(exc)
            result.method_results.append(MethodResult(method=method, ok=False, error=err))
            print(f"  FAILED: {err}")
            log.debug("Traceback for %s/%s:\n%s", recording_name, method, traceback.format_exc())

    if not result.succeeded:
        print(f"  No method succeeded for {recording_name} — skipping selection.")
        return result

    # Compare and select
    try:
        cmp = compare_sync_models(recording_name)
        print()
        print_comparison(cmp)
        sel = select_best_sync_method(recording_name)
        result.selection = sel
        print(f"\n  Selected: {sel.method} (stage: {sel.stage})")
        if apply:
            apply_selection(recording_name, sel, plot=plot)
    except Exception as exc:
        result.selection_error = str(exc)
        log.warning("Selection failed for %s: %s", recording_name, exc)

    return result


# ---------------------------------------------------------------------------
# Session-level orchestration
# ---------------------------------------------------------------------------

def synchronize_session(
    session_name: str,
    stage_in: str = "parsed",
    *,
    methods: list[str] | None = None,
    apply: bool = True,
    plot: bool = False,
) -> list[RecordingResult]:
    """Run sync methods on every recording in a session.

    Recordings are discovered by matching directories whose name starts with
    ``"<session_name>_"`` under ``data/recordings/``.

    Parameters
    ----------
    session_name:
        Session date prefix, e.g. ``"2026-02-26"``.
    stage_in:
        Input stage directory for each recording (default: ``"parsed"``).
    methods:
        Sync methods to run (default: all four).
    apply:
        Copy the selected method's outputs to ``synced/``.
    plot:
        Generate plots for each method and the comparison.

    Returns
    -------
    list[RecordingResult]
        One result per recording, in alphabetical order.
    """
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith(f"{session_name}_")
    )

    if not recordings:
        log.warning("No recordings found matching prefix '%s_'.", session_name)
        return []

    print(f"Session '{session_name}': {len(recordings)} recording(s) found.")

    results: list[RecordingResult] = []
    for rec in recordings:
        r = synchronize_recording_all_methods(
            rec,
            stage_in,
            methods=methods,
            apply=apply,
            plot=plot,
        )
        results.append(r)

    _print_session_summary(session_name, results)
    return results


def _print_session_summary(session_name: str, results: list[RecordingResult]) -> None:
    """Print a compact per-recording outcome table."""
    print(f"\n{'━' * 60}")
    print(f"  Session summary: {session_name}")
    print(f"{'━' * 60}")
    for r in results:
        ok_str = ", ".join(METHOD_LABELS.get(m, m) for m in r.succeeded) or "none"
        fail_str = ", ".join(METHOD_LABELS.get(m, m) for m in r.failed)
        sel_str = f"→ {r.selection.method}" if r.selection else "no selection"
        line = f"  {r.recording_name:<22}  ok=[{ok_str}]  {sel_str}"
        if fail_str:
            line += f"  failed=[{fail_str}]"
        print(line)
    print(f"{'━' * 60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.session",
        description=(
            "Run all (or selected) sync methods on a recording or an entire "
            "session, then select the best result and optionally write it to "
            "synced/."
        ),
    )
    parser.add_argument(
        "name",
        help=(
            "Recording name (e.g. '2026-02-26_5') or session prefix when "
            "--all is used (e.g. '2026-02-26')."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_recordings",
        help="Process all recordings whose name starts with NAME followed by '_'.",
    )
    parser.add_argument(
        "--stage",
        default="parsed",
        metavar="STAGE",
        help="Input stage directory (default: parsed).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["sda", "lida", "calibration", "online"],
        default=None,
        metavar="METHOD",
        help=(
            "Sync methods to run (default: all four). "
            "Choices: sda lida calibration online."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Copy the selected method's outputs to synced/.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots for each method and the comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = _build_arg_parser().parse_args(argv)

    if args.all_recordings:
        synchronize_session(
            session_name=args.name,
            stage_in=args.stage,
            methods=args.methods,
            apply=args.apply,
            plot=args.plot,
        )
    else:
        synchronize_recording_all_methods(
            recording_name=args.name,
            stage_in=args.stage,
            methods=args.methods,
            apply=args.apply,
            plot=args.plot,
        )


if __name__ == "__main__":
    main()
