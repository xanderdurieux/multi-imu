"""Run all sync methods on a recording or session, then select the best result."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from typing import Optional

from common import recordings_root

from .calibration_sync import synchronize_recording_from_calibration
from .lida_sync import synchronize_recording
from .online_sync import synchronize_recording_online
from .sda_sync import synchronize_recording_sda
from .selection import (
    METHOD_LABELS,
    SyncSelectionResult,
    apply_selection,
    compare_sync_models,
    plot_sync_comparison,
    print_comparison,
    print_selection_result,
    select_best_sync_method,
)

log = logging.getLogger(__name__)


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
    """Run sync methods on one recording, then select the best result."""
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


def synchronize_session(
    session_name: str,
    stage_in: str = "parsed",
    *,
    methods: list[str] | None = None,
    apply: bool = True,
    plot: bool = False,
) -> list[RecordingResult]:
    """Run sync methods on every recording in a session."""
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


def run_selection_only(
    recording_name: str,
    *,
    apply: bool = False,
    plot: bool = False,
) -> None:
    """Compare existing method outputs and optionally apply the winner."""
    cmp = compare_sync_models(recording_name)
    print_comparison(cmp)
    result = select_best_sync_method(recording_name)
    print()
    print_selection_result(result)
    if apply:
        apply_selection(recording_name, result, plot=plot)
    elif plot:
        plot_sync_comparison(recording_name)


def run_selection_session(
    session_prefix: str,
    *,
    apply: bool = False,
    plot: bool = False,
) -> None:
    """Run selection for every recording in a session."""
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and d.name.startswith(f"{session_prefix}_")
    )
    if not recordings:
        log.warning("No recordings found matching prefix '%s_'.", session_prefix)
        return

    for rec in recordings:
        print(f"\n=== {rec} ===")
        run_selection_only(rec, apply=apply, plot=plot)
