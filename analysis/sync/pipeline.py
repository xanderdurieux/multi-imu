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
    print_comparison,
    select_best_sync_method,
)

log = logging.getLogger(__name__)

_STAGE_IN = "parsed"


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
    "sda": lambda rec, stage: synchronize_recording_sda(rec, stage),
    "lida": lambda rec, stage: synchronize_recording(rec, stage),
    "calibration": lambda rec, stage: synchronize_recording_from_calibration(rec, stage),
    "online": lambda rec, stage: synchronize_recording_online(rec, stage),
}


def synchronize_recording_all_methods(recording_name: str) -> RecordingResult:
    """Run all four sync methods on *recording_name* ``parsed/`` data, then select and apply."""
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
            runner(recording_name, _STAGE_IN)
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
        apply_selection(recording_name, sel)
    except Exception as exc:
        result.selection_error = str(exc)
        log.warning("Selection failed for %s: %s", recording_name, exc)

    return result


def synchronize_session(session_name: str) -> list[RecordingResult]:
    """Run :func:`synchronize_recording_all_methods` for every ``session_name_*`` recording."""
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
        r = synchronize_recording_all_methods(rec)
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
