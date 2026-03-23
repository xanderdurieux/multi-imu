"""Run all synchronisation methods, compare them, and apply the best result."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common import recording_stage_dir, recordings_root

from .helpers import METHOD_LABELS, METHOD_ORDER, MethodRunResult, MethodSummary, SelectionResult, load_method_summary, stage_for_method
from .plotting import generate_sync_plots
from .sync_cal import synchronize_recording as synchronize_recording_cal
from .sync_lida import synchronize_recording as synchronize_recording_lida
from .sync_online import synchronize_recording as synchronize_recording_online
from .sync_sda import synchronize_recording as synchronize_recording_sda

METHOD_RUNNERS = {
    'sda': synchronize_recording_sda,
    'lida': synchronize_recording_lida,
    'cal': synchronize_recording_cal,
    'online': synchronize_recording_online,
}


@dataclass
class PipelineResult:
    recording_name: str
    method_results: list[MethodRunResult] = field(default_factory=list)
    summaries: dict[str, MethodSummary] = field(default_factory=dict)
    selection: SelectionResult | None = None

    @property
    def succeeded(self) -> list[str]:
        return [result.method for result in self.method_results if result.ok]

    @property
    def failed(self) -> list[str]:
        return [result.method for result in self.method_results if not result.ok]


def compare_methods(recording_name: str, methods: list[str] | None = None) -> dict[str, MethodSummary]:
    method_names = methods or list(METHOD_ORDER)
    return {method: load_method_summary(recording_name, method) for method in method_names}


def _selection_sort_key(summary: MethodSummary) -> tuple[float, int, float]:
    score = summary.corr_offset_and_drift
    score_key = float('-inf') if score is None else float(score)
    return (score_key, -METHOD_ORDER.index(summary.method), -(abs(summary.drift_ppm) if summary.drift_ppm is not None else 0.0))


def select_best_method(recording_name: str, methods: list[str] | None = None) -> SelectionResult:
    summaries = compare_methods(recording_name, methods)
    available = [summary for summary in summaries.values() if summary.available]
    if not available:
        raise ValueError(f'No synchronisation outputs available for {recording_name}.')
    best = max(available, key=_selection_sort_key)
    return SelectionResult(method=best.method, stage=best.stage, summary=best)


def apply_best_method(recording_name: str, selection: SelectionResult, summaries: dict[str, MethodSummary] | None = None) -> Path:
    source_dir = recording_stage_dir(recording_name, selection.stage)
    target_dir = recording_stage_dir(recording_name, 'synced')
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in source_dir.glob('*'):
        if file_path.is_file():
            shutil.copy2(file_path, target_dir / file_path.name)

    summary_map = summaries or compare_methods(recording_name)
    payload = {
        'recording_name': recording_name,
        'selected_method': selection.method,
        'selected_stage': selection.stage,
        'methods': {method: asdict(summary) for method, summary in summary_map.items()},
    }
    output_path = target_dir / 'all_methods.json'
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return output_path


def run_pipeline(
    recording_name: str,
    stage_in: str = 'parsed',
    *,
    methods: list[str] | None = None,
    apply: bool = True,
    plot: bool = False,
) -> PipelineResult:
    chosen_methods = methods or list(METHOD_ORDER)
    result = PipelineResult(recording_name=recording_name)

    for method in chosen_methods:
        runner = METHOD_RUNNERS[method]
        try:
            artifacts = runner(recording_name, stage_in)
            result.method_results.append(MethodRunResult(method=method, ok=True, artifacts=artifacts))
        except Exception as exc:
            result.method_results.append(MethodRunResult(method=method, ok=False, error=str(exc)))

    result.summaries = compare_methods(recording_name, chosen_methods)
    if any(summary.available for summary in result.summaries.values()):
        result.selection = select_best_method(recording_name, chosen_methods)
        if apply:
            apply_best_method(recording_name, result.selection, result.summaries)
        if plot:
            generate_sync_plots(recording_name)
    return result


def run_session(
    session_name: str,
    stage_in: str = 'parsed',
    *,
    methods: list[str] | None = None,
    apply: bool = True,
    plot: bool = False,
) -> list[PipelineResult]:
    recordings = sorted(
        path.name
        for path in recordings_root().iterdir()
        if path.is_dir() and path.name.startswith(f'{session_name}_')
    )
    return [
        run_pipeline(recording_name, stage_in, methods=methods, apply=apply, plot=plot)
        for recording_name in recordings
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m sync.pipeline')
    parser.add_argument('name', help='Recording name or session prefix.')
    parser.add_argument('--all', action='store_true', help='Treat name as a session prefix.')
    parser.add_argument('--stage', default='parsed')
    parser.add_argument('--methods', nargs='*', choices=METHOD_ORDER, default=None)
    parser.add_argument('--no-apply', action='store_true')
    parser.add_argument('--plot', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.all:
        results = run_session(args.name, args.stage, methods=args.methods, apply=not args.no_apply, plot=args.plot)
        for result in results:
            selected = result.selection.method if result.selection else 'none'
            print(f"{result.recording_name}: ok={result.succeeded} failed={result.failed} selected={selected}")
        return

    result = run_pipeline(args.name, args.stage, methods=args.methods, apply=not args.no_apply, plot=args.plot)
    selected = result.selection.method if result.selection else 'none'
    print(f"{result.recording_name}: ok={result.succeeded} failed={result.failed} selected={selected}")


if __name__ == '__main__':
    main()
