"""Compatibility entry point for sync plotting."""

from __future__ import annotations

import argparse

from sync.plotting import generate_sync_plots, plot_alignment, plot_method_comparison


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m visualization.plot_sync')
    parser.add_argument('recording_name')
    parser.add_argument('--no-comparison', action='store_true')
    parser.add_argument('--no-alignment', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    if args.no_comparison and args.no_alignment:
        return
    if not args.no_comparison and not args.no_alignment:
        generate_sync_plots(args.recording_name)
        return
    if not args.no_comparison:
        plot_method_comparison(args.recording_name)
    if not args.no_alignment:
        plot_alignment(args.recording_name)


if __name__ == '__main__':
    main()
