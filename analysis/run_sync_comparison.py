"""Run sync method comparison for all sessions and save results.

Compares SDA+LIDA (synced_lida) vs calibration-based (synced_cal) for every
recording that has at least one synced stage, prints the side-by-side table,
saves a comparison PNG per recording, and writes a summary CSV.

Usage::

    uv run run_sync_comparison.py
    uv run run_sync_comparison.py --sessions 2026-02-26 2025-12-04
    uv run run_sync_comparison.py --plot
"""

from __future__ import annotations

import csv
from pathlib import Path

from common import recordings_root
from sync.compare_sync import compare_sync_models, plot_sync_comparison, print_comparison

SESSIONS = ["2026-02-26", "2025-12-04"]

SUMMARY_CSV = Path(__file__).parent / "data" / "sync_comparison_summary.csv"

SUMMARY_FIELDS = [
    "recording",
    "lida_offset_s",
    "cal_offset_s",
    "delta_offset_s",
    "lida_drift_ppm",
    "cal_drift_ppm",
    "delta_drift_ppm",
    "lida_corr_offset_only",
    "lida_corr_final",
    "cal_corr_offset_only",
    "cal_corr_final",
    "cal_span_s",
    "cal_score_opening",
    "cal_score_closing",
]


def _extract_row(result: dict) -> dict:
    lida = result["lida"]
    cal = result["cal"]

    def _corr(info, key):
        if info is None:
            return ""
        return info.get("correlation", {}).get(key, "")

    def _cal_detail(key, sub=None):
        if cal is None or "calibration" not in cal:
            return ""
        block = cal["calibration"]
        if sub:
            return block.get(sub, {}).get(key, "")
        return block.get(key, "")

    return {
        "recording": result["recording"],
        "lida_offset_s": lida["offset_seconds"] if lida else "",
        "cal_offset_s": cal["offset_seconds"] if cal else "",
        "delta_offset_s": result["delta_offset_s"] if result["delta_offset_s"] is not None else "",
        "lida_drift_ppm": lida["drift_seconds_per_second"] * 1e6 if lida else "",
        "cal_drift_ppm": cal["drift_seconds_per_second"] * 1e6 if cal else "",
        "delta_drift_ppm": result["delta_drift_ppm"] if result["delta_drift_ppm"] is not None else "",
        "lida_corr_offset_only": _corr(lida, "offset_only"),
        "lida_corr_final": _corr(lida, "offset_and_drift"),
        "cal_corr_offset_only": _corr(cal, "offset_only"),
        "cal_corr_final": _corr(cal, "offset_and_drift"),
        "cal_span_s": _cal_detail("calibration_span_s"),
        "cal_score_opening": _cal_detail("score", "opening"),
        "cal_score_closing": _cal_detail("score", "closing"),
    }


def run(sessions: list[str], plot: bool = False) -> None:
    root = recordings_root()
    recordings = []
    for session in sessions:
        recordings.extend(
            sorted(d.name for d in root.iterdir() if d.is_dir() and d.name.startswith(f"{session}_"))
        )

    rows = []
    for rec in recordings:
        result = compare_sync_models(rec)
        if result["lida"] is None and result["cal"] is None:
            print(f"\n[{rec}] No sync data found — skipping.")
            continue
        print_comparison(result)
        if plot:
            plot_sync_comparison(rec)
        rows.append(_extract_row(result))

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written → {SUMMARY_CSV}")


if __name__ == "__main__":
    import sys

    sessions = SESSIONS
    plot = False

    args = sys.argv[1:]
    if "--plot" in args:
        plot = True
        args.remove("--plot")
    if "--sessions" in args:
        idx = args.index("--sessions")
        sessions = args[idx + 1 :]

    run(sessions, plot=plot)
