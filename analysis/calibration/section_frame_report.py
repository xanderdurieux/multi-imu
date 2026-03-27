"""Section-frame validation helper for representative sections.

Usage (from ``analysis/``):

    uv run python -m calibration.section_frame_report \
        data/sections/2026-02-26_r2s1 \
        data/sections/2026-02-26_r3s4

This script re-runs calibration with ``section_horizontal_frame`` for provided
sections, then exports a compact CSV/JSON summary of section-frame confidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibration.calibrate import calibrate_section
from common.paths import sections_root


def _resolve_section_arg(arg: str) -> Path:
    p = Path(arg.strip())
    if p.is_absolute():
        return p
    # Accept direct relative paths (e.g. data/sections/...) and bare folder names.
    if p.exists():
        return p.resolve()
    return (sections_root() / p).resolve()


def run_report(section_args: list[str], *, force_recalibrate: bool = True) -> dict[str, Any]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"sections": []}

    for raw in section_args:
        sec = _resolve_section_arg(raw)
        if not sec.exists():
            payload["sections"].append({"section": str(sec), "error": "not_found"})
            continue

        if force_recalibrate:
            calibrate_section(
                sec,
                frame_alignment="section_horizontal_frame",
                write_plots=True,
            )

        cal_path = sec / "calibrated" / "calibration.json"
        if not cal_path.is_file():
            payload["sections"].append({"section": str(sec), "error": "missing_calibration_json"})
            continue
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        section_block = {"section": str(sec), "sensors": {}}
        for sensor, meta in cal.items():
            ff = meta.get("forward_frame_meta") or {}
            row = {
                "section": str(sec),
                "sensor": sensor,
                "frame_alignment": meta.get("frame_alignment", ""),
                "calibration_quality": meta.get("calibration_quality", ""),
                "gravity_residual_m_per_s2": meta.get("gravity_residual_m_per_s2", None),
                "frame_confidence": ff.get("confidence_score", None),
                "heading_stability": ff.get("heading_stability", None),
                "axis_reliability": ff.get("horizontal_axis_reliability", None),
                "magnetometer_reliability": ff.get("magnetometer_reliability", None),
                "magnetometer_used": ff.get("magnetometer_used", False),
                "straight_motion_confidence": ff.get("straight_motion_confidence", None),
                "fallback": ff.get("fallback", False),
                "fallback_reason": ff.get("fallback_reason", ""),
            }
            rows.append(row)
            section_block["sensors"][sensor] = row
        payload["sections"].append(section_block)

    out_dir = recordings_root().parent
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "reports" / "section_frame_validation.json"
    out_csv = out_dir / "reports" / "section_frame_validation.csv"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return {"json": str(out_json), "csv": str(out_csv), "rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m calibration.section_frame_report")
    parser.add_argument("sections", nargs="+", help="Section paths, relative to recordings root or absolute.")
    parser.add_argument("--no-recalibrate", action="store_true", help="Use existing calibrated outputs only.")
    args = parser.parse_args()

    out = run_report(args.sections, force_recalibrate=not args.no_recalibrate)
    print(f"Wrote section-frame report: {out['rows']} rows")
    print(f"  JSON: {out['json']}")
    print(f"  CSV:  {out['csv']}")


if __name__ == "__main__":
    main()
