"""Quality validation for calibration stage."""

from __future__ import annotations
import pathlib

import json
import logging
import sys
from pathlib import Path

from common import recordings_root

log = logging.getLogger(__name__)

# README thresholds
GOOD_RESIDUAL = 0.2
MARGINAL_RESIDUAL = 0.5
GOOD_N_STATIC = 500


def validate(section_path: Path) -> dict[str, str | float | int]:
    """Validate calibration output for a section.

    Returns dict with status ('good'|'marginal'|'poor') and metrics.
    """
    section_path = Path(section_path)
    calibrated_dir = section_path / "calibrated"
    cal_json = calibrated_dir / "calibration.json"

    result: dict[str, str | float | int] = {"status": "poor", "section": str(section_path)}

    if not cal_json.exists():
        log.warning("No calibration.json at %s", cal_json)
        result["error"] = "calibration.json not found"
        return result

    with cal_json.open("r", encoding="utf-8") as f:
        cal = json.load(f)

    worst_status = "good"
    for sensor, meta in cal.items():
        residual = meta.get("gravity_residual_m_per_s2", 1e6)
        n_static = meta.get("n_static_samples", 0)

        if residual <= GOOD_RESIDUAL and n_static >= GOOD_N_STATIC:
            status = "good"
        elif residual <= MARGINAL_RESIDUAL:
            status = "marginal"
        else:
            status = "poor"

        result[f"{sensor}_gravity_residual_m_per_s2"] = residual
        result[f"{sensor}_n_static_samples"] = n_static
        result[f"{sensor}_status"] = status

        if status == "poor" or (status == "marginal" and worst_status == "good"):
            worst_status = status
        elif status == "marginal":
            worst_status = "marginal"

    result["status"] = worst_status
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: uv run -m calibration.validate <section_path>")
        sys.exit(1)
    path = sys.argv[1].strip()
    if not pathlib.Path(path).is_absolute():
        path = str(pathlib.Path.cwd() / path)
    result = validate(Path(path))
    print(f"Status: {result['status']}")
    for k, v in result.items():
        if k not in ("status", "section"):
            print(f"  {k}: {v}")
