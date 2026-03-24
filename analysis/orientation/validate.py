"""Quality validation for orientation stage."""

from __future__ import annotations
import pathlib

import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

GOOD_G_ERR = 0.3
GOOD_STD_DEG = 2.0
MARGINAL_STD_DEG = 5.0


def validate(section_path: Path) -> dict[str, str | float]:
    """Validate orientation output; return status and metrics."""
    section_path = Path(section_path)
    orient_dir = section_path / "orientation"
    stats_path = orient_dir / "orientation_stats.json"

    result: dict[str, str | float] = {"status": "poor", "section": str(section_path)}

    if not stats_path.exists():
        log.warning("No orientation_stats.json at %s", stats_path)
        result["error"] = "orientation_stats.json not found"
        return result

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    worst = "good"
    for sensor, variants in stats.items():
        for var, meta in variants.items():
            if not var.startswith("__"):
                continue
            g_err = meta.get("g_err_abs_mean", 1e6)
            pitch_std = meta.get("static_pitch_std_deg", 1e6)
            roll_std = meta.get("static_roll_std_deg", 1e6)
            quality = meta.get("quality", "poor")

            key = f"{sensor}{var}"
            result[f"{key}_g_err"] = g_err
            result[f"{key}_quality"] = quality

            if quality == "poor" or (quality == "marginal" and worst == "good"):
                worst = quality
            elif quality == "marginal":
                worst = "marginal"

    result["status"] = worst
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: uv run -m orientation.validate <section_path>")
        sys.exit(1)
    from common import recordings_root
    path = sys.argv[1].strip()
    if not pathlib.Path(path).is_absolute():
        path = str(pathlib.Path.cwd() / path)
    r = validate(Path(path))
    print(f"Status: {r['status']}")
    for k, v in r.items():
        if k not in ("status", "section"):
            print(f"  {k}: {v}")
