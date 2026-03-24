"""Quality validation for features stage."""

from __future__ import annotations
import pathlib

import json
import logging
import sys
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def validate(section_path: Path) -> dict[str, str | float]:
    """Validate features output."""
    section_path = Path(section_path)
    feat_dir = section_path / "features"
    feat_csv = feat_dir / "features.csv"
    stats_json = feat_dir / "features_stats.json"

    result: dict[str, str | float] = {"status": "poor", "section": str(section_path)}

    if not feat_csv.exists():
        log.warning("No features.csv at %s", feat_csv)
        result["error"] = "features.csv not found"
        return result

    df = pd.read_csv(feat_csv)
    if df.empty:
        result["status"] = "marginal"
        result["n_rows"] = 0
        return result

    # Check for all-NaN cross-sensor columns
    cross_cols = [
        "acc_norm_corr", "acc_norm_lag_ms", "acc_energy_ratio",
        "gyro_energy_ratio", "pitch_corr", "pitch_divergence_std",
    ]
    all_nan_count = sum(1 for c in cross_cols if c in df.columns and df[c].isna().all())
    if all_nan_count == len([c for c in cross_cols if c in df.columns]):
        result["status"] = "poor"
        result["all_nan_cross_sensor"] = True
    elif all_nan_count > 0:
        result["status"] = "marginal"
    else:
        result["status"] = "good"

    result["n_rows"] = len(df)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    if len(sys.argv) < 2:
        print("Usage: uv run -m features.validate <section_path>")
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
