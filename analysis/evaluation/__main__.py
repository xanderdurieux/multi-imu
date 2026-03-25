"""Run evaluation report: ``uv run python -m evaluation <features_fused.csv> [out_dir]``."""

from __future__ import annotations

import sys
from pathlib import Path

from .experiments import run_evaluation_report


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: uv run python -m evaluation <features_fused.csv> [out_dir]\n"
            "  out_dir defaults to <csv_parent>/evaluation_report"
        )
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent / "evaluation_report"
    run_evaluation_report(csv_path, out)


if __name__ == "__main__":
    main()
