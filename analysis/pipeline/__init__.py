"""Per-stage pipeline CLI.

Run individual pipeline stages::

    python -m pipeline calibration 2026-02-26_r1s1
    python -m pipeline calibration --recording 2026-02-26_r1
    python -m pipeline orientation 2026-02-26_r1s1 --filter madgwick
    python -m pipeline derived --recording 2026-02-26_r1
    python -m pipeline events --recording 2026-02-26_r1
    python -m pipeline features --recording 2026-02-26_r1
    python -m pipeline exports
"""
