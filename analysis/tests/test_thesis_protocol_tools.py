from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from labels.agreement import compute_inter_rater_agreement
from protocol.thesis_protocol import load_locked_split_manifest


def test_locked_split_manifest_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps({"splits": {"train": ["r1", "r2"], "validation": ["r3"], "test": ["r4"]}}),
        encoding="utf-8",
    )
    splits = load_locked_split_manifest(path)
    assert splits["train"] == {"r1", "r2"}


def test_inter_rater_exports(tmp_path: Path) -> None:
    a = pd.DataFrame(
        [
            {"annotation_id": "a1", "scenario_label": "steady_state", "scope": "section"},
            {"annotation_id": "a2", "scenario_label": "hard_braking", "scope": "section"},
        ]
    )
    b = pd.DataFrame(
        [
            {"annotation_id": "a1", "scenario_label": "steady_state", "scope": "section"},
            {"annotation_id": "a2", "scenario_label": "surface_disturbance", "scope": "section"},
        ]
    )
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    out = tmp_path / "out"
    a.to_csv(a_path, index=False)
    b.to_csv(b_path, index=False)

    paths = compute_inter_rater_agreement(a_path, b_path, out_dir=out)
    assert paths["summary"].is_file()
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["n_overlap"] == 2
