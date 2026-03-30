from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from reporting.bundle import build_thesis_report_bundle, generate_core_figures_only


class ReportingHonestOutputsTest(unittest.TestCase):
    def test_reports_missing_prerequisites_without_placeholders(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reporting-missing-") as td:
            root = Path(td)
            (root / "sections").mkdir(parents=True, exist_ok=True)
            out_dir = root / "out"
            prev = os.environ.get("MULTI_IMU_DATA_ROOT")
            os.environ["MULTI_IMU_DATA_ROOT"] = str(root)
            try:
                payload = build_thesis_report_bundle(out_dir)
            finally:
                if prev is None:
                    os.environ.pop("MULTI_IMU_DATA_ROOT", None)
                else:
                    os.environ["MULTI_IMU_DATA_ROOT"] = prev

            statuses = {x["status"] for x in payload["figures"] + payload["tables"]}
            self.assertIn("missing_prerequisite", statuses)
            for artifact in payload["figures"] + payload["tables"]:
                if artifact["status"] == "missing_prerequisite":
                    self.assertIsNone(artifact["path"])

    def test_core_figure_command_uses_deterministic_filenames(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reporting-core-") as td:
            root = Path(td)
            sections = root / "sections"
            sections.mkdir(parents=True, exist_ok=True)
            sec = sections / "secA"
            (sec / "features").mkdir(parents=True, exist_ok=True)
            (sec / "quality_metadata.json").write_text(
                json.dumps(
                    {
                        "overall_quality_score": 0.8,
                        "sync_confidence": 0.9,
                        "orientation_quality_score": 0.75,
                        "overall_quality_label": "good",
                        "overall_usability_category": "usable",
                        "quality_flags": ["ok"],
                    }
                ),
                encoding="utf-8",
            )
            (sec / "qc_section.json").write_text(json.dumps({"quality_tier": "good"}), encoding="utf-8")
            pd.DataFrame(
                {
                    "scenario_label": ["a", "a", "a", "a", "b", "b", "b", "b"],
                    "cross_sensor__f1": [0.1, 0.2, 0.1, 0.2, 1.0, 1.2, 1.1, 1.0],
                    "cross_sensor__f2": [0.4, 0.5, 0.4, 0.45, 1.6, 1.5, 1.7, 1.6],
                }
            ).to_csv(sec / "features" / "features.csv", index=False)

            out_dir = root / "out"
            prev = os.environ.get("MULTI_IMU_DATA_ROOT")
            os.environ["MULTI_IMU_DATA_ROOT"] = str(root)
            try:
                payload = generate_core_figures_only(out_dir)
            finally:
                if prev is None:
                    os.environ.pop("MULTI_IMU_DATA_ROOT", None)
                else:
                    os.environ["MULTI_IMU_DATA_ROOT"] = prev

            expected_stem = out_dir / "core_thesis_figures" / "thesis_core_01_pipeline_overview.pdf"
            self.assertTrue(expected_stem.is_file())
            self.assertEqual(payload["mode"], "core_figures_only")
            self.assertEqual(len(payload["figures"]), 5)


if __name__ == "__main__":
    unittest.main()
