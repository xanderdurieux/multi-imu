from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class ThesisWorkflowSmokeTest(unittest.TestCase):
    def test_end_to_end_fixture_and_deterministic_evaluation(self) -> None:
        analysis_root = Path(__file__).resolve().parents[1]
        fixture_root = analysis_root / "tests" / "fixtures" / "thesis_smoke"

        with tempfile.TemporaryDirectory(prefix="thesis-smoke-") as td:
            run_root = Path(td) / "fixture"
            shutil.copytree(fixture_root, run_root)

            (run_root / "data" / "recordings").mkdir(parents=True, exist_ok=True)
            (run_root / "data" / "sections").mkdir(parents=True, exist_ok=True)

            cfg_path = run_root / "workflow.fixture.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "workflow",
                    "--config",
                    str(cfg_path),
                    "--no-plots",
                    "--force",
                ],
                cwd=analysis_root,
                check=True,
            )

            data_root = run_root / "data"
            for rec in ("2026-01-15_r1", "2026-01-15_r2", "2026-01-15_r3"):
                rec_root = data_root / "recordings" / rec
                self.assertTrue((rec_root / "parsed" / "sporsa.csv").is_file())
                self.assertTrue((rec_root / "parsed" / "arduino.csv").is_file())
                self.assertTrue((rec_root / "synced" / "sporsa.csv").is_file())
                self.assertTrue((rec_root / "synced" / "arduino.csv").is_file())

            sec_root = data_root / "sections"
            sections = sorted(p for p in sec_root.iterdir() if p.is_dir())
            self.assertGreaterEqual(len(sections), 3)
            for sec in sections:
                self.assertTrue((sec / "calibrated" / "calibration.json").is_file())
                self.assertTrue((sec / "orientation" / "orientation_stats.json").is_file())
                self.assertTrue((sec / "derived" / "derived_signals_meta.json").is_file())
                self.assertTrue((sec / "events" / "event_candidates.csv").is_file())
                self.assertTrue((sec / "features" / "features.csv").is_file())
                self.assertTrue((sec / "qc_section.json").is_file())

            fused_csv = data_root / "exports" / "features_fused.csv"
            self.assertTrue(fused_csv.is_file())
            fused_df = pd.read_csv(fused_csv)
            self.assertGreater(len(fused_df), 0)
            self.assertIn("scenario_label", fused_df.columns)

            eval_cfg = run_root / "evaluation.fixture.json"
            out_a = run_root / "eval_run_a"
            out_b = run_root / "eval_run_b"
            for out in (out_a, out_b):
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "evaluation",
                        str(fused_csv),
                        str(out),
                        "--primary",
                        "--config",
                        str(eval_cfg),
                        "--seed",
                        "123",
                    ],
                    cwd=analysis_root,
                    check=True,
                )

            for filename in (
                "classification_summary.csv",
                "thesis_table_model_metrics.csv",
                "separability_effect_size.csv",
                "separability_within_between_variance.csv",
                "primary_feature_source_comparison.csv",
                "sync_ablation_compact.csv",
                "orientation_downstream_comparison.csv",
                "feature_family_ablation_compact.csv",
            ):
                a = pd.read_csv(out_a / filename)
                b = pd.read_csv(out_b / filename)
                pd.testing.assert_frame_equal(a, b, check_dtype=False, check_like=True)

            for md_name in (
                "PRIMARY_INTERPRETATION.md",
                "SYNC_ABLATION_INTERPRETATION.md",
                "ORIENTATION_DOWNSTREAM_INTERPRETATION.md",
                "FEATURE_FAMILY_INTERPRETATION.md",
                "THESIS_EXPERIMENT_BUNDLE.md",
            ):
                self.assertTrue((out_a / md_name).is_file())

            sum_a = json.loads((out_a / "evaluation_summary.json").read_text(encoding="utf-8"))
            sum_b = json.loads((out_b / "evaluation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(sum_a["seeds"]["evaluation_seed"], 123)
            self.assertEqual(sum_b["seeds"]["evaluation_seed"], 123)
            self.assertEqual(sum_a["dataset"], sum_b["dataset"])

            prov_latest = data_root / "provenance" / "workflow_run_latest.json"
            self.assertTrue(prov_latest.is_file())
            prov = json.loads(prov_latest.read_text(encoding="utf-8"))
            self.assertEqual(prov["seeds"]["evaluation_seed"], 123)


if __name__ == "__main__":
    unittest.main()
