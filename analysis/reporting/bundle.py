from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .figure_gen import (
    make_event_centered_plot,
    make_feature_separability_plot,
    make_orientation_filter_comparison_figure,
    make_pipeline_overview_figure,
    make_success_failure_case_plot,
)
from .table_gen import (
    make_case_study_table,
    make_quality_summary_table,
    make_single_vs_dual_sensor_table,
    make_sync_method_comparison_table,
)


def _captions_and_placement() -> list[dict[str, str]]:
    return [
        {
            "artifact_key": "pipeline_overview",
            "caption": "End-to-end processing workflow from raw session logs to thesis-level report artifacts.",
            "chapter_placement": "Methods chapter / pipeline overview section.",
        },
        {
            "artifact_key": "quality_summary",
            "caption": "Section-level quality summary showing confidence tiers and usability classes across the analyzed dataset.",
            "chapter_placement": "Data quality and preprocessing chapter.",
        },
        {
            "artifact_key": "sync_method_comparison",
            "caption": "Synchronization method comparison using median confidence and residual quality per method.",
            "chapter_placement": "Synchronization methodology chapter.",
        },
        {
            "artifact_key": "orientation_filter_comparison",
            "caption": "Orientation filter trade-off between responsiveness and event separability across processed sections.",
            "chapter_placement": "Orientation estimation chapter.",
        },
        {
            "artifact_key": "event_centered_bike_vs_rider",
            "caption": "Event-centered bike vs rider acceleration response around the top-confidence detected event.",
            "chapter_placement": "Event analysis chapter.",
        },
        {
            "artifact_key": "feature_separability",
            "caption": "Feature separability plot for scenario labels using the two most discriminative feature dimensions.",
            "chapter_placement": "Feature engineering and classification readiness chapter.",
        },
        {
            "artifact_key": "single_vs_dual_sensor_comparison",
            "caption": "Single-sensor vs dual-sensor feature separability comparison summarizing cross-sensor benefit.",
            "chapter_placement": "Ablation study chapter.",
        },
        {
            "artifact_key": "success_failure_case_studies",
            "caption": "Representative success and failure section exemplars selected by overall quality score.",
            "chapter_placement": "Results discussion / qualitative case studies section.",
        },
        {
            "artifact_key": "case_studies",
            "caption": "Tabulated success/failure case metadata with dominant quality flags for interpretability.",
            "chapter_placement": "Results discussion / qualitative case studies section.",
        },
    ]


def build_thesis_report_bundle(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    fig_dir = output_dir / "report_figures"
    table_dir = output_dir / "report_tables"
    debug_dir = output_dir / "draft_debug"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        make_pipeline_overview_figure(fig_dir),
        make_orientation_filter_comparison_figure(fig_dir),
        make_event_centered_plot(fig_dir),
        make_feature_separability_plot(fig_dir),
        make_success_failure_case_plot(fig_dir),
    ]
    tables = [
        make_quality_summary_table(table_dir),
        make_sync_method_comparison_table(table_dir),
        make_single_vs_dual_sensor_table(table_dir),
        make_case_study_table(table_dir),
    ]

    payload = {
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "output_dir": str(output_dir),
        "figures": [{**asdict(f), "path": str(f.path)} for f in figures],
        "tables": [{**asdict(t), "path": str(t.path)} for t in tables],
        "captions": _captions_and_placement(),
        "notes": {
            "final_report_figures": str(fig_dir),
            "draft_debug_figures": str(debug_dir),
            "publication_formats": ["pdf", "png"],
        },
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    captions_md = output_dir / "caption_suggestions.md"
    lines = ["# Thesis Figure/Table Caption Suggestions", ""]
    for item in payload["captions"]:
        lines.extend(
            [
                f"## {item['artifact_key']}",
                f"- **Suggested caption:** {item['caption']}",
                f"- **Suggested placement:** {item['chapter_placement']}",
                "",
            ]
        )
    captions_md.write_text("\n".join(lines), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis-oriented report bundle from processed section outputs.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "thesis_report_bundle")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_thesis_report_bundle(args.output_dir)
    print(json.dumps({"output_dir": payload["output_dir"], "n_figures": len(payload["figures"]), "n_tables": len(payload["tables"])}, indent=2))


if __name__ == "__main__":
    main()
