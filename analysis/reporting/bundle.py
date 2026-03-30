from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .figure_gen import (
    STATUS_FAILED,
    CORE_FIGURE_KEYS,
    generate_core_thesis_figures,
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
            "caption": "Representative success and failure section exemplars selected via QC, confidence, and downstream-separability proxy signals.",
            "chapter_placement": "Results discussion / qualitative case studies section.",
        },
        {
            "artifact_key": "case_studies",
            "caption": "Tabulated success/failure case metadata with compact multi-signal scores and dominant quality flags.",
            "chapter_placement": "Results discussion / qualitative case studies section.",
        },
    ]


def _serialize_artifact(item: object) -> dict:
    payload = asdict(item)
    payload["path"] = str(item.path) if getattr(item, "path", None) else None
    return payload


def _run_and_wrap(builders: list[tuple[str, callable]]) -> list[object]:
    artifacts = []
    for key, fn in builders:
        try:
            artifacts.append(fn())
        except Exception as exc:
            artifacts.append(type("Artifact", (), {"key": key, "path": None, "status": STATUS_FAILED, "note": f"Unhandled error: {exc}", "diagnostics": []})())
    return artifacts


def _write_captions(output_dir: Path, payload: dict) -> None:
    captions_md = output_dir / "caption_suggestions.md"
    status_by_key = {item["key"]: item["status"] for item in payload["figures"] + payload["tables"]}
    lines = ["# Thesis Figure/Table Caption Suggestions", ""]
    for item in payload["captions"]:
        status = status_by_key.get(item["artifact_key"], "missing_prerequisite")
        lines.extend(
            [
                f"## {item['artifact_key']}",
                f"- **Status:** `{status}`",
                f"- **Suggested caption:** {item['caption']}",
                f"- **Suggested placement:** {item['chapter_placement']}",
                "",
            ]
        )
    captions_md.write_text("\n".join(lines), encoding="utf-8")


def build_thesis_report_bundle(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    fig_dir = output_dir / "report_figures"
    table_dir = output_dir / "report_tables"
    debug_dir = output_dir / "draft_debug"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    figures = _run_and_wrap(
        [
            ("pipeline_overview", lambda: make_pipeline_overview_figure(fig_dir)),
            ("orientation_filter_comparison", lambda: make_orientation_filter_comparison_figure(fig_dir)),
            ("event_centered_bike_vs_rider", lambda: make_event_centered_plot(fig_dir)),
            ("feature_separability", lambda: make_feature_separability_plot(fig_dir)),
            ("success_failure_case_studies", lambda: make_success_failure_case_plot(fig_dir)),
        ]
    )
    tables = _run_and_wrap(
        [
            ("quality_summary", lambda: make_quality_summary_table(table_dir)),
            ("sync_method_comparison", lambda: make_sync_method_comparison_table(table_dir)),
            ("single_vs_dual_sensor_comparison", lambda: make_single_vs_dual_sensor_table(table_dir)),
            ("case_studies", lambda: make_case_study_table(table_dir)),
        ]
    )

    payload = {
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "output_dir": str(output_dir),
        "figures": [_serialize_artifact(f) for f in figures],
        "tables": [_serialize_artifact(t) for t in tables],
        "captions": _captions_and_placement(),
        "notes": {
            "final_report_figures": str(fig_dir),
            "draft_debug_figures": str(debug_dir),
            "publication_formats": ["pdf", "png"],
            "status_legend": {
                "real_result": "Artifact created from available data.",
                "skipped": "Intentionally not generated in this mode.",
                "missing_prerequisite": "Inputs were missing; no placeholder artifact was emitted.",
                "failed": "Unexpected runtime failure during generation.",
            },
        },
    }
    (output_dir / "bundle_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_captions(output_dir, payload)
    return payload


def generate_core_figures_only(output_dir: Path) -> dict:
    output_dir = output_dir.resolve()
    core_dir = output_dir / "core_thesis_figures"
    core_dir.mkdir(parents=True, exist_ok=True)
    figures = generate_core_thesis_figures(core_dir)

    payload = {
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "mode": "core_figures_only",
        "output_dir": str(output_dir),
        "core_figure_dir": str(core_dir),
        "required_keys": CORE_FIGURE_KEYS,
        "figures": [_serialize_artifact(f) for f in figures],
        "notes": {
            "deterministic_filenames": True,
            "status_legend": {
                "real_result": "Artifact created from available data.",
                "skipped": "Intentionally not generated in this mode.",
                "missing_prerequisite": "Inputs were missing; no placeholder artifact was emitted.",
                "failed": "Unexpected runtime failure during generation.",
            },
        },
    }
    (output_dir / "core_figures_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis-oriented report bundle from processed section outputs.")
    p.add_argument("--output-dir", type=Path, default=Path("outputs") / "thesis_report_bundle")
    p.add_argument(
        "--core-figures-only",
        action="store_true",
        help="Generate only deterministic core thesis figures into <output-dir>/core_thesis_figures.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.core_figures_only:
        payload = generate_core_figures_only(args.output_dir)
        print(json.dumps({"mode": payload["mode"], "output_dir": payload["output_dir"], "n_figures": len(payload["figures"])}, indent=2))
        return

    payload = build_thesis_report_bundle(args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": payload["output_dir"],
                "n_figures": len(payload["figures"]),
                "n_tables": len(payload["tables"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
