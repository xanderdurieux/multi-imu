"""Export pipeline: aggregate features from sections and write split tables."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.paths import sections_root, recordings_root, analysis_root

logger = logging.getLogger(__name__)

# Quality label ordering (higher index = higher quality)
_QUALITY_ORDER = ["poor", "marginal", "good"]

_METADATA_COLS = {
    "section_id",
    "window_idx",
    "window_start_ms",
    "window_end_ms",
    "window_duration_s",
    "scenario_label",
    "overall_quality_label",
    "quality_tier",
    "calibration_quality",
    "sync_confidence",
}


def aggregate_features(
    recording_names: list[str] | None = None,
    *,
    min_quality_label: str = "marginal",
) -> pd.DataFrame:
    """Collect all features/features.csv from all sections, return combined DataFrame.

    Parameters
    ----------
    recording_names:
        If given, restrict to sections belonging to these recordings.
    min_quality_label:
        Minimum quality threshold. One of "good", "marginal", "poor".
        Rows with overall_quality_label below this threshold are dropped.
    """
    if min_quality_label not in _QUALITY_ORDER:
        raise ValueError(
            f"min_quality_label must be one of {_QUALITY_ORDER}, got {min_quality_label!r}"
        )

    root = sections_root()
    if not root.exists():
        logger.warning("sections_root does not exist: %s", root)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for section_dir in sorted(root.iterdir()):
        if not section_dir.is_dir():
            continue

        # Filter by recording name if requested
        if recording_names is not None:
            match = any(section_dir.name.startswith(rec) for rec in recording_names)
            if not match:
                logger.debug("Skipping section %s (not in recording_names)", section_dir.name)
                continue

        features_csv = section_dir / "features" / "features.csv"
        if not features_csv.exists():
            logger.debug("No features.csv for section %s", section_dir.name)
            continue

        try:
            df = pd.read_csv(features_csv)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", features_csv, exc)
            continue

        # Inject section_id if not present
        if "section_id" not in df.columns:
            df.insert(0, "section_id", section_dir.name)

        frames.append(df)
        logger.debug("Loaded %d rows from %s", len(df), features_csv)

    if not frames:
        logger.warning("No feature files found under %s", root)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Apply quality filter
    if "overall_quality_label" in combined.columns:
        min_idx = _QUALITY_ORDER.index(min_quality_label)
        valid_labels = set(_QUALITY_ORDER[min_idx:])
        before = len(combined)
        combined = combined[combined["overall_quality_label"].isin(valid_labels)].copy()
        logger.info(
            "Quality filter (>= %s): %d → %d rows",
            min_quality_label,
            before,
            len(combined),
        )
    else:
        logger.warning(
            "Column 'overall_quality_label' not found; skipping quality filter"
        )

    combined.reset_index(drop=True, inplace=True)
    logger.info("Aggregated %d rows from %d section(s)", len(combined), len(frames))
    return combined


def export_feature_tables(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Split and write bike-only, rider-only, and fused feature tables.

    Parameters
    ----------
    df:
        Combined feature DataFrame from :func:`aggregate_features`.
    output_dir:
        Directory where CSV files and manifest are written.

    Returns
    -------
    dict mapping table name → absolute Path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = [c for c in df.columns if c in _METADATA_COLS]

    bike_cols = [
        c for c in df.columns
        if c not in _METADATA_COLS and (c.startswith("bike_") or c.startswith("sporsa_"))
    ]
    rider_cols = [
        c for c in df.columns
        if c not in _METADATA_COLS and (c.startswith("rider_") or c.startswith("arduino_"))
    ]

    df_bike = df[meta_cols + bike_cols].copy()
    df_rider = df[meta_cols + rider_cols].copy()
    df_fused = df.copy()

    paths: dict[str, Path] = {}

    for name, frame in [
        ("features_bike", df_bike),
        ("features_rider", df_rider),
        ("features_fused", df_fused),
    ]:
        out_path = output_dir / f"{name}.csv"
        frame.to_csv(out_path, index=False)
        paths[name] = out_path
        logger.info("Wrote %s (%d rows, %d cols)", out_path, len(frame), len(frame.columns))

    # Build manifest
    label_dist: dict[str, int] = {}
    if "scenario_label" in df.columns:
        label_dist = df["scenario_label"].fillna("unlabeled").value_counts().to_dict()

    n_sections = df["section_id"].nunique() if "section_id" in df.columns else 0

    data_root_rel = analysis_root() / "data"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "filtering_policy": "min_quality_label=marginal",
        "total_sections": int(n_sections),
        "total_windows": int(len(df)),
        "label_distribution": {k: int(v) for k, v in label_dist.items()},
        "tables": {
            name: str(path.relative_to(analysis_root()))
            for name, path in paths.items()
        },
    }

    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    paths["export_manifest"] = manifest_path
    logger.info("Wrote manifest to %s", manifest_path)

    return paths


def run_exports(
    recording_names: list[str] | None = None,
    *,
    output_dir: Path | None = None,
    min_quality_label: str = "marginal",
    force: bool = False,
) -> dict[str, Path]:
    """Full export pipeline: aggregate → split → write.

    Parameters
    ----------
    recording_names:
        Optional list of recording names to include. ``None`` means all.
    output_dir:
        Destination directory. Defaults to ``data/exports/``.
    min_quality_label:
        Minimum quality threshold passed to :func:`aggregate_features`.
    force:
        If ``False`` and outputs already exist, skip and return existing paths.
    """
    if output_dir is None:
        output_dir = analysis_root() / "data" / "exports"
    output_dir = Path(output_dir)

    manifest_path = output_dir / "export_manifest.json"
    if not force and manifest_path.exists():
        logger.info("Export already exists at %s (use force=True to re-run)", output_dir)
        existing: dict[str, Path] = {}
        try:
            data = json.loads(manifest_path.read_text())
            for name, rel_path in data.get("tables", {}).items():
                existing[name] = analysis_root() / rel_path
        except Exception:
            pass
        existing["export_manifest"] = manifest_path
        return existing

    logger.info(
        "Running export pipeline (min_quality=%s, recordings=%s)",
        min_quality_label,
        recording_names,
    )

    df = aggregate_features(recording_names, min_quality_label=min_quality_label)
    if df.empty:
        logger.warning("No data after aggregation; nothing to export.")
        return {}

    return export_feature_tables(df, output_dir)
