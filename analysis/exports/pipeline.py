"""Export pipeline: aggregate features, calibration params, and sync params."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.paths import (
    analysis_root,
    data_root,
    project_relative_path,
    read_csv,
    sections_root,
    write_csv,
)
from exports.aggregate import aggregate_calibration_params, aggregate_sync_params

log = logging.getLogger(__name__)

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
        log.warning("sections_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for section_dir in sorted(root.iterdir()):
        if not section_dir.is_dir():
            continue

        # Filter by recording name if requested
        if recording_names is not None:
            match = any(section_dir.name.startswith(rec) for rec in recording_names)
            if not match:
                log.debug("Skipping section %s (not in recording_names)", section_dir.name)
                continue

        features_csv = section_dir / "features" / "features.csv"
        if not features_csv.exists():
            log.debug("No features.csv for section %s", section_dir.name)
            continue

        try:
            df = read_csv(features_csv)
        except Exception as exc:
            log.warning("Failed to read %s: %s", project_relative_path(features_csv), exc)
            continue

        # Inject section_id if not present
        if "section_id" not in df.columns:
            df.insert(0, "section_id", section_dir.name)

        frames.append(df)
        log.debug("Loaded %d rows from %s", len(df), project_relative_path(features_csv))

    if not frames:
        log.warning("No feature files found under %s", project_relative_path(root))
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Apply quality filter
    if "overall_quality_label" in combined.columns:
        min_idx = _QUALITY_ORDER.index(min_quality_label)
        valid_labels = set(_QUALITY_ORDER[min_idx:])
        before = len(combined)
        combined = combined[combined["overall_quality_label"].isin(valid_labels)].copy()
        log.info(
            "Quality filter (>= %s): %d → %d rows",
            min_quality_label,
            before,
            len(combined),
        )
    else:
        log.warning(
            "Column 'overall_quality_label' not found; skipping quality filter"
        )

    combined.reset_index(drop=True, inplace=True)
    log.info("Aggregated %d rows from %d section(s)", len(combined), len(frames))
    return combined


def export_feature_tables(
    df: pd.DataFrame,
    output_dir: Path,
    *,
    recording_names: list[str] | None = None,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    """Split and write bike-only, rider-only, fused, calibration, and sync tables.

    Parameters
    ----------
    df:
        Combined feature DataFrame from :func:`aggregate_features`.
    output_dir:
        Directory where CSV files and manifest are written.
    recording_names:
        Optional list of recording names; passed to the calibration and sync
        aggregators to restrict which recordings are collected.

    Returns
    -------
    Tuple of (paths dict, calibration params DataFrame, sync params DataFrame).
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
        write_csv(frame, out_path)
        paths[name] = out_path
        log.info(
            "Wrote %s (%d rows, %d cols)",
            project_relative_path(out_path),
            len(frame),
            len(frame.columns),
        )

    # Calibration parameters (one row per section)
    cal_df = aggregate_calibration_params(recording_names)
    if not cal_df.empty:
        cal_path = output_dir / "calibration_params.csv"
        cal_df.to_csv(cal_path, index=False)
        paths["calibration_params"] = cal_path
        log.info(
            "Wrote %s (%d sections)",
            project_relative_path(cal_path),
            len(cal_df),
        )
    else:
        log.warning("No calibration params found; calibration_params.csv not written")

    # Sync parameters (one row per recording)
    sync_df = aggregate_sync_params(recording_names)
    if not sync_df.empty:
        sync_path = output_dir / "sync_params.csv"
        sync_df.to_csv(sync_path, index=False)
        paths["sync_params"] = sync_path
        log.info(
            "Wrote %s (%d recordings)",
            project_relative_path(sync_path),
            len(sync_df),
        )
    else:
        log.warning("No sync params found; sync_params.csv not written")

    # Build manifest
    label_dist: dict[str, int] = {}
    if "scenario_label" in df.columns:
        label_dist = df["scenario_label"].fillna("unlabeled").value_counts().to_dict()

    n_sections = df["section_id"].nunique() if "section_id" in df.columns else 0

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "filtering_policy": "min_quality_label=marginal",
        "total_sections": int(n_sections),
        "total_windows": int(len(df)),
        "total_recordings": int(len(sync_df)) if not sync_df.empty else 0,
        "label_distribution": {k: int(v) for k, v in label_dist.items()},
        "tables": {
            name: str(path.relative_to(analysis_root()))
            for name, path in paths.items()
        },
    }

    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    paths["export_manifest"] = manifest_path
    log.info("Wrote manifest to %s", project_relative_path(manifest_path))

    return paths, cal_df, sync_df


def run_exports(
    recording_names: list[str] | None = None,
    *,
    output_dir: Path | None = None,
    min_quality_label: str = "marginal",
    force: bool = False,
    no_plots: bool = False,
) -> dict[str, Path]:
    """Full export pipeline: aggregate → split → write → EDA figures.

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
    no_plots:
        If ``True``, skip EDA figure generation.
    """
    if output_dir is None:
        output_dir = data_root() / "exports"
    output_dir = Path(output_dir)

    manifest_path = output_dir / "export_manifest.json"
    if not force and manifest_path.exists():
        log.info(
            "Export already exists at %s (use force=True to re-run)",
            project_relative_path(output_dir),
        )
        existing: dict[str, Path] = {}
        try:
            data = json.loads(manifest_path.read_text())
            for name, rel_path in data.get("tables", {}).items():
                existing[name] = analysis_root() / rel_path
        except Exception:
            pass
        existing["export_manifest"] = manifest_path
        if not no_plots:
            _run_eda_safe(existing.get("features_fused"), output_dir)
        return existing

    log.info(
        "Running export pipeline (min_quality=%s, recordings=%s)",
        min_quality_label,
        recording_names,
    )

    df = aggregate_features(recording_names, min_quality_label=min_quality_label)
    if df.empty:
        log.warning("No data after aggregation; nothing to export.")
        return {}

    paths, cal_df, sync_df = export_feature_tables(df, output_dir, recording_names=recording_names)

    if not no_plots:
        try:
            from visualization.plot_exports import run_eda
            run_eda(df, output_dir)
        except Exception as exc:
            log.warning("Feature EDA figure generation failed: %s", exc)

        try:
            from visualization.plot_exports import run_calibration_eda, run_sync_eda
            run_calibration_eda(cal_df, output_dir)
            run_sync_eda(sync_df, output_dir)
        except Exception as exc:
            log.warning("Params EDA figure generation failed: %s", exc)

    return paths


def _run_eda_safe(fused_path: Path | None, output_dir: Path) -> None:
    """Load fused features and run EDA if the file exists (best-effort)."""
    if fused_path is None or not fused_path.exists():
        return
    try:
        from visualization.plot_exports import run_eda
        df = read_csv(fused_path)
        if not df.empty:
            run_eda(df, output_dir)
    except Exception as exc:
        log.warning("EDA figure generation failed: %s", exc)
