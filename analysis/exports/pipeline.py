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
    recording_sort_key,
    section_sort_key,
    sections_root,
    write_csv,
)
from exports.aggregate import aggregate_calibration_params, aggregate_orientation_stats, aggregate_parsed_params, aggregate_sync_params

log = logging.getLogger(__name__)

# Quality label ordering (higher index = higher quality)
_QUALITY_ORDER = ["poor", "marginal", "good"]

_METADATA_COLS = {
    "recording_name",
    "session",
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


def _session_from_recording(recording_name: str) -> str:
    """Return session from recording."""
    parts = str(recording_name).rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("r") and parts[1][1:].isdigit():
        return parts[0]
    return str(recording_name)


def _recording_from_section(section_id: str) -> str:
    """Return recording from section."""
    section_id = str(section_id)
    parts = section_id.rsplit("s", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return section_id


def _recording_mask(df: pd.DataFrame, recording_names: list[str]) -> pd.Series:
    """Return rows in *df* belonging to any of *recording_names*."""
    mask = pd.Series(False, index=df.index)
    targets = {str(name) for name in recording_names}

    if "recording_name" in df.columns:
        mask |= df["recording_name"].astype(str).isin(targets)

    if "section_id" in df.columns:
        section_recordings = df["section_id"].astype(str).map(_recording_from_section)
        mask |= section_recordings.isin(targets)

    return mask


def _sort_export_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return sort export df."""
    if df.empty:
        return df

    out = df.copy()
    if "recording_name" not in out.columns and "section_id" in out.columns:
        out["recording_name"] = out["section_id"].astype(str).map(_recording_from_section)

    if "recording_name" in out.columns:
        out["_recording_sort"] = out["recording_name"].astype(str).map(recording_sort_key)
        sort_cols = ["_recording_sort"]
        if "section_id" in out.columns:
            sort_cols.append("section_id")
        if "window_idx" in out.columns:
            sort_cols.append("window_idx")
        out = out.sort_values(sort_cols, kind="stable").drop(columns=["_recording_sort"])
        out = out.reset_index(drop=True)

    return out


def _upsert_export_table(
    new_df: pd.DataFrame,
    path: Path,
    *,
    recording_names: list[str] | None,
) -> pd.DataFrame:
    """Merge new export rows into an existing CSV, replacing matching recordings."""
    if new_df.empty:
        return new_df

    if recording_names is None or not path.exists():
        merged = new_df.copy()
    else:
        try:
            existing = read_csv(path)
        except Exception as exc:
            log.warning("Failed to read existing %s for merge: %s", project_relative_path(path), exc)
            existing = pd.DataFrame()

        if existing.empty:
            merged = new_df.copy()
        else:
            keep = existing.loc[~_recording_mask(existing, recording_names)].copy()
            merged = pd.concat([keep, new_df], ignore_index=True, sort=False)

    return _sort_export_df(merged)


def _recording_count(df: pd.DataFrame) -> int:
    """Return recording count."""
    if "recording_name" in df.columns:
        return int(df["recording_name"].nunique())
    if "section_id" in df.columns:
        return int(df["section_id"].astype(str).map(_recording_from_section).nunique())
    return 0


def aggregate_features(
    recording_names: list[str] | None = None,
    *,
    min_quality_label: str | None = None,
) -> pd.DataFrame:
    """Aggregate features.

    If `min_quality_label` is None then no quality filter is applied and all
    extracted windows are retained. Otherwise only windows with
    `overall_quality_label` >= `min_quality_label` are kept.
    """
    if min_quality_label is not None and min_quality_label not in _QUALITY_ORDER:
        raise ValueError(
            f"min_quality_label must be one of {_QUALITY_ORDER} or None, got {min_quality_label!r}"
        )

    root = sections_root()
    if not root.exists():
        log.warning("sections_root does not exist: %s", project_relative_path(root))
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for section_dir in sorted(root.iterdir(), key=section_sort_key):
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
        if "recording_name" not in df.columns:
            df.insert(0, "recording_name", _recording_from_section(section_dir.name))
        if "session" not in df.columns:
            insert_at = 1 if "recording_name" in df.columns else 0
            df.insert(insert_at, "session", _session_from_recording(_recording_from_section(section_dir.name)))

        frames.append(df)
        log.debug("Loaded %d rows from %s", len(df), project_relative_path(features_csv))

    if not frames:
        log.warning("No feature files found under %s", project_relative_path(root))
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Apply quality filter (optional)
    if min_quality_label is not None:
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
    min_quality_label: str | None = None,
) -> tuple[dict[str, Path], pd.DataFrame, pd.DataFrame]:
    """Export feature tables."""
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
        frame = _upsert_export_table(frame, out_path, recording_names=recording_names)
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
        cal_df = _upsert_export_table(cal_df, cal_path, recording_names=recording_names)
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
        sync_df = _upsert_export_table(sync_df, sync_path, recording_names=recording_names)
        sync_df.to_csv(sync_path, index=False)
        paths["sync_params"] = sync_path
        log.info(
            "Wrote %s (%d recordings)",
            project_relative_path(sync_path),
            len(sync_df),
        )
    else:
        log.warning("No sync params found; sync_params.csv not written")

    # Build manifest from the merged fused table so incremental recording
    # exports report the whole dataset, not just the current subset.
    manifest_df = paths.get("features_fused")
    if manifest_df is not None and manifest_df.exists():
        try:
            df_for_manifest = read_csv(manifest_df)
        except Exception:
            df_for_manifest = df
    else:
        df_for_manifest = df

    n_sections = df_for_manifest["section_id"].nunique() if "section_id" in df_for_manifest.columns else 0
    label_dist = {}
    if "scenario_label" in df_for_manifest.columns:
        label_dist = df_for_manifest["scenario_label"].fillna("unlabeled").value_counts().to_dict()

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "filtering_policy": f"min_quality_label={min_quality_label}",
        "total_sections": int(n_sections),
        "total_windows": int(len(df_for_manifest)),
        "total_recordings": _recording_count(df_for_manifest),
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
    min_quality_label: str | None = None,
    force: bool = False,
    no_plots: bool = False,
) -> dict[str, Path]:
    """Run exports."""
    if output_dir is None:
        output_dir = data_root() / "exports"
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    log.info(
        "Running export pipeline (min_quality=%s, recordings=%s)",
        min_quality_label,
        recording_names,
    )

    # Always write params tables — independent of whether features exist or cache is valid.
    parsed_df = aggregate_parsed_params(recording_names)
    if not parsed_df.empty:
        parsed_path = output_dir / "parsed_recording_summary.csv"
        parsed_df = _upsert_export_table(parsed_df, parsed_path, recording_names=recording_names)
        parsed_df.to_csv(parsed_path, index=False)
        paths["parsed_recording_summary"] = parsed_path
        log.info("Wrote parsed_recording_summary.csv (%d recordings)", len(parsed_df))
    else:
        log.warning("No parsed stats found; parsed_recording_summary.csv not written")

    cal_df = aggregate_calibration_params(recording_names)
    if not cal_df.empty:
        cal_path = output_dir / "calibration_params.csv"
        cal_df = _upsert_export_table(cal_df, cal_path, recording_names=recording_names)
        cal_df.to_csv(cal_path, index=False)
        paths["calibration_params"] = cal_path
        log.info("Wrote calibration_params.csv (%d sections)", len(cal_df))
    else:
        log.warning("No calibration params found; calibration_params.csv not written")

    sync_df = aggregate_sync_params(recording_names)
    if not sync_df.empty:
        sync_path = output_dir / "sync_params.csv"
        sync_df = _upsert_export_table(sync_df, sync_path, recording_names=recording_names)
        sync_df.to_csv(sync_path, index=False)
        paths["sync_params"] = sync_path
        log.info("Wrote sync_params.csv (%d recordings)", len(sync_df))
    else:
        log.warning("No sync params found; sync_params.csv not written")

    orient_df = aggregate_orientation_stats(recording_names)
    if not orient_df.empty:
        orient_path = output_dir / "orientation_stats.csv"
        orient_df = _upsert_export_table(orient_df, orient_path, recording_names=recording_names)
        orient_df.to_csv(orient_path, index=False)
        paths["orientation_stats"] = orient_path
        log.info("Wrote orientation_stats.csv (%d sections)", len(orient_df))
    else:
        log.warning("No orientation stats found; orientation_stats.csv not written")

    # Feature tables: honour the manifest cache.
    manifest_path = output_dir / "export_manifest.json"
    if not force and recording_names is None and manifest_path.exists():
        log.info(
            "Feature export cache exists at %s (use force=True to re-run features)",
            project_relative_path(output_dir),
        )
        try:
            data = json.loads(manifest_path.read_text())
            for name, rel_path in data.get("tables", {}).items():
                paths[name] = analysis_root() / rel_path
        except Exception:
            pass
        paths["export_manifest"] = manifest_path
        if not no_plots:
            _run_eda_safe(paths.get("features_fused"), output_dir)
            _run_params_eda_safe(cal_df, sync_df, orient_df, output_dir)
        return paths

    df = aggregate_features(recording_names, min_quality_label=min_quality_label)
    if df.empty:
        log.warning("No features after aggregation; skipping feature tables and EDA plots.")
        return paths

    feature_paths, _, _ = export_feature_tables(
        df, output_dir, recording_names=recording_names, min_quality_label=min_quality_label
    )
    paths.update(feature_paths)

    if not no_plots:
        try:
            from visualization.plot_exports import run_eda
            fused_path = paths.get("features_fused")
            plot_df = read_csv(fused_path) if fused_path is not None and fused_path.exists() else df
            run_eda(plot_df, output_dir)
        except Exception as exc:
            log.warning("Feature EDA figure generation failed: %s", exc)

        try:
            from visualization.plot_exports import run_calibration_eda, run_orientation_eda, run_sync_eda
            run_calibration_eda(cal_df, output_dir)
            run_sync_eda(sync_df, output_dir)
            run_orientation_eda(orient_df, output_dir)
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


def _run_params_eda_safe(
    cal_df: pd.DataFrame,
    sync_df: pd.DataFrame,
    orient_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Run parameter export EDA figures best-effort."""
    try:
        from visualization.plot_exports import run_calibration_eda, run_orientation_eda, run_sync_eda
        run_calibration_eda(cal_df, output_dir)
        run_sync_eda(sync_df, output_dir)
        run_orientation_eda(orient_df, output_dir)
    except Exception as exc:
        log.warning("Params EDA figure generation failed: %s", exc)
