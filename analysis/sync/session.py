"""Session-stage synchronization CLI.

Usage:
    uv run -m sync.sync_session <session_name>/<stage>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from common import session_stage_dir

from .drift_estimator import DEFAULT_LOCAL_SEARCH_SECONDS, DEFAULT_WINDOW_SECONDS, DEFAULT_WINDOW_STEP_SECONDS
from .sync_streams import DEFAULT_MAX_LAG_SECONDS, DEFAULT_SAMPLE_RATE_HZ, synchronize


def _load_stats_stream_scores(session_name: str, stage_in: str) -> dict[str, float]:
    """Load per-stream quality scores from session_stats.json when available."""
    candidates = [
        session_stage_dir(session_name, stage_in) / "session_stats.json",
        session_stage_dir(session_name, "parsed") / "session_stats.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        streams = data.get("streams")
        if not isinstance(streams, dict):
            continue

        scores: dict[str, float] = {}
        for stem, info in streams.items():
            if not isinstance(info, dict):
                continue
            quality = info.get("quality")
            if not isinstance(quality, dict):
                continue
            score = quality.get("score")
            if score is None:
                continue
            try:
                scores[str(stem)] = float(score)
            except (TypeError, ValueError):
                continue
        return scores
    return {}


def _find_single_csv_by_token(csv_files: list[Path], token: str) -> Path:
    hits = [p for p in csv_files if token.lower() in p.stem.lower()]
    if not hits:
        raise FileNotFoundError(f"No CSV stream found containing '{token}'.")
    if len(hits) > 1:
        names = ", ".join(sorted(p.name for p in hits))
        raise ValueError(f"Multiple CSV streams found for '{token}': {names}")
    return hits[0]


def _pick_reference_csv(csv_files: list[Path], score_by_stem: dict[str, float]) -> Path:
    for stem, _score in sorted(score_by_stem.items(), key=lambda kv: kv[1], reverse=True):
        for path in csv_files:
            if path.stem == stem:
                return path
    for path in csv_files:
        if "sporsa" in path.stem.lower():
            return path
    return csv_files[0]


def _pick_target_csv(csv_files: list[Path], reference_csv: Path, score_by_stem: dict[str, float]) -> Path:
    for path in csv_files:
        if path == reference_csv:
            continue
        if "arduino" in path.stem.lower():
            return path

    for stem, _score in sorted(score_by_stem.items(), key=lambda kv: kv[1], reverse=True):
        for path in csv_files:
            if path == reference_csv:
                continue
            if path.stem == stem:
                return path

    for path in csv_files:
        if path != reference_csv:
            return path
    raise ValueError("Could not determine target stream.")


def synchronize_session_stage(
    session_name: str,
    stage_in: str,
    *,
    reference_sensor: str | None = None,
    target_sensor: str | None = None,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
) -> tuple[Path, Path, Path, Path | None]:
    """Synchronize one stage: choose reference/target streams and write outputs under synced/."""
    in_dir = session_stage_dir(session_name, stage_in)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input stage directory not found: {in_dir}")

    csv_files = sorted(p for p in in_dir.glob("*.csv") if p.is_file())
    if len(csv_files) < 2:
        raise ValueError(f"Need at least two CSV files in {in_dir} to synchronize streams.")

    score_by_stem = _load_stats_stream_scores(session_name, stage_in)

    if reference_sensor:
        reference_csv = _find_single_csv_by_token(csv_files, reference_sensor)
    else:
        reference_csv = _pick_reference_csv(csv_files, score_by_stem)

    if target_sensor:
        target_csv = _find_single_csv_by_token(csv_files, target_sensor)
    else:
        target_csv = _pick_target_csv(csv_files, reference_csv, score_by_stem)

    if reference_csv == target_csv:
        raise ValueError("Reference and target streams must be different files.")

    out_dir = session_stage_dir(session_name, "synced")
    out_dir.mkdir(parents=True, exist_ok=True)
    copied_reference_csv = out_dir / reference_csv.name
    shutil.copy2(reference_csv, copied_reference_csv)

    sync_json_path, target_synced_csv, uniform_csv = synchronize(
        reference_csv=reference_csv,
        target_csv=target_csv,
        output_dir=out_dir,
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        resample_rate_hz=resample_rate_hz,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
    )
    return copied_reference_csv, target_synced_csv, sync_json_path, uniform_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sync.sync_session",
        description=(
            "Synchronize two IMU streams for one session stage using the default configuration. "
            "Default reference is the highest-quality stream from session stats "
            "(usually sporsa), and default target prefers arduino."
        ),
    )
    parser.add_argument(
        "session_name_stage",
        help="Session and stage as '<session_name>/<stage>' (example: 'session_6/parsed').",
    )
    parser.add_argument(
        "--reference-sensor",
        default=None,
        help="Optional sensor token for reference stream selection (substring match).",
    )
    parser.add_argument(
        "--target-sensor",
        default=None,
        help="Optional sensor token for target stream selection (substring match).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    parts = args.session_name_stage.split("/", 1)
    if len(parts) != 2:
        raise SystemExit("session_name_stage must be in format '<session_name>/<stage>'")

    session_name, stage_in = parts
    ref_csv, tgt_synced_csv, sync_json, uniform_csv = synchronize_session_stage(
        session_name=session_name,
        stage_in=stage_in,
        reference_sensor=args.reference_sensor,
        target_sensor=args.target_sensor,
    )
    print(f"reference_csv: {ref_csv}")
    print(f"target_synced_csv: {tgt_synced_csv}")
    if uniform_csv is not None:
        print(f"target_synced_uniform_csv: {uniform_csv}")
    print(f"sync_info_json: {sync_json}")


if __name__ == "__main__":
    main()
