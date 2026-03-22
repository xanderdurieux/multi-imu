"""SDA + LIDA recording-level synchronization."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from common import find_sensor_csv, recording_stage_dir, write_dataframe

from .core import (
    DEFAULT_LOCAL_SEARCH_SECONDS,
    DEFAULT_MIN_FIT_R2,
    DEFAULT_MIN_WINDOW_SCORE,
    DEFAULT_WINDOW_SECONDS,
    DEFAULT_WINDOW_STEP_SECONDS,
    SyncModel,
    apply_sync_model,
    compute_sync_correlations,
    estimate_sync_model,
    load_stream,
    resample_aligned_stream,
    save_sync_model,
)

DEFAULT_SAMPLE_RATE_HZ = 100.0
DEFAULT_MAX_LAG_SECONDS = 60.0


def synchronize(
    reference_csv: Path | str,
    target_csv: Path | str,
    *,
    output_dir: Path | str,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = True,
    use_mag: bool = False,
    lowpass_cutoff_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    """Synchronize target stream to reference stream using SDA + LIDA."""
    ref_path = Path(reference_csv)
    tgt_path = Path(target_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = load_stream(ref_path)
    tgt_df = load_stream(tgt_path)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Reference and target streams must both be non-empty.")

    model = estimate_sync_model(
        ref_df,
        tgt_df,
        reference_name=str(ref_path),
        target_name=str(tgt_path),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
        min_window_score=min_window_score,
        min_fit_r2=min_fit_r2,
        use_acc=use_acc,
        use_gyro=use_gyro,
        use_mag=use_mag,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
    )

    sync_json_path = out_dir / f"{tgt_path.stem}_to_{ref_path.stem}_sync.json"
    save_sync_model(model, sync_json_path)

    correlations = compute_sync_correlations(ref_df, tgt_df, model, sample_rate_hz=sample_rate_hz)
    sync_data = json.loads(sync_json_path.read_text(encoding="utf-8"))
    sync_data["sync_method"] = "sda_lida"
    sync_data["correlation"] = correlations
    sync_json_path.write_text(json.dumps(sync_data, indent=2), encoding="utf-8")

    aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    drop_cols = [
        c for c in ("timestamp_orig", "timestamp_aligned", "timestamp_received")
        if c in aligned_df.columns
    ]
    if drop_cols:
        aligned_df = aligned_df.drop(columns=drop_cols)

    synced_csv_path = out_dir / f"{tgt_path.stem}_synced.csv"
    write_dataframe(aligned_df, synced_csv_path)

    uniform_csv_path: Path | None = None
    if resample_rate_hz is not None:
        uniform_df = resample_aligned_stream(
            aligned_df,
            resample_rate_hz=float(resample_rate_hz),
            timestamp_col="timestamp",
        )
        uniform_csv_path = out_dir / f"{tgt_path.stem}_synced_uniform.csv"
        write_dataframe(uniform_df, uniform_csv_path)

    return sync_json_path, synced_csv_path, uniform_csv_path


def synchronize_recording(
    recording_name: str,
    stage_in: str = "parsed",
    *,
    reference_sensor: str = "sporsa",
    target_sensor: str = "arduino",
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    max_lag_seconds: float = DEFAULT_MAX_LAG_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    window_step_seconds: float = DEFAULT_WINDOW_STEP_SECONDS,
    local_search_seconds: float = DEFAULT_LOCAL_SEARCH_SECONDS,
    min_window_score: float = DEFAULT_MIN_WINDOW_SCORE,
    min_fit_r2: float = DEFAULT_MIN_FIT_R2,
    resample_rate_hz: float | None = None,
    use_acc: bool = True,
    use_gyro: bool = False,
    use_mag: bool = False,
    plot: bool = False,
) -> tuple[Path, Path, Path]:
    """Synchronize two sensor streams for one recording using SDA + LIDA."""
    ref_csv = find_sensor_csv(recording_name, stage_in, reference_sensor)
    tgt_csv = find_sensor_csv(recording_name, stage_in, target_sensor)

    out_dir = recording_stage_dir(recording_name, "synced/lida")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{recording_name}/synced/lida] {reference_sensor} (ref) ← {ref_csv.name}")
    print(f"[{recording_name}/synced/lida] {target_sensor} (target) ← {tgt_csv.name}")

    tmp_dir = out_dir / "_tmp"
    try:
        sync_json_raw, synced_csv_raw, uniform_csv_raw = synchronize(
            reference_csv=ref_csv,
            target_csv=tgt_csv,
            output_dir=tmp_dir,
            sample_rate_hz=sample_rate_hz,
            max_lag_seconds=max_lag_seconds,
            window_seconds=window_seconds,
            window_step_seconds=window_step_seconds,
            local_search_seconds=local_search_seconds,
            min_window_score=min_window_score,
            min_fit_r2=min_fit_r2,
            resample_rate_hz=resample_rate_hz,
            use_acc=use_acc,
            use_gyro=use_gyro,
            use_mag=use_mag,
        )

        ref_out = out_dir / f"{reference_sensor}.csv"
        tgt_out = out_dir / f"{target_sensor}.csv"
        sync_json_out = out_dir / "sync_info.json"

        shutil.copy2(ref_csv, ref_out)
        shutil.move(str(synced_csv_raw), tgt_out)
        shutil.move(str(sync_json_raw), sync_json_out)

        if uniform_csv_raw is not None:
            uniform_out = out_dir / f"{target_sensor}_uniform.csv"
            shutil.move(str(uniform_csv_raw), uniform_out)
            print(f"[{recording_name}/synced/lida] {uniform_out.name}")

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    print(f"[{recording_name}/synced/lida] {ref_out.name}")
    print(f"[{recording_name}/synced/lida] {tgt_out.name}")
    print(f"[{recording_name}/synced/lida] {sync_json_out.name}")

    if plot:
        from visualization import plot_comparison

        stage_ref = f"{recording_name}/synced/lida"
        try:
            plot_comparison.main([stage_ref])
            plot_comparison.main([stage_ref, "--norm"])
        except SystemExit:
            pass

    return ref_out, tgt_out, sync_json_out
