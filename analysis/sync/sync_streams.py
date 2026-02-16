"""Synchronize two processed IMU streams (manual or session mode).

No argparse is used intentionally; options are passed as key=value pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

from common import processed_session_dir
from .common import load_stream
from .drift_estimator import (
    apply_sync_model,
    estimate_sync_model,
    resample_aligned_stream,
    save_sync_model,
)


def _split_combined_sensor_csv(session_dir: Path) -> tuple[Path, Path] | None:
    combined = session_dir / "sensor.csv"
    if not combined.is_file():
        return None

    df = pd.read_csv(combined)
    sensor_col = next(
        (col for col in ("sensor", "source", "device", "stream", "imu") if col in df.columns),
        None,
    )
    if sensor_col is None:
        return None

    sensor = df[sensor_col].astype(str).str.lower()

    ref_mask = (
        sensor.str.contains("sporsa")
        | sensor.str.contains("handlebar")
        | sensor.str.contains("ref")
        | sensor.str.contains("reference")
    )
    tgt_mask = (
        sensor.str.contains("arduino")
        | sensor.str.contains("rider")
        | sensor.str.contains("tgt")
        | sensor.str.contains("target")
    )

    if not ref_mask.any() or not tgt_mask.any():
        return None

    ref_csv = session_dir / "sensor_sporsa.csv"
    tgt_csv = session_dir / "sensor_arduino.csv"

    df.loc[ref_mask].copy().to_csv(ref_csv, index=False)
    df.loc[tgt_mask].copy().to_csv(tgt_csv, index=False)

    return ref_csv, tgt_csv


def _parse_kv_args(items: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if value.lower() in {"true", "false"}:
            options[key] = value.lower() == "true"
            continue

        try:
            if "." in value:
                options[key] = float(value)
            else:
                options[key] = int(value)
            continue
        except ValueError:
            pass

        options[key] = value

    return options


def _pick_session_streams(session_name: str) -> tuple[Path, Path]:
    session_dir = processed_session_dir(session_name)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"processed session not found: {session_dir}")

    def _prefer_csv(name: str) -> Path | None:
        exact = session_dir / f"{name}.csv"
        if exact.is_file():
            return exact

        candidates = sorted(session_dir.glob(f"*{name}*.csv"))
        for c in candidates:
            stem = c.stem.lower()
            if any(token in stem for token in ("sync", "aligned", "resampled", "synced")):
                continue
            return c
        return None

    ref = _prefer_csv("sporsa")
    tgt = _prefer_csv("arduino")

    if ref is None or tgt is None:
        split = _split_combined_sensor_csv(session_dir)
        if split is not None:
            return split
        raise FileNotFoundError(
            f"could not infer sporsa/arduino CSVs under {session_dir}; "
            "also failed to split sensor.csv automatically; "
            "provide explicit files in manual mode"
        )

    return ref, tgt


def synchronize(
    reference_csv: Path,
    target_csv: Path,
    *,
    sample_rate_hz: float = 50.0,
    max_lag_seconds: float = 20.0,
    window_seconds: float = 20.0,
    window_step_seconds: float = 10.0,
    local_search_seconds: float = 2.0,
    write_resampled_hz: float | None = None,
) -> tuple[Path, Path, Path | None]:
    """Estimate sync, write model JSON, and write synchronized CSV."""
    ref_df = load_stream(reference_csv)
    tgt_df = load_stream(target_csv)

    model = estimate_sync_model(
        ref_df,
        tgt_df,
        reference_name=str(reference_csv),
        target_name=str(target_csv),
        sample_rate_hz=sample_rate_hz,
        max_lag_seconds=max_lag_seconds,
        window_seconds=window_seconds,
        window_step_seconds=window_step_seconds,
        local_search_seconds=local_search_seconds,
    )

    base = target_csv.parent
    sync_json = base / f"{target_csv.stem}_to_{reference_csv.stem}_sync.json"
    synced_csv = base / f"{target_csv.stem}_synced.csv"

    save_sync_model(model, sync_json)

    synced_df = apply_sync_model(tgt_df, model, replace_timestamp=True)
    synced_df.to_csv(synced_csv, index=False)

    resampled_csv: Path | None = None
    if write_resampled_hz is not None:
        resampled_df = resample_aligned_stream(
            synced_df,
            rate_hz=write_resampled_hz,
            timestamp_col="timestamp",
        )
        resampled_csv = base / f"{target_csv.stem}_synced_resampled_{write_resampled_hz:g}hz.csv"
        resampled_df.to_csv(resampled_csv, index=False)

    print(f"reference_csv={reference_csv}")
    print(f"target_csv={target_csv}")
    print(f"sync_json={sync_json}")
    print(f"synced_csv={synced_csv}")
    if resampled_csv is not None:
        print(f"resampled_csv={resampled_csv}")

    print(
        "model="
        f"offset_seconds={model.offset_seconds:.6f}, "
        f"drift_seconds_per_second={model.drift_seconds_per_second:.9f}, "
        f"scale={model.scale:.9f}, "
        f"fit_r2={model.fit_r2:.4f}, windows={model.num_windows}"
    )

    return sync_json, synced_csv, resampled_csv


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage:")
        print("  python -m sync.sync_streams <session_name> [key=value ...]")
        print("  python -m sync.sync_streams <reference_csv> <target_csv> [key=value ...]")
        return

    options = _parse_kv_args(argv)

    if len(argv) >= 2 and "=" not in argv[0] and "=" not in argv[1]:
        reference_csv = Path(argv[0])
        target_csv = Path(argv[1])
    elif "=" not in argv[0]:
        session_name = argv[0]
        reference_csv, target_csv = _pick_session_streams(session_name)
    else:
        raise ValueError("invalid arguments; provide session name or <reference_csv> <target_csv>")

    synchronize(
        reference_csv=reference_csv,
        target_csv=target_csv,
        sample_rate_hz=float(options.get("rate_hz", 50.0)),
        max_lag_seconds=float(options.get("max_lag_s", 20.0)),
        window_seconds=float(options.get("window_s", 20.0)),
        window_step_seconds=float(options.get("step_s", 10.0)),
        local_search_seconds=float(options.get("search_s", 2.0)),
        write_resampled_hz=(
            float(options["resample_hz"]) if "resample_hz" in options else None
        ),
    )


if __name__ == "__main__":
    main()
