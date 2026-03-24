"""Per-section feature extraction from calibrated and orientation data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from common import load_dataframe

log = logging.getLogger(__name__)

GRAVITY_M_S2 = 9.81
ACC_COLS = ["ax", "ay", "az"]
GYRO_COLS = ["gx", "gy", "gz"]

PER_SENSOR_FEATURES = [
    "acc_norm_mean",
    "acc_norm_max",
    "acc_norm_energy",
    "jerk_norm_max",
    "gyro_norm_max",
    "gyro_energy",
    "vertical_acc_mean",
    "vertical_acc_std",
]

CROSS_SENSOR_FEATURES = [
    "acc_norm_corr",
    "acc_norm_lag_ms",
    "acc_energy_ratio",
    "gyro_energy_ratio",
    "pitch_corr",
    "pitch_divergence_std",
]


def _acc_norm(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(arr * arr, axis=1))


def _gyro_norm(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.nansum(arr * arr, axis=1))


def _extract_per_sensor_features(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Extract per-sensor features for one window."""
    acc_n = _acc_norm(acc)
    gyro_n = _gyro_norm(gyro)
    vertical_acc = acc[:, 2] if acc.shape[1] > 2 else np.full(len(acc), np.nan)

    jerk = np.abs(np.diff(acc_n, prepend=acc_n[0])) / max(dt, 1e-9)

    return {
        "acc_norm_mean": float(np.nanmean(acc_n)),
        "acc_norm_max": float(np.nanmax(acc_n)),
        "acc_norm_energy": float(np.nansum(acc_n * acc_n)),
        "jerk_norm_max": float(np.nanmax(jerk)),
        "gyro_norm_max": float(np.nanmax(gyro_n)),
        "gyro_energy": float(np.nansum(gyro_n * gyro_n)),
        "vertical_acc_mean": float(np.nanmean(vertical_acc)),
        "vertical_acc_std": float(np.nanstd(vertical_acc)) if len(vertical_acc) > 1 else 0.0,
    }


def _cross_corr_lag_ms(
    a: np.ndarray,
    b: np.ndarray,
    dt_ms: float,
    max_lag_ms: float = 500.0,
) -> float:
    """Lag in ms at max cross-correlation (-max_lag to +max_lag)."""
    if len(a) < 2 or len(b) < 2 or not np.any(np.isfinite(a)) or not np.any(np.isfinite(b)):
        return np.nan
    a = np.nan_to_num(a, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    corr = np.correlate(a, b, mode="full")
    mid = len(corr) // 2
    dt_ms = max(dt_ms, 0.1)
    max_lag_samp = int(max_lag_ms / dt_ms)
    lo = max(0, mid - max_lag_samp)
    hi = min(len(corr), mid + max_lag_samp + 1)
    region = corr[lo:hi]
    idx = np.argmax(region)
    lag_samp = idx - (mid - lo)
    return float(lag_samp * dt_ms)


def _align_to_common_time(
    ts_a: np.ndarray,
    vals_a: np.ndarray,
    ts_b: np.ndarray,
    vals_b: np.ndarray,
    t_center: float,
    half_window_s: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract aligned segments for cross-sensor features. Returns (a, b) or None."""
    t_lo = t_center - half_window_s
    t_hi = t_center + half_window_s
    mask_a = (ts_a >= t_lo) & (ts_a <= t_hi)
    mask_b = (ts_b >= t_lo) & (ts_b <= t_hi)
    if np.sum(mask_a) < 5 or np.sum(mask_b) < 5:
        return None
    # Resample to common grid for correlation (use finer grid)
    grid = np.linspace(t_lo, t_hi, max(20, min(np.sum(mask_a), np.sum(mask_b))))
    a_interp = np.interp(grid, ts_a[mask_a], vals_a[mask_a])
    b_interp = np.interp(grid, ts_b[mask_b], vals_b[mask_b])
    return a_interp, b_interp


def extract_section(
    section_path: Path,
    section_name: str,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    write_plots: bool = True,
) -> pd.DataFrame:
    """Extract features for a section. Returns DataFrame with one row per window."""
    section_path = Path(section_path)
    calibrated_dir = section_path / "calibrated"
    orient_dir = section_path / "orientation"
    features_dir = section_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    # Load calibrated data
    dfs: dict[str, pd.DataFrame] = {}
    orient_dfs: dict[str, pd.DataFrame] = {}
    for sensor in ("sporsa", "arduino"):
        p = calibrated_dir / f"{sensor}.csv"
        if p.exists():
            df = load_dataframe(p)
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            dfs[sensor] = df
        p_orient = orient_dir / f"{sensor}__complementary_orientation.csv"
        if p_orient.exists():
            orient_dfs[sensor] = pd.read_csv(p_orient)

    if not dfs:
        raise FileNotFoundError(f"No calibrated data in {calibrated_dir}")

    t0 = float(dfs[list(dfs.keys())[0]]["timestamp"].iloc[0])
    t_end = float(dfs[list(dfs.keys())[0]]["timestamp"].iloc[-1])
    duration_s = (t_end - t0) / 1000.0

    half_window_s = window_s / 2.0
    rows = []

    t_center_s = half_window_s
    while t_center_s + half_window_s <= duration_s:
        t_center_ms = t0 + t_center_s * 1000.0
        window_start_s = (t_center_ms - t0) / 1000.0 - half_window_s
        window_end_s = window_start_s + window_s

        row: dict[str, Any] = {
            "section": section_name,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "window_center_s": t_center_s,
        }

        for sensor, df in dfs.items():
            ts_raw = df["timestamp"].to_numpy(dtype=float)
            ts = (ts_raw - t0) / 1000.0
            mask = (ts >= window_start_s) & (ts <= window_end_s)
            if np.sum(mask) < 5:
                for f in PER_SENSOR_FEATURES:
                    row[f"{sensor}__{f}"] = np.nan
            else:
                acc = df.loc[mask, ACC_COLS].to_numpy(dtype=float)
                gyro = df.loc[mask, GYRO_COLS].to_numpy(dtype=float)
                dt = float(np.nanmean(np.diff(ts_raw[mask]))) / 1000.0 if np.sum(mask) > 1 else 0.01
                feats = _extract_per_sensor_features(acc, gyro, dt)
                for k, v in feats.items():
                    row[f"{sensor}__{k}"] = v

        # Cross-sensor features
        if "sporsa" in dfs and "arduino" in dfs:
            sporsa_df = dfs["sporsa"]
            arduino_df = dfs["arduino"]
            ts_s = (sporsa_df["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
            ts_a = (arduino_df["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
            acc_n_s = _acc_norm(sporsa_df[ACC_COLS].to_numpy(dtype=float))
            acc_n_a = _acc_norm(arduino_df[ACC_COLS].to_numpy(dtype=float))
            dt_ms = 1000.0 * float(np.nanmean(np.diff(sporsa_df["timestamp"]))) if len(sporsa_df) > 1 else 10.0

            aligned = _align_to_common_time(
                ts_s, acc_n_s, ts_a, acc_n_a, t_center_s, half_window_s
            )
            if aligned is not None:
                a_s, a_a = aligned
                if len(a_s) > 2 and (np.ptp(a_s) > 1e-12 or np.ptp(a_a) > 1e-12):
                    try:
                        corr = stats.pearsonr(a_s, a_a)[0]
                    except Exception:
                        corr = np.nan
                else:
                    corr = np.nan
                row["acc_norm_corr"] = corr if np.isfinite(corr) else np.nan
                dt_interp_ms = 1000.0 * window_s / max(len(a_s), 1)
                row["acc_norm_lag_ms"] = _cross_corr_lag_ms(a_s, a_a, dt_interp_ms)
            else:
                row["acc_norm_corr"] = np.nan
                row["acc_norm_lag_ms"] = np.nan

            e_s = row.get("sporsa__acc_norm_energy", np.nan)
            e_a = row.get("arduino__acc_norm_energy", np.nan)
            row["acc_energy_ratio"] = e_s / e_a if np.isfinite(e_a) and e_a != 0 else np.nan

            g_s = row.get("sporsa__gyro_energy", np.nan)
            g_a = row.get("arduino__gyro_energy", np.nan)
            row["gyro_energy_ratio"] = g_s / g_a if np.isfinite(g_a) and g_a != 0 else np.nan

            if "sporsa" in orient_dfs and "arduino" in orient_dfs:
                od_s = orient_dfs["sporsa"]
                od_a = orient_dfs["arduino"]
                ts_os = (od_s["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
                ts_oa = (od_a["timestamp"].to_numpy(dtype=float) - t0) / 1000.0
                pitch_s = od_s["pitch_deg"].to_numpy(dtype=float)
                pitch_a = od_a["pitch_deg"].to_numpy(dtype=float)
                al = _align_to_common_time(ts_os, pitch_s, ts_oa, pitch_a, t_center_s, half_window_s)
                if al is not None:
                    ps, pa = al
                    if len(ps) > 2 and (np.ptp(ps) > 1e-12 or np.ptp(pa) > 1e-12):
                        try:
                            row["pitch_corr"] = stats.pearsonr(ps, pa)[0]
                        except Exception:
                            row["pitch_corr"] = np.nan
                    else:
                        row["pitch_corr"] = np.nan
                    row["pitch_divergence_std"] = float(np.nanstd(ps - pa))
                else:
                    row["pitch_corr"] = np.nan
                    row["pitch_divergence_std"] = np.nan
            else:
                row["pitch_corr"] = np.nan
                row["pitch_divergence_std"] = np.nan
        else:
            for f in CROSS_SENSOR_FEATURES:
                row[f] = np.nan

        row["scenario_label"] = ""
        rows.append(row)

        t_center_s += hop_s

    out_df = pd.DataFrame(rows)
    out_path = features_dir / "features.csv"
    out_df.to_csv(out_path, index=False)

    # features_stats.json
    stats_dict: dict[str, dict[str, float]] = {}
    for col in out_df.select_dtypes(include=[np.number]).columns:
        vals = out_df[col].dropna()
        stats_dict[col] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "nan_fraction": float(out_df[col].isna().mean()),
        }
    with (features_dir / "features_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats_dict, f, indent=2)

    if write_plots and len(out_df) > 0:
        from .plots import plot_features_timeline
        plot_features_timeline(features_dir, out_df)

    return out_df


def _parse_arg(name: str) -> tuple[str, str | None, bool]:
    """Return (recording, section, is_session)."""
    name = name.strip().rstrip("/").replace("\\", "/")
    parts = name.split("/")
    rec = parts[0]
    sec = None
    for i, p in enumerate(parts):
        if p == "sections" and i + 1 < len(parts):
            sec = parts[i + 1]
            break
    is_session = "_" not in rec or rec.split("_")[-1].isdigit() is False
    return rec, sec, is_session


def extract_from_args(
    name: str,
    *,
    all_sections: bool = False,
    all_recordings: bool = False,
    window_s: float = 1.0,
    hop_s: float = 0.5,
) -> list[Path]:
    """Run feature extraction from CLI args."""
    from common import recording_dir, recordings_root

    rec, sec, _ = _parse_arg(name)

    if all_recordings:
        rec_dirs = sorted(
            d for d in (recordings_root()).iterdir()
            if d.is_dir() and d.name.startswith(rec + "_")
        )
    else:
        rec_dir = recording_dir(rec)
        if not rec_dir.exists():
            raise FileNotFoundError(f"Recording not found: {rec_dir}")
        rec_dirs = [rec_dir]

    done = []
    for rdir in rec_dirs:
        rec_name = rdir.name
        sections_root = rdir / "sections"
        if not sections_root.exists():
            log.warning("No sections in %s", rec_name)
            continue
        if all_sections or all_recordings:
            section_dirs = sorted(
                d for d in sections_root.iterdir()
                if d.is_dir() and d.name.startswith("section_")
            )
        else:
            if sec is None:
                raise ValueError("Specify section or use --all-sections / --all")
            section_dirs = [sections_root / sec]
            if not section_dirs[0].exists():
                raise FileNotFoundError(f"Section not found: {section_dirs[0]}")

        for sdir in section_dirs:
            section_name = f"{rec_name}/{sdir.name}"
            cal_dir = sdir / "calibrated"
            if not (cal_dir / "calibration.json").exists():
                log.warning("Skipping %s (no calibration)", section_name)
                continue
            extract_section(sdir, section_name, window_s=window_s, hop_s=hop_s)
            done.append(sdir)

    return done


if __name__ == "__main__":
    import argparse
    import logging
    parser = argparse.ArgumentParser(prog="python -m features.extract")
    parser.add_argument("name", help="Section path, recording, or session")
    parser.add_argument("--all-sections", action="store_true")
    parser.add_argument("--all", action="store_true", dest="all_recordings")
    parser.add_argument("--window", type=float, default=1.0)
    parser.add_argument("--hop", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    extract_from_args(
        args.name,
        all_sections=args.all_sections,
        all_recordings=args.all_recordings,
        window_s=args.window,
        hop_s=args.hop,
    )
