"""Bias calibration ordering experiment.

Tests whether applying static bias calibration *before* vs *after*
synchronisation changes the final alignment quality (Pearson r of acc_norm).

For each selected recording the experiment runs:

  1. Raw sync  — SDA+LIDA on the original parsed CSVs (no bias correction).
  2. Pre-sync bias-corrected — subtract static bias from the first 5 s
     of each sensor CSV *before* running SDA+LIDA.

The final correlation (offset + drift) is compared between conditions.
Expected result: negligible difference, confirming that calibration ordering
does not matter for timestamp synchronisation.

Usage::

    uv run run_bias_order_experiment.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import recordings_root
from orientation.calibration import estimate_bias_from_dataframe_static_segment, apply_calibration_bias
from sync.common import add_vector_norms, load_stream, resample_stream
from sync.drift_estimator import estimate_sync_model, apply_sync_model

SESSIONS = ["2026-02-26", "2025-12-04"]
OUT_DIR = Path(__file__).parent / "data"

# Recordings with reliably good LIDA sync to use (avoids confounding with poor coarse offset)
SELECTED_RECORDINGS = [
    "2025-12-04_3",
    "2025-12-04_6",
    "2025-12-04_7",
    "2025-12-04_8",
    "2026-02-26_4",
    "2026-02-26_6",
    "2026-02-26_8",
]
STATIC_WINDOW_MS = 5000.0
SAMPLE_RATE_HZ = 100.0


def _acc_norm_correlation(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> float | None:
    ref_r = add_vector_norms(resample_stream(ref_df, sample_rate_hz))
    tgt_r = add_vector_norms(resample_stream(tgt_df, sample_rate_hz))
    ref_ts = ref_r["timestamp"].to_numpy(dtype=float)
    tgt_ts = tgt_r["timestamp"].to_numpy(dtype=float)
    lo = max(float(ref_ts[0]), float(tgt_ts[0]))
    hi = min(float(ref_ts[-1]), float(tgt_ts[-1]))
    if lo >= hi:
        return None
    ref_acc = ref_r.loc[(ref_ts >= lo) & (ref_ts <= hi), "acc_norm"].to_numpy(dtype=float)
    tgt_acc = tgt_r.loc[(tgt_ts >= lo) & (tgt_ts <= hi), "acc_norm"].to_numpy(dtype=float)
    n = min(len(ref_acc), len(tgt_acc))
    if n < 10:
        return None
    x, y = ref_acc[:n], tgt_acc[:n]
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return None
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def _apply_bias_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate and remove static bias from the first 5 s of a DataFrame."""
    t0 = float(df["timestamp"].iloc[0])
    calib = estimate_bias_from_dataframe_static_segment(
        df,
        start_time=t0,
        end_time=t0 + STATIC_WINDOW_MS,
        expected_gravity_body=[0.0, 0.0, -9.81],
    )
    return apply_calibration_bias(df, accel_bias=calib.accel_bias, gyro_bias=calib.gyro_bias)


def _sync_and_score(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    recording: str,
    condition: str,
) -> dict:
    """Run SDA+LIDA sync and return quality metrics."""
    try:
        model = estimate_sync_model(
            ref_df, tgt_df,
            reference_name=recording,
            target_name=recording,
            sample_rate_hz=SAMPLE_RATE_HZ,
            max_lag_seconds=60.0,
        )
        aligned_df = apply_sync_model(tgt_df, model, replace_timestamp=True)

        # Correlation offset-only (no drift)
        offset_only = apply_sync_model(tgt_df, replace(model, drift_seconds_per_second=0.0),
                                       replace_timestamp=True)
        corr_offset = _acc_norm_correlation(ref_df, offset_only)
        corr_full = _acc_norm_correlation(ref_df, aligned_df)

        return {
            "recording": recording,
            "condition": condition,
            "offset_s": float(model.offset_seconds),
            "drift_ppm": float(model.drift_seconds_per_second * 1e6),
            "corr_offset_only": corr_offset,
            "corr_offset_drift": corr_full,
        }
    except Exception as e:
        return {
            "recording": recording,
            "condition": condition,
            "error": str(e),
        }


def run_experiment(recordings: list[str]) -> list[dict]:
    root = recordings_root()
    results = []

    for rec in recordings:
        ref_csv = root / rec / "parsed" / "sporsa.csv"
        tgt_csv = root / rec / "parsed" / "arduino.csv"
        if not ref_csv.exists() or not tgt_csv.exists():
            print(f"  [{rec}] parsed CSVs not found — skipping")
            continue

        ref_df = load_stream(ref_csv)
        tgt_df = load_stream(tgt_csv)

        print(f"  [{rec}] raw sync...")
        raw_result = _sync_and_score(ref_df, tgt_df, rec, "raw")

        print(f"  [{rec}] bias-corrected sync...")
        try:
            ref_cal = _apply_bias_calibration(ref_df.copy())
            tgt_cal = _apply_bias_calibration(tgt_df.copy())
            cal_result = _sync_and_score(ref_cal, tgt_cal, rec, "bias_corrected")
        except Exception as e:
            cal_result = {"recording": rec, "condition": "bias_corrected", "error": str(e)}

        results.append({"recording": rec, "raw": raw_result, "bias_corrected": cal_result})
        print(
            f"    corr (raw):       offset={raw_result.get('corr_offset_only', 'N/A')!r}  "
            f"full={raw_result.get('corr_offset_drift', 'N/A')!r}"
        )
        print(
            f"    corr (bias-corr): offset={cal_result.get('corr_offset_only', 'N/A')!r}  "
            f"full={cal_result.get('corr_offset_drift', 'N/A')!r}"
        )

    return results


def print_experiment_table(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  BIAS CALIBRATION ORDERING EXPERIMENT")
    print("  Q: Does applying bias calibration BEFORE sync change alignment quality?")
    print("=" * 80)
    header = (f"  {'Recording':<20} {'Condition':<18} "
              f"{'Corr offset':>12} {'Corr full':>12} {'Drift (ppm)':>13}")
    print(header)
    print("  " + "-" * 76)
    for r in results:
        rec = r["recording"]
        for cond_key in ("raw", "bias_corrected"):
            m = r.get(cond_key, {})
            if "error" in m:
                print(f"  {rec:<20} {cond_key:<18} ERROR: {m['error']}")
                continue
            corr_off = f"{m['corr_offset_only']:.4f}" if m.get("corr_offset_only") is not None else "  N/A"
            corr_full = f"{m['corr_offset_drift']:.4f}" if m.get("corr_offset_drift") is not None else "  N/A"
            drift = f"{m.get('drift_ppm', 0.0):.1f}"
            label = "raw" if cond_key == "raw" else "bias-corrected"
            print(f"  {rec:<20} {label:<18} {corr_off:>12} {corr_full:>12} {drift:>13}")
    print("=" * 80)
    # Compute mean difference
    deltas_offset = []
    deltas_full = []
    for r in results:
        raw = r.get("raw", {})
        cal = r.get("bias_corrected", {})
        ro = raw.get("corr_offset_only")
        co = cal.get("corr_offset_only")
        rf = raw.get("corr_offset_drift")
        cf = cal.get("corr_offset_drift")
        if ro is not None and co is not None:
            deltas_offset.append(co - ro)
        if rf is not None and cf is not None:
            deltas_full.append(cf - rf)
    if deltas_offset:
        print(f"\n  Mean delta corr_offset (bias_corrected - raw): "
              f"{np.mean(deltas_offset):+.4f}  (std={np.std(deltas_offset):.4f})")
    if deltas_full:
        print(f"  Mean delta corr_full   (bias_corrected - raw): "
              f"{np.mean(deltas_full):+.4f}  (std={np.std(deltas_full):.4f})")
    print()


def plot_experiment(results: list[dict], out_dir: Path) -> Path:
    recordings = [r["recording"] for r in results if "raw" in r and "bias_corrected" in r]
    raw_full = [r["raw"].get("corr_offset_drift") for r in results if "raw" in r]
    cal_full = [r["bias_corrected"].get("corr_offset_drift") for r in results if "bias_corrected" in r]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bias Calibration Ordering Experiment", fontsize=13)

    x = np.arange(len(recordings))
    width = 0.35
    ax = axes[0]
    raw_ok = [v if v is not None else 0.0 for v in raw_full]
    cal_ok = [v if v is not None else 0.0 for v in cal_full]
    ax.bar(x - width / 2, raw_ok, width, label="Raw (no bias correction)", color="steelblue", alpha=0.8)
    ax.bar(x + width / 2, cal_ok, width, label="Bias-corrected before sync", color="tomato", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.split("_", 1)[1] if "_" in r else r for r in recordings],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (acc_norm, offset+drift)")
    ax.set_title("Final alignment correlation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Scatter: raw vs bias-corrected
    ax2 = axes[1]
    valid = [(r, c) for r, c in zip(raw_ok, cal_ok)]
    rv = np.array([x[0] for x in valid])
    cv = np.array([x[1] for x in valid])
    ax2.scatter(rv, cv, s=70, color="steelblue", zorder=3)
    lo = min(np.min(rv), np.min(cv)) - 0.02
    hi = max(np.max(rv), np.max(cv)) + 0.02
    ax2.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="no change")
    ax2.set_xlabel("Corr (raw)")
    ax2.set_ylabel("Corr (bias-corrected)")
    ax2.set_title("Scatter: raw vs bias-corrected correlation")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bias_order_experiment.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bias ordering experiment plot saved -> {out_path}")
    return out_path


def save_results_json(results: list[dict], out_dir: Path) -> Path:
    out_path = out_dir / "bias_order_experiment.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Bias ordering experiment JSON saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    print(f"Running bias ordering experiment on {len(SELECTED_RECORDINGS)} recordings...\n")
    results = run_experiment(SELECTED_RECORDINGS)
    print_experiment_table(results)
    plot_experiment(results, OUT_DIR)
    save_results_json(results, OUT_DIR)
