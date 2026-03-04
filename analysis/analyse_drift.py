"""Arduino crystal drift characterisation from all sync_info.json files.

Reads drift estimates from both SDA+LIDA (synced_lida) and calibration-based
(synced_cal) methods across every available recording, prints a statistical
summary, and saves a distribution plot.

Usage::

    uv run analyse_drift.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import recordings_root

SESSIONS = ["2026-02-26", "2025-12-04"]
OUT_DIR = Path(__file__).parent / "data"

MAX_PLAUSIBLE_PPM = 10_000.0


def _load_drift_ppm(recording_name: str, stage: str) -> float | None:
    root = recordings_root()
    path = root / recording_name / stage / "sync_info.json"
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        drift_s_per_s = float(info.get("drift_seconds_per_second", 0.0))
        return drift_s_per_s * 1e6
    except Exception:
        return None


def _load_cal_span(recording_name: str, stage: str = "synced_cal") -> float | None:
    root = recordings_root()
    path = root / recording_name / stage / "sync_info.json"
    if not path.exists():
        return None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        return float(info.get("calibration", {}).get("calibration_span_s", 0.0))
    except Exception:
        return None


def collect_drift_data(sessions: list[str]) -> dict:
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and any(d.name.startswith(f"{s}_") for s in sessions)
    )
    rows: list[dict] = []
    for rec in recordings:
        lida_ppm = _load_drift_ppm(rec, "synced_lida")
        cal_ppm = _load_drift_ppm(rec, "synced_cal")
        cal_span = _load_cal_span(rec, "synced_cal")
        rows.append({
            "recording": rec,
            "lida_ppm": lida_ppm,
            "cal_ppm": cal_ppm,
            "cal_span_s": cal_span,
        })
    return {"rows": rows, "recordings": recordings}


def _filter_plausible(values: list, max_ppm: float = MAX_PLAUSIBLE_PPM) -> np.ndarray:
    arr = np.array([v for v in values if v is not None], dtype=float)
    return arr[np.abs(arr) <= max_ppm]


def print_summary(data: dict) -> None:
    rows = data["rows"]
    lida_vals = _filter_plausible([r["lida_ppm"] for r in rows])
    cal_vals = _filter_plausible([
        r["cal_ppm"] for r in rows
        if r["cal_ppm"] is not None and r["cal_span_s"] is not None and r["cal_span_s"] >= 60.0
    ])

    print("\n" + "=" * 60)
    print("  ARDUINO CRYSTAL DRIFT CHARACTERISATION")
    print("=" * 60)

    def _stats(label: str, vals: np.ndarray) -> None:
        if vals.size == 0:
            print(f"\n  {label}: no data")
            return
        print(f"\n  {label}  (n={vals.size})")
        print(f"    Mean   : {np.mean(vals):+.1f} ppm")
        print(f"    Median : {np.median(vals):+.1f} ppm")
        print(f"    Std    : {np.std(vals):.1f} ppm")
        print(f"    Min    : {np.min(vals):+.1f} ppm")
        print(f"    Max    : {np.max(vals):+.1f} ppm")
        print(f"    P5-P95 : {np.percentile(vals, 5):+.1f} to {np.percentile(vals, 95):+.1f} ppm")

    _stats("SDA+LIDA (all, |drift| <= 10000 ppm)", lida_vals)
    _stats("Cal-sync (span>=60s, |drift| <= 10000 ppm)", cal_vals)

    print("\n  Per-recording table:")
    print(f"  {'Recording':<20} {'LIDA (ppm)':>12} {'Cal (ppm)':>12} {'Cal span (s)':>14}  Note")
    print("  " + "-" * 72)
    for r in rows:
        lida = f"{r['lida_ppm']:+.1f}" if r["lida_ppm"] is not None else "  N/A"
        cal = f"{r['cal_ppm']:+.1f}" if r["cal_ppm"] is not None else "  N/A"
        span = f"{r['cal_span_s']:.1f}" if r["cal_span_s"] is not None else "  N/A"
        note = ""
        if r["lida_ppm"] is not None and abs(r["lida_ppm"]) > MAX_PLAUSIBLE_PPM:
            note += "[LIDA implausible] "
        if r["cal_ppm"] is not None and abs(r["cal_ppm"]) > MAX_PLAUSIBLE_PPM:
            note += "[Cal implausible] "
        if r["cal_span_s"] is not None and r["cal_span_s"] < 60.0:
            note += "[span<60s]"
        print(f"  {r['recording']:<20} {lida:>12} {cal:>12} {span:>14}  {note}")
    print("=" * 60 + "\n")


def plot_drift_distribution(data: dict, out_dir: Path) -> Path:
    rows = data["rows"]
    lida_ok = _filter_plausible([r["lida_ppm"] for r in rows])
    cal_ok_span = _filter_plausible([
        r["cal_ppm"] for r in rows
        if r["cal_ppm"] is not None and r["cal_span_s"] is not None and r["cal_span_s"] >= 60.0
    ])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Arduino Crystal Drift Distribution", fontsize=13)

    ax = axes[0]
    if lida_ok.size > 0:
        ax.hist(lida_ok, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(lida_ok), color="navy", lw=1.5, linestyle="--",
                   label=f"mean={np.mean(lida_ok):.0f}")
        ax.axvline(np.median(lida_ok), color="cornflowerblue", lw=1.5, linestyle=":",
                   label=f"median={np.median(lida_ok):.0f}")
    ax.set_xlabel("Drift (ppm)")
    ax.set_ylabel("Count")
    ax.set_title(f"SDA+LIDA  (n={lida_ok.size})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if cal_ok_span.size > 0:
        ax.hist(cal_ok_span, bins=12, color="tomato", edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(cal_ok_span), color="darkred", lw=1.5, linestyle="--",
                   label=f"mean={np.mean(cal_ok_span):.0f}")
        ax.axvline(np.median(cal_ok_span), color="salmon", lw=1.5, linestyle=":",
                   label=f"median={np.median(cal_ok_span):.0f}")
    ax.set_xlabel("Drift (ppm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cal-sync (span>=60s, n={cal_ok_span.size})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    both = [
        (r["lida_ppm"], r["cal_ppm"], r["cal_span_s"], r["recording"])
        for r in rows
        if r["lida_ppm"] is not None and r["cal_ppm"] is not None
    ]
    if both:
        lida_b = np.array([x[0] for x in both], dtype=float)
        cal_b = np.array([x[1] for x in both], dtype=float)
        spans = [x[2] for x in both]
        labels = [x[3] for x in both]
        plausible = (np.abs(lida_b) <= MAX_PLAUSIBLE_PPM) & (np.abs(cal_b) <= MAX_PLAUSIBLE_PPM)
        colors = ["steelblue" if (s is not None and s >= 60.0) else "orange" for s in spans]
        for i, (lv, cv, lab, col) in enumerate(zip(lida_b, cal_b, labels, colors)):
            if plausible[i]:
                ax.scatter(lv, cv, c=col, s=60, zorder=3)
                ax.annotate(lab, (lv, cv), fontsize=6, ha="left", va="bottom",
                            xytext=(3, 2), textcoords="offset points")
        all_vals = np.concatenate([lida_b[plausible], cal_b[plausible]])
        if len(all_vals) > 0:
            lo, hi = np.min(all_vals) - 200, np.max(all_vals) + 200
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="1:1")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
    ax.set_xlabel("LIDA drift (ppm)")
    ax.set_ylabel("Cal drift (ppm)")
    ax.set_title("LIDA vs Cal-sync drift\n(blue=span>=60s, orange=span<60s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drift_distribution.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Drift distribution plot saved -> {out_path}")
    return out_path


def save_drift_json(data: dict, out_dir: Path) -> Path:
    rows = data["rows"]
    lida_ok = _filter_plausible([r["lida_ppm"] for r in rows])
    cal_ok = _filter_plausible([
        r["cal_ppm"] for r in rows
        if r["cal_ppm"] is not None and r["cal_span_s"] is not None and r["cal_span_s"] >= 60.0
    ])

    def _stat(arr):
        if arr.size == 0:
            return None
        return {
            "n": int(arr.size),
            "mean_ppm": float(np.mean(arr)),
            "median_ppm": float(np.median(arr)),
            "std_ppm": float(np.std(arr)),
            "p5_ppm": float(np.percentile(arr, 5)),
            "p95_ppm": float(np.percentile(arr, 95)),
        }

    summary = {
        "lida": _stat(lida_ok),
        "cal_span_ge_60s": _stat(cal_ok),
        "recordings": data["rows"],
    }
    out_path = out_dir / "drift_characterisation.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Drift characterisation JSON saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    data = collect_drift_data(SESSIONS)
    print_summary(data)
    plot_drift_distribution(data, OUT_DIR)
    save_drift_json(data, OUT_DIR)
