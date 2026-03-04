"""Three-method sync comparison: SDA+LIDA vs Cal-sync vs Online-single-anchor.

For recordings that have all three stages (synced_lida, synced_cal, synced_online),
compare the estimated offset, drift, and final correlation side-by-side.

The online method uses only the opening calibration anchor with a pre-characterised
drift rate. This script shows how much accuracy is lost by not having the closing
calibration.

Usage::

    uv run run_three_method_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import recordings_root
from sync.online_sync import synchronize_recording_online

SESSIONS = ["2026-02-26", "2025-12-04"]
OUT_DIR = Path(__file__).parent / "data"


def _load_sync_info(recording_name: str, stage: str) -> dict | None:
    root = recordings_root()
    path = root / recording_name / stage / "sync_info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_online_stage(recording_name: str) -> bool:
    """Run online sync if synced_online stage not present. Returns True on success."""
    root = recordings_root()
    sync_json = root / recording_name / "synced_online" / "sync_info.json"
    if sync_json.exists():
        return True
    try:
        synchronize_recording_online(recording_name, stage_in="parsed", plot=False)
        return True
    except Exception as e:
        return False


def compare_three_methods(recording_name: str) -> dict | None:
    lida = _load_sync_info(recording_name, "synced_lida")
    cal = _load_sync_info(recording_name, "synced_cal")
    online = _load_sync_info(recording_name, "synced_online")

    if lida is None and cal is None and online is None:
        return None

    def _ppm(info):
        if info is None:
            return None
        return float(info.get("drift_seconds_per_second", 0.0)) * 1e6

    def _corr(info):
        if info is None:
            return None, None
        c = info.get("correlation", {})
        return c.get("offset_only"), c.get("offset_and_drift")

    lida_off = lida["offset_seconds"] if lida else None
    cal_off = cal["offset_seconds"] if cal else None
    online_off = online["offset_seconds"] if online else None

    return {
        "recording": recording_name,
        "lida_offset_s": lida_off,
        "cal_offset_s": cal_off,
        "online_offset_s": online_off,
        "delta_cal_lida_s": (cal_off - lida_off) if (lida_off is not None and cal_off is not None) else None,
        "delta_online_cal_s": (online_off - cal_off) if (cal_off is not None and online_off is not None) else None,
        "lida_drift_ppm": _ppm(lida),
        "cal_drift_ppm": _ppm(cal),
        "online_drift_ppm": _ppm(online),
        "lida_corr_full": _corr(lida)[1],
        "cal_corr_full": _corr(cal)[1],
        "online_corr_full": _corr(online)[1],
    }


def print_three_method_table(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("  THREE-METHOD SYNC COMPARISON: SDA+LIDA  |  Cal-sync  |  Online (opening anchor only)")
    print("=" * 100)
    hdr = (f"  {'Recording':<20} {'LIDA off':>12} {'Cal off':>12} {'Online off':>12} "
           f"{'Δcal-lida':>10} {'Δon-cal':>10}  {'LIDA r':>8} {'Cal r':>8} {'Online r':>8}")
    print(hdr)
    print("  " + "-" * 95)
    for r in results:
        def _f(v, fmt):
            return f"{v:{fmt}}" if v is not None else "  N/A"
        print(
            f"  {r['recording']:<20} "
            f"{_f(r['lida_offset_s'], '.3f'):>12} "
            f"{_f(r['cal_offset_s'], '.3f'):>12} "
            f"{_f(r['online_offset_s'], '.3f'):>12} "
            f"{_f(r['delta_cal_lida_s'], '+.3f'):>10} "
            f"{_f(r['delta_online_cal_s'], '+.3f'):>10}  "
            f"{_f(r['lida_corr_full'], '.4f'):>8} "
            f"{_f(r['cal_corr_full'], '.4f'):>8} "
            f"{_f(r['online_corr_full'], '.4f'):>8}"
        )
    print("=" * 100 + "\n")

    online_vs_cal = [
        abs(r["delta_online_cal_s"])
        for r in results
        if r["delta_online_cal_s"] is not None
    ]
    if online_vs_cal:
        print(f"  Online vs Cal-sync offset difference:")
        print(f"    Mean   |Δ|: {np.mean(online_vs_cal):.3f} s")
        print(f"    Median |Δ|: {np.median(online_vs_cal):.3f} s")
        print(f"    Max    |Δ|: {np.max(online_vs_cal):.3f} s")
    print()


def plot_three_method_comparison(results: list[dict], out_dir: Path) -> Path:
    valid = [r for r in results if r["lida_offset_s"] is not None]
    if not valid:
        return None

    recordings = [r["recording"] for r in valid]
    x = np.arange(len(recordings))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Three-method Synchronisation Comparison", fontsize=13)

    ax = axes[0]
    corr_lida = [r["lida_corr_full"] or 0.0 for r in valid]
    corr_cal = [r["cal_corr_full"] or 0.0 for r in valid]
    corr_online = [r["online_corr_full"] or 0.0 for r in valid]
    w = 0.25
    ax.bar(x - w, corr_lida, w, label="SDA+LIDA", color="steelblue", alpha=0.8)
    ax.bar(x, corr_cal, w, label="Cal-sync", color="tomato", alpha=0.8)
    ax.bar(x + w, corr_online, w, label="Online (opening only)", color="forestgreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r.split("_", 1)[1] if "_" in r else r for r in recordings],
                       rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (acc_norm, offset+drift)")
    ax.set_title("Final alignment correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    delta_online_cal = [r["delta_online_cal_s"] for r in valid if r["delta_online_cal_s"] is not None]
    recs_with_delta = [r["recording"] for r in valid if r["delta_online_cal_s"] is not None]
    if delta_online_cal:
        colors = ["tomato" if abs(d) > 1.0 else "steelblue" for d in delta_online_cal]
        bars = ax.bar(range(len(delta_online_cal)), delta_online_cal, color=colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1.0, color="gray", lw=0.8, linestyle="--", label="|Δ|=1s threshold")
        ax.axhline(-1.0, color="gray", lw=0.8, linestyle="--")
        ax.set_xticks(range(len(delta_online_cal)))
        ax.set_xticklabels([r.split("_", 1)[1] if "_" in r else r for r in recs_with_delta],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Offset difference (s)")
        ax.set_title("Online − Cal-sync offset (s)\n(blue=|Δ|<1s, red=|Δ|>1s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "three_method_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Three-method comparison plot saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    root = recordings_root()
    recordings = sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and any(d.name.startswith(f"{s}_") for s in SESSIONS)
    )

    print("Ensuring synced_online stage exists for all eligible recordings...")
    for rec in recordings:
        has_lida = (root / rec / "synced_lida" / "sync_info.json").exists()
        if has_lida:
            ok = _ensure_online_stage(rec)
            status = "ok" if ok else "failed (no cal sequences)"
            print(f"  [{rec}] {status}")

    results = [r for rec in recordings if (r := compare_three_methods(rec)) is not None]
    print_three_method_table(results)
    plot_three_method_comparison(results, OUT_DIR)

    out_path = OUT_DIR / "three_method_comparison.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Three-method comparison JSON saved -> {out_path}")
