"""Dynamic orientation filter comparison for cycling sections.

This module compares all available orientation variants already computed in
``section/orientation/*.csv`` and reports dynamic-usefulness metrics.

Focus metrics (per sensor):
- Inter-filter agreement over time (pairwise quaternion angle distance)
- Smoothness vs responsiveness tradeoff
- Drift behaviour on long low-dynamic windows
- Consistency of derived roll/pitch features
- Event-interpretation usefulness and feature separability proxy
- Magnetometer heading reliability flags

Outputs are written under ``<section>/orientation/comparison_dynamic``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import load_dataframe
from common.paths import iter_sections_for_recording

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: int
    end: int

    @property
    def n(self) -> int:
        return max(0, self.end - self.start)


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion angular distance in degrees, sign-invariant."""
    dots = np.abs(np.sum(q1 * q2, axis=1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def _find_longest_segment(mask: np.ndarray, min_len: int) -> Segment | None:
    best: Segment | None = None
    s = None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif (not v) and s is not None:
            seg = Segment(s, i)
            if seg.n >= min_len and (best is None or seg.n > best.n):
                best = seg
            s = None
    if s is not None:
        seg = Segment(s, len(mask))
        if seg.n >= min_len and (best is None or seg.n > best.n):
            best = seg
    return best


def _slice_time_df(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    out = df.iloc[start:end].copy()
    if not out.empty:
        t0 = float(out["timestamp"].iloc[0])
        out["time_s"] = (out["timestamp"].astype(float) - t0) / 1000.0
    return out


def _load_orientation_variants(section_path: Path, sensor: str) -> dict[str, pd.DataFrame]:
    orient_dir = section_path / "orientation"
    if not orient_dir.exists():
        return {}
    out: dict[str, pd.DataFrame] = {}
    for p in sorted(orient_dir.glob(f"{sensor}__*.csv")):
        variant = p.stem.split("__", 1)[1]
        df = load_dataframe(p)
        req = {"timestamp", "q0", "q1", "q2", "q3", "roll_deg", "pitch_deg", "yaw_deg"}
        if df.empty or not req.issubset(df.columns):
            continue
        out[variant] = df.sort_values("timestamp").reset_index(drop=True)
    return out


def _align_variants(variant_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Align all variant frames by timestamp inner-join."""
    if not variant_dfs:
        return {}
    ts_sets = [set(df["timestamp"].astype(float).to_numpy()) for df in variant_dfs.values()]
    common_ts = sorted(set.intersection(*ts_sets))
    if not common_ts:
        return {}
    idx_ts = pd.Index(common_ts)
    out: dict[str, pd.DataFrame] = {}
    for name, df in variant_dfs.items():
        sdf = df.set_index("timestamp").loc[idx_ts].reset_index()
        out[name] = sdf
    return out


def _motion_masks(cal_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (static_mask, dynamic_mask) from calibrated body-frame proxies."""
    gyro = cal_df[["gx", "gy", "gz"]].to_numpy(dtype=float)
    acc = cal_df[["ax", "ay", "az"]].to_numpy(dtype=float)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)
    static = (gyro_norm < 0.08) & (np.abs(acc_norm - 9.81) < 0.12 * 9.81)
    dynamic = (gyro_norm > 0.25) | (np.abs(acc_norm - 9.81) > 0.18 * 9.81)
    return static, dynamic


def _align_motion_mask(mask_ts: np.ndarray, mask: np.ndarray, target_ts: np.ndarray) -> np.ndarray:
    if len(mask_ts) == 0:
        return np.zeros(len(target_ts), dtype=bool)
    idx = np.searchsorted(mask_ts, target_ts, side="right") - 1
    idx = np.clip(idx, 0, len(mask) - 1)
    return mask[idx]


def _mag_unreliable_mask(cal_df: pd.DataFrame) -> np.ndarray:
    if not {"mx", "my", "mz"}.issubset(cal_df.columns):
        return np.zeros(len(cal_df), dtype=bool)
    mag = cal_df[["mx", "my", "mz"]].to_numpy(dtype=float)
    mag_norm = np.linalg.norm(mag, axis=1)
    if len(mag_norm) == 0 or not np.isfinite(mag_norm).any():
        return np.zeros(len(cal_df), dtype=bool)
    median_norm = float(np.nanmedian(mag_norm))
    rel_dev = np.abs(mag_norm - median_norm) / max(median_norm, 1e-6)

    heading = np.degrees(np.arctan2(mag[:, 1], mag[:, 0]))
    heading_unwrap = np.unwrap(np.radians(heading))
    dhead = np.abs(np.diff(np.degrees(heading_unwrap), prepend=np.degrees(heading_unwrap[0])))

    unreliable = (rel_dev > 0.2) | (dhead > 25.0)
    return np.asarray(unreliable, dtype=bool)


def _event_proxy(dynamic_mask: np.ndarray) -> np.ndarray:
    # Binary proxy used for feature-separability checks.
    return dynamic_mask.astype(int)


def _compute_metrics(
    aligned: dict[str, pd.DataFrame],
    static_mask: np.ndarray,
    dynamic_mask: np.ndarray,
    mag_unrel_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    variants = sorted(aligned.keys())
    ts = aligned[variants[0]]["timestamp"].to_numpy(dtype=float)
    dt = np.nanmedian(np.diff(ts)) / 1000.0 if len(ts) > 1 else 0.01
    dt = max(float(dt), 1e-4)

    # --- Inter-filter agreement ---
    pair_rows: list[dict[str, Any]] = []
    per_t_agreement = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a, b = variants[i], variants[j]
            qa = aligned[a][["q0", "q1", "q2", "q3"]].to_numpy(dtype=float)
            qb = aligned[b][["q0", "q1", "q2", "q3"]].to_numpy(dtype=float)
            ang = _quat_angle_deg(qa, qb)
            per_t_agreement.append(ang)
            pair_rows.append({
                "variant_a": a,
                "variant_b": b,
                "agreement_mean_deg": float(np.nanmean(ang)),
                "agreement_p95_deg": float(np.nanpercentile(ang, 95)),
                "agreement_dynamic_mean_deg": float(np.nanmean(ang[dynamic_mask])) if dynamic_mask.any() else np.nan,
            })
    pairwise_df = pd.DataFrame(pair_rows)
    interfilter_series = np.nanmean(np.vstack(per_t_agreement), axis=0) if per_t_agreement else np.zeros(len(ts))

    # --- Per-filter dynamics ---
    event = _event_proxy(dynamic_mask)
    rows: list[dict[str, Any]] = []
    rp_columns = []
    rp_values = []
    for v in variants:
        df = aligned[v]
        roll = np.unwrap(np.radians(df["roll_deg"].to_numpy(dtype=float)))
        pitch = np.unwrap(np.radians(df["pitch_deg"].to_numpy(dtype=float)))
        yaw = np.unwrap(np.radians(df["yaw_deg"].to_numpy(dtype=float)))

        droll = np.gradient(roll, dt)
        dpitch = np.gradient(pitch, dt)
        dyaw = np.gradient(yaw, dt)

        smoothness = float(np.nanstd(np.gradient(droll, dt)) + np.nanstd(np.gradient(dpitch, dt)))
        responsiveness = float(np.nanpercentile(np.abs(np.degrees(droll)) + np.abs(np.degrees(dpitch)), 95))
        tradeoff = responsiveness / max(smoothness, 1e-6)

        # drift: compare earliest and latest stable blocks
        drift = np.nan
        sidx = np.flatnonzero(static_mask)
        if len(sidx) >= 40:
            k = min(80, len(sidx) // 3)
            s1 = sidx[:k]
            s2 = sidx[-k:]
            drift = float(
                np.sqrt(
                    (np.degrees(np.nanmean(roll[s2]) - np.nanmean(roll[s1]))) ** 2
                    + (np.degrees(np.nanmean(pitch[s2]) - np.nanmean(pitch[s1]))) ** 2
                )
            )

        # derived-feature consistency payload
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        rp_columns.extend([f"{v}__roll", f"{v}__pitch"])
        rp_values.extend([roll_deg, pitch_deg])

        # Event separability proxy from roll/pitch-rate magnitude.
        rate_mag = np.abs(np.degrees(droll)) + np.abs(np.degrees(dpitch))
        dyn = rate_mag[event == 1]
        sta = rate_mag[event == 0]
        if len(dyn) >= 10 and len(sta) >= 10:
            mu1, mu0 = float(np.nanmean(dyn)), float(np.nanmean(sta))
            sd1, sd0 = float(np.nanstd(dyn)), float(np.nanstd(sta))
            pooled = np.sqrt((sd1 ** 2 + sd0 ** 2) / 2.0)
            separability = abs(mu1 - mu0) / max(pooled, 1e-6)
        else:
            separability = np.nan

        rows.append({
            "variant": v,
            "smoothness_score_rad_s2": smoothness,
            "responsiveness_p95_deg_s": responsiveness,
            "smoothness_responsiveness_ratio": tradeoff,
            "drift_static_endpoint_deg": drift,
            "roll_std_dynamic_deg": float(np.nanstd(roll_deg[dynamic_mask])) if dynamic_mask.any() else np.nan,
            "pitch_std_dynamic_deg": float(np.nanstd(pitch_deg[dynamic_mask])) if dynamic_mask.any() else np.nan,
            "event_separability_index": separability,
            "mag_unreliable_fraction": float(np.mean(mag_unrel_mask)),
        })

    per_filter_df = pd.DataFrame(rows)

    # Consistency across variants of derived roll/pitch (timepoint-wise spread)
    if rp_values:
        stack = np.vstack(rp_values)  # shape (2V, N)
        spread = np.nanstd(stack, axis=0)
        consistency = {
            "roll_pitch_feature_spread_mean_deg": float(np.nanmean(spread)),
            "roll_pitch_feature_spread_p95_deg": float(np.nanpercentile(spread, 95)),
            "interfilter_agreement_mean_deg": float(np.nanmean(interfilter_series)),
            "interfilter_agreement_p95_deg": float(np.nanpercentile(interfilter_series, 95)),
        }
    else:
        consistency = {}

    return per_filter_df, pairwise_df, consistency


def _plot_overlay(
    aligned: dict[str, pd.DataFrame],
    out_path: Path,
    section_name: str,
    sensor: str,
    segment: Segment,
    segment_label: str,
    mag_flag: np.ndarray,
) -> None:
    vars_sorted = sorted(aligned.keys())
    base = aligned[vars_sorted[0]]
    sl = slice(segment.start, segment.end)
    t = (base["timestamp"].to_numpy(dtype=float)[sl] - float(base["timestamp"].iloc[segment.start])) / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    for v in vars_sorted:
        df = aligned[v]
        axes[0].plot(t, df["roll_deg"].to_numpy(dtype=float)[sl], label=v, linewidth=1.2)
        axes[1].plot(t, df["pitch_deg"].to_numpy(dtype=float)[sl], label=v, linewidth=1.2)
        axes[2].plot(t, df["yaw_deg"].to_numpy(dtype=float)[sl], label=v, linewidth=1.2)

    if len(mag_flag) == len(base):
        m = mag_flag[sl]
        if np.any(m):
            ms = np.flatnonzero(m)
            if len(ms):
                t_m = t[ms]
                for ax in axes:
                    ax.scatter(t_m, np.interp(t_m, t, ax.lines[0].get_ydata()), s=12, c="red", alpha=0.35, marker="x")

    axes[0].set_ylabel("Roll [deg]")
    axes[1].set_ylabel("Pitch [deg]")
    axes[2].set_ylabel("Yaw [deg]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.25)

    fig.suptitle(f"{section_name} · {sensor} · {segment_label} overlay")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _choose_segment(mask: np.ndarray, fallback_len: int = 500) -> Segment:
    seg = _find_longest_segment(mask, min_len=100)
    if seg is not None:
        return seg
    n = len(mask)
    if n == 0:
        return Segment(0, 0)
    m = min(fallback_len, n)
    s = max(0, (n - m) // 2)
    return Segment(s, s + m)


def evaluate_section(section_path: Path, *, sensors: tuple[str, ...] = ("sporsa", "arduino")) -> dict[str, Any]:
    section_path = Path(section_path)
    out_root = section_path / "orientation" / "comparison_dynamic"
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"section": section_path.name, "sensors": {}}

    for sensor in sensors:
        aligned = _align_variants(_load_orientation_variants(section_path, sensor))
        if len(aligned) < 2:
            log.info("%s/%s: need >=2 orientation variants, skipping", section_path.name, sensor)
            continue

        cal_path = section_path / "calibrated" / f"{sensor}.csv"
        if not cal_path.exists():
            log.info("%s/%s: calibrated CSV missing", section_path.name, sensor)
            continue
        cal_df = load_dataframe(cal_path).sort_values("timestamp").reset_index(drop=True)

        ts_o = aligned[next(iter(aligned))]["timestamp"].to_numpy(dtype=float)
        ts_c = cal_df["timestamp"].to_numpy(dtype=float)
        static_c, dynamic_c = _motion_masks(cal_df)
        mag_bad_c = _mag_unreliable_mask(cal_df)
        static = _align_motion_mask(ts_c, static_c, ts_o)
        dynamic = _align_motion_mask(ts_c, dynamic_c, ts_o)
        mag_bad = _align_motion_mask(ts_c, mag_bad_c, ts_o)

        per_filter_df, pairwise_df, consistency = _compute_metrics(aligned, static, dynamic, mag_bad)

        sensor_out = out_root / sensor
        sensor_out.mkdir(parents=True, exist_ok=True)
        pf_path = sensor_out / "summary_per_filter.csv"
        pw_path = sensor_out / "summary_pairwise_agreement.csv"
        per_filter_df.to_csv(pf_path, index=False)
        pairwise_df.to_csv(pw_path, index=False)

        # static and dynamic overlay plots
        static_seg = _choose_segment(static)
        dynamic_seg = _choose_segment(dynamic)
        p_static = sensor_out / "overlay_static.png"
        p_dynamic = sensor_out / "overlay_dynamic.png"
        _plot_overlay(aligned, p_static, section_path.name, sensor, static_seg, "static-focused", mag_bad)
        _plot_overlay(aligned, p_dynamic, section_path.name, sensor, dynamic_seg, "dynamic-focused", mag_bad)

        recommended = None
        if not per_filter_df.empty:
            rank = per_filter_df.copy()
            rank["score"] = (
                0.40 * rank["smoothness_responsiveness_ratio"].rank(pct=True)
                + 0.25 * (1.0 - rank["drift_static_endpoint_deg"].fillna(rank["drift_static_endpoint_deg"].max()).rank(pct=True))
                + 0.20 * rank["event_separability_index"].fillna(0.0).rank(pct=True)
                + 0.15 * (1.0 - rank["mag_unreliable_fraction"].rank(pct=True))
            )
            best_row = rank.sort_values("score", ascending=False).iloc[0]
            recommended = str(best_row["variant"])

        summary["sensors"][sensor] = {
            "n_variants": len(aligned),
            "consistency": consistency,
            "magnetometer_unreliable_fraction": float(np.mean(mag_bad)) if len(mag_bad) else np.nan,
            "recommended_default_variant": recommended,
            "artifacts": {
                "per_filter_table": str(pf_path.relative_to(section_path)),
                "pairwise_table": str(pw_path.relative_to(section_path)),
                "overlay_static": str(p_static.relative_to(section_path)),
                "overlay_dynamic": str(p_dynamic.relative_to(section_path)),
            },
        }

    summary_path = out_root / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _recommendation_text(all_summaries: list[dict[str, Any]]) -> str:
    votes: dict[str, int] = {}
    mag_fracs: list[float] = []
    for s in all_summaries:
        for data in s.get("sensors", {}).values():
            v = data.get("recommended_default_variant")
            if v:
                votes[v] = votes.get(v, 0) + 1
            mf = data.get("magnetometer_unreliable_fraction")
            if mf is not None and np.isfinite(mf):
                mag_fracs.append(float(mf))

    if votes:
        winner = max(votes.items(), key=lambda kv: kv[1])[0]
    else:
        winner = "complementary_orientation"
    mag_mean = float(np.mean(mag_fracs)) if mag_fracs else np.nan

    conditions = (
        "Use magnetometer-assisted filters only when magnetic reliability flags stay low "
        f"(dataset mean unreliable fraction ≈ {mag_mean:.2f}). "
        "When flagged sections are frequent, prefer gyro+acc filters for robust relative orientation."
    )
    return (
        f"Recommended default: **{winner}** based on smoothness/responsiveness, drift, "
        "and event-separability aggregate scores. " + conditions
    )


def run(recording_or_section: str, *, all_sections: bool = False) -> Path:
    p = Path(recording_or_section)
    if p.exists() and p.is_dir() and (p / "orientation").exists():
        sections = [p]
        rec_name = p.name
        out_root = p / "orientation" / "comparison_dynamic"
    else:
        if all_sections:
            sections = iter_sections_for_recording(recording_or_section)
            if not sections:
                raise FileNotFoundError(f"No sections found for recording: {recording_or_section}")
            rec_name = recording_or_section
            out_root = sections[0].parent / f"{rec_name}__comparison_dynamic"
            out_root.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError("Pass a section path, or pass <recording_name> with --all-sections")

    all_summaries = []
    for sec in sections:
        summary = evaluate_section(sec)
        all_summaries.append(summary)

    if all_sections:
        rows = []
        for s in all_summaries:
            section = s.get("section")
            for sensor, data in s.get("sensors", {}).items():
                c = data.get("consistency", {})
                rows.append({
                    "section": section,
                    "sensor": sensor,
                    "recommended_default_variant": data.get("recommended_default_variant"),
                    "magnetometer_unreliable_fraction": data.get("magnetometer_unreliable_fraction"),
                    **c,
                })
        pd.DataFrame(rows).to_csv(out_root / "comparison_overall_summary.csv", index=False)

        interp = _recommendation_text(all_summaries)
        md = [
            "# Dynamic orientation filter comparison",
            "",
            f"Recording: `{rec_name}`",
            "",
            "## Recommendation",
            "",
            interp,
            "",
            "## Notes for thesis",
            "",
            "- Inter-filter agreement is reported as quaternion angular distance (mean and p95).",
            "- Smoothness vs responsiveness uses orientation angular acceleration noise vs p95 roll/pitch rate response.",
            "- Drift is endpoint difference between earliest/latest static windows.",
            "- Feature consistency is cross-filter spread of roll/pitch trajectories.",
            "- Event separability index compares dynamic vs static roll/pitch-rate distributions.",
            "- Heading reliability flags are triggered by magnetometer norm distortion and sudden heading jumps.",
        ]
        (out_root / "THESIS_INTERPRETATION.md").write_text("\n".join(md), encoding="utf-8")

    return out_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python -m orientation.compare_dynamic")
    parser.add_argument("target", help="Section path, or recording name when --all-sections")
    parser.add_argument("--all-sections", action="store_true", help="Run over all sections of a recording")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = run(args.target, all_sections=args.all_sections)
    print(f"Wrote dynamic orientation comparison artifacts to: {out}")
