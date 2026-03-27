"""Extended section-level quality assessment (calibration, sync proxy, features, usability)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import load_dataframe

log = logging.getLogger(__name__)


def _interpolated_fraction_proxy(df: pd.DataFrame, *, timestamp_col: str = "timestamp") -> float:
    """Upper-bound proxy: fraction of gaps > 2× median Δt (not true interpolation flags)."""
    ts = pd.to_numeric(df[timestamp_col], errors="coerce").dropna().to_numpy()
    if len(ts) < 3:
        return float("nan")
    dt = np.diff(ts)
    med = float(np.nanmedian(dt))
    if med <= 0:
        return float("nan")
    bad = np.sum(dt > 2.0 * med)
    return float(bad / max(len(dt), 1))


def assess_section(
    section_path: Path,
    *,
    max_gravity_residual: float = 0.8,
    max_interp_fraction: float = 0.15,
    min_duration_s: float = 5.0,
    min_windows: int = 3,
    max_feature_cv: float = 3.0,
    orientation_variant: str = "complementary_orientation",
) -> dict[str, Any]:
    """Return tier ``good`` / ``marginal`` / ``poor`` and human-readable *reasons*.

    Thresholds are intentionally simple baselines; override for stricter thesis gates.
    """
    section_path = Path(section_path)
    reasons: list[str] = []
    tier = 2  # 2=good, 1=marginal, 0=poor

    def downgrade(to: int, msg: str) -> None:
        nonlocal tier
        reasons.append(msg)
        tier = min(tier, to)

    cal_json = section_path / "calibrated" / "calibration.json"
    if cal_json.is_file():
        cal = json.loads(cal_json.read_text(encoding="utf-8"))
        for sensor, block in cal.items():
            if not isinstance(block, dict):
                continue
            res = float(block.get("gravity_residual_m_per_s2", 99))
            if res > max_gravity_residual:
                downgrade(1, f"{sensor}: high gravity residual ({res:.3f} m/s²)")
            fq = block.get("forward_frame_meta") or {}
            if fq.get("fallback"):
                downgrade(1, f"{sensor}: forward-frame alignment fell back (weak horizontal motion)")
            conf = fq.get("confidence_score", None)
            if conf is not None:
                try:
                    conf_f = float(conf)
                except Exception:
                    conf_f = float("nan")
                if np.isfinite(conf_f) and conf_f < 0.35:
                    downgrade(1, f"{sensor}: low section-frame confidence ({conf_f:.2f})")
            cq = block.get("calibration_quality", "")
            if cq == "poor":
                downgrade(0, f"{sensor}: calibration_quality=poor")
            elif cq == "marginal":
                downgrade(1, f"{sensor}: calibration_quality=marginal")
    else:
        downgrade(0, "missing calibrated/calibration.json")

    # Per-sensor stream usability
    for sensor in ("sporsa", "arduino"):
        csv_p = section_path / f"{sensor}.csv"
        if not csv_p.is_file():
            downgrade(0, f"missing {sensor}.csv")
            continue
        df = load_dataframe(csv_p)
        if df.empty or "timestamp" not in df.columns:
            downgrade(0, f"{sensor}: empty or no timestamp")
            continue
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        duration_s = (ts.max() - ts.min()) / 1000.0
        if duration_s < min_duration_s:
            downgrade(1, f"{sensor}: short duration ({duration_s:.1f}s)")
        interp = _interpolated_fraction_proxy(df)
        if np.isfinite(interp) and interp > max_interp_fraction:
            downgrade(1, f"{sensor}: irregular timing proxy {interp:.2%} (gap >2× median)")

    feat_csv = section_path / "features" / "features.csv"
    if feat_csv.is_file():
        fdf = pd.read_csv(feat_csv)
        if len(fdf) < min_windows:
            downgrade(1, f"few feature windows ({len(fdf)})")
        col = "sporsa__acc_norm_mean"
        if col in fdf.columns:
            v = fdf[col].dropna().to_numpy()
            if len(v) > 2:
                m = float(np.nanmean(v))
                s = float(np.nanstd(v))
                cv = s / m if abs(m) > 1e-6 else float("inf")
                if cv > max_feature_cv:
                    downgrade(1, f"high window-to-window CV on {col} ({cv:.2f})")
        unlabeled = fdf["scenario_label"].isna() | (fdf["scenario_label"].astype(str).str.strip() == "")
        if unlabeled.all() and len(fdf):
            downgrade(1, "all feature windows unlabeled")
    else:
        downgrade(1, "features not computed yet")

    # Orientation quality (reference sensor)
    orient_stats = section_path / "orientation" / "orientation_stats.json"
    if orient_stats.is_file():
        ost = json.loads(orient_stats.read_text(encoding="utf-8"))
        ref = ost.get("sporsa", {})
        key = f"__{orientation_variant}"
        block = ref.get(key, {})
        q = block.get("quality", "")
        if q == "poor":
            downgrade(1, f"sporsa orientation quality poor ({orientation_variant})")
        elif q == "marginal":
            downgrade(1, f"sporsa orientation quality marginal ({orientation_variant})")

    label = ("poor", "marginal", "good")[tier]
    out: dict[str, Any] = {
        "section_path": str(section_path),
        "quality_tier": label,
        "reasons": reasons,
    }
    return out


def write_section_qc(section_path: Path, **kwargs: Any) -> Path:
    """Run :func:`assess_section` and write ``qc_section.json`` into *section_path*."""
    section_path = Path(section_path)
    res = assess_section(section_path, **kwargs)
    out = section_path / "qc_section.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return out
