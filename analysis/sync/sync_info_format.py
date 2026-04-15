"""Read/write layout for ``sync_info.json`` (nested schema + legacy flat)."""

from __future__ import annotations

from typing import Any

SYNC_INFO_SCHEMA_VERSION = 2


def build_sync_info_dict(
    *,
    model: dict[str, Any],
    meta: dict[str, Any],
    correlation: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the on-disk ``sync_info.json`` structure (schema v2)."""
    method_meta = {k: v for k, v in meta.items() if k not in ("calibration", "adaptive")}
    payload: dict[str, Any] = {
        "schema_version": SYNC_INFO_SCHEMA_VERSION,
        "model": dict(model),
        "method": method_meta,
        "correlation": dict(correlation),
    }
    if meta.get("calibration") is not None:
        payload["calibration"] = meta["calibration"]
    if meta.get("adaptive") is not None:
        payload["adaptive"] = meta["adaptive"]
    return payload


def flatten_sync_info_dict(info: dict[str, Any] | None) -> dict[str, Any] | None:
    """Merge v2 nested ``sync_info`` into the legacy flat dict shape."""
    if info is None:
        return None
    if info.get("schema_version") == SYNC_INFO_SCHEMA_VERSION or "model" in info:
        model = dict(info.get("model") or {})
        method = dict(info.get("method") or {})
        out: dict[str, Any] = {**model, **method}
        if "calibration" in info:
            out["calibration"] = info["calibration"]
        if "adaptive" in info:
            out["adaptive"] = info["adaptive"]
        if "correlation" in info:
            out["correlation"] = info["correlation"]
        return out
    return info
