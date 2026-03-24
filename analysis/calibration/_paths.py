"""Local path helpers for calibration (plan: do not extend ``common.paths``)."""

from __future__ import annotations

from pathlib import Path

from static_calibration.imu_static import calibration_data_dir


def static_calibration_json_path() -> Path:
    """Path to ``arduino_imu_calibration.json`` under static calibration data dir."""
    return calibration_data_dir() / "arduino_imu_calibration.json"
