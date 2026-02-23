"""Sensor metadata labels and components."""

from __future__ import annotations

SENSOR_COMPONENTS: dict[str, tuple[str, str, str]] = {
    "acc": ("ax", "ay", "az"),
    "gyro": ("gx", "gy", "gz"),
    "mag": ("mx", "my", "mz"),
}

SENSOR_LABELS: dict[str, tuple[str, str]] = {
    "acc": ("Accelerometer", "m/s²"),
    "gyro": ("Gyroscope", "deg/s"),
    "mag": ("Magnetometer", "µT"),
}
