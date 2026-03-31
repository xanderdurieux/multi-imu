"""Sensor label and component configuration for plots."""

from __future__ import annotations

# Maps sensor type → list of (column_name, display_label, color) tuples
SENSOR_COMPONENTS: dict[str, list[tuple[str, str, str]]] = {
    "sporsa": [
        ("ax", "ax", "#1f77b4"),
        ("ay", "ay", "#ff7f0e"),
        ("az", "az", "#2ca02c"),
    ],
    "arduino": [
        ("ax", "ax", "#d62728"),
        ("ay", "ay", "#9467bd"),
        ("az", "az", "#8c564b"),
    ],
}

# Human-readable sensor display names
SENSOR_DISPLAY_NAMES: dict[str, str] = {
    "sporsa": "Bike (Sporsa)",
    "arduino": "Rider (Arduino)",
}

# Standard colors for each sensor
SENSOR_COLORS: dict[str, str] = {
    "sporsa": "#1f77b4",
    "arduino": "#ff7f0e",
}
