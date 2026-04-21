"""Backward-compatible re-export of the split plot_exports_* modules.

The original monolithic file was split into three focused siblings:

- ``plot_exports_features``   — feature EDA (label distributions, PCA, violins)
- ``plot_exports_calibration`` — per-section calibration audits
- ``plot_exports_sync``       — per-recording / per-session sync audits
- ``plot_exports_orientation`` — per-recording / per-session orientation audits

Shared helpers live in ``_exports_common``.
"""

from __future__ import annotations

from visualization.plot_exports_calibration import (
    plot_calibration_quality_overview,
    plot_forward_confidence,
    plot_gravity_residuals,
    plot_sensor_biases,
    plot_static_calibration_reference,
    run_calibration_eda,
)
from visualization.plot_exports_features import (
    plot_feature_correlation,
    plot_feature_distributions_by_label,
    plot_label_distribution,
    plot_pca_by_label,
    plot_quality_distribution,
    plot_section_overview,
    run_eda,
)
from visualization.plot_exports_orientation import (
    plot_orientation_method_quality_counts,
    plot_orientation_method_residual_heatmap,
    plot_orientation_method_selection,
    plot_orientation_quality_overview,
    plot_orientation_selected_residuals,
    run_orientation_eda,
)
from visualization.plot_exports_sync import (
    plot_sync_calibration_anchor_overview,
    plot_sync_correlation_overview,
    plot_sync_drift_overview,
    plot_sync_method_availability,
    plot_sync_method_heatmap,
    plot_sync_method_selection,
    plot_sync_offset_overview,
    plot_sync_session_corr,
    plot_sync_session_drift,
    plot_sync_session_strip,
    run_sync_eda,
)

__all__ = [
    "plot_calibration_quality_overview",
    "plot_feature_correlation",
    "plot_feature_distributions_by_label",
    "plot_forward_confidence",
    "plot_gravity_residuals",
    "plot_label_distribution",
    "plot_orientation_method_quality_counts",
    "plot_orientation_method_residual_heatmap",
    "plot_orientation_method_selection",
    "plot_orientation_quality_overview",
    "plot_orientation_selected_residuals",
    "plot_pca_by_label",
    "plot_quality_distribution",
    "plot_section_overview",
    "plot_sensor_biases",
    "plot_static_calibration_reference",
    "plot_sync_calibration_anchor_overview",
    "plot_sync_correlation_overview",
    "plot_sync_drift_overview",
    "plot_sync_method_availability",
    "plot_sync_method_heatmap",
    "plot_sync_method_selection",
    "plot_sync_offset_overview",
    "plot_sync_session_corr",
    "plot_sync_session_drift",
    "plot_sync_session_strip",
    "run_calibration_eda",
    "run_eda",
    "run_orientation_eda",
    "run_sync_eda",
]
