"""Compatibility shim — EDA figures have moved to visualization.plot_exports."""

from visualization.plot_exports import (  # noqa: F401
    plot_feature_correlation,
    plot_feature_distributions_by_label,
    plot_label_distribution,
    plot_pca_by_label,
    plot_quality_distribution,
    plot_section_overview,
    run_eda,
)
