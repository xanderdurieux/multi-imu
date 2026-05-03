"""Evaluation package: train and evaluate models on extracted features."""

from evaluation.confusion_analysis import (
    analyze_confusion_matrix,
    write_confusion_analysis,
)
from evaluation.experiments import run_evaluation
from evaluation.imu_contribution import (
    compute_imu_contribution,
    per_class_f1_deltas,
    write_imu_contribution,
)
from evaluation.permutation_importance import (
    aggregate_by_sensor_group,
    compute_permutation_importance_grouped,
    write_permutation_importance,
)
from evaluation.sweep import run_evaluation_sweep

__all__ = [
    "run_evaluation",
    "run_evaluation_sweep",
    "analyze_confusion_matrix",
    "write_confusion_analysis",
    "compute_imu_contribution",
    "per_class_f1_deltas",
    "write_imu_contribution",
    "compute_permutation_importance_grouped",
    "aggregate_by_sensor_group",
    "write_permutation_importance",
]
