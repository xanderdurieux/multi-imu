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
from evaluation.label_grid import resolve_label_cols, run_label_grid_evaluation
from evaluation.trained_model import (
    TrainedModel,
    list_trained_models,
    load_trained_model,
    save_trained_model,
)

__all__ = [
    "run_evaluation",
    "run_label_grid_evaluation",
    "resolve_label_cols",
    "TrainedModel",
    "load_trained_model",
    "save_trained_model",
    "list_trained_models",
    "analyze_confusion_matrix",
    "write_confusion_analysis",
    "compute_imu_contribution",
    "per_class_f1_deltas",
    "write_imu_contribution",
    "compute_permutation_importance_grouped",
    "aggregate_by_sensor_group",
    "write_permutation_importance",
]
