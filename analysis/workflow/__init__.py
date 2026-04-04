"""Main workflow orchestration for the dual-IMU cycling pipeline.

Run all pipeline stages end-to-end from a single config file::

    python -m workflow configs/workflow.thesis.json
    python -m workflow configs/workflow.thesis.json --force
    python -m workflow configs/workflow.thesis.json --stage calibration
"""

from .config import WorkflowConfig, load_workflow_config
from .runner import run_pipeline

__all__ = ["WorkflowConfig", "load_workflow_config", "run_pipeline"]
