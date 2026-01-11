# aim/dreamer/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Dreamer Module - DAG-based pipeline orchestration.

Provides step-by-step execution of scenarios with Redis-backed state
management and queue-based job distribution.
"""

from .server.api import (
    start_pipeline,
    get_status,
    cancel_pipeline,
    delete_pipeline,
    resume_pipeline,
    refresh_pipeline,
    list_pipelines,
    generate_pipeline_id,
    PipelineStatus,
    ResumeResult,
)
from .core.models import (
    PipelineState,
    StepResult,
    StepJob,
    StepStatus,
    Scenario,
    StepDefinition,
    StepConfig,
    StepOutput,
    ScenarioContext,
    MemoryAction,
)
from .server.worker import DreamerWorker, run_worker
from .core.executor import RetryableError
from .server.state import StateStore
from .server.scheduler import Scheduler
from .core.scenario import load_scenario, render_template, build_template_context

__all__ = [
    # API
    "start_pipeline",
    "get_status",
    "cancel_pipeline",
    "delete_pipeline",
    "resume_pipeline",
    "refresh_pipeline",
    "list_pipelines",
    "generate_pipeline_id",
    "PipelineStatus",
    "ResumeResult",
    # Models
    "PipelineState",
    "StepResult",
    "StepJob",
    "StepStatus",
    "Scenario",
    "StepDefinition",
    "StepConfig",
    "StepOutput",
    "ScenarioContext",
    "MemoryAction",
    # Worker
    "DreamerWorker",
    "run_worker",
    # Executor
    "RetryableError",
    # State
    "StateStore",
    # Scheduler
    "Scheduler",
    # Scenario
    "load_scenario",
    "render_template",
    "build_template_context",
]
