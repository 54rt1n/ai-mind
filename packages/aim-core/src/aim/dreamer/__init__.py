# aim/dreamer/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Dreamer Module - Strategy-based scenario execution.

Provides step-by-step execution of scenarios using the strategy pattern
with ScenarioBuilder, ScenarioFramework, ScenarioExecutor, and ScenarioState.

The legacy distributed server infrastructure has been moved to aim_legacy.dreamer.
"""

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
    # Dialogue models
    SpeakerType,
    DialogueSpeaker,
    DialogueTurn,
    DialogueStepDefinition,
    # Step definition types
    NewStepDefinition,
    ContextOnlyStepDefinition,
    StandardStepDefinition,
    ToolCallingStepDefinition,
    RenderingStepDefinition,
)
from .core.executor import RetryableError
from .core.scenario import load_scenario, render_template, build_template_context
from .core.builder import ScenarioBuilder, load_scenario_framework
from .core.framework import ScenarioFramework, DialogueConfig
from .core.state import ScenarioState
from .core.strategy import (
    BaseStepStrategy,
    ScenarioExecutor,
    ScenarioStepResult,
    ContextOnlyStrategy,
    StandardStrategy,
    ToolCallingStrategy,
    RenderingStrategy,
    DialogueStrategy,
    StepFactory,
)

__all__ = [
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
    # Dialogue models
    "SpeakerType",
    "DialogueSpeaker",
    "DialogueTurn",
    "DialogueStepDefinition",
    # Step definition types
    "NewStepDefinition",
    "ContextOnlyStepDefinition",
    "StandardStepDefinition",
    "ToolCallingStepDefinition",
    "RenderingStepDefinition",
    # Executor
    "RetryableError",
    # Scenario (legacy loader)
    "load_scenario",
    "render_template",
    "build_template_context",
    # New system
    "ScenarioBuilder",
    "load_scenario_framework",
    "ScenarioFramework",
    "DialogueConfig",
    "ScenarioState",
    # Strategies
    "BaseStepStrategy",
    "ScenarioExecutor",
    "ScenarioStepResult",
    "ContextOnlyStrategy",
    "StandardStrategy",
    "ToolCallingStrategy",
    "RenderingStrategy",
    "DialogueStrategy",
    "StepFactory",
]
