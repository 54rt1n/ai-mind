# aim/dreamer/core/strategy/__init__.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Step execution strategies for strategy-based scenario execution."""

from .base import (
    BaseStepStrategy,
    ScenarioExecutor,
    ScenarioStepResult,
)
from .factory import StepFactory
from .context_only import ContextOnlyStrategy
from .standard import StandardStrategy
from .tool_calling import ToolCallingStrategy
from .rendering import RenderingStrategy
from .dialogue import DialogueStrategy
from .functions import (
    execute_context_actions,
    load_memory_docs,
    load_step_docs,
)


__all__ = [
    # Base classes
    "BaseStepStrategy",
    "ScenarioExecutor",
    "ScenarioStepResult",
    # Factory
    "StepFactory",
    # Strategy implementations
    "ContextOnlyStrategy",
    "StandardStrategy",
    "ToolCallingStrategy",
    "RenderingStrategy",
    "DialogueStrategy",
    # Shared helper functions
    "execute_context_actions",
    "load_memory_docs",
    "load_step_docs",
]
