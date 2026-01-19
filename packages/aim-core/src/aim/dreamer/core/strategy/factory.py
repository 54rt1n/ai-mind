# aim/dreamer/core/strategy/factory.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""StepFactory - creates appropriate strategy for each step type."""

import logging
from typing import TYPE_CHECKING

from .base import BaseStepStrategy, ScenarioExecutor
from .context_only import ContextOnlyStrategy
from .standard import StandardStrategy
from .tool_calling import ToolCallingStrategy
from .rendering import RenderingStrategy

if TYPE_CHECKING:
    from ..models import NewStepDefinition


logger = logging.getLogger(__name__)


class StepFactory:
    """Factory for creating step strategies based on step type.

    Maps step definition types to strategy classes:
    - context_only -> ContextOnlyStrategy
    - standard -> StandardStrategy
    - tool_calling -> ToolCallingStrategy
    - rendering -> RenderingStrategy
    """

    # Map step type to strategy class
    _strategy_map = {
        "context_only": ContextOnlyStrategy,
        "standard": StandardStrategy,
        "tool_calling": ToolCallingStrategy,
        "rendering": RenderingStrategy,
    }

    @classmethod
    def create(
        cls,
        executor: ScenarioExecutor,
        step_def: "NewStepDefinition",
    ) -> BaseStepStrategy:
        """Create the appropriate strategy for a step definition.

        Args:
            executor: ScenarioExecutor with shared resources
            step_def: Step definition to create strategy for

        Returns:
            Configured strategy instance

        Raises:
            ValueError: If step type is unknown
        """
        step_type = step_def.type

        if step_type not in cls._strategy_map:
            raise ValueError(
                f"Unknown step type '{step_type}'. "
                f"Valid types: {list(cls._strategy_map.keys())}"
            )

        strategy_class = cls._strategy_map[step_type]

        logger.debug(
            f"Creating {strategy_class.__name__} for step '{step_def.id}' "
            f"(type={step_type})"
        )

        return strategy_class(executor=executor, step_def=step_def)

    @classmethod
    def get_strategy_class(cls, step_type: str) -> type:
        """Get the strategy class for a step type.

        Args:
            step_type: The step type string

        Returns:
            Strategy class

        Raises:
            ValueError: If step type is unknown
        """
        if step_type not in cls._strategy_map:
            raise ValueError(
                f"Unknown step type '{step_type}'. "
                f"Valid types: {list(cls._strategy_map.keys())}"
            )
        return cls._strategy_map[step_type]

    @classmethod
    def register(cls, step_type: str, strategy_class: type) -> None:
        """Register a custom strategy class for a step type.

        Allows extending the factory with custom strategies.

        Args:
            step_type: The step type identifier
            strategy_class: Strategy class (must inherit from BaseStepStrategy)
        """
        if not issubclass(strategy_class, BaseStepStrategy):
            raise TypeError(
                f"Strategy class must inherit from BaseStepStrategy, "
                f"got {strategy_class}"
            )
        cls._strategy_map[step_type] = strategy_class
        logger.info(f"Registered {strategy_class.__name__} for step type '{step_type}'")
