# aim/dreamer/core/strategy/context_only.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""ContextOnlyStrategy - memory actions only, no LLM, no output."""

import logging
from typing import TYPE_CHECKING

from .base import BaseStepStrategy, ScenarioStepResult, ScenarioExecutor
from .functions import execute_context_actions

if TYPE_CHECKING:
    from ..models import ContextOnlyStepDefinition


logger = logging.getLogger(__name__)


class ContextOnlyStrategy(BaseStepStrategy):
    """Executes context DSL actions only - no LLM call, no document output.

    Used for steps that only need to load memory context for subsequent steps.
    For example, gathering relevant memories before a decision step.

    Attributes:
        step_def: ContextOnlyStepDefinition with context actions and next
    """

    step_def: "ContextOnlyStepDefinition"

    async def execute(self) -> ScenarioStepResult:
        """Execute context actions and advance to next step.

        1. Execute memory DSL actions from step_def.context
        2. Store resulting doc references in state.memory_refs
        3. Return result with next step from step_def.next

        Returns:
            ScenarioStepResult with success=True and next step
        """
        step_id = self.step_def.id
        executor = self.executor

        logger.info(f"ContextOnlyStrategy executing step '{step_id}'")

        # Execute context DSL if present (clears and populates memory_refs)
        if self.step_def.context:
            execute_context_actions(executor, self.step_def)

        # Get next step
        next_step = self._get_next_step()

        logger.info(
            f"Step '{step_id}' complete, "
            f"loaded {len(executor.state.memory_refs)} memory refs, "
            f"next_step='{next_step}'"
        )

        return ScenarioStepResult(
            success=True,
            next_step=next_step,
            state_changed=True,  # We modified memory_refs
            doc_created=False,   # No document output
        )
