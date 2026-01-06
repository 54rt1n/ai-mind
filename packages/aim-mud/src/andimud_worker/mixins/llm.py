# aim/app/mud/worker/llm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""LLM interaction for the MUD worker.

Handles calling the LLM provider (single attempt, no blocking retries).
Extracted from worker.py lines 1191-1228
"""

import logging
from typing import TYPE_CHECKING

from aim.llm.models import LanguageModelV2
from aim.llm.model_set import ModelSet

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class LLMMixin:
    """Mixin for LLM interaction methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    def _init_llm_provider(self: "MUDAgentWorker") -> None:
        """Initialize the ModelSet with persona-level overrides.

        Originally from worker.py lines 811-839

        Creates a ModelSet that manages multiple LLM providers for different
        task types (default, thought, tool) with persona-level model overrides.
        """
        # Create ModelSet with persona overrides
        self.model_set = ModelSet.from_config(
            config=self.chat_config,
            persona=self.persona
        )

        # Store default model info for logging/compatibility
        self.model_name = self.model_set.default_model
        models = LanguageModelV2.index_models(self.chat_config)
        self.model = models.get(self.model_name)

        logger.info(
            f"Initialized ModelSet for {self.persona.persona_id}: "
            f"default={self.model_set.default_model}, "
            f"chat={self.model_set.chat_model}, "
            f"tool={self.model_set.tool_model}, "
            f"thought={self.model_set.thought_model}, "
            f"codex={self.model_set.codex_model}"
        )

    async def _call_llm(
        self: "MUDAgentWorker",
        chat_turns: list[dict[str, str]],
        role: str = "default",
        max_retries: int = 3
    ) -> str:
        """Call the LLM - single attempt, let exceptions propagate.

        Args:
            chat_turns: List of chat turns (system/user/assistant messages).
            role: Model role - "default", "chat", "tool", "thought", etc.
            max_retries: Ignored (kept for API compatibility).

        Returns:
            The complete LLM response as a string.

        Raises:
            Exception: If LLM call fails.
        """
        try:
            # Get appropriate provider for role
            provider = self.model_set.get_provider(role)

            chunks = []
            for chunk in provider.stream_turns(chat_turns, self.chat_config):
                if chunk:
                    chunks.append(chunk)
            return "".join(chunks)

        except Exception as e:
            logger.error(f"LLM call failed (role={role}): {e}")
            raise
