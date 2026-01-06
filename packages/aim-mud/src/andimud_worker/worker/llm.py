# aim/app/mud/worker/llm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""LLM interaction for the MUD worker.

Handles calling the LLM provider (single attempt, no blocking retries).
Extracted from worker.py lines 1191-1228
"""

import logging
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .main import MUDAgentWorker


logger = logging.getLogger(__name__)


class LLMMixin:
    """Mixin for LLM interaction methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

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
