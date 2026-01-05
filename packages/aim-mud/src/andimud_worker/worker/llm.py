# aim/app/mud/worker/llm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""LLM interaction for the MUD worker.

Handles calling the LLM provider with retry logic.
Extracted from worker.py lines 1191-1228
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from aim.llm.llm import is_retryable_error


if TYPE_CHECKING:
    from .main import MUDAgentWorker


logger = logging.getLogger(__name__)


class LLMMixin:
    """Mixin for LLM interaction methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    async def _call_llm(self: "MUDAgentWorker", chat_turns: list[dict[str, str]], max_retries: int = 3) -> str:
        """Call the LLM with chat turns and return the response.

        Originally from worker.py lines 1191-1228

        Implements retry logic with exponential backoff for transient errors,
        following the pattern from aim/refiner/engine.py.

        Args:
            chat_turns: List of chat turns (system/user/assistant messages).
            max_retries: Maximum number of retry attempts for transient errors.

        Returns:
            The complete LLM response as a string.

        Raises:
            Exception: If max retries exceeded or non-retryable error occurs.
        """
        for attempt in range(max_retries):
            try:
                chunks = []
                for chunk in self._llm_provider.stream_turns(chat_turns, self.chat_config):
                    if chunk:
                        chunks.append(chunk)
                return "".join(chunks)

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}/{max_retries}): {e}")

                # Check if error is retryable and we have retries left
                if is_retryable_error(e) and attempt < max_retries - 1:
                    delay = min(30 * (2 ** attempt), 120)  # 30s, 60s, 120s max
                    logger.info(f"Retryable error, waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                else:
                    raise

        # Should not reach here, but just in case
        raise RuntimeError(f"LLM call failed after {max_retries} attempts")
