# aim/app/mud/strategy.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Strategy classes for MUD agent LLM turn building.

This module provides strategy classes for building LLM turns in MUD's
two-phase architecture:

1. MUDDecisionStrategy (Phase 1): Fast decision making for speak/move/take/drop/give
   tool selection without CVM memory queries. The "fast path".

2. MUDResponseStrategy (Phase 2): Full response with consciousness block and
   CVM memory retrieval. Used after Phase 1 decides {"speak": {}}.

The strategy pattern allows different turn construction depending on the phase,
with Phase 2 extending XMLMemoryTurnStrategy for memory-augmented responses.
"""

import logging
from typing import Optional

from ...adapter import build_current_context, entries_to_chat_turns
from ..manager import MUDConversationManager
from aim_mud_types import MUDSession
from aim.agents.persona import Persona
from aim.chat.manager import ChatManager
from aim.chat.strategy.xmlmemory import XMLMemoryTurnStrategy
from aim.utils.tokens import count_tokens
from aim.utils.xml import XmlFormatter

logger = logging.getLogger(__name__)


class MUDResponseStrategy(XMLMemoryTurnStrategy):
    """Phase 2: Full response with consciousness block and CVM memory.

    Extends XMLMemoryTurnStrategy to provide memory-augmented turn building
    for MUD agents after Phase 1 decides {"speak": {}}. This is the "slow path"
    that includes:

    - Consciousness block with location-aware memory queries
    - Token budgeting: system + memory + conversation (~50%)
    - Wakeup message on fresh sessions
    - History compression if > 50%

    The strategy delegates most work to the parent XMLMemoryTurnStrategy's
    chat_turns_for() method, which handles:
    - Building consciousness block via get_conscious_memory()
    - Wakeup insertion for fresh sessions
    - History compression when over budget
    - Token budgeting across all components

    Attributes:
        conversation_manager: MUDConversationManager for Redis history access.
        _cached_system_message: Cached system message (from PhaseTwoMixin).
    """

    def __init__(self, chat: ChatManager):
        """Initialize the response strategy.

        Args:
            chat: ChatManager instance with CVM and config.
        """
        super().__init__(chat)
        # Initialize attributes expected by PhaseTwoMixin
        self._cached_system_message = None
        # Conversation manager
        self.conversation_manager: Optional[MUDConversationManager] = None

    def set_conversation_manager(self, cm: MUDConversationManager) -> None:
        """Set the conversation manager for history retrieval.

        Args:
            cm: MUDConversationManager instance.
        """
        self.conversation_manager = cm

    async def build_turns(
        self,
        persona: Persona,
        user_input: str,
        session: MUDSession,
        coming_online: bool = False,
        max_context_tokens: int = 128000,
        max_output_tokens: int = 4096,
        memory_query: str = "",
    ) -> list[dict[str, str]]:
        """Build turns for Phase 2 full response.

        Assembles context for memory-augmented response generation:
        1. Get conversation history from Redis (within token budget)
        2. Merge format guidance into last user turn if needed (avoids double user turns)
        3. Calculate content_len for token budgeting
        4. Set location context for memory queries
        5. Delegate to parent's chat_turns_for() for heavy lifting

        Token budget (matching XMLMemoryTurnStrategy pattern):
        - usable = max_context - max_output - system_prompt - 1024 (safety margin)
        - content_len = history + user_input + wakeup (external tokens)
        - Memory queries get remaining after fixed elements
        - History compressed if > 50% of usable tokens
        - Memory reranker: 60% conversations+insights, 40% broad

        Note on double user turns:
        In Phase 2, events are already pushed to conversation history as a user turn.
        The user_input here is format guidance (e.g., "[~~ FORMAT... ~~]").
        If we pass both to chat_turns_for(), we get two consecutive user turns, which
        is invalid for LLM APIs. To fix this, we append the format guidance to the
        last user turn in history instead of adding a new turn.

        Args:
            persona: The agent's persona for system prompt and wakeup.
            user_input: The current user input (format guidance for Phase 2).
            session: Current MUD session with world state.
            coming_online: Whether this is a fresh session (for wakeup).
            max_context_tokens: Maximum context window size.
            max_output_tokens: Reserved tokens for response.
            memory_query: Optional query to enhance CVM memory search.

        Returns:
            List of chat turns ready for LLM inference.
        """
        # Use parent's calculation which accounts for system prompt tokens
        usable = self._calc_max_context_tokens(max_context_tokens, max_output_tokens)

        # Get conversation history from Redis
        history = await self._get_conversation_history(token_budget=usable // 2)

        # Merge format guidance into last user turn to avoid double user turns.
        # In Phase 2, history already contains the user turn with events. The
        # user_input is format guidance that should be appended to that turn,
        # not added as a separate turn.
        effective_user_input = user_input
        if history and history[-1]["role"] == "user" and user_input and user_input.strip():
            history[-1]["content"] += "\n\n" + user_input
            effective_user_input = ""  # Don't add separate user turn

        # Calculate content_len (external tokens for budget calc)
        content_len = sum(count_tokens(h["content"]) for h in history)
        content_len += count_tokens(effective_user_input)
        content_len += count_tokens(persona.get_wakeup() or "")

        # Set location for memory queries (room state affects search)
        if session.world_state and session.world_state.room_state:
            self.chat.current_location = session.world_state.to_xml(include_self=False)

        # Set Phase 2 system message (without tools) from mixin
        self.chat.config.system_message = self.get_system_message(persona)

        # Delegate to parent XMLMemoryTurnStrategy
        return self.chat_turns_for(
            persona=persona,
            user_input=effective_user_input,
            history=history,
            content_len=content_len,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            query=memory_query,
        )

    async def _get_conversation_history(
        self,
        token_budget: int,
    ) -> list[dict[str, str]]:
        """Get history from Redis conversation manager.

        Retrieves entries from the conversation manager and converts
        them to chat turn format for use in turn building.

        Args:
            token_budget: Maximum tokens to include in history.

        Returns:
            List of chat turns in role/content format.
        """
        if not self.conversation_manager:
            return []
        entries = await self.conversation_manager.get_history(token_budget)
        return [{"role": e.role, "content": e.content} for e in entries]

    def get_consciousness_tail(self, formatter: XmlFormatter) -> XmlFormatter:
        """Add world state XML at end of consciousness block.

        Renders the current world state (room description, entities, inventory)
        at the PraxOS level (not nested under Active Memory) so the agent has
        current context.

        Args:
            formatter: XmlFormatter instance to extend

        Returns:
            Modified formatter with world state added
        """
        # Only add if we have location context set
        if self.chat.current_location and self.chat.current_location.strip():
            formatter.add_element(
                self.hud_name, "Current World State",
                content=self.chat.current_location,
                priority=1,
                noindent=True
            )

        return formatter

    def get_system_message(self, persona: "Persona") -> str:
        """Build and return Phase 2 system message without decision tools.

        Args:
            persona: Persona to build system message for

        Returns:
            System message XML string without tools
        """
        # Check for cached message
        if hasattr(self, '_cached_system_message') and self._cached_system_message is not None:
            return self._cached_system_message

        # Build base persona XML (NO tools)
        xml = XmlFormatter()
        xml = persona.xml_decorator(
            xml,
            disable_guidance=False,
            disable_pif=False,
            conversation_length=0,
        )

        # Do NOT add decision tools for Phase 2

        # Cache and return
        self._cached_system_message = xml.render()
        return self._cached_system_message

    def _calc_max_context_tokens(self, max_context_tokens: int, max_output_tokens: int) -> int:
        """Calculate usable context tokens using phase-specific system message.

        Overrides parent to use get_system_message() instead of self.chat.config.system_message.
        """
        system_tokens = 0
        if hasattr(self, 'chat') and self.chat and hasattr(self.chat, 'persona'):
            system_message = self.get_system_message(self.chat.persona)
            if system_message and isinstance(system_message, str):
                system_tokens = self.count_tokens(system_message)
        return max_context_tokens - max_output_tokens - system_tokens - 1024