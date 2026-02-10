# andimud_worker/conversation/code/response.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Code response strategy for Phase 2 narrative generation.

CodeResponseStrategy extends XMLCodeTurnStrategy for MUD code agents, providing:
- Conversation history from Redis via MUDConversationManager
- Code consciousness with focused code, call graph, and semantic search
- World state XML in consciousness tail

This is the "slow path" - includes code consciousness building via CVM queries.
"""

import logging
from typing import Optional

from aim_code.strategy import XMLCodeTurnStrategy, DEFAULT_CONSCIOUSNESS_BUDGET
from aim_code.graph import CodeGraph
from aim.chat.manager import ChatManager
from aim.chat.strategy.base import DEFAULT_MAX_CONTEXT, DEFAULT_MAX_OUTPUT
from aim.agents.persona import Persona
from aim.utils.xml import XmlFormatter
from aim.utils.tokens import count_tokens
from aim_mud_types import MUDSession

from ..manager import MUDConversationManager
from ...adapter import entries_to_chat_turns

logger = logging.getLogger(__name__)


class CodeResponseStrategy(XMLCodeTurnStrategy):
    """Phase 2: Narrative generation for code agent.

    Extends XMLCodeTurnStrategy with MUD-specific functionality:
    - conversation_manager for Redis history access
    - World state XML in consciousness tail
    - Code consciousness with memory_query for semantic search

    Key differences from MUDResponseStrategy:
    - Inherits from XMLCodeTurnStrategy (not XMLMemoryTurnStrategy)
    - Uses code consciousness (focused code + call graph + semantic search)
    - No conversation memory - only code documents in CVM

    Attributes:
        conversation_manager: MUDConversationManager for Redis history.
        _cached_system_message: Cached system message for performance.
    """

    def __init__(self, chat: ChatManager):
        """Initialize code response strategy.

        Args:
            chat: ChatManager instance with CVM pointing to code index.
        """
        super().__init__(chat)
        self.conversation_manager: Optional[MUDConversationManager] = None
        self._cached_system_message: Optional[str] = None

        # Interface compatibility with MUDResponseStrategy
        # Thought content is accessed by ProfileMixin for thought injection
        self.thought_content: str = ""

    def set_conversation_manager(self, cm: MUDConversationManager) -> None:
        """Set the conversation manager for Redis history access.

        Args:
            cm: MUDConversationManager instance.
        """
        self.conversation_manager = cm

    def get_consciousness_head(self, formatter: XmlFormatter) -> XmlFormatter:
        """Add content at head of consciousness block.

        Currently a no-op for code agents. Override if additional
        head content is needed.

        Args:
            formatter: XmlFormatter instance to extend.

        Returns:
            Unmodified formatter.
        """
        return formatter

    def get_consciousness_tail(self, formatter: XmlFormatter) -> XmlFormatter:
        """Add world state XML at end of consciousness block.

        Renders current world state (room, entities, inventory) so the
        agent has context for narrative generation.

        Args:
            formatter: XmlFormatter instance to extend.

        Returns:
            Modified formatter with world state added.
        """
        if self.chat.current_location and self.chat.current_location.strip():
            formatter.add_element(
                self.hud_name,
                "Current World State",
                content=self.chat.current_location,
                priority=1,
                noindent=True,
            )
        return formatter

    def get_code_consciousness(
        self,
        persona: Persona,
        query: str,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
        token_budget: int = DEFAULT_CONSCIOUSNESS_BUDGET,
    ) -> tuple[str, int]:
        """Build code consciousness for Phase 2 with world state.

        Extends parent implementation to include:
        - PraxOS header
        - Persona thoughts
        - Code context from parent (focused code, call graph, semantic search)
        - World state via tail hook

        Args:
            persona: Agent persona for thoughts.
            query: Query text for semantic code search.
            max_context_tokens: Context limit.
            max_output_tokens: Output limit.
            token_budget: Maximum tokens for consciousness content.

        Returns:
            Tuple of (consciousness_content, memory_count).
        """
        # Build header with PraxOS, persona thoughts, and dynamic thought content
        header_formatter = XmlFormatter()
        header_formatter = self.get_consciousness_head(header_formatter)

        header_formatter.add_element(
            "PraxOS",
            content="--== PraxOS Conscious Memory **Online** ==--",
            nowrap=True,
            priority=3,
        )

        # Add dynamic thought content if present (from recent reasoning)
        if self.thought_content and self.thought_content.strip():
            header_formatter.add_element(
                self.hud_name,
                "recent_reasoning",
                content=self.thought_content,
                priority=2,
                noindent=True,
            )
            logger.debug(
                "Injected thought_content into consciousness header (%d chars)",
                len(self.thought_content),
            )

        # Add static persona thoughts
        for thought in persona.thoughts:
            header_formatter.add_element(
                self.hud_name,
                "thought",
                content=thought,
                nowrap=True,
                priority=2,
            )

        header_content = header_formatter.render()

        # Get code context from parent (focused code, call graph, semantic search, specs)
        code_consciousness, memory_count = super().get_code_consciousness(
            persona, query, max_context_tokens, max_output_tokens, token_budget
        )

        # Build world state tail
        tail_formatter = XmlFormatter()
        tail_formatter = self.get_consciousness_tail(tail_formatter)
        tail_content = tail_formatter.render()

        # Combine all parts
        parts = [header_content]
        if code_consciousness.strip():
            parts.append(code_consciousness)
        if tail_content.strip():
            parts.append(tail_content)

        rendered = "\n".join(parts)
        # Ensure memory_count >= 1 so consciousness is always included
        return rendered, max(memory_count, 1)

    async def build_turns(
        self,
        persona: Persona,
        user_input: str,
        session: MUDSession,
        coming_online: bool = False,
        memory_query: Optional[str] = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
    ) -> list[dict[str, str]]:
        """Build turns for Phase 2 full response with code consciousness.

        Assembles context for code-aware response generation:
        1. Get conversation history from Redis (within token budget)
        2. Merge format guidance into last user turn if needed
        3. Set location context for consciousness tail
        4. Build code consciousness (focused code + call graph + semantic search)

        Note on double user turns:
        In Phase 2, events are already pushed to conversation history as a user turn.
        The user_input here is format guidance. If we pass both to chat_turns_for(),
        we get two consecutive user turns, which is invalid for LLM APIs. To fix this,
        we append the format guidance to the last user turn in history.

        Args:
            persona: The agent's persona for system prompt.
            user_input: The current user input (format guidance for Phase 2).
            session: Current MUD session with world state.
            coming_online: Whether this is a fresh session.
            memory_query: Optional query to enhance code semantic search.
            max_context_tokens: Maximum context window size.
            max_output_tokens: Reserved tokens for response.

        Returns:
            List of chat turns ready for LLM inference.
        """
        # Reload code graph to pick up any file changes since last turn
        self.reload_code_graph()

        usable = self._calc_max_context_tokens(max_context_tokens, max_output_tokens)
        history = await self._get_conversation_history(token_budget=usable // 2)

        # Merge format guidance into last user turn to avoid double user turns.
        effective_user_input = user_input
        if (
            history
            and history[-1]["role"] == "user"
            and user_input
            and user_input.strip()
        ):
            history[-1]["content"] += "\n\n" + user_input
            effective_user_input = ""

        # Set location context for consciousness tail
        if session.world_state:
            self.chat.current_location = session.world_state.to_xml(include_self=False)

        self.chat.config.system_message = self.get_system_message(persona)

        # Use parent's chat_turns_for with code consciousness
        # The query parameter drives semantic code search
        return self.chat_turns_for(
            persona=persona,
            user_input=effective_user_input,
            history=history,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
        )

    async def _get_conversation_history(
        self,
        token_budget: int,
    ) -> list[dict[str, str]]:
        """Get conversation history from Redis within token budget.

        Args:
            token_budget: Maximum tokens to include in history.

        Returns:
            List of chat turns in role/content format.
        """
        if not self.conversation_manager:
            return []
        entries = await self.conversation_manager.get_history(token_budget=token_budget)
        return entries_to_chat_turns(entries)

    def get_system_message(self, persona: Persona) -> str:
        """Build and return Phase 2 system message without tools.

        Args:
            persona: Persona to build system message for.

        Returns:
            System message XML string without tool definitions.
        """
        if self._cached_system_message is not None:
            return self._cached_system_message

        xml = XmlFormatter()
        xml = persona.xml_decorator(
            xml,
            disable_guidance=False,
            disable_pif=False,
            conversation_length=0,
        )

        # Phase 2: No tools in system message

        self._cached_system_message = xml.render()
        return self._cached_system_message

    def _calc_max_context_tokens(
        self, max_context_tokens: int, max_output_tokens: int
    ) -> int:
        """Calculate usable context tokens.

        Reserves tokens for output, system prompt, and safety margin.

        Args:
            max_context_tokens: Total context window size.
            max_output_tokens: Reserved tokens for output.

        Returns:
            Usable tokens for content.
        """
        system_tokens = 0
        if hasattr(self, "chat") and self.chat and hasattr(self.chat, "persona"):
            system_message = self.get_system_message(self.chat.persona)
            if system_message and isinstance(system_message, str):
                system_tokens = count_tokens(system_message)
        return max_context_tokens - max_output_tokens - system_tokens - 1024
