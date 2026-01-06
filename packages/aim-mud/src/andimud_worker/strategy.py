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

from .adapter import build_current_context, entries_to_chat_turns
from .conversation import MUDConversationManager
from .session import MUDSession
from aim.agents.persona import Persona
from aim.chat.manager import ChatManager
from aim.chat.strategy.xmlmemory import XMLMemoryTurnStrategy
from aim.tool.formatting import ToolUser
from aim.utils.tokens import count_tokens
from aim.utils.xml import XmlFormatter

logger = logging.getLogger(__name__)


class MUDDecisionStrategy:
    """Phase 1: Fast decision making for act/move/take tool selection.

    This strategy builds turns for the first phase of MUD's two-phase architecture:
    - Conversation history (from Redis rolling buffer via MUDConversationManager)
    - System prompt: persona + ToolUser.xml_decorator() (tool definitions)
    - User turn: world state + decision guidance (exits/inventory/targets)
    - NO CVM memory queries (fast path)

    Attributes:
        conversation_manager: Manages Redis conversation list for history.
        tool_user: Optional ToolUser for tool definition formatting.
    """

    def __init__(self, conversation_manager: MUDConversationManager):
        """Initialize the decision strategy.

        Args:
            conversation_manager: MUDConversationManager for conversation history.
        """
        self.conversation_manager = conversation_manager
        self.tool_user: Optional[ToolUser] = None

    def set_tool_user(self, tool_user: ToolUser) -> None:
        """Set the ToolUser for tool definition formatting.

        Args:
            tool_user: ToolUser instance with available tools.
        """
        self.tool_user = tool_user

    async def build_turns(
        self,
        persona: Persona,
        session: MUDSession,
        idle_mode: bool = False,
    ) -> list[dict[str, str]]:
        """Build complete turn array for Phase 1 LLM inference.

        Assembles:
        1. System prompt with persona and tool definitions
        2. Conversation history from Redis (within token budget)
        3. Current context with decision guidance

        Args:
            persona: The agent's persona for system prompt generation.
            session: Current MUD session with world state.
            idle_mode: Whether this is a spontaneous (idle) turn.

        Returns:
            List of chat turns ready for LLM inference.
        """
        turns: list[dict[str, str]] = []

        # 1. Get conversation history from Redis
        history = await self._get_conversation_history(token_budget=8000)
        turns.extend(history)

        # 2. Build user turn with decision guidance
        guidance = self._build_decision_guidance(session)
        user_content = build_current_context(
            session,
            idle_mode=idle_mode,
            guidance=guidance,
            include_format_guidance=False,  # Phase 1: JSON tool calls only, no ESH
        )
        turns.append({"role": "user", "content": user_content})

        return turns

    def _build_decision_guidance(self, session: MUDSession) -> str:
        """Build guidance for Phase 1 tool selection.

        Generates a structured tool use turn guidance with:
        - Clear header indicating this is a tool use turn
        - OpenAI-style function signatures from ToolUser
        - Current world state context (exits, objects, inventory, targets)
        - Contextual examples based on available options

        Args:
            session: Current MUD session with world state.

        Returns:
            Formatted guidance string for tool selection.
        """
        parts = []

        # Header
        parts.append("[~~ Tool Guidance: Tool Use Turn ~~]")
        parts.append("")
        parts.append("You are in a tool use turn. Your response is going to be used to determine your next action.")
        parts.append("Tool use involves you generating a JSON block like the following:")
        parts.append("")

        # Get tool signatures from ToolUser
        if self.tool_user:
            tool_guidance = self.tool_user.get_tool_guidance()
            if tool_guidance:
                parts.append(tool_guidance)
                parts.append("")

        # Extract current world state context
        exits: list[str] = []
        room_objects: list[str] = []
        present_targets: list[str] = []
        inventory_items: list[str] = []

        # Get exits from current room
        room = session.current_room if session else None
        if room and room.exits:
            exits = list(room.exits.keys())

        # Get entities and inventory from world state or session
        world_state = session.world_state if session else None
        if world_state:
            for entity in world_state.entities_present:
                if entity.is_self:
                    continue
                if entity.entity_type in ("player", "ai", "npc"):
                    if entity.name:
                        present_targets.append(entity.name)
                else:
                    if entity.name:
                        room_objects.append(entity.name)
            for item in world_state.inventory:
                if item.name:
                    inventory_items.append(item.name)
        else:
            # Fall back to session entities if no world_state
            if session:
                for entity in session.entities_present:
                    if entity.is_self:
                        continue
                    if entity.entity_type in ("player", "ai", "npc"):
                        if entity.name:
                            present_targets.append(entity.name)
                    else:
                        if entity.name:
                            room_objects.append(entity.name)

        # Add current context
        parts.append("Current Context:")
        if exits:
            parts.append(f"  Available exits: {', '.join(exits)}")
        if room_objects:
            parts.append(f"  Objects present: {', '.join(room_objects)}")
        if inventory_items:
            parts.append(f"  Your inventory: {', '.join(inventory_items)}")
        if present_targets:
            parts.append(f"  People present: {', '.join(present_targets)}")
        if not any([exits, room_objects, inventory_items, present_targets]):
            parts.append("  (No special options available)")
        parts.append("")

        # Add contextual examples
        if exits or room_objects or inventory_items or present_targets:
            parts.append("Contextual Examples:")
            if exits:
                parts.append(f'  Move: {{"move": {{"location": "{exits[0]}"}}}}')
            if room_objects:
                parts.append(f'  Take: {{"take": {{"object": "{room_objects[0]}"}}}}')
            if inventory_items:
                parts.append(f'  Drop: {{"drop": {{"object": "{inventory_items[0]}"}}}}')
            if inventory_items and present_targets:
                parts.append(f'  Give: {{"give": {{"object": "{inventory_items[0]}", "target": "{present_targets[0]}"}}}}')
            parts.append("")

        parts.append("Just follow the instructions. Thanks!")
        parts.append("")
        parts.append("[/~~Tool Guidance~~/]")

        return "\n".join(parts)

    def _build_agent_action_hints(self, session: MUDSession) -> list[str]:
        """Build dynamic hints for @agent actions based on current world state.

        This method extracts context hints without the full format instructions,
        useful for building abbreviated guidance.

        Args:
            session: Current MUD session with world state.

        Returns:
            List of hint strings about available actions.
        """
        exits: list[str] = []
        room_objects: list[str] = []
        present_targets: list[str] = []
        inventory_items: list[str] = []

        # Get exits from current room
        room = session.current_room if session else None
        if room and room.exits:
            exits = list(room.exits.keys())

        # Get entities and inventory from world state or session
        world_state = session.world_state if session else None
        if world_state:
            for entity in world_state.entities_present:
                if entity.is_self:
                    continue
                if entity.entity_type in ("player", "ai", "npc"):
                    if entity.name:
                        present_targets.append(entity.name)
                else:
                    if entity.name:
                        room_objects.append(entity.name)
            for item in world_state.inventory:
                if item.name:
                    inventory_items.append(item.name)
        else:
            # Fall back to session entities if no world_state
            if session:
                for entity in session.entities_present:
                    if entity.is_self:
                        continue
                    if entity.entity_type in ("player", "ai", "npc"):
                        if entity.name:
                            present_targets.append(entity.name)
                    else:
                        if entity.name:
                            room_objects.append(entity.name)

        # Build hints
        hints: list[str] = []
        if exits:
            hints.append(f"Valid move locations: {', '.join(exits)}")
        if room_objects:
            hints.append(f"Objects present: {', '.join(room_objects)}")
        if inventory_items:
            hints.append(f"Inventory: {', '.join(inventory_items)}")
        if present_targets:
            hints.append(f"Valid give targets: {', '.join(present_targets)}")

        return hints

    async def _get_conversation_history(
        self,
        token_budget: int,
    ) -> list[dict[str, str]]:
        """Get conversation history from Redis within token budget.

        Retrieves entries from the conversation manager and converts
        them to chat turn format.

        Args:
            token_budget: Maximum tokens to include in history.

        Returns:
            List of chat turns in chronological order.
        """
        entries = await self.conversation_manager.get_history(token_budget=token_budget)
        return entries_to_chat_turns(entries)


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
    """

    def __init__(self, chat: ChatManager):
        """Initialize the response strategy.

        Args:
            chat: ChatManager instance with CVM and config.
        """
        super().__init__(chat)
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
