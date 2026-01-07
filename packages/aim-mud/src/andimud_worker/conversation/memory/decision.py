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
from pathlib import Path
from typing import Optional

from ...adapter import build_current_context, entries_to_chat_turns
from ..manager import MUDConversationManager
from aim_mud_types import MUDSession
from aim.agents.persona import Persona
from aim.chat.manager import ChatManager
from aim.chat.strategy.xmlmemory import XMLMemoryTurnStrategy
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from aim.utils.tokens import count_tokens
from aim.utils.xml import XmlFormatter

logger = logging.getLogger(__name__)


class MUDDecisionStrategy(XMLMemoryTurnStrategy):
    """Phase 1: Fast decision making for act/move/take tool selection.

    This strategy extends XMLMemoryTurnStrategy to provide lightweight consciousness
    for MUD agents during Phase 1 (fast path):
    - Conversation history (from Redis rolling buffer via MUDConversationManager)
    - Lightweight consciousness block (world state XML only, no CVM queries)
    - System prompt: persona + ToolUser.xml_decorator() (tool definitions)
    - User turn: world state + decision guidance (exits/inventory/targets)

    The strategy reuses the parent's chat_turns_for() infrastructure but overrides
    get_conscious_memory() to skip expensive memory queries.

    Attributes:
        conversation_manager: Manages Redis conversation list for history.
        _tool_user: ToolUser for tool definition formatting.
        _cached_system_message: Cached system message.
    """

    def __init__(self, chat: ChatManager):
        """Initialize the decision strategy.

        Args:
            chat: ChatManager instance with config and CVM (inherited from parent).
        """
        super().__init__(chat)
        # Initialize tool-related attributes
        self.tool_user = None
        self._cached_system_message = None
        # Conversation manager
        self.conversation_manager: Optional[MUDConversationManager] = None

    def set_conversation_manager(self, cm: MUDConversationManager) -> None:
        """Set the conversation manager for Redis history access.

        Args:
            cm: MUDConversationManager instance.
        """
        self.conversation_manager = cm

    def set_tool_user(self, tool_user) -> None:
        """Set the tool user for tool formatting.

        Args:
            tool_user: ToolUser instance.
        """
        self.tool_user = tool_user
        self._cached_system_message = None  # Clear cache to rebuild with new tools

    # Tool loading is handled by init_tools() method below

    def get_conscious_memory(
        self,
        persona: Persona,
        query: Optional[str] = None,
        user_queries: list[str] = [],
        assistant_queries: list[str] = [],
        content_len: int = 0,
        thought_stream: list[str] = [],
        max_context_tokens: int = 128000,
        max_output_tokens: int = 4096
    ) -> tuple[str, int]:
        """Build lightweight consciousness for Phase 1 (fast path).

        Phase 1 skips expensive CVM memory queries and only includes:
        - PraxOS header
        - Persona thoughts
        - World state XML (via get_consciousness_tail hook)
        - Memory count: 0

        This provides structured world state without the latency of memory retrieval.

        Args:
            persona: Agent's persona for thoughts.
            query: Ignored (Phase 1 doesn't use memory queries).
            user_queries: Ignored.
            assistant_queries: Ignored.
            content_len: External tokens for token budgeting (not used in Phase 1).
            thought_stream: Ignored.
            max_context_tokens: Ignored.
            max_output_tokens: Ignored.

        Returns:
            Tuple of (consciousness_content, memory_count).
            memory_count is always 0 for Phase 1 (no memory retrieval).
        """
        formatter = XmlFormatter()

        # Hook: Allow subclasses to add content at head (currently unused)
        formatter = self.get_consciousness_head(formatter)

        # Add PraxOS header
        formatter.add_element(
            "PraxOS",
            content="--== PraxOS Conscious Memory **Online** ==--",
            nowrap=True,
            priority=3
        )

        # Add persona thoughts
        for thought in persona.thoughts:
            formatter.add_element(
                self.hud_name,
                "thought",
                content=thought,
                nowrap=True,
                priority=2
            )

        # NO MEMORY QUERIES - Phase 1 is fast path
        # World state will be added by get_consciousness_tail() hook

        # Memory count is 0 (no memories retrieved)
        formatter.add_element(
            self.hud_name,
            "Memory Count",
            content="0",
            nowrap=True,
            priority=1
        )

        # Hook: Add world state XML at tail
        formatter = self.get_consciousness_tail(formatter)

        return formatter.render(), 0

    def get_consciousness_tail(self, formatter: XmlFormatter) -> XmlFormatter:
        """Add world state XML at end of consciousness block.

        Renders the current world state (room description, entities, inventory)
        at the PraxOS level after Memory Count so the agent has current context.

        This hook is shared between Phase 1 (lightweight consciousness) and
        Phase 2 (full consciousness with memories).

        Args:
            formatter: XmlFormatter instance to extend.

        Returns:
            Modified formatter with world state added.
        """
        # Only add if we have location context set
        if self.chat.current_location and self.chat.current_location.strip():
            formatter.add_element(
                self.hud_name,
                "Current World State",
                content=self.chat.current_location,
                priority=1,
                noindent=True
            )

        return formatter

    async def build_turns(
        self,
        persona: Persona,
        session: MUDSession,
        idle_mode: bool = False,
        action_guidance: str = "",
        user_guidance: str = "",
    ) -> list[dict[str, str]]:
        """Build complete turn array for Phase 1 LLM inference.

        Assembles:
        1. Lightweight consciousness block (world state only, no CVM queries)
        2. Conversation history from Redis (within token budget)
        3. Current context with decision guidance

        Delegates to parent's chat_turns_for() which handles:
        - Calling get_conscious_memory() (Phase 1 override)
        - Prepending consciousness block
        - Token budgeting and wakeup (if needed)

        Args:
            persona: The agent's persona for system prompt generation.
            session: Current MUD session with world state.
            idle_mode: Whether this is a spontaneous (idle) turn.
            action_guidance: Optional guidance from prior action results to include
                at the start of the user turn.
            user_guidance: Optional guidance from user (@choose) to include.

        Returns:
            List of chat turns ready for LLM inference.
        """
        # 1. Get conversation history from Redis
        if not self.conversation_manager:
            raise ValueError("conversation_manager not set - call set_conversation_manager() first")

        history = await self._get_conversation_history(token_budget=8000)

        # 2. Build user turn with decision guidance
        guidance = self._build_decision_guidance(session)
        # Append user guidance if provided (@choose)
        if user_guidance:
            guidance = f"{guidance}\n\n[Link Guidance: {user_guidance}]"
        user_input = build_current_context(
            session,
            idle_mode=idle_mode,
            guidance=guidance,
            include_format_guidance=False,  # Phase 1: JSON tool calls only, no ESH
            action_guidance=action_guidance,
        )

        # 3. Calculate content_len for token budgeting
        content_len = sum(count_tokens(h["content"]) for h in history)
        content_len += count_tokens(user_input)

        # 4. Set location context for consciousness tail hook
        if session.world_state and session.world_state.room_state:
            self.chat.current_location = session.world_state.to_xml(include_self=False)

        # 5. Set Phase 1 system message (with tools) from mixin
        self.chat.config.system_message = self.get_system_message(persona)

        # 6. Delegate to parent XMLMemoryTurnStrategy
        # This will call our overridden get_conscious_memory() (lightweight)
        # and get_consciousness_tail() (world state XML)
        return self.chat_turns_for(
            persona=persona,
            user_input=user_input,
            history=history,
            content_len=content_len,
            max_context_tokens=128000,  # Could be made configurable
            max_output_tokens=4096,     # Could be made configurable
            query="",  # No query - Phase 1 doesn't use memory search
        )

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

    def init_tools(self, tool_file: str, tools_path: str):
        """Initialize decision tools from file.

        Args:
            tool_file: Path to tool definition file (absolute or relative)
            tools_path: Base path for resolving relative tool files
        """
        tool_path = Path(tool_file)
        if not tool_path.is_absolute():
            # Resolve relative to tools_path
            if "/" not in str(tool_path):
                tool_path = Path(tools_path) / tool_path
            elif not tool_path.exists():
                candidate = Path(tools_path) / tool_path.name
                if candidate.exists():
                    tool_path = candidate

        loader = ToolLoader(tools_path)
        tools = loader.load_tool_file(str(tool_path))
        if not tools:
            raise ValueError(f"No tools loaded from {tool_path}")

        self.tool_user = ToolUser(tools)
        self._cached_system_message = None  # Clear cache to rebuild with new tools

    def get_system_message(self, persona: "Persona") -> str:
        """Build and return Phase 1 system message with decision tools.

        Args:
            persona: Persona to build system message for

        Returns:
            System message XML string with tools
        """
        # Check for cached message
        if hasattr(self, '_cached_system_message') and self._cached_system_message is not None:
            return self._cached_system_message

        # Build base persona XML
        xml = XmlFormatter()
        xml = persona.xml_decorator(
            xml,
            disable_guidance=False,
            disable_pif=False,
            conversation_length=0,
        )

        # Add decision tools (ONLY for Phase 1)
        if hasattr(self, 'tool_user') and self.tool_user:
            xml = self.tool_user.xml_decorator(xml)

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
