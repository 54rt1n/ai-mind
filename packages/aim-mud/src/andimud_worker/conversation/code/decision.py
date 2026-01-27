# andimud_worker/conversation/code/decision.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Code decision strategy for Phase 1 tool selection.

CodeDecisionStrategy extends XMLCodeTurnStrategy for MUD code agents, providing:
- Conversation history from Redis via MUDConversationManager
- Tool definitions via ToolUser (base tools + aura tools)
- Lightweight consciousness with world state XML
- FocusTool for setting code focus

This is the "fast path" - no expensive CVM memory queries, just tool selection.
"""

import logging
from pathlib import Path
from typing import Optional

from aim_code.strategy import XMLCodeTurnStrategy, FocusRequest, DEFAULT_CONSCIOUSNESS_BUDGET
from aim_code.graph import CodeGraph, generate_mermaid
from aim_code.tools import FocusTool
from aim.chat.manager import ChatManager
from aim.chat.strategy.base import DEFAULT_MAX_CONTEXT, DEFAULT_MAX_OUTPUT
from aim.agents.persona import Persona
from aim.utils.xml import XmlFormatter
from aim.utils.tokens import count_tokens
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from aim_mud_types import MUDSession, AURA_RINGABLE

from ..manager import MUDConversationManager
from ...adapter import build_current_context, entries_to_chat_turns

logger = logging.getLogger(__name__)


class CodeDecisionStrategy(XMLCodeTurnStrategy):
    """Phase 1: Tool selection for code agent.

    Extends XMLCodeTurnStrategy with MUD-specific functionality:
    - conversation_manager for Redis history access
    - tool_user for tool definitions and formatting
    - aura_tools for room-injected tools
    - World state XML in consciousness tail

    Key differences from MUDDecisionStrategy:
    - Inherits from XMLCodeTurnStrategy (not XMLMemoryTurnStrategy)
    - No CVM memory queries (code docs only)
    - Includes FocusTool for code navigation

    Attributes:
        conversation_manager: MUDConversationManager for Redis history.
        tool_user: ToolUser for tool definition formatting.
        _base_tools: Core tools loaded from tool file.
        _aura_tools: Room-injected tools from auras.
        _active_auras: Current active aura names for deduplication.
        _focus_tool: FocusTool instance for code navigation.
        _cached_system_message: Cached system message for performance.
    """

    def __init__(self, chat: ChatManager):
        """Initialize code decision strategy.

        Args:
            chat: ChatManager instance with CVM pointing to code index.
        """
        super().__init__(chat)
        self.conversation_manager: Optional[MUDConversationManager] = None
        self.tool_user: Optional[ToolUser] = None
        self._base_tools: list = []
        self._aura_tools: list = []
        self._active_auras: list[str] = []
        self._focus_tool: Optional[FocusTool] = None
        self._cached_system_message: Optional[str] = None

        # Interface compatibility with MUDDecisionStrategy
        # These attributes are accessed by mixins but not used for code agents
        self._active_plan = None
        self._redis_client = None
        self._agent_id: Optional[str] = None
        self._plan_tool_impl = None
        self._emote_allowed = True
        self._workspace_active = False
        self._can_speak = True
        self._can_move = True
        self._aura_blacklist: set[str] = set()
        self.thought_content: str = ""

    def set_conversation_manager(self, cm: MUDConversationManager) -> None:
        """Set the conversation manager for Redis history access.

        Args:
            cm: MUDConversationManager instance.
        """
        self.conversation_manager = cm

    # =========================================================================
    # Interface compatibility methods
    # These are called by mixins but are no-ops for code agents
    # =========================================================================

    def set_emote_allowed(self, allowed: bool) -> None:
        """Enable or disable the emote tool for decision turns."""
        allowed = bool(allowed)
        if self._emote_allowed == allowed:
            return
        self._emote_allowed = allowed
        self._refresh_tool_user()

    def set_workspace_active(self, active: bool) -> None:
        """Enable or disable the close_book tool based on workspace content.

        When workspace has content (e.g., from reading a book), close_book
        becomes available. When workspace is empty, close_book is hidden.

        Args:
            active: True if workspace has content, False otherwise.
        """
        active = bool(active)
        if self._workspace_active == active:
            return
        self._workspace_active = active
        self._refresh_tool_user()

    def set_can_speak(self, can_speak: bool) -> None:
        """Enable or disable the speak tool based on persona capability.

        When can_speak is False, the speak tool is filtered from available tools.
        This is used for passive agents (like stock research bots) that only
        respond when directly addressed rather than speaking proactively.

        Args:
            can_speak: True if persona can speak, False to disable speak tool.
        """
        can_speak = bool(can_speak)
        if self._can_speak == can_speak:
            return
        self._can_speak = can_speak
        self._refresh_tool_user()

    def set_can_move(self, can_move: bool) -> None:
        """Enable or disable the move tool based on persona capability.

        When can_move is False, the move tool is filtered from available tools.
        This is used for stationary agents that should not leave their location.

        Args:
            can_move: True if persona can move, False to disable move tool.
        """
        can_move = bool(can_move)
        if self._can_move == can_move:
            return
        self._can_move = can_move
        self._refresh_tool_user()

    def set_aura_blacklist(self, blacklist: list[str]) -> None:
        """Set the list of auras this persona cannot use.

        Blacklisted auras will be filtered out before loading tools.

        Args:
            blacklist: List of aura names to block (case-insensitive).
        """
        normalized = {str(a).strip().lower() for a in (blacklist or []) if str(a).strip()}
        if normalized == self._aura_blacklist:
            return
        self._aura_blacklist = normalized
        # Re-apply current auras with new blacklist
        if self._active_auras:
            self._refresh_tool_user()

    def set_context(self, redis_client, agent_id: str) -> None:
        """Set Redis context for plan tool execution.

        No-op for code agents - plans are MUD-specific.

        Args:
            redis_client: Async Redis client.
            agent_id: Current agent ID.
        """
        self._redis_client = redis_client
        self._agent_id = agent_id

    def get_plan_tool_impl(self):
        """Get plan tool implementation. Returns None for code agents."""
        return self._plan_tool_impl

    def get_plan_guidance(self) -> str:
        """Get plan guidance. Returns empty string for code agents."""
        return ""

    def _build_agent_action_hints(self, session: MUDSession) -> list[str]:
        """Build action hints for @agent. Returns empty list for code agents."""
        return []

    def init_tools(self, tool_file: str, tools_path: str) -> None:
        """Initialize tools from file, including focus tool.

        Loads base tools from the specified file, plus the focus tool from
        config/tools/code/focus.yaml. Creates FocusTool for code navigation.

        Args:
            tool_file: Path to tool definition file (absolute or relative).
            tools_path: Base path for resolving relative tool files.
        """
        tool_path = Path(tool_file)
        if not tool_path.is_absolute():
            if "/" not in str(tool_path):
                tool_path = Path(tools_path) / tool_path
            elif not tool_path.exists():
                candidate = Path(tools_path) / tool_path.name
                if candidate.exists():
                    tool_path = candidate

        loader = ToolLoader(tools_path)
        base_tools = loader.load_tool_file(str(tool_path))
        if not base_tools:
            raise ValueError(f"No tools loaded from {tool_path}")

        # Load focus tool for code agents
        focus_tool_path = Path(tools_path) / "code" / "focus.yaml"
        if focus_tool_path.exists():
            focus_tools = loader.load_tool_file(str(focus_tool_path))
            if focus_tools:
                base_tools = base_tools + focus_tools
                logger.info("Loaded focus tool from %s", focus_tool_path)

        self._base_tools = base_tools

        # Initialize focus tool implementation
        self._focus_tool = FocusTool(self)

        self._refresh_tool_user()

    def get_focus_tool(self) -> Optional[FocusTool]:
        """Get the FocusTool instance if available.

        Returns:
            FocusTool instance or None if not initialized.
        """
        return self._focus_tool

    def update_aura_tools(self, auras: list[str], tools_path: str) -> None:
        """Update tools based on room auras.

        Loads additional tools from aura-specific YAML files. Tools are
        deduplicated by name, with first occurrence winning.

        Args:
            auras: List of active aura names in current room.
            tools_path: Base path for aura tool files.
        """
        normalized = sorted(
            {str(a).strip().lower() for a in (auras or []) if str(a).strip()}
        )
        # Filter out blacklisted auras
        if self._aura_blacklist:
            normalized = [a for a in normalized if a not in self._aura_blacklist]
        if normalized == self._active_auras:
            return
        self._active_auras = normalized
        self._aura_tools = []

        if not normalized:
            self._refresh_tool_user()
            return

        loader = ToolLoader(tools_path)
        tools_by_name: dict[str, object] = {}
        for aura in normalized:
            tool_file = Path(tools_path) / "auras" / f"{aura}.yaml"
            if not tool_file.exists():
                logger.warning("Aura tools file not found: %s", tool_file)
                continue
            try:
                aura_tools = loader.load_tool_file(str(tool_file)) or []
            except Exception as exc:
                logger.warning("Failed to load aura tools from %s: %s", tool_file, exc)
                continue
            for tool in aura_tools:
                name = getattr(tool.function, "name", None)
                if name and name not in tools_by_name:
                    tools_by_name[name] = tool
        self._aura_tools = list(tools_by_name.values())
        self._refresh_tool_user()

    def _refresh_tool_user(self) -> None:
        """Rebuild ToolUser with current tool set and filters.

        Applies the same filters as MUDDecisionStrategy:
        - Removes emote if not allowed
        - Removes close_book if workspace not active
        """
        if not self._base_tools:
            self.tool_user = None
            self._cached_system_message = None
            return
        tools = self._base_tools + self._aura_tools

        # Filter emote if not allowed
        if not self._emote_allowed:
            tools = [t for t in tools if getattr(t.function, "name", None) != "emote"]

        # Filter close_book if workspace not active
        if not self._workspace_active:
            tools = [t for t in tools if getattr(t.function, "name", None) != "close_book"]

        # Filter speak tool if persona cannot speak
        if not self._can_speak:
            tools = [t for t in tools if getattr(t.function, "name", None) != "speak"]

        # Filter move tool if persona cannot move
        if not self._can_move:
            tools = [t for t in tools if getattr(t.function, "name", None) != "move"]

        self.tool_user = ToolUser(tools)
        self._cached_system_message = None

    def get_available_tool_names(self) -> list[str]:
        """Return current tool names available to the decision LLM.

        Returns:
            List of tool names.
        """
        if not self.tool_user or not getattr(self.tool_user, "tools", None):
            return []
        return [
            tool.function.name
            for tool in self.tool_user.tools
            if tool and tool.function and tool.function.name
        ]

    def is_aura_tool(self, tool_name: str) -> bool:
        """Check if tool_name came from an aura.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if the tool is from an aura, False otherwise.
        """
        return any(
            getattr(tool.function, "name", None) == tool_name
            for tool in self._aura_tools
        )

    def get_consciousness_head(self, formatter: XmlFormatter) -> XmlFormatter:
        """Add content at head of consciousness block.

        Currently a no-op for code agents. Override if plan context
        or other head content is needed.

        Args:
            formatter: XmlFormatter instance to extend.

        Returns:
            Unmodified formatter.
        """
        return formatter

    def get_consciousness_tail(self, formatter: XmlFormatter) -> XmlFormatter:
        """Add world state XML at end of consciousness block.

        Renders current world state (room, entities, inventory) so the
        agent has context for tool selection.

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
        persona: "Persona",
        query: str,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
        token_budget: int = DEFAULT_CONSCIOUSNESS_BUDGET,
    ) -> tuple[str, int]:
        """Build consciousness for Phase 1 with focused code context.

        When focus is set, includes:
        - PraxOS header
        - Persona thoughts
        - Focused code source
        - Call graph mermaid diagram
        - World state XML

        Skips expensive semantic CVM search (that's for Phase 2).

        Args:
            persona: Agent persona for thoughts.
            query: Query text (unused in Phase 1 - no semantic search).
            max_context_tokens: Context limit (unused).
            max_output_tokens: Output limit (unused).

        Returns:
            Tuple of (consciousness_content, memory_count).
            memory_count reflects actual content added.
        """
        formatter = XmlFormatter()
        memory_count = 0

        # Add head content (currently empty for code agents)
        formatter = self.get_consciousness_head(formatter)

        # Add PraxOS header like MUDDecisionStrategy
        formatter.add_element(
            "PraxOS",
            content="--== PraxOS Conscious Memory **Online** ==--",
            nowrap=True,
            priority=3,
        )

        # Add persona thoughts
        for thought in persona.thoughts:
            formatter.add_element(
                self.hud_name,
                "thought",
                content=thought,
                nowrap=True,
                priority=2,
            )

        # Include focused code and call graph when focus is set
        if self.focus:
            # Focused code source
            focused_source, focus_count = self._get_focused_source()
            if focused_source:
                formatter.add_element(
                    "code", "focused",
                    content=focused_source,
                    noindent=True,
                )
                memory_count += focus_count

            # Call graph mermaid diagram
            if self.code_graph:
                focused_symbols = self._get_focused_symbols()
                if focused_symbols:
                    edges = self.code_graph.get_neighborhood(
                        symbols=focused_symbols,
                        height=self.focus.height,
                        depth=self.focus.depth,
                    )
                    if edges:
                        mermaid = generate_mermaid(edges)
                        formatter.add_element(
                            "code", "call_graph",
                            content=f"```mermaid\n{mermaid}\n```",
                            noindent=True,
                        )

        # Add world state via tail hook
        formatter = self.get_consciousness_tail(formatter)

        rendered = formatter.render()
        # Return at least 1 to ensure consciousness is included
        return rendered, max(memory_count, 1)

    async def build_turns(
        self,
        persona: Persona,
        session: MUDSession,
        idle_mode: bool = False,
        action_guidance: str = "",
        user_guidance: str = "",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT,
    ) -> list[dict[str, str]]:
        """Build complete turn array for Phase 1 LLM inference.

        Assembles:
        1. Conversation history from Redis (within token budget)
        2. Current context with decision guidance
        3. Lightweight consciousness (world state, no CVM queries)

        Args:
            persona: The agent's persona for system prompt generation.
            session: Current MUD session with world state.
            idle_mode: Whether this is a spontaneous (idle) turn.
            action_guidance: Optional guidance from prior action results.
            user_guidance: Optional guidance from user (@choose).
            max_context_tokens: Maximum context window size.
            max_output_tokens: Reserved tokens for response.

        Returns:
            List of chat turns ready for LLM inference.

        Raises:
            ValueError: If conversation_manager is not set.
        """
        if not self.conversation_manager:
            raise ValueError(
                "conversation_manager not set - call set_conversation_manager() first"
            )

        # Reload code graph to pick up any file changes since last turn
        self.reload_code_graph()

        history = await self._get_conversation_history(token_budget=8000)

        # Build user turn with decision guidance
        guidance = self._build_decision_guidance(session)
        if user_guidance:
            guidance = f"{guidance}\n\n[Link Guidance: {user_guidance}]"
        user_input = build_current_context(
            session,
            idle_mode=idle_mode,
            guidance=guidance,
            include_format_guidance=False,  # Phase 1: JSON tool calls only
            action_guidance=action_guidance,
        )

        # Set location context for consciousness tail hook
        if session.world_state:
            self.chat.current_location = session.world_state.to_xml(include_self=False)

        self.chat.config.system_message = self.get_system_message(persona)

        return self.chat_turns_for(
            persona=persona,
            user_input=user_input,
            history=history,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
        )

    def _build_decision_guidance(self, session: MUDSession) -> str:
        """Build guidance for Phase 1 tool selection.

        Generates structured tool use guidance with:
        - Clear header indicating tool use turn
        - Tool signatures from ToolUser
        - Current world state context
        - Contextual examples based on available options

        Args:
            session: Current MUD session with world state.

        Returns:
            Formatted guidance string for tool selection.
        """
        parts = []

        # Header
        parts.append("[~~ Tool Guidance: Code Tool Use Turn ~~]")
        parts.append("")
        parts.append(
            "Select a tool to interact with code or the world. "
            "Available tools include focus (set code focus), and room aura tools."
        )
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
        aura_descriptions: list[str] = []
        ringable_sources: list[str] = []

        room = session.current_room if session else None
        if room and room.exits:
            exits = list(room.exits.keys())
        if room and getattr(room, "auras", None):
            for aura in room.auras:
                if isinstance(aura, dict):
                    name = aura.get("name", "") or ""
                    source = aura.get("source", "") or ""
                else:
                    name = getattr(aura, "name", "") or ""
                    source = getattr(aura, "source", "") or ""
                if not name:
                    continue
                if source:
                    aura_descriptions.append(f"{name} (source: {source})")
                else:
                    aura_descriptions.append(name)
                if name.upper() == AURA_RINGABLE and source:
                    ringable_sources.append(source)

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
        if aura_descriptions:
            parts.append(f"  Auras: {', '.join(aura_descriptions)}")
        if not any(
            [exits, room_objects, inventory_items, present_targets, aura_descriptions]
        ):
            parts.append("  (No special options available)")
        parts.append("")

        parts.append("Just follow the instructions. Thanks!")
        parts.append("")
        parts.append("[/~~Tool Guidance~~/]")

        return "\n".join(parts)

    async def _get_conversation_history(
        self,
        token_budget: int,
    ) -> list[dict[str, str]]:
        """Get conversation history from Redis within token budget.

        Args:
            token_budget: Maximum tokens to include in history.

        Returns:
            List of chat turns in chronological order.
        """
        entries = await self.conversation_manager.get_history(token_budget=token_budget)
        return entries_to_chat_turns(entries)

    def get_system_message(self, persona: Persona) -> str:
        """Build and return Phase 1 system message with tools.

        Args:
            persona: Persona to build system message for.

        Returns:
            System message XML string with tool definitions.
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

        if self.tool_user:
            xml = self.tool_user.xml_decorator(xml)

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
