# aim/app/mud/worker/turns/strategy/agent.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent turn processor: single-phase with full guidance."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path

from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from aim.utils.xml import XmlFormatter
from aim_mud_types import MUDAction, MUDTurnRequest, MUDEvent
from aim.utils.think import extract_think_tags
from ...adapter import build_current_context
from ..response import (
    sanitize_response,
    parse_agent_action_response,
)
from ..validation import resolve_move_location, resolve_target_name, get_ringable_objects
from .base import BaseTurnProcessor
from ...tools.helper import ToolHelper
from ...config import MUDConfig
from aim.config import ChatConfig
from ...exceptions import AbortRequestedException

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin

logger = logging.getLogger(__name__)


class AgentTurnProcessor(BaseTurnProcessor):
    """Single-phase turn processing with agent guidance strategy.

    Skips decision phase entirely and provides full guidance upfront.
    Single LLM call with TOOL role and agent action schema.
    All actions (move, take, drop, give, describe, ring) decided together.
    """

    def __init__(self, worker: "TurnsMixin", tool_helper: ToolHelper):
        """Initialize with worker and set user_guidance to empty string.

        Args:
            worker: MUDAgentWorker instance
        """
        super().__init__(worker)
        self.user_guidance = ""
        self.required_tool = ""  # If set, filter allowed tools to only this tool
        self._tool_helper = tool_helper

    def build_system_message(self) -> str:
        """Build the system message for the agent turn.

        Returns:
            The system message
        """
        xml = XmlFormatter()
        xml = self.worker.persona.xml_decorator(xml, disable_guidance=False, disable_pif=False, conversation_length=0)
        xml = self._tool_helper.decorate_xml(xml)
        return xml.render()

    def _build_agent_guidance(self, user_guidance: str) -> str:
        """Build comprehensive guidance for @agent action selection.

        Generates structured guidance similar to Phase 1 decision guidance:
        - Clear header indicating this is a tool use turn
        - OpenAI-style function signatures from ToolUser
        - Agent action spec instructions
        - Current world state context (exits, objects, inventory, targets)
        - User guidance if provided (appended at the end)

        Args:
            user_guidance: Optional user-provided guidance string

        Returns:
            Formatted guidance string for agent action selection
        """
        parts = []

        # Add JSON tool use header - same as Phase 1
        parts.append("[~~ Tool Guidance: Tool Use Turn ~~]")
        parts.append("")
        parts.append("You are in a tool use turn. Your response is going to be used to determine your next action.")
        parts.append("Tool use involves you generating a JSON block like the following:")
        parts.append("")

        # Get ToolUser guidance (OpenAI-style function signatures)
        if self._tool_helper and self._tool_helper._tool_user:
            tool_guidance = self._tool_helper._tool_user.get_tool_guidance()
            if tool_guidance:
                parts.append(tool_guidance)
                parts.append("")

        # Get basic agent action hints from worker (filter to required_tool if set)
        basic_guidance = self.worker._build_agent_guidance("", self.required_tool)
        if basic_guidance:
            parts.append(basic_guidance)

        guidance = "\n".join(parts)

        # Append user guidance at the end - same as Phase 1
        if user_guidance:
            guidance = f"{guidance}\n\n[Link Guidance: {user_guidance}]"

        return guidance

    def _refresh_aura_tools(self) -> None:
        """Update tool helper with aura tools based on current room state."""
        if not self.worker or not self.worker.session or not self.worker.session.current_room:
            return
        room = self.worker.session.current_room
        auras = []
        aura_source_ids: dict[str, str] = {}
        for aura in getattr(room, "auras", []) or []:
            if isinstance(aura, dict):
                name = aura.get("name")
                source_id = aura.get("source_id", "")
            else:
                name = getattr(aura, "name", None)
                source_id = getattr(aura, "source_id", "")
            if name:
                auras.append(name)
                if source_id:
                    aura_source_ids[name.lower()] = source_id
        if auras and hasattr(self._tool_helper, "update_aura_tools"):
            self._tool_helper.update_aura_tools(
                auras, self.worker.chat_config.tools_path, aura_source_ids
            )
        elif hasattr(self._tool_helper, "update_aura_tools"):
            self._tool_helper.update_aura_tools([], self.worker.chat_config.tools_path)

    async def _decide_action(self, turn_request: MUDTurnRequest, events: list[MUDEvent]) -> tuple[list[MUDAction], str]:
        """Execute agent-guided decision strategy.

        Args:
            turn_request: Current turn request with sequence_id
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking)
        """
        thinking_parts: list[str] = []
        actions_taken: list[MUDAction] = []

        # Refresh aura tools for @agent context
        self._refresh_aura_tools()

        # Filter tools FIRST if required_tool is set (before building any guidance)
        logger.info("AgentTurnProcessor required_tool='%s'", self.required_tool)
        if self.required_tool:
            if not self._tool_helper.filter_to_tool(self.required_tool):
                logger.warning("Required tool '%s' not found, proceeding with all tools", self.required_tool)
        else:
            logger.info("No required_tool set, using all tools")

        # Build comprehensive guidance including ToolUser signatures
        guidance = self._build_agent_guidance(self.user_guidance)
        coming_online = await self.worker._is_fresh_session()

        # Build user input with current context and agent guidance
        # Note: action_guidance intentionally omitted - agent turns are user-directed
        # and don't need self-action context (handled by regular phased turns)
        user_input = build_current_context(
            self.worker.session,
            idle_mode=False,
            guidance=guidance,
            coming_online=coming_online,
            include_events=False,
            include_format_guidance=False,  # @agent: JSON tool calls only, like Phase 1
        )

        # Use response strategy for full context (consciousness + memory)
        chat_turns = await self.worker._response_strategy.build_turns(
            persona=self.worker.persona,
            user_input=user_input,
            session=self.worker.session,
            coming_online=coming_online,
            max_context_tokens=self.worker.model.max_tokens,
            max_output_tokens=self.worker.chat_config.max_tokens,
        )

        # Build system message with agent action tools (override Phase 2's clean message)
        self.worker.chat_config.system_message = self.build_system_message()

        allowed = {a.get("name", "").lower() for a in self.worker._agent_action_list() if a.get("name")}
        # Allow aura-injected tools even if not in mud_agent.yaml
        if hasattr(self._tool_helper, "_aura_tools") and self._tool_helper._aura_tools:
            allowed = set(allowed)
            for tool in self._tool_helper._aura_tools:
                name = getattr(tool.function, "name", None)
                if name:
                    allowed.add(name)

        # Filter allowed set to required tool if specified
        logger.info("Initial allowed set: %s", allowed)
        if self.required_tool:
            required_lower = self.required_tool.lower()
            if required_lower in allowed:
                allowed = {required_lower}
                logger.info("Filtered allowed set to: %s", allowed)
            else:
                logger.warning("Required tool '%s' not in allowed tools: %s", self.required_tool, allowed)
        else:
            logger.info("No required_tool, keeping all allowed: %s", allowed)

        # Create heartbeat callback for @agent action generation
        async def heartbeat_callback() -> None:
            """Refresh turn request heartbeat during long-running @agent generation."""
            result = await self.worker.atomic_heartbeat_update()

            if result == 0:
                logger.debug("Turn request deleted during @agent turn, stopping heartbeat")
            elif result == -1:
                logger.error("Corrupted turn_request during @agent heartbeat")

        try:
            for attempt in range(self.worker.config.decision_max_retries):
                # Check for abort before @agent LLM call
                if await self.worker._check_abort_requested():
                    raise AbortRequestedException("Turn aborted before @agent action")

                # @agent uses agent role (structured agent actions)
                response = await self.worker._call_llm(
                    chat_turns,
                    role="agent",
                    heartbeat_callback=heartbeat_callback
                )
                cleaned, think_content = extract_think_tags(response)
                cleaned = sanitize_response(cleaned)
                cleaned = cleaned.strip()
                if think_content:
                    thinking_parts.append(think_content)

                action, args, error = parse_agent_action_response(cleaned)
                logger.info("Parsed action='%s' from response, allowed=%s", action, allowed)
                if not action:
                    logger.warning(
                        "Invalid @agent response (attempt %d/%d): %s",
                        attempt + 1,
                        self.worker.config.decision_max_retries,
                        error,
                    )
                    continue

                if allowed and action not in allowed:
                    logger.warning(
                        "REJECTING @agent action '%s' not in allowed %s (attempt %d/%d)",
                        action,
                        allowed,
                        attempt + 1,
                        self.worker.config.decision_max_retries,
                    )
                    continue

                logger.info("ACCEPTING action '%s' (in allowed set)", action)

                # Valid action -> emit
                if action == "move":
                    location = args.get("location") or args.get("direction")
                    resolved = resolve_move_location(self.worker.session, location)
                    if resolved:
                        action_obj = MUDAction(tool="move", args={"location": resolved})
                        actions_taken.append(action_obj)
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent move missing/invalid location")
                    break

                if action == "take":
                    obj = args.get("object")
                    if obj:
                        action_obj = MUDAction(tool="get", args={"object": obj})
                        actions_taken.append(action_obj)
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent take missing object")
                    break

                if action == "drop":
                    obj = args.get("object")
                    if obj:
                        action_obj = MUDAction(tool="drop", args={"object": obj})
                        actions_taken.append(action_obj)
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent drop missing object")
                    break

                if action == "give":
                    obj = args.get("object")
                    target = args.get("target")
                    if obj and target:
                        action_obj = MUDAction(tool="give", args={"object": obj, "target": target})
                        actions_taken.append(action_obj)
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent give missing object or target")
                    break

                if action == "desc_room":
                    description = args.get("description")
                    if description:
                        action_obj = MUDAction(
                            tool="desc_room",
                            args={"description": description},
                        )
                        actions_taken.append(action_obj)
                        emote_text = "took a moment to reimagine the room."
                        actions_taken.append(
                            MUDAction(tool="emote", args={"action": emote_text})
                        )
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent desc_room missing description")
                    break

                if action == "desc_object":
                    target = args.get("target")
                    description = args.get("description")
                    if target and description:
                        # Validate target is an object, not the room
                        room = self.worker.session.current_room if self.worker.session else None
                        if room and room.room_id == target:
                            logger.warning("Agent desc_object target is room ID %s, use desc_room instead", target)
                            break

                        action_obj = MUDAction(
                            tool="desc_object",
                            args={"target": target, "description": description},
                        )
                        actions_taken.append(action_obj)
                        object_name = resolve_target_name(self.worker.session, target)
                        emote_text = f"adjusted the {object_name}."
                        actions_taken.append(
                            MUDAction(tool="emote", args={"action": emote_text})
                        )
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent desc_object missing target or description")
                    break

                if action == "ring":
                    obj = (args.get("object") or "").strip()
                    ringables = get_ringable_objects(self.worker.session)
                    if not obj:
                        if len(ringables) == 1:
                            obj = ringables[0]
                        else:
                            logger.warning("Agent ring missing object; no action emitted")
                            continue
                    if ringables and obj.lower() not in [r.lower() for r in ringables]:
                        logger.warning("Agent ring invalid target '%s'; no action emitted", obj)
                        continue
                    action_obj = MUDAction(tool="ring", args={"object": obj})
                    actions_taken.append(action_obj)
                    await self.worker._emit_actions(actions_taken)
                    break

                # Check if it's an aura tool - just emit the action, Evennia executes
                if self._tool_helper.is_aura_tool(action):
                    action_obj = MUDAction(tool=action, args=args)
                    actions_taken.append(action_obj)
                    await self.worker._emit_actions(actions_taken)
                    break

        except Exception as e:
            logger.error(f"Error during @agent turn processing: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] @agent turn processing failed: {e}")
            # Emit a graceful emote when LLM fails
            action_obj = MUDAction(tool="emote", args={"action": "was at a loss for words."})
            actions_taken.append(action_obj)
            await self.worker._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()
        return actions_taken, thinking

    @classmethod
    def from_config(cls, worker: "TurnsMixin", chat_config: ChatConfig, mud_config: MUDConfig) -> "AgentTurnProcessor":
        """Create an AgentTurnProcessor from a configuration."""
        tool_helper = ToolHelper.from_file(mud_config.agent_tool_file, chat_config.tools_path)
        return cls(worker, tool_helper)
