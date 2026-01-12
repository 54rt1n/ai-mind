# aim/app/mud/worker/turns/strategy/agent.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agent turn processor: single-phase with full guidance."""

import logging
from typing import TYPE_CHECKING
from pathlib import Path

from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from aim.utils.xml import XmlFormatter
from aim_mud_types import MUDAction, MUDEvent, MUDTurnRequest, EventType
from aim.utils.think import extract_think_tags
from ...adapter import build_current_context
from ..response import (
    sanitize_response,
    normalize_response,
    has_emotional_state_header,
    parse_agent_action_response,
)
from ..validation import resolve_move_location, resolve_target_name
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
    All actions (speak, move, take, drop, give, describe) decided together.
    """

    def __init__(self, worker: "TurnsMixin", tool_helper: ToolHelper):
        """Initialize with worker and set user_guidance to empty string.

        Args:
            worker: MUDAgentWorker instance
        """
        super().__init__(worker)
        self.user_guidance = ""
        self._tool_helper = tool_helper

    def _create_event(
        self,
        turn_request: MUDTurnRequest,
        event_type: EventType,
        content: str,
        target: str = None,
    ) -> MUDEvent:
        """Create self-action event with formatted guidance content.

        Creates event, then formats it with format_self_action_guidance()
        to generate the detailed guidance box that appears in prompts.

        Args:
            turn_request: Current turn request with sequence_id.
            event_type: Type of event (MOVEMENT, OBJECT, etc.).
            content: Event content/description (e.g., "moved from X to Y").
            target: Optional target (object name, direction, etc.).

        Returns:
            MUDEvent with formatted content and is_self_action=True metadata.
        """
        from ...adapter import format_self_action_guidance
        from aim_mud_types import ActorType

        # Create basic event
        event = MUDEvent(
            event_type=event_type,
            actor=self.worker.persona.name,
            actor_id=self.worker.config.agent_id,
            actor_type=ActorType.AI,
            room_id=(
                self.worker.session.current_room.room_id
                if self.worker.session.current_room
                else "unknown"
            ),
            room_name=(
                self.worker.session.current_room.name
                if self.worker.session.current_room and self.worker.session.current_room.name
                else "Unknown Location"
            ),
            content=content,
            target=target,
            sequence_id=turn_request.sequence_id,
            metadata={"is_self_action": True},
            world_state=self.worker.session.world_state,
        )

        # Format with existing guidance formatter
        formatted_content = format_self_action_guidance(
            [event],
            world_state=self.worker.session.world_state
        )

        # Update event with formatted content
        event.content = formatted_content

        return event

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
        - OpenAI-style function signatures from ToolUser
        - Agent action spec instructions
        - Current world state context (exits, objects, inventory, targets)
        - User guidance if provided

        Args:
            user_guidance: Optional user-provided guidance string

        Returns:
            Formatted guidance string for agent action selection
        """
        parts = []

        # Get ToolUser guidance (OpenAI-style function signatures)
        if self._tool_helper and self._tool_helper._tool_user:
            tool_guidance = self._tool_helper._tool_user.get_tool_guidance()
            if tool_guidance:
                parts.append(tool_guidance)
                parts.append("")

        # Get basic agent action hints from worker
        basic_guidance = self.worker._build_agent_guidance(user_guidance)
        if basic_guidance:
            parts.append(basic_guidance)

        return "\n".join(parts)

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

        try:
            for attempt in range(self.worker.config.decision_max_retries):
                # Check for abort before @agent LLM call
                if await self.worker._check_abort_requested():
                    raise AbortRequestedException("Turn aborted before @agent action")

                # @agent uses agent role (structured agent actions)
                response = await self.worker._call_llm(chat_turns, role="agent")
                cleaned, think_content = extract_think_tags(response)
                cleaned = sanitize_response(cleaned)
                cleaned = cleaned.strip()
                if think_content:
                    thinking_parts.append(think_content)

                action, args, error = parse_agent_action_response(cleaned)
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
                        "Invalid @agent action '%s' (attempt %d/%d)",
                        action,
                        attempt + 1,
                        self.worker.config.decision_max_retries,
                    )
                    continue

                # Valid action -> emit
                if action == "speak":
                    text = args.get("text", "")
                    # Validate emotional state header for speak actions
                    if not has_emotional_state_header(text):
                        logger.warning(
                            "Agent speak missing Emotional State header (attempt %d/%d)",
                            attempt + 1,
                            self.worker.config.decision_max_retries,
                        )
                        if attempt < self.worker.config.decision_max_retries - 1:
                            persona_name = self.worker.session.persona_id if self.worker.session else "Agent"
                            format_guidance = (
                                f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                                f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotion+> ==] then continue with prose.]"
                            )
                            if chat_turns and chat_turns[-1]["role"] == "user":
                                chat_turns[-1]["content"] += format_guidance
                            else:
                                chat_turns.append({"role": "user", "content": format_guidance})
                            continue
                    normalized = normalize_response(text)
                    if normalized:
                        action_obj = MUDAction(tool="speak", args={"text": normalized})
                        actions_taken.append(action_obj)
                        await self.worker._emit_actions(actions_taken)
                    else:
                        logger.info("Agent speak had no text; no action emitted")
                    break

                if action == "move":
                    location = args.get("location") or args.get("direction")
                    resolved = resolve_move_location(self.worker.session, location)
                    if resolved:
                        # Capture source location before move
                        if self.worker.session.current_room and self.worker.session.current_room.name:
                            source_location = self.worker.session.current_room.name
                        else:
                            logger.warning(
                                f"Current room not set for agent {self.worker.agent_id} during move action"
                            )
                            source_location = "Unknown Location"

                        action_obj = MUDAction(tool="move", args={"location": resolved})
                        actions_taken.append(action_obj)

                        # Create and write self-action event immediately
                        event = self._create_event(
                            turn_request,
                            EventType.MOVEMENT,
                            f"moved from {source_location} to {resolved}",
                            target=resolved,
                        )
                        # Store source and destination in metadata for formatting
                        event.metadata["source_location"] = source_location
                        event.metadata["destination_location"] = resolved

                        await self.worker._write_self_event(event)
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

                if action == "describe":
                    target = args.get("target")
                    description = args.get("description")
                    if target and description:
                        action_obj = MUDAction(
                            tool="describe",
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
                        logger.warning("Agent describe missing target or description")
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