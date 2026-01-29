# aim/app/mud/worker/turns/processor/decision.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Phase 1 decision processor.

This processor calls the LLM with the decision role (fast, cheap)
to choose an action. It returns a DecisionResult enum that routing
logic will use to determine next steps.

This processor does NOT emit actions itself - it only decides.
"""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDEvent, MUDTurnRequest
from aim_mud_types.models.decision import DecisionType, DecisionResult

from ...exceptions import AbortRequestedException
from .base import BaseTurnProcessor

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin

logger = logging.getLogger(__name__)


class DecisionProcessor(BaseTurnProcessor):
    """Phase 1: Decision making processor.

    Calls LLM with decision role to choose action using tool calling.
    Maps the tool name to a DecisionType enum and stores the result
    on worker._last_decision for routing logic.

    Does NOT emit actions - that happens in command handlers based on
    the DecisionType returned.

    Attributes:
        user_guidance: Optional guidance from @choose command
    """

    def __init__(self, worker: "TurnsMixin"):
        """Initialize decision processor.

        Args:
            worker: MUDAgentWorker instance with TurnsMixin
        """
        super().__init__(worker)
        self.user_guidance = ""

    async def _decide_action(
        self,
        turn_request: MUDTurnRequest,
        events: list[MUDEvent]
    ) -> tuple[list[MUDAction], str]:
        """Make decision and store result for routing.

        This method:
        1. Calls LLM with decision role (fast tool selection)
        2. Maps tool name to DecisionType enum
        3. Stores DecisionResult on worker._last_decision
        4. Returns empty actions list (routing happens in commands)

        Args:
            turn_request: Current turn request
            events: List of events to process

        Returns:
            Tuple of (empty_actions_list, thinking_text)

        Raises:
            AbortRequestedException: If turn is aborted before decision
        """
        metadata = turn_request.metadata or {}
        logger.info(
            "Decision turn start: id=%s reason=%s status=%s seq=%s events=%d room_auras=%s",
            turn_request.turn_id,
            turn_request.reason.value if hasattr(turn_request.reason, "value") else turn_request.reason,
            turn_request.status.value if hasattr(turn_request.status, "value") else turn_request.status,
            turn_request.sequence_id,
            len(events),
            metadata.get("room_auras"),
        )

        room_auras = metadata.get("room_auras")
        if (
            room_auras is not None
            and self.worker._decision_strategy
            and self.worker.chat_config
        ):
            if isinstance(room_auras, (list, tuple, set)):
                self.worker._decision_strategy.update_aura_tools(
                    list(room_auras),
                    self.worker.chat_config.tools_path,
                )
        if self.worker._decision_strategy:
            tool_names = self.worker._decision_strategy.get_available_tool_names()
            logger.info("Decision available tools (%d): %s", len(tool_names), ", ".join(tool_names))

        idle_mode = len(events) == 0

        # Check for abort before decision LLM call
        if await self.worker._check_abort_requested():
            raise AbortRequestedException("Turn aborted before decision")

        # Phase 1: Call decision strategy
        decision_tool, decision_args, decision_raw, decision_thinking, decision_cleaned = (
            await self.worker._decide_action(
                idle_mode=idle_mode,
                role="decision",
                action_guidance="",
                user_guidance=self.user_guidance,
            )
        )

        # Map string tool to DecisionType enum
        decision_type = self._parse_decision_type(
            decision_tool,
            decision_args
        )

        # Package as DecisionResult
        result = DecisionResult(
            decision_type=decision_type,
            args=decision_args,
            thinking=decision_thinking,
            raw_response=decision_raw,
            cleaned_response=decision_cleaned,
            should_flush=decision_type == DecisionType.SPEAK,
            aura_tool_name=decision_tool if decision_type == DecisionType.AURA_TOOL else None
        )

        # Store result on worker for routing logic
        self.worker._last_decision = result

        logger.info(
            "Decision made: %s (tool=%s, args=%s)",
            decision_type.name,
            decision_tool,
            decision_args
        )

        # Return empty actions - routing happens in commands based on DecisionType
        return [], decision_thinking

    def _parse_decision_type(
        self,
        tool_name: str,
        args: dict
    ) -> DecisionType:
        """Map string tool name to DecisionType enum.

        Args:
            tool_name: Name of the tool chosen by LLM
            args: Arguments for the tool (unused, for signature compatibility)

        Returns:
            DecisionType enum member
        """
        # Check if this is an aura tool (dynamic tool from room)
        if self.worker._decision_strategy.is_aura_tool(tool_name):
            return DecisionType.AURA_TOOL

        # Map standard tool names to DecisionType
        mapping = {
            "move": DecisionType.MOVE,
            "take": DecisionType.TAKE,
            "drop": DecisionType.DROP,
            "give": DecisionType.GIVE,
            "speak": DecisionType.SPEAK,
            "emote": DecisionType.EMOTE,
            "wait": DecisionType.WAIT,
            "think": DecisionType.THINK,
            "plan": DecisionType.PLAN,
            "plan_update": DecisionType.PLAN_UPDATE,
            "close_book": DecisionType.CLOSE_BOOK,
            "focus": DecisionType.FOCUS,
            "confused": DecisionType.CONFUSED,
        }

        decision_type = mapping.get(tool_name.lower(), DecisionType.CONFUSED)

        if decision_type == DecisionType.CONFUSED:
            logger.warning(
                "Unknown decision tool '%s'; mapping to CONFUSED",
                tool_name
            )

        return decision_type
