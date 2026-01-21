# aim/app/mud/worker/turns/strategy/base.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Base turn processor with template method pattern.

execute() is the framework that orchestrates the turn.
Subclasses override _decide_action() to implement their decision strategy.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDEvent, MUDTurn, MUDTurnRequest
from aim_mud_types.helper import _utc_now

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin

logger = logging.getLogger(__name__)


class BaseTurnProcessor(ABC):
    """Base class for turn processing strategies.

    Template method pattern:
    - execute() is the framework (final)
    - _decide_action() is the extension point (abstract, overridden by subclasses)
    - finalize_turn() is shared helper

    NOTE: setup_turn() removed - commands call setup_turn_context() helper instead.
    """

    def __init__(self, worker: "TurnsMixin"):
        """Initialize processor with worker reference.

        Args:
            worker: MUDAgentWorker instance (TurnsMixin)
        """
        self.worker = worker

    async def execute(self, turn_request: MUDTurnRequest, events: list[MUDEvent]) -> None:
        """Framework method that orchestrates the turn.

        NOTE: Commands should call setup_turn_context() BEFORE calling execute().

        Template method that calls:
        1. _decide_action() - Strategy-specific decision logic (abstract)
        2. finalize_turn() - Save conversation, create turn record

        Args:
            turn_request: Current turn request with sequence_id
            events: List of events to process
        """
        actions_taken, thinking = await self._decide_action(turn_request, events)
        await self.finalize_turn(actions_taken, thinking, events)

    @abstractmethod
    async def _decide_action(self, turn_request: MUDTurnRequest, events: list[MUDEvent]) -> tuple[list[MUDAction], str]:
        """Strategy-specific decision logic (overridden by subclasses).

        Implementations determine:
        - How to process events
        - Which LLM calls to make
        - How to validate responses
        - What actions to emit

        Args:
            turn_request: Current turn request with sequence_id
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking_text)
        """
        pass

    async def finalize_turn(
        self,
        actions_taken: list[MUDAction],
        thinking: str,
        events: list[MUDEvent],
    ) -> None:
        """Common teardown for any turn processing.

        Pushes assistant turn to conversation history (if speak action),
        creates turn record, and cleans up session state.

        Args:
            actions_taken: List of actions that were executed
            thinking: Concatenated thinking/reasoning from all phases
            events: Original events that were processed
        """
        # Push assistant turn to conversation list - ONLY for speak actions
        # Non-speak actions are mechanical tool calls, not narrative content
        if self.worker.conversation_manager:
            for action in actions_taken:
                if action.tool == "speak":
                    speak_text = action.args.get("text", "")
                    if speak_text:
                        await self.worker.conversation_manager.push_assistant_turn(
                            content=speak_text,
                            think=thinking if thinking else None,
                            actions=actions_taken,
                        )
                    break

        # Track emote usage so we can suppress repeated emotes for the same drain
        if any(action.tool == "emote" for action in actions_taken):
            if hasattr(self.worker, "_mark_emote_used_in_drain"):
                self.worker._mark_emote_used_in_drain()

        # Create turn record
        turn = MUDTurn(
            timestamp=_utc_now(),
            events_received=events,
            room_context=self.worker.session.current_room,
            entities_context=self.worker.session.entities_present,
            thinking=thinking,
            actions_taken=actions_taken,
        )

        # Add turn to session history
        self.worker.session.add_turn(turn)
        self.worker.session.clear_pending_events()

        logger.info(
            f"Turn processed. Actions: {len(actions_taken)}. "
            f"Session now has {len(self.worker.session.recent_turns)} turns"
        )
