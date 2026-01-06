# aim/app/mud/worker/turns/strategy/base.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Base turn processor with template method pattern.

execute() is the framework that orchestrates the turn.
Subclasses override _decide_action() to implement their decision strategy.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDEvent, MUDTurn
from aim_mud_types.helper import _utc_now

from ...adapter import format_self_action_guidance

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin

logger = logging.getLogger(__name__)


class BaseTurnProcessor(ABC):
    """Base class for turn processing strategies.

    Template method pattern:
    - execute() is the framework (final)
    - _decide_action() is the extension point (abstract, overridden by subclasses)
    - setup_turn() and finalize_turn() are shared helpers
    """

    _action_guidance: str = ""

    def __init__(self, worker: "TurnsMixin"):
        """Initialize processor with worker reference.

        Args:
            worker: MUDAgentWorker instance (TurnsMixin)
        """
        self.worker = worker

    async def execute(self, events: list[MUDEvent]) -> None:
        """Framework method that orchestrates the entire turn.

        Template method that calls:
        1. setup_turn() - Load world state, log events, update session
        2. _decide_action() - Strategy-specific decision logic (abstract)
        3. finalize_turn() - Save conversation, create turn record

        Args:
            events: List of events to process
        """
        await self.setup_turn(events)
        actions_taken, thinking = await self._decide_action(events)
        await self.finalize_turn(actions_taken, thinking, events)

    @abstractmethod
    async def _decide_action(self, events: list[MUDEvent]) -> tuple[list[MUDAction], str]:
        """Strategy-specific decision logic (overridden by subclasses).

        Implementations determine:
        - How to process events
        - Which LLM calls to make
        - How to validate responses
        - What actions to emit

        Args:
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking_text)
        """
        pass

    async def setup_turn(self, events: list[MUDEvent]) -> None:
        """Common setup for any turn processing.

        Loads world state, logs events, updates session, and pushes user turn
        to conversation history.

        Args:
            events: List of events to process
        """
        # Refresh world state snapshot from agent + room profiles
        room_id, character_id = await self.worker._load_agent_world_state()
        if not room_id and self.worker.session.current_room and self.worker.session.current_room.room_id:
            room_id = self.worker.session.current_room.room_id
        if not room_id and events:
            room_id = events[-1].room_id
        await self.worker._load_room_profile(room_id, character_id)

        # Capture pending self-actions for both guidance and document inclusion
        self._action_guidance = ""
        self_actions: list[MUDEvent] = []
        if self.worker.session and self.worker.session.pending_self_actions:
            self_actions = self.worker.session.pending_self_actions.copy()
            self._action_guidance = format_self_action_guidance(self_actions)
            logger.info(
                f"Prepared action guidance for {len(self_actions)} self-actions"
            )
            # Clear pending self-actions (they will be presented this turn)
            self.worker.session.pending_self_actions = []

        # Log event details for debugging
        for event in events:
            logger.info(
                f"  Event: {event.event_type.value} | "
                f"Actor: {event.actor} | "
                f"Room: {event.room_name or event.room_id} | "
                f"Content: {event.content[:100] if event.content else '(none)'}..."
            )

        # Step 1: Update session context from events
        self.worker.session.pending_events = events
        if events:
            latest = events[-1]
            self.worker.session.last_event_time = latest.timestamp

        # Combine self-actions with external events for mud-world document
        # Self-actions go first (they happened before we received new events)
        all_events_for_doc = self_actions + events

        # Push user turn to conversation list
        if self.worker.conversation_manager and all_events_for_doc:
            await self.worker.conversation_manager.push_user_turn(
                events=all_events_for_doc,
                world_state=self.worker.session.world_state,
                room_id=self.worker.session.current_room.room_id if self.worker.session.current_room else None,
                room_name=self.worker.session.current_room.name if self.worker.session.current_room else None,
            )

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

        # Clear action guidance (it has been consumed)
        self._action_guidance = ""
