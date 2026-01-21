# aim/app/mud/worker/commands/helpers.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Helper functions for command execution.

This module contains shared setup logic used by commands before processor execution.
The setup_turn_context() function was extracted from BaseTurnProcessor.setup_turn()
to separate business logic from the processor base class.

Commands call setup_turn_context() once at the start, then processors use the
prepared state from worker.session and worker.conversation_manager.
"""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDEvent

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker

logger = logging.getLogger(__name__)


async def setup_turn_context(
    worker: "MUDAgentWorker",
    events: list[MUDEvent]
) -> None:
    """Setup turn context - called by commands before processors.

    This is the business logic that was previously in BaseTurnProcessor.setup_turn().
    Commands call this once at the start, then all processors use the prepared state.

    Performs the following setup steps:
    1. Update decision tool availability based on drained events
    2. Refresh world state snapshot from agent + room profiles
    3. Log event details for debugging
    4. Update session context from events
    5. Push user turn to conversation history

    Args:
        worker: Worker instance with session, conversation_manager, etc.
        events: Events to process

    Example:
        ```python
        # In a command's execute() method:
        await setup_turn_context(worker, events)
        processor = PhasedTurnProcessor(worker)
        await processor.execute(turn_request, events)
        ```
    """
    # Update decision tool availability based on drained events
    if hasattr(worker, "_refresh_emote_tools"):
        worker._refresh_emote_tools(events)

    # Refresh world state snapshot from agent + room profiles
    room_id, character_id = await worker._load_agent_world_state()
    if not room_id and worker.session.current_room and worker.session.current_room.room_id:
        room_id = worker.session.current_room.room_id
    if not room_id and events:
        room_id = events[-1].room_id
    await worker._load_room_profile(room_id, character_id)

    # Log event details for debugging
    for event in events:
        logger.info(
            f"  Event: {event.event_type.value} | "
            f"Actor: {event.actor} | "
            f"Room: {event.room_name or event.room_id} | "
            f"Content: {event.content[:100] if event.content else '(none)'}..."
        )

    # Update session context from events
    worker.session.pending_events = events
    if events:
        latest = events[-1]
        worker.session.last_event_time = latest.timestamp

    # Push user turn to conversation list
    if worker.conversation_manager and events:
        await worker.conversation_manager.push_user_turn(
            events=events,
            world_state=worker.session.world_state,
            room_id=worker.session.current_room.room_id if worker.session.current_room else None,
            room_name=worker.session.current_room.name if worker.session.current_room else None,
        )
