# tests/unit/worker/test_self_action_formatting.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests to verify rich formatted guidance boxes are preserved in self-action events.

This test suite validates that when self-action events with rich formatting
(guidance boxes with box-drawing characters) are processed by the conversation
manager, the formatting is preserved verbatim rather than being converted to
plain prose.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from andimud_worker.conversation.manager import MUDConversationManager
from aim_mud_types import MUDEvent, EventType, ActorType, RoomState, WorldState
from aim.constants import DOC_MUD_WORLD


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.rpush = AsyncMock()
    redis.llen = AsyncMock(return_value=0)
    redis.lrange = AsyncMock(return_value=[])
    return redis


@pytest.fixture
def conversation_manager(mock_redis):
    """Create a conversation manager with mocked Redis."""
    return MUDConversationManager(
        redis=mock_redis,
        agent_id="test_agent",
        persona_id="andi",
        max_tokens=50000,
    )


@pytest.fixture
def rich_formatted_movement_event():
    """Create a self-action movement event with rich formatted guidance box."""
    return MUDEvent(
        event_type=EventType.MOVEMENT,
        actor="Andi",
        actor_id="test_agent",
        actor_type=ActorType.AI,
        room_id="#124",
        room_name="The Kitchen",
        content="""╔════════════════════════════════════════════════════════════════════════════╗
║                           !! IMPORTANT NOTICE !!                           ║
║  You have just taken an action. The environment has responded with a      ║
║  description below. You do NOT need to repeat or acknowledge this action  ║
║  unless directly relevant to your next decision or response.              ║
╚════════════════════════════════════════════════════════════════════════════╝

You moved north to The Kitchen.

The Kitchen is a warm kitchen with the smell of bread baking.""",
        metadata={"is_self_action": True},
        world_state=WorldState(
            room_state=RoomState(
                room_id="#124",
                name="The Kitchen",
                description="A warm kitchen with the smell of bread baking.",
                exits={"south": "#123"}
            ),
            entities_present=[],
            inventory=[]
        ),
    )


@pytest.mark.asyncio
async def test_self_action_preserves_rich_formatting(
    conversation_manager, mock_redis, rich_formatted_movement_event
):
    """Test that rich formatted guidance boxes are preserved in self-action events."""
    # Process the self-action event
    await conversation_manager.push_user_turn(
        events=[rich_formatted_movement_event],
    )

    # Verify Redis was called to push the event
    assert mock_redis.rpush.called, "Redis rpush should be called"

    # Get the actual content that was pushed
    call_args = mock_redis.rpush.call_args
    pushed_json = call_args[0][1]  # Second argument to rpush is the JSON data

    # Parse the JSON to get the content
    import json
    pushed_entry = json.loads(pushed_json)
    content = pushed_entry["content"]

    # Verify the formatting is preserved
    assert "╔════" in content, "Top border of guidance box should be preserved"
    assert "!! IMPORTANT NOTICE !!" in content, "Header text should be preserved"
    assert "╚════" in content, "Bottom border of guidance box should be preserved"
    assert "You moved north to The Kitchen" in content, "Movement description should be present"

    # Verify it's NOT converted to plain prose
    assert "You moved" in content, "Original content should be present"
    # The old behavior would have stripped the box and just said "You moved to The Kitchen."


@pytest.mark.asyncio
async def test_self_action_uses_event_content_directly(
    conversation_manager, mock_redis, rich_formatted_movement_event
):
    """Test that self-action events use event.content directly, not format_self_event()."""
    # Process the event
    await conversation_manager.push_user_turn(
        events=[rich_formatted_movement_event],
    )

    # Get the pushed content
    import json
    call_args = mock_redis.rpush.call_args
    pushed_json = call_args[0][1]
    pushed_entry = json.loads(pushed_json)
    content = pushed_entry["content"]

    # The content should be exactly what was in event.content
    # (minus any additional processing by the conversation manager)
    assert content == rich_formatted_movement_event.content, \
        "Content should match event.content exactly"


@pytest.mark.asyncio
async def test_regular_event_not_affected(conversation_manager, mock_redis):
    """Test that regular (non-self-action) events still work correctly."""
    # Create a regular event (not a self-action)
    regular_event = MUDEvent(
        event_type=EventType.SPEECH,
        actor="Prax",
        actor_id="#prax_1",
        actor_type=ActorType.PLAYER,
        room_id="#123",
        content="Prax says, 'Hello there!'",
        metadata={},  # No is_self_action
        world_state=None,
    )

    # Process it
    await conversation_manager.push_user_turn(
        events=[regular_event],
    )

    # Verify it was processed
    assert mock_redis.rpush.called

    # Get the content
    import json
    call_args = mock_redis.rpush.call_args
    pushed_json = call_args[0][1]
    pushed_entry = json.loads(pushed_json)
    content = pushed_entry["content"]

    # Regular events should have their content preserved as-is
    assert "Prax says, 'Hello there!'" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
