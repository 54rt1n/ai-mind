# packages/aim-mud/tests/mud_tests/unit/mediator/test_movement_event_filtering.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for metadata-based movement event filtering.

Tests the fix where:
1. Mediator uses event.metadata.get("actor_agent_id") to identify self-agents
2. Movement events are ALWAYS filtered from the actor (worker has self-action)
3. No room lookup needed when metadata is present
4. Defensive fallback to room lookup when metadata is absent
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import MUDEvent, EventType, ActorType, RedisKeys


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
    )


@pytest.fixture
def mediator(mock_redis, mediator_config):
    """Create a mediator service with mocked Redis."""
    # Setup mock redis incr for sequence ID
    seq_counter = [0]
    async def mock_incr(key):
        seq_counter[0] += 1
        return seq_counter[0]
    mock_redis.incr = mock_incr

    # Setup mock hget to return None for sleeping state (agents are awake)
    # This is critical - RedisMUDClient.get_agent_is_sleeping() calls hget
    async def mock_hget(key, field=None):
        # Return None for sleeping state (not sleeping)
        return None
    mock_redis.hget = AsyncMock(side_effect=mock_hget)

    # Setup mock hexists to return False (events not already processed)
    async def mock_hexists(key, field):
        return False
    mock_redis.hexists = AsyncMock(side_effect=mock_hexists)

    # Setup mock hset for marking events as processed
    mock_redis.hset = AsyncMock(return_value=1)

    service = MediatorService(mock_redis, mediator_config)
    service.register_agent("andi")
    service.register_agent("nova")

    # Mock _try_handle_control_command to return False (not a control command)
    async def mock_control_command(event):
        return False
    service._try_handle_control_command = mock_control_command

    return service


class TestMovementEventMetadataFiltering:
    """Test movement event filtering using metadata for actor identification."""

    @pytest.mark.asyncio
    async def test_movement_event_uses_metadata_for_actor_id(self, mediator, mock_redis):
        """Test that movement events prefer metadata over room lookup for actor identification."""
        # Create movement event with actor_agent_id in metadata
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#3",
            actor_type=ActorType.AI,
            room_id="room:kitchen",
            room_name="Kitchen",
            content="moved from Living Room to Kitchen",
            target="Kitchen",
            metadata={"actor_agent_id": "andi"},  # Metadata provides actor
        )

        # Track queue_event_for_compilation calls
        queued_events = []
        async def mock_queue_event(event, observer_agent_ids, self_action_agent_id):
            for agent_id in observer_agent_ids:
                queued_events.append((agent_id, False))
            if self_action_agent_id:
                queued_events.append((self_action_agent_id, True))

        # Create a spy to track if _agent_id_from_actor is called
        with patch.object(mediator, '_agent_id_from_actor', wraps=mediator._agent_id_from_actor) as spy:
            # Mock _agents_from_room_profile
            with patch.object(mediator, '_agents_from_room_profile', return_value=["andi", "nova"]):
                # Mock _maybe_assign_turn to avoid turn assignment logic
                with patch.object(mediator, '_maybe_assign_turn', return_value=False):
                    # Mock queue_event_for_compilation to track calls
                    with patch.object(mediator, 'queue_event_for_compilation', side_effect=mock_queue_event):
                        # Process the event
                        msg_id = "1704096000000-0"
                        event_data = {
                            b"data": json.dumps(event.to_redis_dict()).encode()
                        }
                        await mediator._process_event(msg_id, event_data)

                        # Assert _agent_id_from_actor was NOT called (metadata was used)
                        spy.assert_not_called()

                        # Assert event was queued for compilation
                        assert len(queued_events) >= 1

    @pytest.mark.asyncio
    async def test_movement_event_filters_actor_from_distribution(self, mediator, mock_redis):
        """Test that movement events filter out the actor agent from distribution."""
        # Create movement event with metadata indicating andi is the actor
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#3",
            actor_type=ActorType.AI,
            room_id="room:kitchen",
            room_name="Kitchen",
            content="moved from Living Room to Kitchen",
            target="Kitchen",
            metadata={"actor_agent_id": "andi"},
        )

        # Track queue_event_for_compilation calls
        queued_events = []
        async def mock_queue_event(event, observer_agent_ids, self_action_agent_id):
            for agent_id in observer_agent_ids:
                queued_events.append((agent_id, False))
            if self_action_agent_id:
                queued_events.append((self_action_agent_id, True))

        # Mock _agents_from_room_profile to return both agents
        with patch.object(mediator, '_agents_from_room_profile', return_value=["andi", "nova"]):
            # Mock _maybe_assign_turn
            with patch.object(mediator, '_maybe_assign_turn', return_value=False):
                # Mock queue_event_for_compilation to track calls
                with patch.object(mediator, 'queue_event_for_compilation', side_effect=mock_queue_event):
                    # Process the event
                    msg_id = "1704096000000-1"
                    event_data = {
                        b"data": json.dumps(event.to_redis_dict()).encode()
                    }
                    await mediator._process_event(msg_id, event_data)

                    # Assert events were queued for compilation
                    assert len(queued_events) >= 1

                    # Extract which agents received the event
                    distributed_agents = [agent_id for agent_id, _ in queued_events]

                    # Assert only nova received the event as third-person
                    # (andi filtered from observers, but movement doesn't give self-action)
                    assert "nova" in distributed_agents

    @pytest.mark.asyncio
    async def test_non_movement_event_still_uses_room_lookup_fallback(self, mediator, mock_redis):
        """Test that non-movement events use room lookup when metadata is absent."""
        # Create speech event WITHOUT actor_agent_id in metadata
        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Andi",
            actor_id="#3",
            actor_type=ActorType.AI,
            room_id="room:kitchen",
            room_name="Kitchen",
            content="Hello, Nova!",
            metadata={},  # No actor_agent_id
        )

        # Track queue_event_for_compilation calls
        queued_events = []
        async def mock_queue_event(event, observer_agent_ids, self_action_agent_id):
            for agent_id in observer_agent_ids:
                queued_events.append((agent_id, False))
            if self_action_agent_id:
                queued_events.append((self_action_agent_id, True))

        # Create a spy to track if _agent_id_from_actor is called
        with patch.object(mediator, '_agent_id_from_actor', return_value="andi") as spy:
            with patch.object(mediator, '_agents_from_room_profile', return_value=["andi", "nova"]):
                with patch.object(mediator, '_maybe_assign_turn', return_value=False):
                    with patch.object(mediator, 'queue_event_for_compilation', side_effect=mock_queue_event):
                        # Process the event
                        msg_id = "1704096000000-2"
                        event_data = {
                            b"data": json.dumps(event.to_redis_dict()).encode()
                        }
                        await mediator._process_event(msg_id, event_data)

                        # Assert _agent_id_from_actor WAS called (fallback to room lookup)
                        spy.assert_called_once_with(event.room_id, event.actor_id)

                        # SPEECH events now get self-action routing, so both agents receive
                        # the event: nova gets it normally, andi gets it with is_self_action=True
                        distributed_agents = [agent_id for agent_id, _ in queued_events]

                        # Both agents receive the event (andi as self-action, nova normally)
                        assert "nova" in distributed_agents
                        assert "andi" in distributed_agents

    @pytest.mark.asyncio
    async def test_movement_event_falls_back_to_lookup_when_metadata_missing(self, mediator, mock_redis):
        """Test that movement events fall back to room lookup when metadata is absent."""
        # Create movement event WITHOUT actor_agent_id in metadata
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#3",
            actor_type=ActorType.AI,
            room_id="room:kitchen",
            room_name="Kitchen",
            content="moved from Living Room to Kitchen",
            target="Kitchen",
            metadata={},  # No actor_agent_id
        )

        # Track queue_event_for_compilation calls
        queued_events = []
        async def mock_queue_event(event, observer_agent_ids, self_action_agent_id):
            for agent_id in observer_agent_ids:
                queued_events.append((agent_id, False))
            if self_action_agent_id:
                queued_events.append((self_action_agent_id, True))

        # Create a spy to track if _agent_id_from_actor is called
        with patch.object(mediator, '_agent_id_from_actor', return_value="andi") as spy:
            with patch.object(mediator, '_agents_from_room_profile', return_value=["andi", "nova"]):
                with patch.object(mediator, '_maybe_assign_turn', return_value=False):
                    with patch.object(mediator, 'queue_event_for_compilation', side_effect=mock_queue_event):
                        # Process the event
                        msg_id = "1704096000000-3"
                        event_data = {
                            b"data": json.dumps(event.to_redis_dict()).encode()
                        }
                        await mediator._process_event(msg_id, event_data)

                        # Assert _agent_id_from_actor WAS called (fallback)
                        spy.assert_called_once_with(event.room_id, event.actor_id)

                        # Actor should be filtered from third-person observers
                        distributed_agents = [agent_id for agent_id, _ in queued_events]

                        assert "nova" in distributed_agents


class TestMovementEventFilteringEdgeCases:
    """Test edge cases for movement event filtering."""

    @pytest.mark.asyncio
    async def test_movement_event_with_empty_metadata(self, mediator, mock_redis):
        """Test movement event when metadata has no actor_agent_id key."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#3",
            actor_type=ActorType.AI,
            room_id="room:kitchen",
            room_name="Kitchen",
            content="moved from Living Room to Kitchen",
            target="Kitchen",
            metadata={},  # Empty metadata dict
        )

        # Track queue_event_for_compilation calls
        queued_events = []
        async def mock_queue_event(event, observer_agent_ids, self_action_agent_id):
            for agent_id in observer_agent_ids:
                queued_events.append((agent_id, False))
            if self_action_agent_id:
                queued_events.append((self_action_agent_id, True))

        # Ensure metadata handling doesn't crash
        with patch.object(mediator, '_agent_id_from_actor', return_value="andi"):
            with patch.object(mediator, '_agents_from_room_profile', return_value=["andi", "nova"]):
                with patch.object(mediator, '_maybe_assign_turn', return_value=False):
                    with patch.object(mediator, 'queue_event_for_compilation', side_effect=mock_queue_event):
                        msg_id = "1704096000000-4"
                        event_data = {
                            b"data": json.dumps(event.to_redis_dict()).encode()
                        }
                        await mediator._process_event(msg_id, event_data)

                        # Should not crash and should queue events for compilation
                        assert len(queued_events) >= 1

    @pytest.mark.asyncio
    async def test_movement_event_single_agent_filters_self(self, mediator, mock_redis):
        """Test movement event when only the actor is in the room (should filter to nobody)."""
        event = MUDEvent(
            event_type=EventType.MOVEMENT,
            actor="Andi",
            actor_id="#3",
            actor_type=ActorType.AI,
            room_id="room:bedroom",
            room_name="Bedroom",
            content="moved from Kitchen to Bedroom",
            target="Bedroom",
            metadata={"actor_agent_id": "andi"},
        )

        # Track queue_event_for_compilation calls
        queued_events = []
        async def mock_queue_event(event, observer_agent_ids, self_action_agent_id):
            for agent_id in observer_agent_ids:
                queued_events.append((agent_id, False))
            if self_action_agent_id:
                queued_events.append((self_action_agent_id, True))

        # Mock room with only andi
        with patch.object(mediator, '_agents_from_room_profile', return_value=["andi"]):
            with patch.object(mediator, '_maybe_assign_turn', return_value=False):
                with patch.object(mediator, 'queue_event_for_compilation', side_effect=mock_queue_event):
                    msg_id = "1704096000000-5"
                    event_data = {
                        b"data": json.dumps(event.to_redis_dict()).encode()
                    }
                    await mediator._process_event(msg_id, event_data)

                    # Should not queue any events for compilation (andi filtered out, no other agents)
                    assert len(queued_events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
