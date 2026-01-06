# packages/aim-mud/tests/unit/mediator/test_abort.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for mediator abort handling."""

import pytest
from unittest.mock import AsyncMock, Mock
import json

from andimud_mediator.mediator import MediatorService, MediatorConfig
from aim_mud_types import RedisKeys


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.expire = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)  # CAS success
    return redis


class TestMediatorMaybeAssignTurn:
    """Test _maybe_assign_turn abort handling."""

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_returns_false_when_abort_requested(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn returns False when status is abort_requested."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock turn_request with abort_requested status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"turn_id": b"abort_turn_123"
        })

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        assert result is False
        # Should not call hset to assign new turn
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_returns_false_when_assigned(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn returns False when status is assigned."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock turn_request with assigned status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"assigned",
            b"turn_id": b"current_turn_456"
        })

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        assert result is False
        # Should not call hset to assign new turn
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_returns_false_when_in_progress(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn returns False when status is in_progress."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock turn_request with in_progress status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"active_turn_789"
        })

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        assert result is False
        # Should not call hset to assign new turn
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_allows_assignment_when_ready(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn allows assignment when status is ready."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock turn_request with ready status (worker completed turn and returned to ready)
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"previous_turn"
        })

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        assert result is True
        # Should call eval (Lua script) to assign new turn with CAS
        mock_redis.eval.assert_called_once()
        call_args = mock_redis.eval.call_args[0]
        # Check that script was called with "assigned" status
        assert "assigned" in str(call_args)

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_allows_assignment_when_done(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn allows assignment when status is done (edge case).

        In the new architecture, workers transition done->ready in the finally block.
        However, there's a brief moment where status might be "done" before the transition.
        The mediator allows assignment in this case (status is not actively blocked).
        """
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock turn_request with done status (worker hasn't transitioned to ready yet)
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"done",
            b"turn_id": b"completed_turn"
        })

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        # Technically allowed (not in blocked list), though workers should transition to "ready"
        assert result is True

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_returns_false_when_no_turn(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn returns False when no turn_request exists (worker offline)."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock empty turn_request (worker never announced presence)
        mock_redis.hgetall = AsyncMock(return_value={})

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        # Should return False - no turn_request means worker offline
        assert result is False

    @pytest.mark.asyncio
    async def test_maybe_assign_turn_handles_string_keys(
        self, mock_redis, mediator_config
    ):
        """Test _maybe_assign_turn handles string keys (decode_responses=True)."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Mock turn_request with string keys
        mock_redis.hgetall = AsyncMock(return_value={
            "status": "abort_requested",
            "turn_id": "abort_turn_str"
        })

        result = await mediator._maybe_assign_turn("test_agent", reason="events")

        assert result is False


class TestMediatorAbortEventProcessing:
    """Test mediator event processing with abort status."""

    @pytest.mark.asyncio
    async def test_process_event_skips_turn_assignment_for_abort_requested(
        self, mock_redis, mediator_config
    ):
        """Test mediator does not assign turn when agent has abort_requested status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("test_agent")

        # Mock agent in room
        mock_redis.hget = AsyncMock(
            return_value=json.dumps([
                {
                    "entity_id": "#3",
                    "name": "TestAgent",
                    "entity_type": "ai",
                    "agent_id": "test_agent"
                }
            ]).encode("utf-8")
        )

        # Mock turn_request with abort_requested status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"turn_id": b"abort_turn"
        })

        sample_event = {
            "type": "speech",
            "actor": "Prax",
            "actor_type": "player",
            "room_id": "#123",
            "room_name": "Test Room",
            "content": "Hello!",
            "timestamp": "2026-01-01T12:00:00+00:00",
        }

        data = {b"data": json.dumps(sample_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should still deliver event to agent stream
        assert mock_redis.xadd.call_count >= 1

        # Check if turn_request was assigned
        hset_calls = [call for call in mock_redis.hset.call_args_list
                      if RedisKeys.agent_turn_request("test_agent") in str(call)]

        # Should NOT assign new turn (only mark event as processed)
        # The only hset should be for events_processed
        for call in mock_redis.hset.call_args_list:
            if RedisKeys.agent_turn_request("test_agent") in str(call[0]):
                # If there's a turn_request hset, it should not be a new assignment
                pytest.fail("Turn request should not be assigned when status is abort_requested")

    @pytest.mark.asyncio
    async def test_process_event_assigns_turn_after_worker_ready(
        self, mock_redis, mediator_config
    ):
        """Test mediator can assign turn when worker is ready (completed abort and returned to ready)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("test_agent")

        # Mock agent in room
        mock_redis.hget = AsyncMock(
            return_value=json.dumps([
                {
                    "entity_id": "#3",
                    "name": "TestAgent",
                    "entity_type": "ai",
                    "agent_id": "test_agent"
                }
            ]).encode("utf-8")
        )

        # Mock turn_request with ready status (worker completed abort and returned to ready)
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"old_abort_turn"
        })

        sample_event = {
            "type": "speech",
            "actor": "Prax",
            "actor_type": "player",
            "room_id": "#123",
            "room_name": "Test Room",
            "content": "Are you listening?",
            "timestamp": "2026-01-01T12:01:00+00:00",
        }

        data = {b"data": json.dumps(sample_event).encode()}
        await mediator._process_event("1704096000001-0", data)

        # Should assign new turn since status is ready (using Lua script/eval)
        eval_calls = mock_redis.eval.call_args_list
        # Check that eval was called (for CAS turn assignment)
        assert len(eval_calls) >= 1

        # Verify the script was called with "assigned" status
        call_args = eval_calls[0][0]
        assert "assigned" in str(call_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
