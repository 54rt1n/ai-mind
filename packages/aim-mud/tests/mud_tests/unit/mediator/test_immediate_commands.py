# packages/aim-mud/tests/mud_tests/unit/mediator/test_immediate_commands.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for immediate command status assignment in the mediator.

Tests that the mediator correctly assigns EXECUTE status for immediate commands
(FLUSH, CLEAR, NEW) and ASSIGNED status for regular turns.
"""

import pytest
from unittest.mock import AsyncMock

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import RedisKeys, TurnReason, TurnRequestStatus


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
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.set = AsyncMock(return_value=True)
    redis.xtrim = AsyncMock(return_value=0)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    redis.hkeys = AsyncMock(return_value=[])
    redis.hdel = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)  # Lua script success by default
    redis.incr = AsyncMock(return_value=1)  # For sequence ID
    redis.aclose = AsyncMock()
    return redis


class TestMediatorImmediateCommandStatusAssignment:
    """Test that mediator assigns correct status based on reason."""

    @pytest.mark.asyncio
    async def test_flush_reason_gets_execute_status(self, mock_redis, mediator_config):
        """Test that FLUSH reason gets EXECUTE status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Agent is ready for turn assignment
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)  # CAS success

        # Assign turn with FLUSH reason
        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.FLUSH)

        assert result is True
        mock_redis.eval.assert_called_once()

        # Verify Lua script was called with EXECUTE status
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]

        # The status field is in the serialized fields passed to Lua
        # Check that "execute" status is in the arguments
        assert TurnRequestStatus.EXECUTE.value in str(args)

    @pytest.mark.asyncio
    async def test_clear_reason_gets_execute_status(self, mock_redis, mediator_config):
        """Test that CLEAR reason gets EXECUTE status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.CLEAR)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.EXECUTE.value in str(args)

    @pytest.mark.asyncio
    async def test_new_reason_gets_execute_status(self, mock_redis, mediator_config):
        """Test that NEW reason gets EXECUTE status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.NEW)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.EXECUTE.value in str(args)

    @pytest.mark.asyncio
    async def test_events_reason_gets_assigned_status(self, mock_redis, mediator_config):
        """Test that EVENTS reason gets ASSIGNED status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.EVENTS)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.ASSIGNED.value in str(args)

    @pytest.mark.asyncio
    async def test_idle_reason_gets_assigned_status(self, mock_redis, mediator_config):
        """Test that IDLE reason gets ASSIGNED status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.IDLE)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.ASSIGNED.value in str(args)

    @pytest.mark.asyncio
    async def test_dream_reason_gets_assigned_status(self, mock_redis, mediator_config):
        """Test that DREAM reason gets ASSIGNED status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.DREAM)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.ASSIGNED.value in str(args)

    @pytest.mark.asyncio
    async def test_agent_reason_gets_assigned_status(self, mock_redis, mediator_config):
        """Test that AGENT reason gets ASSIGNED status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.AGENT)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.ASSIGNED.value in str(args)

    @pytest.mark.asyncio
    async def test_choose_reason_gets_assigned_status(self, mock_redis, mediator_config):
        """Test that CHOOSE reason gets ASSIGNED status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.CHOOSE)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.ASSIGNED.value in str(args)

    @pytest.mark.asyncio
    async def test_retry_reason_gets_assigned_status(self, mock_redis, mediator_config):
        """Test that RETRY reason gets ASSIGNED status."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert TurnRequestStatus.ASSIGNED.value in str(args)

    @pytest.mark.asyncio
    async def test_reason_string_value_passed_to_lua(self, mock_redis, mediator_config):
        """Test that reason string value is correctly passed to Lua script."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"reason": b"events",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._maybe_assign_turn("andi", reason=TurnReason.FLUSH)

        assert result is True
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]

        # Check that the reason string "flush" is in the arguments
        assert "flush" in str(args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
