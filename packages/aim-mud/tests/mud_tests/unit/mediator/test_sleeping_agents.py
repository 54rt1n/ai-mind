# packages/aim-mud/tests/mud_tests/unit/mediator/test_sleeping_agents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for sleeping agent turn assignment.

Sleeping agents should receive IDLE turns (to potentially wake up)
but NOT other turn types (EVENTS, DREAM, etc.).

Paused agents receive NO turns at all.
"""

import pytest
from unittest.mock import AsyncMock

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import RedisKeys, TurnReason
from aim_mud_types.helper import _utc_now


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
    redis.incr = AsyncMock(side_effect=lambda key: 1)  # Sequence counter
    return redis


def _make_ready_turn_request():
    """Create a ready turn request for testing."""
    return {
        b"status": b"ready",
        b"turn_id": b"turn_123",
        b"reason": b"events",
        b"heartbeat_at": _utc_now().isoformat().encode(),
        b"assigned_at": _utc_now().isoformat().encode(),
        b"sequence_id": b"1",
        b"attempt_count": b"0",
    }


class TestSleepingAgentTurnAssignment:
    """Test that sleeping agents can receive IDLE turns but not other turn types."""

    @pytest.mark.asyncio
    async def test_sleeping_agent_receives_idle_turn(self, mock_redis, mediator_config):
        """Sleeping agents should be able to receive IDLE turns."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Agent is sleeping (is_sleeping = "true")
        async def hget_side_effect(key, field=None):
            if field == "is_sleeping":
                return b"true"
            return None
        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        # Agent has a ready turn_request
        mock_redis.hgetall = AsyncMock(return_value=_make_ready_turn_request())

        # Assign IDLE turn
        result = await mediator._maybe_assign_turn("test_agent", reason=TurnReason.IDLE)

        # Should succeed - sleeping agents CAN receive IDLE turns
        assert result is True

    @pytest.mark.skip(reason="Sleeping check moved to event routing level; _maybe_assign_turn now allows all turn types for sleeping agents")
    @pytest.mark.asyncio
    async def test_sleeping_agent_blocked_from_events_turn(self, mock_redis, mediator_config):
        """Sleeping agents should NOT receive EVENTS turns."""
        pass

    @pytest.mark.skip(reason="Sleeping check moved to event routing level; _maybe_assign_turn now allows all turn types for sleeping agents")
    @pytest.mark.asyncio
    async def test_sleeping_agent_blocked_from_dream_turn(self, mock_redis, mediator_config):
        """Sleeping agents should NOT receive DREAM turns."""
        pass

    @pytest.mark.skip(reason="Sleeping check moved to event routing level; _maybe_assign_turn now allows all turn types for sleeping agents")
    @pytest.mark.asyncio
    async def test_sleeping_agent_blocked_from_agent_turn(self, mock_redis, mediator_config):
        """Sleeping agents should NOT receive AGENT turns."""
        pass

    @pytest.mark.asyncio
    async def test_awake_agent_receives_all_turn_types(self, mock_redis, mediator_config):
        """Non-sleeping agents should receive all turn types."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Agent is NOT sleeping (is_sleeping = None or "false")
        mock_redis.hget = AsyncMock(return_value=None)
        mock_redis.hgetall = AsyncMock(return_value=_make_ready_turn_request())

        # All turn types should work
        for reason in [TurnReason.EVENTS, TurnReason.IDLE, TurnReason.AGENT]:
            # Reset mocks for each iteration
            mock_redis.eval.reset_mock()
            mock_redis.eval.return_value = 1

            result = await mediator._maybe_assign_turn("test_agent", reason=reason)
            assert result is True, f"Awake agent should receive {reason.value} turn"


class TestPausedAgentTurnAssignment:
    """Test that paused agents receive NO turns at all."""

    @pytest.mark.asyncio
    async def test_paused_agent_blocked_from_all_turns(self, mock_redis, mediator_config):
        """Paused agents should NOT receive any turns, including IDLE."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Agent is paused
        mock_redis.get = AsyncMock(return_value=b"1")  # mud:agent:X:paused = "1"
        mock_redis.hget = AsyncMock(return_value=None)  # Not sleeping
        mock_redis.hgetall = AsyncMock(return_value=_make_ready_turn_request())

        # No turn type should work
        for reason in [TurnReason.EVENTS, TurnReason.IDLE, TurnReason.AGENT, TurnReason.DREAM]:
            result = await mediator._maybe_assign_turn("test_agent", reason=reason)
            assert result is False, f"Paused agent should NOT receive {reason.value} turn"


class TestSleepingVsPausedDistinction:
    """Test the distinction between sleeping and paused states."""

    @pytest.mark.asyncio
    async def test_sleeping_allows_idle_paused_does_not(self, mock_redis, mediator_config):
        """Verify sleeping allows IDLE turns while paused blocks all turns."""
        mediator = MediatorService(mock_redis, mediator_config)
        mock_redis.hgetall = AsyncMock(return_value=_make_ready_turn_request())

        # Case 1: Sleeping agent (is_sleeping=true, paused=false)
        mock_redis.get = AsyncMock(return_value=None)  # Not paused
        async def sleeping_hget(key, field=None):
            if field == "is_sleeping":
                return b"true"
            return None
        mock_redis.hget = AsyncMock(side_effect=sleeping_hget)

        result = await mediator._maybe_assign_turn("test_agent", reason=TurnReason.IDLE)
        assert result is True, "Sleeping agent should receive IDLE turn"

        # Case 2: Paused agent (paused=true, is_sleeping=false)
        mock_redis.get = AsyncMock(return_value=b"1")  # Paused
        mock_redis.hget = AsyncMock(return_value=None)  # Not sleeping
        mock_redis.eval.reset_mock()

        result = await mediator._maybe_assign_turn("test_agent", reason=TurnReason.IDLE)
        assert result is False, "Paused agent should NOT receive IDLE turn"
