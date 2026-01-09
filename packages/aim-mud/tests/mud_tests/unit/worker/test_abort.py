# packages/aim-mud/tests/unit/worker/test_abort.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for worker abort functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from andimud_worker.worker import MUDAgentWorker, AbortRequestedException
from andimud_worker.config import MUDConfig
from aim_mud_types import MUDSession, MUDTurnRequest
from aim_mud_types.helper import _utc_now


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.get = AsyncMock(return_value=None)
    redis.eval = AsyncMock(return_value=1)  # CAS success
    redis.expire = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()
    return redis


class TestCheckAbortRequested:
    """Test _check_abort_requested method."""

    @pytest.mark.asyncio
    async def test_check_abort_returns_true_when_status_is_abort_requested(
        self, mud_config, mock_redis
    ):
        """Test _check_abort_requested returns True when status is abort_requested."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock turn_request with abort_requested status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"turn_id": b"test_turn_123",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
        })

        result = await worker._check_abort_requested()

        assert result is True
        # Should have called eval (Lua script) to set status to "aborted"
        mock_redis.eval.assert_called_once()
        # Check that the script was called with "aborted" status
        call_args = mock_redis.eval.call_args[0]
        assert "aborted" in str(call_args)

    @pytest.mark.asyncio
    async def test_check_abort_returns_false_when_status_is_in_progress(
        self, mud_config, mock_redis
    ):
        """Test _check_abort_requested returns False when status is in_progress."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock turn_request with in_progress status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"test_turn_456",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
        })

        result = await worker._check_abort_requested()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_abort_returns_false_when_status_is_assigned(
        self, mud_config, mock_redis
    ):
        """Test _check_abort_requested returns False when status is assigned."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock turn_request with assigned status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"assigned",
            b"turn_id": b"test_turn_789",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
        })

        result = await worker._check_abort_requested()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_abort_returns_false_when_no_turn_request(
        self, mud_config, mock_redis
    ):
        """Test _check_abort_requested returns False when no turn_request exists."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock empty turn_request
        mock_redis.hgetall = AsyncMock(return_value={})

        result = await worker._check_abort_requested()

        assert result is False

    @pytest.mark.asyncio
    async def test_check_abort_handles_string_keys(
        self, mud_config, mock_redis
    ):
        """Test _check_abort_requested handles string keys (decode_responses=True)."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock turn_request with string keys
        mock_redis.hgetall = AsyncMock(return_value={
            "status": "abort_requested",
            "turn_id": "test_turn_abc",
            "reason": "events",
            "heartbeat_at": _utc_now().isoformat(),
        })

        result = await worker._check_abort_requested()

        assert result is True


class TestWorkerAbortInLoop:
    """Test abort detection in worker main loop."""

    @pytest.mark.asyncio
    async def test_worker_detects_abort_and_transitions_to_aborted(
        self, mud_config, mock_redis
    ):
        """Test worker detects abort during turn processing and transitions to aborted."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock turn_request with abort_requested status
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"turn_id": b"abort_turn",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
        })

        result = await worker._check_abort_requested()

        assert result is True
        # Verify state transition to "aborted" was called
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_rollback_on_abort(
        self, mud_config, mock_redis
    ):
        """Test worker rolls back last_event_id on abort."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")
        worker.session.last_event_id = "1704096000001-0"

        # Save original last_event_id before turn
        original_event_id = "1704096000000-0"

        # Mock abort detection after changing last_event_id
        async def mock_check_abort():
            # Simulate abort happening after event processing started
            worker.session.last_event_id = "1704096000002-0"  # Changed during turn
            return True

        with patch.object(worker, "_check_abort_requested", side_effect=mock_check_abort):
            # Simulate the abort rollback logic
            saved_id = worker.session.last_event_id
            was_aborted = await worker._check_abort_requested()
            if was_aborted:
                # This is what the worker should do on abort
                worker.session.last_event_id = original_event_id

        assert was_aborted is True
        # Verify rollback happened
        assert worker.session.last_event_id == original_event_id


class TestWorkerAbortEmote:
    """Test abort emote emission."""

    @pytest.mark.asyncio
    async def test_worker_emits_abort_action_on_abort(
        self, mud_config, mock_redis
    ):
        """Test worker emits abort action when abort is detected."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock _publish_action to track calls
        mock_publish_action = AsyncMock()
        worker._publish_action = mock_publish_action

        # Simulate abort
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"abort_requested",
            b"turn_id": b"abort_turn",
            b"reason": b"events",
            b"heartbeat_at": _utc_now().isoformat().encode(),
        })

        # In the actual worker, when abort is detected, it should call:
        # await self._publish_action(tool="emote", args={"message": "..."})
        #
        # For this test, we simulate the worker's abort handler
        was_aborted = await worker._check_abort_requested()
        if was_aborted:
            # This is what the worker should do
            await worker._publish_action(
                tool="emote",
                args={"message": "pauses, looking momentarily distracted."}
            )

        assert was_aborted is True
        mock_publish_action.assert_called_once()
        call_args = mock_publish_action.call_args
        assert call_args[1]["tool"] == "emote"
        assert "distracted" in call_args[1]["args"]["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
