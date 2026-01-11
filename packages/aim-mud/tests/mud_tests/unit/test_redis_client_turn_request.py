# packages/aim-mud/tests/mud_tests/unit/test_redis_client_turn_request.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for TurnRequestMixin.

Tests turn request CRUD operations with CAS support.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from aim_mud_types import MUDTurnRequest, TurnRequestStatus, TurnReason, RedisKeys
from aim_mud_types.client import RedisMUDClient


@pytest.fixture
def sample_turn_request():
    """Create a sample turn request for testing."""
    return MUDTurnRequest(
        turn_id="turn123",
        status=TurnRequestStatus.ASSIGNED,
        reason=TurnReason.EVENTS,
        sequence_id=1000,
        assigned_at=datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
        heartbeat_at=datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def client():
    """Create RedisMUDClient with mocked Redis."""
    mock_redis = AsyncMock()
    return RedisMUDClient(mock_redis)


class TestGetTurnRequest:
    """Tests for get_turn_request method."""

    @pytest.mark.asyncio
    async def test_get_turn_request_success(self, client, sample_turn_request):
        """Should fetch and deserialize turn request."""
        agent_id = "andi"
        client.redis.hgetall.return_value = {
            b"turn_id": b"turn123",
            b"status": b"assigned",
            b"reason": b"events",
            b"sequence_id": b"1000",
            b"assigned_at": b"2026-01-10T12:00:00+00:00",
            b"heartbeat_at": b"2026-01-10T12:00:00+00:00",
            b"attempt_count": b"0",
        }

        result = await client.get_turn_request(agent_id)

        assert result is not None
        assert result.turn_id == "turn123"
        assert result.status == TurnRequestStatus.ASSIGNED
        assert result.reason == TurnReason.EVENTS
        assert result.sequence_id == 1000
        client.redis.hgetall.assert_called_once_with(
            RedisKeys.agent_turn_request(agent_id)
        )

    @pytest.mark.asyncio
    async def test_get_turn_request_not_found(self, client):
        """Should return None when turn request doesn't exist."""
        client.redis.hgetall.return_value = {}

        result = await client.get_turn_request("andi")

        assert result is None


class TestCreateTurnRequest:
    """Tests for create_turn_request method."""

    @pytest.mark.asyncio
    async def test_create_turn_request_success(self, client, sample_turn_request):
        """Should create turn request when key doesn't exist."""
        agent_id = "andi"
        client.redis.eval.return_value = 1

        result = await client.create_turn_request(agent_id, sample_turn_request)

        assert result is True
        # Verify Lua script was called with correct key
        call_args = client.redis.eval.call_args
        assert RedisKeys.agent_turn_request(agent_id) in call_args[0]

    @pytest.mark.asyncio
    async def test_create_turn_request_already_exists(self, client, sample_turn_request):
        """Should return False when turn request already exists."""
        agent_id = "andi"
        client.redis.eval.return_value = 0

        result = await client.create_turn_request(agent_id, sample_turn_request)

        assert result is False


class TestUpdateTurnRequest:
    """Tests for update_turn_request method."""

    @pytest.mark.asyncio
    async def test_update_turn_request_success(self, client, sample_turn_request):
        """Should update turn request when CAS matches."""
        agent_id = "andi"
        sample_turn_request.status = TurnRequestStatus.IN_PROGRESS
        client.redis.eval.return_value = 1

        result = await client.update_turn_request(
            agent_id,
            sample_turn_request,
            expected_turn_id="turn123"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_turn_request_cas_failure(self, client, sample_turn_request):
        """Should return False when turn_id doesn't match (CAS failure)."""
        agent_id = "andi"
        sample_turn_request.status = TurnRequestStatus.IN_PROGRESS
        client.redis.eval.return_value = 0

        result = await client.update_turn_request(
            agent_id,
            sample_turn_request,
            expected_turn_id="turn123"
        )

        assert result is False


class TestHeartbeatTurnRequest:
    """Tests for heartbeat_turn_request method."""

    @pytest.mark.asyncio
    async def test_heartbeat_turn_request_success(self, client):
        """Should update heartbeat_at timestamp."""
        agent_id = "andi"
        client.redis.eval.return_value = 1

        result = await client.heartbeat_turn_request(agent_id)

        assert result is True
        # Verify the call was made
        assert client.redis.eval.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
