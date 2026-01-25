# packages/aim-mud/tests/mud_tests/unit/test_idle_thought_mixins.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for IdleMixin and ThoughtMixin."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from aim_mud_types import RedisKeys
from aim_mud_types.client import RedisMUDClient, SyncRedisMUDClient
from aim_mud_types.models.coordination import ThoughtState


@pytest.fixture
def client():
    """Create RedisMUDClient with mocked Redis."""
    mock_redis = AsyncMock()
    return RedisMUDClient(mock_redis)


@pytest.fixture
def sync_client():
    """Create SyncRedisMUDClient with mocked Redis."""
    mock_redis = MagicMock()
    return SyncRedisMUDClient(mock_redis)


class TestIdleMixin:
    """Tests for IdleMixin."""

    @pytest.mark.asyncio
    async def test_is_idle_active_true(self, client):
        """Should return True when idle active flag is set to true."""
        client.redis.get.return_value = b"true"

        result = await client.is_idle_active("andi")

        assert result is True
        client.redis.get.assert_called_once_with(
            RedisKeys.agent_idle_active("andi")
        )

    @pytest.mark.asyncio
    async def test_is_idle_active_false(self, client):
        """Should return False when idle active flag is set to false."""
        client.redis.get.return_value = b"false"

        result = await client.is_idle_active("andi")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_idle_active_missing(self, client):
        """Should return False when idle active flag is not set."""
        client.redis.get.return_value = None

        result = await client.is_idle_active("andi")

        assert result is False

    @pytest.mark.asyncio
    async def test_is_idle_active_various_truthy_values(self, client):
        """Should accept various truthy string values."""
        truthy_values = [b"1", b"true", b"yes", b"on", b"TRUE", b"Yes", b"ON"]
        for value in truthy_values:
            client.redis.get.return_value = value
            result = await client.is_idle_active("andi")
            assert result is True, f"Expected True for {value!r}"

    @pytest.mark.asyncio
    async def test_is_idle_active_various_falsy_values(self, client):
        """Should return False for non-truthy values."""
        falsy_values = [b"0", b"false", b"no", b"off", b"", b"random"]
        for value in falsy_values:
            client.redis.get.return_value = value
            result = await client.is_idle_active("andi")
            assert result is False, f"Expected False for {value!r}"

    @pytest.mark.asyncio
    async def test_set_idle_active_true(self, client):
        """Should set idle active flag to true."""
        await client.set_idle_active("andi", True)

        client.redis.set.assert_called_once_with(
            RedisKeys.agent_idle_active("andi"),
            "true"
        )

    @pytest.mark.asyncio
    async def test_set_idle_active_false(self, client):
        """Should set idle active flag to false."""
        await client.set_idle_active("andi", False)

        client.redis.set.assert_called_once_with(
            RedisKeys.agent_idle_active("andi"),
            "false"
        )


class TestThoughtMixin:
    """Tests for ThoughtMixin."""

    @pytest.mark.asyncio
    async def test_has_active_thought_true(self, client):
        """Should return True when thought exists."""
        client.redis.exists.return_value = 1

        result = await client.has_active_thought("andi")

        assert result is True
        client.redis.exists.assert_called_once_with(
            RedisKeys.agent_thought("andi")
        )

    @pytest.mark.asyncio
    async def test_has_active_thought_false(self, client):
        """Should return False when thought doesn't exist."""
        client.redis.exists.return_value = 0

        result = await client.has_active_thought("andi")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_thought_state_success(self, client):
        """Should return ThoughtState from hash data."""
        # Now uses hgetall returning a hash
        client.redis.hgetall.return_value = {
            b"agent_id": b"andi",
            b"content": b"Remember to check on the garden",
            b"source": b"dreamer",
            b"created_at": b"1704067200",
            b"actions_since_generation": b"3",
        }

        result = await client.get_thought_state("andi")

        assert result is not None
        assert result.content == "Remember to check on the garden"
        assert result.source == "dreamer"
        assert result.actions_since_generation == 3

    @pytest.mark.asyncio
    async def test_get_thought_state_not_found(self, client):
        """Should return None when no thought exists."""
        client.redis.hgetall.return_value = {}

        result = await client.get_thought_state("andi")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_thought_legacy_success(self, client):
        """Should return thought data as dict via legacy method."""
        client.redis.hgetall.return_value = {
            b"agent_id": b"andi",
            b"content": b"Test thought",
            b"source": b"manual",
            b"created_at": b"1704067200",
            b"actions_since_generation": b"0",
        }

        result = await client.get_thought("andi")

        assert result is not None
        assert result["content"] == "Test thought"
        assert result["source"] == "manual"

    @pytest.mark.asyncio
    async def test_get_thought_legacy_not_found(self, client):
        """Should return None when no thought exists (legacy)."""
        client.redis.hgetall.return_value = {}

        result = await client.get_thought("andi")

        assert result is None

    @pytest.mark.asyncio
    async def test_should_generate_thought_no_thought(self, client):
        """Should return True when no thought exists."""
        client.redis.hgetall.return_value = {}

        result = await client.should_generate_thought("andi")

        assert result is True

    @pytest.mark.asyncio
    async def test_should_generate_thought_action_threshold(self, client):
        """Should return True when action threshold met."""
        import time
        client.redis.hgetall.return_value = {
            b"agent_id": b"andi",
            b"content": b"Test thought",
            b"source": b"reasoning",
            b"created_at": str(int(time.time())).encode(),  # Fresh
            b"actions_since_generation": b"5",  # Threshold met
        }

        result = await client.should_generate_thought("andi")

        assert result is True

    @pytest.mark.asyncio
    async def test_should_generate_thought_time_threshold(self, client):
        """Should return True when time threshold met."""
        import time
        old_time = int(time.time()) - 400  # 6+ minutes ago
        client.redis.hgetall.return_value = {
            b"agent_id": b"andi",
            b"content": b"Test thought",
            b"source": b"reasoning",
            b"created_at": str(old_time).encode(),
            b"actions_since_generation": b"0",
        }

        result = await client.should_generate_thought("andi")

        assert result is True

    @pytest.mark.asyncio
    async def test_should_generate_thought_throttle_active(self, client):
        """Should return False when neither threshold met."""
        import time
        client.redis.hgetall.return_value = {
            b"agent_id": b"andi",
            b"content": b"Test thought",
            b"source": b"reasoning",
            b"created_at": str(int(time.time())).encode(),  # Fresh
            b"actions_since_generation": b"2",  # Below threshold
        }

        result = await client.should_generate_thought("andi")

        assert result is False

    @pytest.mark.asyncio
    async def test_increment_thought_action_counter(self, client):
        """Should increment action counter atomically."""
        client.redis.hincrby.return_value = 3

        result = await client.increment_thought_action_counter("andi")

        assert result == 3
        client.redis.hincrby.assert_called_once_with(
            RedisKeys.agent_thought("andi"),
            "actions_since_generation",
            1
        )


class TestSyncThoughtMixin:
    """Tests for SyncThoughtMixin."""

    def test_get_thought_state_success(self, sync_client):
        """Should return ThoughtState from hash data."""
        sync_client.redis.hgetall.return_value = {
            b"agent_id": b"andi",
            b"content": b"Remember to check on the garden",
            b"source": b"dreamer",
            b"created_at": b"1704067200",
            b"actions_since_generation": b"3",
        }

        result = sync_client.get_thought_state("andi")

        assert result is not None
        assert result.content == "Remember to check on the garden"
        assert result.source == "dreamer"
        assert result.actions_since_generation == 3

    def test_get_thought_state_not_found(self, sync_client):
        """Should return None when no thought exists."""
        sync_client.redis.hgetall.return_value = {}

        result = sync_client.get_thought_state("andi")

        assert result is None

    def test_save_thought_state_success(self, sync_client):
        """Should save thought state via _create_hash and set TTL."""
        sync_client.redis.eval.return_value = 1  # Lua script returns success

        thought = ThoughtState(
            agent_id="andi",
            content="Focus on emotional connections",
            source="manual",
            actions_since_generation=0,
        )

        result = sync_client.save_thought_state(thought, ttl_seconds=7200)

        assert result is True
        # Verify Lua script was called (via _create_hash)
        sync_client.redis.eval.assert_called_once()
        # Verify TTL was set
        sync_client.redis.expire.assert_called_once_with(
            RedisKeys.agent_thought("andi"),
            7200
        )

    def test_save_thought_state_no_ttl(self, sync_client):
        """Should save thought state without TTL when ttl_seconds=0."""
        sync_client.redis.eval.return_value = 1

        thought = ThoughtState(
            agent_id="andi",
            content="Test content",
            source="manual",
        )

        result = sync_client.save_thought_state(thought, ttl_seconds=0)

        assert result is True
        sync_client.redis.eval.assert_called_once()
        # TTL should NOT be set when ttl_seconds=0
        sync_client.redis.expire.assert_not_called()

    def test_delete_thought_state_success(self, sync_client):
        """Should delete thought state and return True when key existed."""
        sync_client.redis.delete.return_value = 1  # Key was deleted

        result = sync_client.delete_thought_state("andi")

        assert result is True
        sync_client.redis.delete.assert_called_once_with(
            RedisKeys.agent_thought("andi")
        )

    def test_delete_thought_state_not_found(self, sync_client):
        """Should return False when thought didn't exist."""
        sync_client.redis.delete.return_value = 0  # No key deleted

        result = sync_client.delete_thought_state("andi")

        assert result is False

    def test_has_active_thought_true(self, sync_client):
        """Should return True when thought exists."""
        sync_client.redis.exists.return_value = 1

        result = sync_client.has_active_thought("andi")

        assert result is True
        sync_client.redis.exists.assert_called_once_with(
            RedisKeys.agent_thought("andi")
        )

    def test_has_active_thought_false(self, sync_client):
        """Should return False when thought doesn't exist."""
        sync_client.redis.exists.return_value = 0

        result = sync_client.has_active_thought("andi")

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
