# packages/aim-mud/tests/mud_tests/unit/test_idle_thought_mixins.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for IdleMixin and ThoughtMixin."""

import json
import pytest
from unittest.mock import AsyncMock

from aim_mud_types import RedisKeys
from aim_mud_types.client import RedisMUDClient


@pytest.fixture
def client():
    """Create RedisMUDClient with mocked Redis."""
    mock_redis = AsyncMock()
    return RedisMUDClient(mock_redis)


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
    async def test_get_thought_success(self, client):
        """Should return thought data as dict."""
        thought_data = {
            "content": "Remember to check on the garden",
            "source": "dreamer",
            "timestamp": 1704067200
        }
        client.redis.get.return_value = json.dumps(thought_data).encode()

        result = await client.get_thought("andi")

        assert result == thought_data
        client.redis.get.assert_called_once_with(
            RedisKeys.agent_thought("andi")
        )

    @pytest.mark.asyncio
    async def test_get_thought_not_found(self, client):
        """Should return None when no thought exists."""
        client.redis.get.return_value = None

        result = await client.get_thought("andi")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_thought_invalid_json(self, client):
        """Should return None for invalid JSON."""
        client.redis.get.return_value = b"not valid json"

        result = await client.get_thought("andi")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_thought_string_value(self, client):
        """Should handle string value (not bytes)."""
        thought_data = {
            "content": "Test thought",
            "source": "manual",
            "timestamp": 1704067200
        }
        # Return string instead of bytes
        client.redis.get.return_value = json.dumps(thought_data)

        result = await client.get_thought("andi")

        assert result == thought_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
