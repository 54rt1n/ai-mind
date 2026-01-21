# packages/aim-mud/tests/mud_tests/unit/mediator/test_idle_detection.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for idle detection using player activity tracking.

Tests the _is_player_activity_idle() method in the AgentsMixin which checks
if player activity has been idle for a threshold period using the tracked
LAST_PLAYER_ACTIVITY timestamp in Redis.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import RedisKeys


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
        system_idle_seconds=15,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.set = AsyncMock(return_value=True)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def mediator(mock_redis, mediator_config):
    """Create a mediator instance with mocked Redis."""
    return MediatorService(mock_redis, mediator_config)


class TestPlayerActivityIdleDetection:
    """Test player activity idle detection using tracked timestamps."""

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_no_activity_recorded(self, mediator, mock_redis):
        """Test returns True when no player activity is recorded (key missing)."""
        # Arrange: Redis GET returns None (key doesn't exist)
        mock_redis.get.return_value = None

        # Act
        result = await mediator._is_player_activity_idle(300)

        # Assert
        assert result is True
        mock_redis.get.assert_called_once_with(RedisKeys.LAST_PLAYER_ACTIVITY)

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_recent_activity(self, mediator, mock_redis):
        """Test returns False with recent player activity (within threshold)."""
        # Arrange: Set recent activity (30 seconds ago)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        recent_ms = now_ms - 30000  # 30 seconds ago
        mock_redis.get.return_value = str(recent_ms).encode('utf-8')

        # Act: Check if idle for 300 seconds
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should NOT be idle (only 30s passed)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_old_activity(self, mediator, mock_redis):
        """Test returns True with old player activity (past threshold)."""
        # Arrange: Set old activity (400 seconds ago)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        old_ms = now_ms - 400000  # 400 seconds ago
        mock_redis.get.return_value = str(old_ms).encode('utf-8')

        # Act: Check if idle for 300 seconds
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should be idle (400s > 300s threshold)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_handles_string(self, mediator, mock_redis):
        """Test handles string return value from Redis (not bytes)."""
        # Arrange: Redis returns string instead of bytes
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        old_ms = now_ms - 400000
        mock_redis.get.return_value = str(old_ms)  # String, not bytes

        # Act
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should handle string correctly
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_handles_bytes(self, mediator, mock_redis):
        """Test handles bytes return value from Redis."""
        # Arrange: Redis returns bytes
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        recent_ms = now_ms - 30000
        mock_redis.get.return_value = str(recent_ms).encode('utf-8')

        # Act
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should handle bytes correctly
        assert result is False

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_handles_exception(self, mediator, mock_redis):
        """Test returns True on exception (treats as idle)."""
        # Arrange: Redis GET raises exception
        mock_redis.get.side_effect = Exception("Redis connection error")

        # Act
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should treat error as idle
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_handles_invalid_data(self, mediator, mock_redis):
        """Test returns True when Redis contains invalid data."""
        # Arrange: Redis returns non-numeric data
        mock_redis.get.return_value = b"invalid_timestamp"

        # Act
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should treat invalid data as idle (exception caught)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_exact_threshold(self, mediator, mock_redis):
        """Test boundary condition: activity exactly at threshold."""
        # Arrange: Activity exactly 300 seconds ago
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        threshold_ms = now_ms - 300000  # Exactly 300 seconds
        mock_redis.get.return_value = str(threshold_ms).encode('utf-8')

        # Act: Check if idle for 300 seconds
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should be idle (>= threshold)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_one_ms_before_threshold(self, mediator, mock_redis):
        """Test boundary condition: activity one millisecond before threshold."""
        # Arrange: Activity 299.999 seconds ago (just under threshold)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        threshold_ms = now_ms - 299999  # One millisecond before 300s
        mock_redis.get.return_value = str(threshold_ms).encode('utf-8')

        # Act: Check if idle for 300 seconds
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should NOT be idle (just under threshold)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_one_ms_after_threshold(self, mediator, mock_redis):
        """Test boundary condition: activity one millisecond after threshold."""
        # Arrange: Activity 300.001 seconds ago (just over threshold)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        threshold_ms = now_ms - 300001  # One millisecond after 300s
        mock_redis.get.return_value = str(threshold_ms).encode('utf-8')

        # Act: Check if idle for 300 seconds
        result = await mediator._is_player_activity_idle(300)

        # Assert: Should be idle (over threshold)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_zero_threshold(self, mediator, mock_redis):
        """Test edge case: zero idle threshold."""
        # Arrange: Activity 1 millisecond ago
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        recent_ms = now_ms - 1
        mock_redis.get.return_value = str(recent_ms).encode('utf-8')

        # Act: Check if idle for 0 seconds
        result = await mediator._is_player_activity_idle(0)

        # Assert: Should be idle (any delay > 0ms threshold)
        assert result is True

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_large_threshold(self, mediator, mock_redis):
        """Test with very large idle threshold."""
        # Arrange: Activity 1 hour ago
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        old_ms = now_ms - 3600000  # 1 hour ago
        mock_redis.get.return_value = str(old_ms).encode('utf-8')

        # Act: Check if idle for 2 hours (7200 seconds)
        result = await mediator._is_player_activity_idle(7200)

        # Assert: Should NOT be idle (1h < 2h threshold)
        assert result is False

    @pytest.mark.asyncio
    async def test_is_player_activity_idle_uses_correct_redis_key(self, mediator, mock_redis):
        """Test that the method queries the correct Redis key."""
        # Arrange
        mock_redis.get.return_value = None

        # Act
        await mediator._is_player_activity_idle(300)

        # Assert: Verify it uses LAST_PLAYER_ACTIVITY key
        mock_redis.get.assert_called_once_with(RedisKeys.LAST_PLAYER_ACTIVITY)
        assert RedisKeys.LAST_PLAYER_ACTIVITY == "mud:last_player_activity"
