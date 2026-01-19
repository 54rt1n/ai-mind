# tests/unit/mediator/test_dream_commands.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for dream command handlers in the mediator.

The mediator only handles the @dreamer control command. All dream execution
and manual dream commands (analyze, journal, etc.) are handled by the worker.
"""

import pytest
from unittest.mock import AsyncMock, patch

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from andimud_mediator.patterns import DREAMER_PATTERN
from aim_mud_types import MUDEvent, EventType, ActorType


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
    redis.eval = AsyncMock(return_value=1)
    redis.aclose = AsyncMock()
    return redis


def create_test_event(content: str, event_type: EventType = EventType.SYSTEM) -> MUDEvent:
    """Create a test MUDEvent with required fields."""
    return MUDEvent(
        event_type=event_type,
        actor="system",
        actor_type=ActorType.SYSTEM,
        room_id="test-room",
        content=content
    )


class TestDreamerPattern:
    """Test the DREAMER_PATTERN regex."""

    def test_pattern_on(self):
        """Test pattern matches with 'on'."""
        match = DREAMER_PATTERN.match("@dreamer andi on")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "on"

    def test_pattern_off(self):
        """Test pattern matches with 'off'."""
        match = DREAMER_PATTERN.match("@dreamer andi off")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "off"

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = DREAMER_PATTERN.match("@DREAMER ANDI ON")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "ON"


class TestHandleDreamerCommand:
    """Test _handle_dreamer_command in mediator."""

    @pytest.mark.asyncio
    async def test_enable_dreamer_unregistered_agent(self, mock_redis, mediator_config):
        """Test that enabling fails for unregistered agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Try to enable for unregistered agent
        success = await mediator._handle_dreamer_command("val", True)

        assert success is False

    @pytest.mark.asyncio
    async def test_enable_dreamer_success(self, mock_redis, mediator_config):
        """Test successful dreamer enable."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock RedisMUDClient methods
        with patch('aim_mud_types.client.RedisMUDClient.get_dreamer_state',
                   return_value=None):
            with patch('aim_mud_types.client.RedisMUDClient.update_dreamer_state_fields') as mock_update:
                success = await mediator._handle_dreamer_command("andi", True)

                assert success is True
                # Should set enabled=True and default thresholds
                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["enabled"] is True
                assert "idle_threshold_seconds" in call_kwargs
                assert "token_threshold" in call_kwargs

    @pytest.mark.asyncio
    async def test_disable_dreamer_success(self, mock_redis, mediator_config):
        """Test successful dreamer disable."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock RedisMUDClient methods
        with patch('aim_mud_types.client.RedisMUDClient.get_dreamer_state',
                   return_value={"enabled": True}):
            with patch('aim_mud_types.client.RedisMUDClient.update_dreamer_state_fields') as mock_update:
                success = await mediator._handle_dreamer_command("andi", False)

                assert success is True
                # Should only set enabled=False (no threshold updates)
                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["enabled"] is False
                assert "idle_threshold_seconds" not in call_kwargs

    @pytest.mark.asyncio
    async def test_enable_dreamer_with_existing_state(self, mock_redis, mediator_config):
        """Test enabling when state already exists (don't reset thresholds)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock existing dreamer state
        existing_state = {
            "enabled": False,
            "idle_threshold_seconds": 7200,
            "token_threshold": 15000,
        }

        with patch('aim_mud_types.client.RedisMUDClient.get_dreamer_state',
                   return_value=existing_state):
            with patch('aim_mud_types.client.RedisMUDClient.update_dreamer_state_fields') as mock_update:
                success = await mediator._handle_dreamer_command("andi", True)

                assert success is True
                # Should only set enabled=True (preserve existing thresholds)
                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["enabled"] is True
                assert "idle_threshold_seconds" not in call_kwargs


class TestTryHandleControlCommand:
    """Test _try_handle_control_command in mediator."""

    @pytest.mark.asyncio
    async def test_handles_dreamer_on(self, mock_redis, mediator_config):
        """Test that @dreamer on command is handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        event = create_test_event("@dreamer andi on")

        with patch.object(mediator, '_handle_dreamer_command', return_value=True) as mock_handle:
            result = await mediator._try_handle_control_command(event)

            assert result is True
            mock_handle.assert_called_once_with("andi", True)

    @pytest.mark.asyncio
    async def test_handles_dreamer_off(self, mock_redis, mediator_config):
        """Test that @dreamer off command is handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        event = create_test_event("@dreamer andi off")

        with patch.object(mediator, '_handle_dreamer_command', return_value=True) as mock_handle:
            result = await mediator._try_handle_control_command(event)

            assert result is True
            mock_handle.assert_called_once_with("andi", False)

    @pytest.mark.asyncio
    async def test_ignores_non_system_events(self, mock_redis, mediator_config):
        """Test that non-SYSTEM events are ignored."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = create_test_event("@dreamer andi on", event_type=EventType.SPEECH)

        result = await mediator._try_handle_control_command(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_ignores_non_command_content(self, mock_redis, mediator_config):
        """Test that non-command content is ignored."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = create_test_event("Just a regular message")

        result = await mediator._try_handle_control_command(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_ignores_other_commands(self, mock_redis, mediator_config):
        """Test that other dream commands are ignored (handled by worker)."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Test that analyze command is NOT handled by mediator
        event = create_test_event("@analyze andi = conv_123")

        result = await mediator._try_handle_control_command(event)

        # Should return False - not a mediator command
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
