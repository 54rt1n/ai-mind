# tests/unit/mud/test_mediator.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD mediator service.

The mediator routes events from mud:events to per-agent streams.
Action execution is handled by Evennia's ActionConsumer, not the mediator.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

from aim.app.mud.mediator import (
    MediatorService,
    MediatorConfig,
    run_mediator,
    parse_args,
)
from aim_mud_types import (
    MUDEvent,
    RedisKeys,
)


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
    redis.hexists = AsyncMock(return_value=False)  # Events not already processed
    redis.hkeys = AsyncMock(return_value=[])
    redis.hdel = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def sample_speech_event():
    """Sample speech event data as it would appear from Redis."""
    return {
        "type": "speech",
        "actor": "Prax",
        "actor_type": "player",
        "room_id": "#123",
        "room_name": "The Garden",
        "content": "Hello, Andi!",
        "timestamp": "2026-01-01T12:00:00+00:00",
    }


@pytest.fixture
def sample_movement_event():
    """Sample movement event for AI actor."""
    return {
        "type": "movement",
        "actor": "andi",
        "actor_type": "ai",
        "room_id": "#124",
        "room_name": "The Library",
        "content": "Andi arrives from the garden.",
        "timestamp": "2026-01-01T12:01:00+00:00",
    }


class TestMediatorConfig:
    """Test MediatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MediatorConfig()

        assert config.redis_url == "redis://localhost:6379"
        assert config.event_stream == RedisKeys.MUD_EVENTS
        assert config.event_poll_timeout == 5.0
        assert config.evennia_api_url == "http://localhost:4001"
        assert config.pause_key == RedisKeys.MEDIATOR_PAUSE

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MediatorConfig(
            redis_url="redis://custom:6380",
            event_poll_timeout=2.0,
        )

        assert config.redis_url == "redis://custom:6380"
        assert config.event_poll_timeout == 2.0


class TestMediatorServiceInit:
    """Test MediatorService initialization."""

    def test_init_with_default_config(self, mock_redis):
        """Test initialization with default config."""
        mediator = MediatorService(mock_redis)

        assert mediator.redis == mock_redis
        assert mediator.config is not None
        assert mediator.running is False
        assert mediator.registered_agents == set()
        assert mediator.last_event_id == "0"

    def test_init_with_custom_config(self, mock_redis, mediator_config):
        """Test initialization with custom config."""
        mediator = MediatorService(mock_redis, mediator_config)

        assert mediator.config == mediator_config
        assert mediator.config.event_poll_timeout == 0.1


class TestMediatorAgentRegistration:
    """Test agent registration and tracking."""

    def test_register_agent_without_room(self, mock_redis, mediator_config):
        """Test registering an agent without initial room."""
        mediator = MediatorService(mock_redis, mediator_config)

        mediator.register_agent("andi")

        assert "andi" in mediator.registered_agents

    def test_register_multiple_agents(self, mock_redis, mediator_config):
        """Test registering multiple agents."""
        mediator = MediatorService(mock_redis, mediator_config)

        mediator.register_agent("andi")
        mediator.register_agent("roommate")

        assert len(mediator.registered_agents) == 2

    def test_unregister_agent(self, mock_redis, mediator_config):
        """Test unregistering an agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mediator.unregister_agent("andi")

        assert "andi" not in mediator.registered_agents


class TestMediatorPause:
    """Test mediator pause functionality."""

    @pytest.mark.asyncio
    async def test_is_paused_returns_false_when_not_set(
        self, mock_redis, mediator_config
    ):
        """Test _is_paused returns False when key not set."""
        mock_redis.get = AsyncMock(return_value=None)
        mediator = MediatorService(mock_redis, mediator_config)

        result = await mediator._is_paused()

        assert result is False
        mock_redis.get.assert_called_once_with(mediator_config.pause_key)

    @pytest.mark.asyncio
    async def test_is_paused_returns_false_when_zero(
        self, mock_redis, mediator_config
    ):
        """Test _is_paused returns False when key is 0."""
        mock_redis.get = AsyncMock(return_value=b"0")
        mediator = MediatorService(mock_redis, mediator_config)

        result = await mediator._is_paused()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_paused_returns_true_when_set(
        self, mock_redis, mediator_config
    ):
        """Test _is_paused returns True when key is 1."""
        mock_redis.get = AsyncMock(return_value=b"1")
        mediator = MediatorService(mock_redis, mediator_config)

        result = await mediator._is_paused()

        assert result is True


class TestMediatorEnrichEvent:
    """Test event enrichment."""

    @pytest.mark.asyncio
    async def test_enrich_event_adds_room_state(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test that enrich_event does not add room state."""
        mediator = MediatorService(mock_redis, mediator_config)
        event = MUDEvent.from_dict(sample_speech_event)
        event.event_id = "1704096000000-0"

        enriched = await mediator.enrich_event(event)

        assert enriched["enriched"] is False
        assert "room_state" not in enriched
        assert "world_state" not in enriched
        assert enriched["id"] == "1704096000000-0"

    @pytest.mark.asyncio
    async def test_enrich_event_preserves_original_fields(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test that enrichment preserves original event fields."""
        mediator = MediatorService(mock_redis, mediator_config)
        event = MUDEvent.from_dict(sample_speech_event)

        enriched = await mediator.enrich_event(event)

        assert enriched["type"] == "speech"
        assert enriched["actor"] == "Prax"
        assert enriched["room_id"] == "#123"
        assert enriched["content"] == "Hello, Andi!"


class TestMediatorEventRouting:
    """Test event routing logic."""

    @pytest.mark.asyncio
    async def test_process_event_distributes_to_agents_in_room(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test events are distributed to agents in the same room."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.register_agent("other")
        mock_redis.hget = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "entity_id": "#3",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "andi",
                    }
                ]
            ).encode("utf-8")
        )

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should have called xadd once for andi, not for other
        assert mock_redis.xadd.call_count == 1
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == RedisKeys.agent_events("andi")

    @pytest.mark.asyncio
    async def test_process_event_no_agents_to_notify(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test no distribution when no agents are in room."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hget = AsyncMock(return_value=None)

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should not have called xadd
        assert mock_redis.xadd.call_count == 0

    @pytest.mark.asyncio
    async def test_process_event_handles_missing_data(
        self, mock_redis, mediator_config
    ):
        """Test graceful handling of missing data field."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Should not raise, just log warning
        await mediator._process_event("1704096000000-0", {b"other": b"value"})

        assert mock_redis.xadd.call_count == 0

    @pytest.mark.asyncio
    async def test_process_event_handles_string_keys(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test handling of string keys (decode_responses=True mode)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hget = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "entity_id": "#3",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "andi",
                    }
                ]
            )
        )

        # String keys instead of bytes
        data = {"data": json.dumps(sample_speech_event)}
        await mediator._process_event("1704096000000-0", data)

    @pytest.mark.asyncio
    async def test_process_event_assigns_turn_request(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test that a turn_request is written when delivering an event."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hget = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "entity_id": "#3",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "andi",
                    }
                ]
            ).encode("utf-8")
        )

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should be called twice: once for turn_request, once for events_processed
        assert mock_redis.hset.call_count == 2

        # First call should be for turn_request
        first_call = mock_redis.hset.call_args_list[0]
        assert first_call[0][0] == RedisKeys.agent_turn_request("andi")
        mapping = first_call.kwargs.get("mapping") or first_call[0][1]
        assert mapping["status"] == "assigned"

        # Second call should be for events_processed
        second_call = mock_redis.hset.call_args_list[1]
        assert second_call[0][0] == RedisKeys.EVENTS_PROCESSED
        assert second_call[0][1] == "1704096000000-0"

        mock_redis.expire.assert_called_once_with(
            RedisKeys.agent_turn_request("andi"),
            mediator_config.turn_request_ttl_seconds,
        )

    @pytest.mark.asyncio
    async def test_process_event_skips_turn_request_when_active(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test that mediator does not overwrite active turn_request."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hget = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "entity_id": "#3",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "andi",
                    }
                ]
            ).encode("utf-8")
        )
        mock_redis.hgetall = AsyncMock(
            return_value={b"status": b"in_progress", b"turn_id": b"abc"}
        )

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should be called once for events_processed (turn_request skipped)
        assert mock_redis.hset.call_count == 1
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == RedisKeys.EVENTS_PROCESSED
        assert call_args[0][1] == "1704096000000-0"

        assert mock_redis.xadd.call_count == 1


class TestMediatorEventRouter:
    """Test the event router loop."""

    @pytest.mark.asyncio
    async def test_run_event_router_processes_events(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test event router processes stream events."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.running = True  # Must set running before the loop
        call_count = [0]
        mock_redis.hget = AsyncMock(
            return_value=json.dumps(
                [
                    {
                        "entity_id": "#3",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "andi",
                    }
                ]
            ).encode("utf-8")
        )

        # Track processed events for trim operation
        processed_events = []

        async def hset_side_effect(key, *args, **kwargs):
            # Track when events are marked as processed
            if key == RedisKeys.EVENTS_PROCESSED and args:
                processed_events.append(args[0])
            return 1

        async def hkeys_side_effect(key):
            # Return processed event IDs for trim operation
            if key == RedisKeys.EVENTS_PROCESSED:
                return processed_events
            return []

        mock_redis.hset = AsyncMock(side_effect=hset_side_effect)
        mock_redis.hkeys = AsyncMock(side_effect=hkeys_side_effect)

        async def xread_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns an event
                return [
                    (
                        b"mud:events",
                        [
                            (
                                b"1704096000000-0",
                                {b"data": json.dumps(sample_speech_event).encode()},
                            )
                        ],
                    )
                ]
            # Subsequent calls stop the loop
            mediator.running = False
            return []

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)

        await mediator.run_event_router()

        # Should have processed the event
        assert mock_redis.xadd.call_count >= 1
        assert mediator.last_event_id == "1704096000000-0"

    @pytest.mark.asyncio
    async def test_run_event_router_respects_pause(
        self, mock_redis, mediator_config
    ):
        """Test event router pauses when flag is set."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True  # Must set running before the loop
        pause_checks = [0]

        async def get_side_effect(key):
            pause_checks[0] += 1
            if pause_checks[0] >= 3:
                mediator.running = False
                return None
            return b"1"  # Paused

        mock_redis.get = AsyncMock(side_effect=get_side_effect)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await mediator.run_event_router()

            # Should have slept while paused
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_event_router_handles_errors(
        self, mock_redis, mediator_config
    ):
        """Test event router continues after errors."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True  # Must set running before the loop
        call_count = [0]

        async def xread_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")
            if call_count[0] >= 3:
                mediator.running = False
            return []

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)

        await mediator.run_event_router()

        # Should have continued after error
        assert call_count[0] >= 2


class TestMediatorStartStop:
    """Test mediator start and stop."""

    @pytest.mark.asyncio
    async def test_start_runs_event_router(self, mock_redis, mediator_config):
        """Test start runs the event router task."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        call_count = [0]

        async def xread_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 3:
                mediator.running = False
            return []

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)

        await mediator.start()

        # Event router should have run
        assert call_count[0] >= 1

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, mock_redis, mediator_config):
        """Test stop sets running flag to False."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True

        await mediator.stop()

        assert mediator.running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_event_task(self, mock_redis, mediator_config):
        """Test stop cancels the event router task."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True

        # Create mock task
        mock_event_task = AsyncMock()
        mock_event_task.done = Mock(return_value=False)
        mock_event_task.cancel = Mock()

        mediator._event_task = mock_event_task

        await mediator.stop()

        mock_event_task.cancel.assert_called_once()


class TestRunMediator:
    """Test run_mediator entry point."""

    @pytest.mark.asyncio
    async def test_run_mediator_registers_agents(self, mediator_config):
        """Test that run_mediator registers agents."""
        with patch("redis.asyncio.from_url") as mock_from_url, patch(
            "aim.app.mud.mediator.MediatorService"
        ) as mock_service_class:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            mock_service = AsyncMock()
            mock_service.start = AsyncMock()
            mock_service.register_agent = Mock()
            mock_service_class.return_value = mock_service

            await run_mediator(mediator_config, ["andi", "roommate"])

            # Verify agents were registered
            assert mock_service.register_agent.call_count == 2
            mock_service.register_agent.assert_any_call("andi")
            mock_service.register_agent.assert_any_call("roommate")

            # Verify cleanup
            mock_redis.aclose.assert_called_once()


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_parse_args_defaults(self):
        """Test parsing with defaults."""
        with patch("sys.argv", ["mediator"]):
            args = parse_args()

            assert args.agents == []
            assert args.redis_url == "redis://localhost:6379"
            assert args.log_level == "INFO"
            # These default to None; MediatorConfig provides actual defaults
            assert args.event_timeout is None

    def test_parse_args_with_agents(self):
        """Test parsing with agent list."""
        with patch("sys.argv", ["mediator", "--agents", "andi", "roommate"]):
            args = parse_args()

            assert args.agents == ["andi", "roommate"]

    def test_parse_args_all_options(self):
        """Test parsing all options."""
        with patch(
            "sys.argv",
            [
                "mediator",
                "--agents",
                "andi",
                "--redis-url",
                "redis://custom:6380",
                "--log-level",
                "DEBUG",
                "--event-timeout",
                "2.0",
            ],
        ):
            args = parse_args()

            assert args.agents == ["andi"]
            assert args.redis_url == "redis://custom:6380"
            assert args.log_level == "DEBUG"
            assert args.event_timeout == 2.0


class TestSignalHandlers:
    """Test signal handler setup."""

    def test_setup_signal_handlers(self, mock_redis, mediator_config):
        """Test that signal handlers are registered."""
        import signal as signal_module

        mediator = MediatorService(mock_redis, mediator_config)

        with patch.object(signal_module, "signal") as mock_signal:
            mediator.setup_signal_handlers()

            assert mock_signal.call_count == 2
            signals_registered = [call[0][0] for call in mock_signal.call_args_list]
            assert signal_module.SIGINT in signals_registered
            assert signal_module.SIGTERM in signals_registered


class TestMediatorGracefulShutdown:
    """Test graceful shutdown scenarios."""

    @pytest.mark.asyncio
    async def test_event_router_handles_cancellation(
        self, mock_redis, mediator_config
    ):
        """Test event router handles CancelledError gracefully."""
        async def xread_side_effect(*args, **kwargs):
            raise asyncio.CancelledError()

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True

        # Should not raise
        await mediator.run_event_router()

        assert True  # Reached end without exception


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
