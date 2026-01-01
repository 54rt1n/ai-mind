# tests/unit/mud/test_mediator.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD mediator service."""

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
    MUDAction,
    EventType,
    ActorType,
    RedisKeys,
)


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
        action_poll_timeout=0.05,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
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


@pytest.fixture
def sample_action_data():
    """Sample action data from an agent."""
    return {
        "agent_id": "andi",
        "command": "say Hello, Papa!",
        "tool": "say",
        "args": {"message": "Hello, Papa!"},
        "priority": 5,
    }


class TestMediatorConfig:
    """Test MediatorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MediatorConfig()

        assert config.redis_url == "redis://localhost:6379"
        assert config.event_stream == RedisKeys.MUD_EVENTS
        assert config.action_stream == RedisKeys.MUD_ACTIONS
        assert config.event_poll_timeout == 5.0
        assert config.action_poll_timeout == 5.0
        assert config.evennia_api_url == "http://localhost:4001"
        assert config.pause_key == RedisKeys.MEDIATOR_PAUSE

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MediatorConfig(
            redis_url="redis://custom:6380",
            event_poll_timeout=2.0,
            action_poll_timeout=0.5,
        )

        assert config.redis_url == "redis://custom:6380"
        assert config.event_poll_timeout == 2.0
        assert config.action_poll_timeout == 0.5


class TestMediatorServiceInit:
    """Test MediatorService initialization."""

    def test_init_with_default_config(self, mock_redis):
        """Test initialization with default config."""
        mediator = MediatorService(mock_redis)

        assert mediator.redis == mock_redis
        assert mediator.config is not None
        assert mediator.running is False
        assert mediator.agent_rooms == {}
        assert mediator.registered_agents == set()
        assert mediator.last_event_id == "0"
        assert mediator.last_action_id == "0"

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
        assert "andi" not in mediator.agent_rooms

    def test_register_agent_with_room(self, mock_redis, mediator_config):
        """Test registering an agent with initial room."""
        mediator = MediatorService(mock_redis, mediator_config)

        mediator.register_agent("andi", initial_room="#123")

        assert "andi" in mediator.registered_agents
        assert mediator.agent_rooms["andi"] == "#123"

    def test_register_multiple_agents(self, mock_redis, mediator_config):
        """Test registering multiple agents."""
        mediator = MediatorService(mock_redis, mediator_config)

        mediator.register_agent("andi", "#123")
        mediator.register_agent("roommate", "#124")

        assert len(mediator.registered_agents) == 2
        assert mediator.agent_rooms["andi"] == "#123"
        assert mediator.agent_rooms["roommate"] == "#124"

    def test_unregister_agent(self, mock_redis, mediator_config):
        """Test unregistering an agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#123")

        mediator.unregister_agent("andi")

        assert "andi" not in mediator.registered_agents
        assert "andi" not in mediator.agent_rooms

    def test_update_agent_room(self, mock_redis, mediator_config):
        """Test updating agent's room."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#123")

        mediator.update_agent_room("andi", "#124")

        assert mediator.agent_rooms["andi"] == "#124"

    def test_get_agents_in_room(self, mock_redis, mediator_config):
        """Test getting agents in a specific room."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#123")
        mediator.register_agent("roommate", "#123")
        mediator.register_agent("other", "#124")

        agents = mediator.get_agents_in_room("#123")

        assert len(agents) == 2
        assert "andi" in agents
        assert "roommate" in agents
        assert "other" not in agents

    def test_get_agents_in_empty_room(self, mock_redis, mediator_config):
        """Test getting agents from an empty room."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#123")

        agents = mediator.get_agents_in_room("#999")

        assert agents == []


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
        """Test that enrich_event adds room state placeholder."""
        mediator = MediatorService(mock_redis, mediator_config)
        event = MUDEvent.from_dict(sample_speech_event)
        event.event_id = "1704096000000-0"

        enriched = await mediator.enrich_event(event)

        assert enriched["enriched"] is True
        assert "room_state" in enriched
        assert enriched["room_state"]["room_id"] == "#123"
        assert enriched["room_state"]["name"] == "The Garden"
        assert "entities_present" in enriched
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
        mediator.register_agent("andi", "#123")
        mediator.register_agent("other", "#999")

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should have called xadd once for andi, not for other
        assert mock_redis.xadd.call_count == 1
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == RedisKeys.agent_events("andi")

    @pytest.mark.asyncio
    async def test_process_event_includes_agents_without_room(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test agents without room assignment see all events."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")  # No room set
        mediator.register_agent("placed", "#999")  # Different room

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should have called xadd once for andi (no room), not for placed
        assert mock_redis.xadd.call_count == 1
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == RedisKeys.agent_events("andi")

    @pytest.mark.asyncio
    async def test_process_event_no_agents_to_notify(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test no distribution when no agents are in room."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#999")  # Different room

        data = {b"data": json.dumps(sample_speech_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Should not have called xadd
        assert mock_redis.xadd.call_count == 0

    @pytest.mark.asyncio
    async def test_process_event_updates_agent_location_on_movement(
        self, mock_redis, mediator_config, sample_movement_event
    ):
        """Test that movement events update agent location."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#123")

        data = {b"data": json.dumps(sample_movement_event).encode()}
        await mediator._process_event("1704096000000-0", data)

        # Agent room should be updated to the new room
        assert mediator.agent_rooms["andi"] == "#124"

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
        mediator.register_agent("andi", "#123")

        # String keys instead of bytes
        data = {"data": json.dumps(sample_speech_event)}
        await mediator._process_event("1704096000000-0", data)

        assert mock_redis.xadd.call_count == 1


class TestMediatorActionExecution:
    """Test action queueing and execution."""

    @pytest.mark.asyncio
    async def test_queue_action_adds_to_agent_queue(
        self, mock_redis, mediator_config, sample_action_data
    ):
        """Test actions are queued per agent."""
        mediator = MediatorService(mock_redis, mediator_config)

        data = {b"data": json.dumps(sample_action_data).encode()}
        await mediator._queue_action("1704096000000-0", data)

        assert len(mediator.action_queues["andi"]) == 1
        assert mediator.action_queues["andi"][0]["command"] == "say Hello, Papa!"

    @pytest.mark.asyncio
    async def test_queue_action_handles_missing_agent_id(
        self, mock_redis, mediator_config
    ):
        """Test graceful handling of missing agent_id."""
        mediator = MediatorService(mock_redis, mediator_config)

        data = {b"data": json.dumps({"command": "say test"}).encode()}
        await mediator._queue_action("1704096000000-0", data)

        # Should not have added to any queue
        assert len(mediator.action_queues) == 0

    @pytest.mark.asyncio
    async def test_execute_round_robin_single_agent(
        self, mock_redis, mediator_config, sample_action_data
    ):
        """Test round-robin execution with single agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.action_queues["andi"].append(sample_action_data)

        await mediator._execute_round_robin()

        # Queue should be empty and removed
        assert "andi" not in mediator.action_queues

    @pytest.mark.asyncio
    async def test_execute_round_robin_multiple_agents(
        self, mock_redis, mediator_config
    ):
        """Test round-robin alternates between agents."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.action_queues["andi"].append({"agent_id": "andi", "command": "a1"})
        mediator.action_queues["andi"].append({"agent_id": "andi", "command": "a2"})
        mediator.action_queues["roommate"].append(
            {"agent_id": "roommate", "command": "r1"}
        )

        # First execution - should execute from first agent
        await mediator._execute_round_robin()
        # Second execution - should execute from second agent
        await mediator._execute_round_robin()
        # Third execution - should be back to first agent
        await mediator._execute_round_robin()

        # Both agents should have had actions executed
        # andi: 2 actions, roommate: 1 action, total 3 executions
        # After 3 round-robins: andi should have 0 left, roommate queue removed
        assert "roommate" not in mediator.action_queues
        assert "andi" not in mediator.action_queues

    @pytest.mark.asyncio
    async def test_execute_round_robin_empty_queues(
        self, mock_redis, mediator_config
    ):
        """Test round-robin with no pending actions."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Should not raise
        await mediator._execute_round_robin()

        assert mediator.last_agent_index == 0

    @pytest.mark.asyncio
    async def test_execute_action_logs_command(
        self, mock_redis, mediator_config, sample_action_data
    ):
        """Test action execution logs the command."""
        mediator = MediatorService(mock_redis, mediator_config)

        # Should not raise - just logs
        await mediator._execute_action(sample_action_data)


class TestMediatorEventRouter:
    """Test the event router loop."""

    @pytest.mark.asyncio
    async def test_run_event_router_processes_events(
        self, mock_redis, mediator_config, sample_speech_event
    ):
        """Test event router processes stream events."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi", "#123")
        mediator.running = True  # Must set running before the loop
        call_count = [0]

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


class TestMediatorActionExecutor:
    """Test the action executor loop."""

    @pytest.mark.asyncio
    async def test_run_action_executor_processes_actions(
        self, mock_redis, mediator_config, sample_action_data
    ):
        """Test action executor processes stream actions."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True  # Must set running before the loop
        call_count = [0]

        async def xread_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns an action
                return [
                    (
                        b"mud:actions",
                        [
                            (
                                b"1704096000000-0",
                                {b"data": json.dumps(sample_action_data).encode()},
                            )
                        ],
                    )
                ]
            if call_count[0] >= 3:
                mediator.running = False
            return []

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)

        await mediator.run_action_executor()

        # Should have processed the action
        assert mediator.last_action_id == "1704096000000-0"
        # Action should have been executed and removed from queue
        assert "andi" not in mediator.action_queues

    @pytest.mark.asyncio
    async def test_run_action_executor_respects_pause(
        self, mock_redis, mediator_config
    ):
        """Test action executor pauses when flag is set."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True  # Must set running before the loop
        pause_checks = [0]

        async def get_side_effect(key):
            pause_checks[0] += 1
            if pause_checks[0] >= 3:
                mediator.running = False
                return None
            return b"1"

        mock_redis.get = AsyncMock(side_effect=get_side_effect)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await mediator.run_action_executor()

            assert mock_sleep.call_count >= 1


class TestMediatorStartStop:
    """Test mediator start and stop."""

    @pytest.mark.asyncio
    async def test_start_runs_both_tasks(self, mock_redis, mediator_config):
        """Test start runs both router and executor tasks."""
        # Create mediator first so closure can reference it
        mediator = MediatorService(mock_redis, mediator_config)
        call_count = [0]

        async def xread_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 5:
                mediator.running = False
            return []

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)

        await mediator.start()

        # Both tasks should have run
        assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, mock_redis, mediator_config):
        """Test stop sets running flag to False."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True

        await mediator.stop()

        assert mediator.running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks(self, mock_redis, mediator_config):
        """Test stop cancels running tasks."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True

        # Create mock tasks
        mock_event_task = AsyncMock()
        mock_event_task.done = Mock(return_value=False)
        mock_event_task.cancel = Mock()

        mock_action_task = AsyncMock()
        mock_action_task.done = Mock(return_value=False)
        mock_action_task.cancel = Mock()

        mediator._event_task = mock_event_task
        mediator._action_task = mock_action_task

        await mediator.stop()

        mock_event_task.cancel.assert_called_once()
        mock_action_task.cancel.assert_called_once()


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
            assert args.action_timeout is None

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
                "--action-timeout",
                "0.5",
            ],
        ):
            args = parse_args()

            assert args.agents == ["andi"]
            assert args.redis_url == "redis://custom:6380"
            assert args.log_level == "DEBUG"
            assert args.event_timeout == 2.0
            assert args.action_timeout == 0.5


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

    @pytest.mark.asyncio
    async def test_action_executor_handles_cancellation(
        self, mock_redis, mediator_config
    ):
        """Test action executor handles CancelledError gracefully."""
        async def xread_side_effect(*args, **kwargs):
            raise asyncio.CancelledError()

        mock_redis.xread = AsyncMock(side_effect=xread_side_effect)
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.running = True

        # Should not raise
        await mediator.run_action_executor()

        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
