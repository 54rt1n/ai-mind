# tests/unit/mud/test_worker.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD agent worker."""

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from andimud_worker.worker import MUDAgentWorker, run_worker, parse_args
from andimud_worker.config import MUDConfig
from andimud_worker.session import (
    MUDSession,
    MUDEvent,
    MUDTurn,
    RoomState,
    EntityState,
    EventType,
    ActorType,
)
from aim_mud_types import RedisKeys
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from andimud_worker.strategy import MUDDecisionStrategy


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
        spontaneous_check_interval=60.0,
        spontaneous_action_interval=300.0,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "0"})
    redis.xrange = AsyncMock(return_value=[])
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def sample_event_data():
    """Sample event data as it would appear from Redis."""
    return {
        "type": "speech",
        "actor": "Prax",
        "actor_type": "player",
        "room_id": "#123",
        "room_name": "The Garden",
        "content": "Hello, Andi!",
        "timestamp": "2026-01-01T12:00:00+00:00",
        "room_state": {
            "room_id": "#123",
            "name": "The Garden",
            "description": "A serene garden.",
            "exits": {"north": "#124"},
        },
        "entities_present": [
            {"entity_id": "prax_1", "name": "Prax", "entity_type": "player"},
            {"entity_id": "andi_1", "name": "Andi", "entity_type": "ai", "is_self": True},
        ],
    }


class TestMUDAgentWorkerInit:
    """Test MUDAgentWorker initialization."""

    def test_init_sets_attributes(self, mud_config, mock_redis):
        """Test that __init__ properly sets all attributes."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        assert worker.config == mud_config
        assert worker.redis == mock_redis
        assert worker.running is False
        assert worker.cvm is None
        assert worker.roster is None
        assert worker.persona is None
        assert worker.session is None
        assert worker.chat_config is None
        assert worker._decision_strategy is None
        assert worker._response_strategy is None

    def test_init_with_chat_config(self, mud_config, mock_redis):
        """Test that __init__ stores provided chat_config."""
        mock_chat_config = MagicMock()
        mock_chat_config.default_model = "test-model"

        worker = MUDAgentWorker(
            config=mud_config,
            redis_client=mock_redis,
            chat_config=mock_chat_config,
        )

        assert worker.chat_config == mock_chat_config


class TestMUDAgentWorkerInitLLMProvider:
    """Test MUDAgentWorker _init_llm_provider method."""

    def test_init_llm_provider_raises_when_no_model(self, mud_config, mock_redis):
        """Test _init_llm_provider raises ValueError when default_model is None."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Set up chat_config with no default_model
        mock_chat_config = MagicMock()
        mock_chat_config.default_model = None
        worker.chat_config = mock_chat_config

        with pytest.raises(ValueError, match="No model specified"):
            worker._init_llm_provider()

    def test_init_llm_provider_raises_when_model_not_found(self, mud_config, mock_redis):
        """Test _init_llm_provider raises ValueError when model is not available."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Set up chat_config with an invalid model
        mock_chat_config = MagicMock()
        mock_chat_config.default_model = "nonexistent-model-xyz"
        worker.chat_config = mock_chat_config

        with patch("andimud_worker.worker.LanguageModelV2.index_models") as mock_index:
            mock_index.return_value = {"valid-model": MagicMock()}

            with pytest.raises(ValueError, match="nonexistent-model-xyz not available"):
                worker._init_llm_provider()


class TestMUDAgentWorkerCallLLM:
    """Test MUDAgentWorker _call_llm method with retry logic."""

    @pytest.mark.asyncio
    async def test_call_llm_returns_response(self, mud_config, mock_redis):
        """Test _call_llm returns concatenated chunks on success."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = MagicMock()

        # Mock LLM provider to return chunks
        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(return_value=iter(["Hello, ", "world!"]))
        worker._llm_provider = mock_llm

        result = await worker._call_llm([{"role": "user", "content": "Hi"}])

        assert result == "Hello, world!"
        mock_llm.stream_turns.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_retries_on_retryable_error(self, mud_config, mock_redis):
        """Test _call_llm retries on retryable errors with backoff."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = MagicMock()

        # Mock LLM to fail once then succeed
        call_count = [0]

        def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Connection reset by peer")
            return iter(["Success!"])

        mock_llm = MagicMock()
        mock_llm.stream_turns = mock_stream
        worker._llm_provider = mock_llm

        with patch("andimud_worker.worker.is_retryable_error", return_value=True):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await worker._call_llm(
                    [{"role": "user", "content": "Hi"}],
                    max_retries=3
                )

        assert result == "Success!"
        assert call_count[0] == 2
        mock_sleep.assert_called_once()
        # First retry delay should be 30s
        assert mock_sleep.call_args[0][0] == 30

    @pytest.mark.asyncio
    async def test_call_llm_raises_non_retryable_error(self, mud_config, mock_redis):
        """Test _call_llm raises immediately on non-retryable errors."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = MagicMock()

        # Mock LLM to raise a non-retryable error
        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=ValueError("Invalid input"))
        worker._llm_provider = mock_llm

        with patch("andimud_worker.worker.is_retryable_error", return_value=False):
            with pytest.raises(ValueError, match="Invalid input"):
                await worker._call_llm([{"role": "user", "content": "Hi"}])

        # Should only try once for non-retryable errors
        assert mock_llm.stream_turns.call_count == 1

    @pytest.mark.asyncio
    async def test_call_llm_raises_after_max_retries(self, mud_config, mock_redis):
        """Test _call_llm raises after exhausting all retries."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = MagicMock()

        # Mock LLM to always fail with retryable error
        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=ConnectionError("Network down"))
        worker._llm_provider = mock_llm

        with patch("andimud_worker.worker.is_retryable_error", return_value=True):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ConnectionError, match="Network down"):
                    await worker._call_llm(
                        [{"role": "user", "content": "Hi"}],
                        max_retries=3
                    )

        # Should try max_retries times
        assert mock_llm.stream_turns.call_count == 3

    @pytest.mark.asyncio
    async def test_call_llm_exponential_backoff(self, mud_config, mock_redis):
        """Test _call_llm uses exponential backoff for retries."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.chat_config = MagicMock()

        # Mock LLM to fail twice then succeed
        call_count = [0]

        def mock_stream(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError("Timeout")
            return iter(["Finally!"])

        mock_llm = MagicMock()
        mock_llm.stream_turns = mock_stream
        worker._llm_provider = mock_llm

        with patch("andimud_worker.worker.is_retryable_error", return_value=True):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                result = await worker._call_llm(
                    [{"role": "user", "content": "Hi"}],
                    max_retries=3
                )

        assert result == "Finally!"
        assert call_count[0] == 3
        assert mock_sleep.call_count == 2

        # Check exponential backoff: 30s, 60s (capped at 120s)
        delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert delays == [30, 60]


class TestMUDAgentWorkerStop:
    """Test MUDAgentWorker stop method."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_to_false(self, mud_config, mock_redis):
        """Test that stop() sets running flag to False."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.running = True

        await worker.stop()

        assert worker.running is False


class TestMUDAgentWorkerPause:
    """Test MUDAgentWorker pause functionality."""

    @pytest.mark.asyncio
    async def test_is_paused_returns_false_when_not_set(self, mud_config, mock_redis):
        """Test _is_paused returns False when key not set."""
        mock_redis.get = AsyncMock(return_value=None)
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        result = await worker._is_paused()

        assert result is False
        mock_redis.get.assert_called_once_with(mud_config.pause_key)

    @pytest.mark.asyncio
    async def test_is_paused_returns_false_when_zero(self, mud_config, mock_redis):
        """Test _is_paused returns False when key is 0."""
        mock_redis.get = AsyncMock(return_value=b"0")
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        result = await worker._is_paused()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_paused_returns_true_when_set(self, mud_config, mock_redis):
        """Test _is_paused returns True when key is 1."""
        mock_redis.get = AsyncMock(return_value=b"1")
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        result = await worker._is_paused()

        assert result is True


class TestMUDAgentWorkerSpontaneous:
    """Test MUDAgentWorker spontaneous action logic."""

    def test_should_act_spontaneously_no_session(self, mud_config, mock_redis):
        """Test spontaneous action returns False when no session."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = None

        result = worker._should_act_spontaneously()

        assert result is False

    def test_should_act_spontaneously_no_last_event(self, mud_config, mock_redis):
        """Test spontaneous action returns False when no events seen."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        result = worker._should_act_spontaneously()

        assert result is False

    def test_should_act_spontaneously_recent_action(self, mud_config, mock_redis):
        """Test spontaneous action returns False when acted recently."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        now = datetime.now(timezone.utc)
        worker.session = MUDSession(
            agent_id="test",
            persona_id="test",
            last_action_time=now,
            last_event_time=now,
        )

        result = worker._should_act_spontaneously()

        assert result is False

    def test_should_act_spontaneously_long_silence(self, mud_config, mock_redis):
        """Test spontaneous action returns True after long silence."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        # Set last event time to 10 minutes ago (> 300s default)
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        worker.session = MUDSession(
            agent_id="test",
            persona_id="test",
            last_action_time=old_time,
            last_event_time=old_time,
        )

        result = worker._should_act_spontaneously()

        assert result is True


class TestMUDAgentWorkerPermissions:
    """Test persona permission filtering."""

    def test_is_superuser_persona_false(self, mud_config, mock_redis):
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.persona = MagicMock()
        worker.persona.attributes = {"mud_role": "player"}

        assert worker._is_superuser_persona() is False

    def test_is_superuser_persona_true(self, mud_config, mock_redis):
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.persona = MagicMock()
        worker.persona.attributes = {"mud_role": "superuser"}

        assert worker._is_superuser_persona() is True


class TestMUDAgentWorkerDrainEvents:
    """Test MUDAgentWorker drain_events method."""

    @pytest.mark.asyncio
    async def test_drain_events_no_messages(self, mud_config, mock_redis):
        """Test drain_events returns empty list when no messages."""
        mock_redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "0"})
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert events == []
        mock_redis.xinfo_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_drain_events_parses_messages(
        self, mud_config, mock_redis, sample_event_data
    ):
        """Test drain_events correctly parses stream messages."""
        # Simulate Redis stream response format
        message_id = b"1704096000000-0"
        message_data = {b"data": json.dumps(sample_event_data).encode("utf-8")}
        mock_redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "1704096000000-0"})
        mock_redis.xrange = AsyncMock(return_value=[(message_id, message_data)])

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert len(events) == 1
        event = events[0]
        assert event.event_type == EventType.SPEECH
        assert event.actor == "Prax"
        assert event.room_id == "#123"
        assert event.content == "Hello, Andi!"
        assert worker.session.last_event_id == "1704096000000-0"

    @pytest.mark.asyncio
    async def test_drain_events_multiple_messages(
        self, mud_config, mock_redis, sample_event_data
    ):
        """Test drain_events handles multiple messages."""
        event1 = sample_event_data.copy()
        event1["content"] = "First message"

        event2 = sample_event_data.copy()
        event2["content"] = "Second message"
        event2["type"] = "emote"

        mock_redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "1704096000001-0"})
        mock_redis.xrange = AsyncMock(
            return_value=[
                (b"1704096000000-0", {b"data": json.dumps(event1).encode()}),
                (b"1704096000001-0", {b"data": json.dumps(event2).encode()}),
            ]
        )

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert len(events) == 2
        assert events[0].content == "First message"
        assert events[1].event_type == EventType.EMOTE
        assert worker.session.last_event_id == "1704096000001-0"

    @pytest.mark.asyncio
    async def test_drain_events_handles_redis_error(self, mud_config, mock_redis):
        """Test drain_events handles Redis errors gracefully."""
        import redis.asyncio as redis_lib

        mock_redis.xinfo_stream = AsyncMock(
            side_effect=redis_lib.RedisError("Connection lost")
        )

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert events == []

    @pytest.mark.asyncio
    async def test_drain_events_drains_multiple_batches(
        self, mud_config, mock_redis, sample_event_data
    ):
        """Test drain_events drains multiple batches in one call."""
        event1 = sample_event_data.copy()
        event1["content"] = "First message"

        event2 = sample_event_data.copy()
        event2["content"] = "Second message"

        event3 = sample_event_data.copy()
        event3["content"] = "Third message"

        mock_redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "1704096000002-0"})
        mock_redis.xrange = AsyncMock(
            side_effect=[
                [
                    (b"1704096000000-0", {b"data": json.dumps(event1).encode()}),
                ],
                [
                    (b"1704096000001-0", {b"data": json.dumps(event2).encode()}),
                    (b"1704096000002-0", {b"data": json.dumps(event3).encode()}),
                ],
            ]
        )

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert len(events) == 3
        assert events[0].content == "First message"
        assert events[2].content == "Third message"

    @pytest.mark.asyncio
    async def test_drain_events_handles_malformed_json(
        self, mud_config, mock_redis
    ):
        """Test drain_events skips messages with malformed JSON."""
        mock_redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "1704096000000-0"})
        mock_redis.xrange = AsyncMock(
            return_value=[(b"1704096000000-0", {b"data": b"not valid json"})]
        )

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert events == []

    @pytest.mark.asyncio
    async def test_drain_events_handles_missing_data_field(
        self, mud_config, mock_redis
    ):
        """Test drain_events skips messages without data field."""
        mock_redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": "1704096000000-0"})
        mock_redis.xrange = AsyncMock(
            return_value=[(b"1704096000000-0", {b"other_field": b"value"})]
        )

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        events = await worker.drain_events(timeout=1.0)

        assert events == []


class TestMUDAgentWorkerProcessTurn:
    """Test MUDAgentWorker process_turn method."""

    @pytest.fixture
    def fully_mocked_worker(self, mud_config, mock_redis):
        """Create a worker with all dependencies mocked for process_turn tests."""
        async def hgetall_side_effect(key):
            if key == RedisKeys.agent_profile("test"):
                return {
                    b"room_id": b"#123",
                    b"character_id": b"andi_1",
                    b"inventory": b"[]",
                }
            if key == RedisKeys.room_profile("#123"):
                room_state = {
                    "room_id": "#123",
                    "name": "The Garden",
                    "description": "A serene garden.",
                    "exits": {"north": "#124"},
                    "tags": [],
                }
                entities = [
                    {"entity_id": "prax_1", "name": "Prax", "entity_type": "player"},
                    {
                        "entity_id": "andi_1",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "test_agent",
                    },
                ]
                return {
                    b"room_state": json.dumps(room_state).encode("utf-8"),
                    b"entities_present": json.dumps(entities).encode("utf-8"),
                }
            return {}

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Mock persona
        mock_persona = MagicMock()
        mock_persona.system_prompt = MagicMock(return_value="System prompt")
        worker.persona = mock_persona

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=[
            iter(['{"speak": {}}']),
            iter(["[== Andi's Emotional State: +Warmth+ ==]\n\nHello!"]),
        ])
        worker._llm_provider = mock_llm

        # Mock chat config
        worker.chat_config = MagicMock()
        worker.chat_config.max_tokens = 4096

        # Mock model for token limits
        worker.model = MagicMock()
        worker.model.max_tokens = 128000

        # Load decision tools
        loader = ToolLoader("config/tools")
        tools = loader.load_tool_file("config/tools/mud_phase1.yaml")
        worker._decision_tool_user = ToolUser(tools)

        # Mock conversation manager and initialize decision strategy
        mock_conversation_manager = AsyncMock()
        mock_conversation_manager.get_history = AsyncMock(return_value=[])
        mock_conversation_manager.get_total_tokens = AsyncMock(return_value=0)
        worker.conversation_manager = mock_conversation_manager

        # Initialize decision strategy (required since we now use it in _decide_action)
        worker._decision_strategy = MUDDecisionStrategy(mock_conversation_manager)
        worker._decision_strategy.set_tool_user(worker._decision_tool_user)

        # Initialize response strategy (mocked for Phase 2 response generation)
        mock_response_strategy = AsyncMock()
        mock_response_strategy.build_turns = AsyncMock(return_value=[
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello!"},
        ])
        worker._response_strategy = mock_response_strategy

        return worker

    @pytest.mark.asyncio
    async def test_process_turn_empty_events(self, fully_mocked_worker):
        """Test process_turn with no events."""
        await fully_mocked_worker.process_turn([])

        assert len(fully_mocked_worker.session.recent_turns) == 1
        assert len(fully_mocked_worker.session.pending_events) == 0

    @pytest.mark.asyncio
    async def test_process_turn_with_events(
        self, fully_mocked_worker, sample_event_data
    ):
        """Test process_turn updates session context from events."""
        event = MUDEvent.from_dict(sample_event_data)

        await fully_mocked_worker.process_turn([event])

        # Verify session was updated
        assert len(fully_mocked_worker.session.recent_turns) == 1
        assert fully_mocked_worker.session.current_room is not None
        assert fully_mocked_worker.session.current_room.name == "The Garden"
        assert len(fully_mocked_worker.session.entities_present) == 2
        assert fully_mocked_worker.session.last_action_time is not None

    @pytest.mark.asyncio
    async def test_process_turn_adds_turn_to_history(
        self, fully_mocked_worker, sample_event_data
    ):
        """Test process_turn adds a turn record to session history."""
        event = MUDEvent.from_dict(sample_event_data)

        await fully_mocked_worker.process_turn([event])

        turn = fully_mocked_worker.session.recent_turns[0]
        assert len(turn.events_received) == 1
        assert turn.events_received[0].actor == "Prax"
        # Turn should have actions since mocked LLM returns text
        assert len(turn.actions_taken) == 1
        assert turn.actions_taken[0].tool == "speak"

    @pytest.mark.asyncio
    async def test_process_turn_clears_pending_events(
        self, fully_mocked_worker, sample_event_data
    ):
        """Test process_turn clears pending events after processing."""
        event = MUDEvent.from_dict(sample_event_data)
        # Pre-populate pending events
        fully_mocked_worker.session.pending_events = [event]

        await fully_mocked_worker.process_turn([event])

        assert len(fully_mocked_worker.session.pending_events) == 0


class TestMUDAgentWorkerStart:
    """Test MUDAgentWorker start method."""

    @pytest.mark.asyncio
    async def test_start_initializes_resources(self, mud_config, mock_redis):
        """Test that start() initializes shared resources."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Mock the resource initialization
        mock_cvm = Mock()
        mock_roster = Mock()
        mock_persona = Mock()
        mock_roster.get_persona = Mock(return_value=mock_persona)

        call_count = [0]

        async def turn_request_side_effect():
            call_count[0] += 1
            if call_count[0] >= 2:
                worker.running = False
            return {}

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "_get_turn_request", new=AsyncMock(side_effect=turn_request_side_effect)
        ), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            mock_cvm_class.from_config.return_value = mock_cvm
            mock_roster_class.from_config.return_value = mock_roster

            await worker.start()

            # Verify resources were initialized
            assert worker.cvm == mock_cvm
            assert worker.roster == mock_roster
            assert worker.persona == mock_persona
            assert worker.session is not None
            assert worker.session.agent_id == mud_config.agent_id

    @pytest.mark.asyncio
    async def test_start_paused_worker_sleeps(self, mud_config, mock_redis):
        """Test that paused worker sleeps instead of processing."""
        call_count = [0]

        async def get_side_effect(key):
            call_count[0] += 1
            if call_count[0] >= 3:
                # Stop after a few pause checks
                worker.running = False
                return None
            return b"1"  # Paused

        mock_redis.get = AsyncMock(side_effect=get_side_effect)

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "_get_turn_request", new=AsyncMock(return_value={})
        ):
            mock_cvm_class.from_config.return_value = Mock()
            mock_roster = Mock()
            mock_roster.get_persona = Mock(return_value=Mock())
            mock_roster_class.from_config.return_value = mock_roster

            await worker.start()

            # Verify sleep was called (worker was paused)
            assert mock_sleep.call_count >= 1

    @pytest.mark.asyncio
    async def test_start_handles_loop_errors(self, mud_config, mock_redis):
        """Test that start() continues after errors in loop."""
        call_count = [0]

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        async def turn_request_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Test error")
            if call_count[0] >= 3:
                worker.running = False
            return {}

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "_get_turn_request", new=AsyncMock(side_effect=turn_request_side_effect)
        ), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            mock_cvm_class.from_config.return_value = Mock()
            mock_roster = Mock()
            mock_roster.get_persona = Mock(return_value=Mock())
            mock_roster_class.from_config.return_value = mock_roster

            # Should not raise - errors are caught and logged
            await worker.start()

            # Verify worker continued after error
            assert call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_start_processes_assigned_turn(self, mud_config, mock_redis):
        """Test worker processes a turn when turn_request is assigned."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "process_turn", new=AsyncMock()
        ) as mock_process, patch.object(
            worker, "drain_events", new=AsyncMock(return_value=[])
        ), patch.object(
            worker, "_get_turn_request", new=AsyncMock(return_value={"turn_id": "t1", "status": "assigned"})
        ), patch.object(
            worker, "_set_turn_request_state", new=AsyncMock()
        ) as mock_set_state, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            mock_cvm_class.from_config.return_value = Mock()
            mock_roster = Mock()
            mock_roster.get_persona = Mock(return_value=Mock())
            mock_roster_class.from_config.return_value = mock_roster

            # Stop after one loop
            async def stop_after_first(*args, **kwargs):
                worker.running = False

            mock_process.side_effect = stop_after_first

            await worker.start()

            mock_process.assert_called_once()
            assert mock_set_state.call_count >= 2

    @pytest.mark.asyncio
    async def test_start_processes_agent_turn_request(self, mud_config, mock_redis):
        """Test worker processes a turn when turn_request has reason='agent'."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Mock events to return
        mock_events = [
            MUDEvent(
                event_id="evt_1",
                event_type=EventType.SPEECH,
                actor="Prax",
                actor_type=ActorType.PLAYER,
                room_id="#123",
                room_name="The Garden",
                content="Hello, Andi!",
                timestamp=datetime.now(timezone.utc),
            )
        ]

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "process_agent_turn", new=AsyncMock()
        ) as mock_process_agent, patch.object(
            worker, "drain_events", new=AsyncMock(return_value=mock_events)
        ) as mock_drain, patch.object(
            worker, "_get_turn_request", new=AsyncMock(return_value={
                "turn_id": "t1",
                "status": "assigned",
                "reason": "agent",
                "guidance": "test guidance"
            })
        ), patch.object(
            worker, "_set_turn_request_state", new=AsyncMock()
        ) as mock_set_state, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            mock_cvm_class.from_config.return_value = Mock()
            mock_roster = Mock()
            mock_roster.get_persona = Mock(return_value=Mock())
            mock_roster_class.from_config.return_value = mock_roster

            # Stop after one loop
            async def stop_after_first(*args, **kwargs):
                worker.running = False

            mock_process_agent.side_effect = stop_after_first

            await worker.start()

            # Verify drain_events was called
            mock_drain.assert_called_once_with(timeout=0)

            # Verify process_agent_turn was called with events and guidance
            mock_process_agent.assert_called_once_with(mock_events, "test guidance")

            # Verify turn state was set to done
            assert mock_set_state.call_count >= 2
            # Check that one of the calls was to set state to "done"
            done_calls = [call for call in mock_set_state.call_args_list if call[0][1] == "done"]
            assert len(done_calls) >= 1

    @pytest.mark.asyncio
    async def test_start_processes_choose_turn_request(self, mud_config, mock_redis):
        """Test worker processes a turn when turn_request has reason='choose'."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Mock events to return
        mock_events = [
            MUDEvent(
                event_id="evt_2",
                event_type=EventType.SPEECH,
                actor="Nova",
                actor_type=ActorType.PLAYER,
                room_id="#124",
                room_name="The Library",
                content="What do you think?",
                timestamp=datetime.now(timezone.utc),
            )
        ]

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "process_agent_turn", new=AsyncMock()
        ) as mock_process_agent, patch.object(
            worker, "drain_events", new=AsyncMock(return_value=mock_events)
        ) as mock_drain, patch.object(
            worker, "_get_turn_request", new=AsyncMock(return_value={
                "turn_id": "t2",
                "status": "assigned",
                "reason": "choose",
                "guidance": "choose guidance"
            })
        ), patch.object(
            worker, "_set_turn_request_state", new=AsyncMock()
        ) as mock_set_state, patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            mock_cvm_class.from_config.return_value = Mock()
            mock_roster = Mock()
            mock_roster.get_persona = Mock(return_value=Mock())
            mock_roster_class.from_config.return_value = mock_roster

            # Stop after one loop
            async def stop_after_first(*args, **kwargs):
                worker.running = False

            mock_process_agent.side_effect = stop_after_first

            await worker.start()

            # Verify drain_events was called
            mock_drain.assert_called_once_with(timeout=0)

            # Verify process_agent_turn was called with events and guidance
            mock_process_agent.assert_called_once_with(mock_events, "choose guidance")

            # Verify turn state was set to done
            assert mock_set_state.call_count >= 2
            # Check that one of the calls was to set state to "done"
            done_calls = [call for call in mock_set_state.call_args_list if call[0][1] == "done"]
            assert len(done_calls) >= 1

    @pytest.mark.asyncio
    async def test_start_processes_agent_turn_without_guidance(self, mud_config, mock_redis):
        """Test worker processes agent turn with default empty guidance."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        with patch(
            "andimud_worker.worker.ConversationModel"
        ) as mock_cvm_class, patch(
            "andimud_worker.worker.Roster"
        ) as mock_roster_class, patch.object(
            worker, "_init_llm_provider"
        ), patch.object(
            worker, "process_agent_turn", new=AsyncMock()
        ) as mock_process_agent, patch.object(
            worker, "drain_events", new=AsyncMock(return_value=[])
        ), patch.object(
            worker, "_get_turn_request", new=AsyncMock(return_value={
                "turn_id": "t3",
                "status": "assigned",
                "reason": "agent"
                # No guidance key
            })
        ), patch.object(
            worker, "_set_turn_request_state", new=AsyncMock()
        ), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            mock_cvm_class.from_config.return_value = Mock()
            mock_roster = Mock()
            mock_roster.get_persona = Mock(return_value=Mock())
            mock_roster_class.from_config.return_value = mock_roster

            # Stop after one loop
            async def stop_after_first(*args, **kwargs):
                worker.running = False

            mock_process_agent.side_effect = stop_after_first

            await worker.start()

            # Verify process_agent_turn was called with empty guidance
            mock_process_agent.assert_called_once_with([], "")


class TestRunWorker:
    """Test run_worker entry point."""

    @pytest.mark.asyncio
    async def test_run_worker_creates_client_and_starts(self, mud_config):
        """Test that run_worker creates Redis client and starts worker."""
        with patch("redis.asyncio.from_url") as mock_from_url, patch(
            "andimud_worker.worker.MUDAgentWorker"
        ) as mock_worker_class:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            mock_worker = AsyncMock()
            mock_worker.start = AsyncMock()
            mock_worker_class.return_value = mock_worker

            await run_worker(mud_config)

            # Verify Redis client was created
            mock_from_url.assert_called_once_with(
                mud_config.redis_url,
                decode_responses=False,
            )

            # Verify worker was created and started
            # chat_config is optional and defaults to None when not passed
            mock_worker_class.assert_called_once_with(mud_config, mock_redis, None)
            mock_worker.start.assert_called_once()

            # Verify Redis client was closed
            mock_redis.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_worker_passes_chat_config(self, mud_config):
        """Test that run_worker passes chat_config to worker when provided."""
        with patch("redis.asyncio.from_url") as mock_from_url, patch(
            "andimud_worker.worker.MUDAgentWorker"
        ) as mock_worker_class:
            mock_redis = AsyncMock()
            mock_from_url.return_value = mock_redis

            mock_worker = AsyncMock()
            mock_worker.start = AsyncMock()
            mock_worker_class.return_value = mock_worker

            # Create a mock ChatConfig
            mock_chat_config = MagicMock()
            mock_chat_config.default_model = "test-model"

            await run_worker(mud_config, chat_config=mock_chat_config)

            # Verify chat_config was passed to worker
            mock_worker_class.assert_called_once_with(
                mud_config, mock_redis, mock_chat_config
            )
            mock_worker.start.assert_called_once()


class TestParseArgs:
    """Test CLI argument parsing."""

    def test_parse_args_required(self):
        """Test parsing required arguments."""
        with patch(
            "sys.argv",
            ["worker", "--agent-id", "andi", "--persona-id", "andi"],
        ):
            args = parse_args()

            assert args.agent_id == "andi"
            assert args.persona_id == "andi"
            assert args.redis_url == "redis://localhost:6379"
            assert args.log_level == "INFO"
            assert args.verbose is False
            # Optional args default to None (use .env values)
            assert args.env_file is None
            assert args.model is None
            assert args.temperature is None
            assert args.max_tokens is None

    def test_parse_args_all_options(self):
        """Test parsing all optional arguments."""
        with patch(
            "sys.argv",
            [
                "worker",
                "--agent-id",
                "test",
                "--persona-id",
                "test",
                "--env-file",
                "/custom/.env",
                "--redis-url",
                "redis://custom:6380",
                "--log-level",
                "DEBUG",
                "-v",
                "--memory-path",
                "/custom/path",
                "--spontaneous-interval",
                "600",
                "--model",
                "gpt-4",
                "--temperature",
                "0.5",
                "--max-tokens",
                "2048",
            ],
        ):
            args = parse_args()

            assert args.agent_id == "test"
            assert args.persona_id == "test"
            assert args.env_file == "/custom/.env"
            assert args.redis_url == "redis://custom:6380"
            assert args.log_level == "DEBUG"
            assert args.verbose is True
            assert args.memory_path == "/custom/path"
            assert args.spontaneous_interval == 600.0
            assert args.model == "gpt-4"
            assert args.temperature == 0.5
            assert args.max_tokens == 2048


class TestSignalHandlers:
    """Test signal handler setup."""

    def test_setup_signal_handlers(self, mud_config, mock_redis):
        """Test that signal handlers are registered."""
        import signal as signal_module

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        with patch.object(signal_module, "signal") as mock_signal:
            worker.setup_signal_handlers()

            # Verify both SIGINT and SIGTERM handlers were registered
            assert mock_signal.call_count == 2
            signals_registered = [call[0][0] for call in mock_signal.call_args_list]
            assert signal_module.SIGINT in signals_registered
            assert signal_module.SIGTERM in signals_registered


class TestGetRoomObjects:
    """Test _get_room_objects method including container contents."""

    def test_get_room_objects_direct_items(self, mud_config, mock_redis):
        """Test _get_room_objects returns items directly in the room."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Set up entities with a direct object
        worker.session.entities_present = [
            EntityState(entity_id="shelf_1", name="Shelf", entity_type="object"),
            EntityState(entity_id="andi_1", name="Andi", entity_type="ai", is_self=True),
        ]

        objects = worker._get_room_objects()

        assert "Shelf" in objects
        assert "Andi" not in objects  # Self should be excluded

    def test_get_room_objects_includes_container_contents(self, mud_config, mock_redis):
        """Test _get_room_objects returns items inside containers.

        When the room profile includes items with a 'container' field,
        those items should be available for take validation.
        """
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        # Set up entities including an item in a container
        # This simulates what _build_room_profile now produces
        worker.session.entities_present = [
            EntityState(entity_id="shelf_1", name="Shelf", entity_type="object"),
            EntityState(
                entity_id="cup_1",
                name="Kintsugi Cup",
                entity_type="object",
                # The container field indicates this is inside the Shelf
            ),
            EntityState(entity_id="andi_1", name="Andi", entity_type="ai", is_self=True),
        ]

        objects = worker._get_room_objects()

        assert "Shelf" in objects
        assert "Kintsugi Cup" in objects  # Container contents should be included
        assert len(objects) == 2

    def test_get_room_objects_excludes_characters(self, mud_config, mock_redis):
        """Test _get_room_objects excludes players, AIs, and NPCs."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MUDSession(agent_id="test", persona_id="test")

        worker.session.entities_present = [
            EntityState(entity_id="shelf_1", name="Shelf", entity_type="object"),
            EntityState(entity_id="prax_1", name="Prax", entity_type="player"),
            EntityState(entity_id="nova_1", name="Nova", entity_type="ai"),
            EntityState(entity_id="guard_1", name="Guard", entity_type="npc"),
        ]

        objects = worker._get_room_objects()

        assert objects == ["Shelf"]
        assert "Prax" not in objects
        assert "Nova" not in objects
        assert "Guard" not in objects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
