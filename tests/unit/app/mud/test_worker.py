# tests/unit/app/mud/test_worker.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUDAgentWorker.process_turn and helper methods."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aim.app.mud.config import MUDConfig
from aim.app.mud.session import MUDSession, MUDTurn, RoomState, EntityState
from aim.app.mud.worker import MUDAgentWorker
from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters
from aim.tool.formatting import ToolUser, ToolCallResult
from aim_mud_types import EventType, MUDEvent, MUDAction


@pytest.fixture
def mud_config():
    """Create a MUDConfig for testing."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="andi",
        redis_url="redis://localhost:6379",
        model="test-model",
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.xadd = AsyncMock(return_value="1234567890-0")
    return redis_mock


@pytest.fixture
def sample_room_state():
    """Create a sample room state."""
    return RoomState(
        room_id="room_1",
        name="The Garden",
        description="A peaceful garden with a silver fountain.",
        exits={"north": "room_2", "south": "room_3"},
    )


@pytest.fixture
def sample_entity_state():
    """Create a sample entity state."""
    return EntityState(
        entity_id="entity_1",
        name="Prax",
        entity_type="player",
        is_self=False,
    )


@pytest.fixture
def sample_events():
    """Create sample MUD events."""
    return [
        MUDEvent(
            event_id="evt_1",
            event_type=EventType.SPEECH,
            actor="Prax",
            content="Hello, Andi!",
            room_id="room_1",
            room_name="The Garden",
        ),
    ]


@pytest.fixture
def sample_mud_tools():
    """Create sample MUD tools for testing."""
    return [
        Tool(
            type="mud",
            function=ToolFunction(
                name="say",
                description="Speak audibly to everyone in your current location",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={
                        "message": {
                            "type": "string",
                            "description": "What you want to say",
                        }
                    },
                    required=["message"],
                ),
            ),
        ),
        Tool(
            type="mud",
            function=ToolFunction(
                name="emote",
                description="Perform an action or express emotion",
                parameters=ToolFunctionParameters(
                    type="object",
                    properties={
                        "action": {
                            "type": "string",
                            "description": "The action to perform",
                        }
                    },
                    required=["action"],
                ),
            ),
        ),
    ]


@pytest.fixture
def worker(mud_config, mock_redis):
    """Create a MUDAgentWorker with mocked dependencies."""
    worker = MUDAgentWorker(mud_config, mock_redis)

    # Set up minimal session
    worker.session = MUDSession(
        agent_id=mud_config.agent_id,
        persona_id=mud_config.persona_id,
    )

    return worker


class TestExtractThinking:
    """Tests for _extract_thinking method."""

    def test_extract_thinking_from_think_tags(self, worker):
        """Should extract content from <think> tags."""
        response = "<think>I should greet them warmly.</think>\n{\"say\": {\"message\": \"Hello!\"}}"

        result = worker._extract_thinking(response)

        assert result == "I should greet them warmly."

    def test_extract_thinking_before_json(self, worker):
        """Should extract content before first JSON object."""
        response = "Let me respond to Prax.\n\n{\"say\": {\"message\": \"Hello, Prax!\"}}"

        result = worker._extract_thinking(response)

        assert result == "Let me respond to Prax."

    def test_extract_thinking_empty_when_no_json(self, worker):
        """Should return empty string when no JSON found."""
        response = "Just some text without any JSON."

        result = worker._extract_thinking(response)

        assert result == ""

    def test_extract_thinking_empty_when_json_at_start(self, worker):
        """Should return empty string when JSON is at the start."""
        response = "{\"say\": {\"message\": \"Hello!\"}}"

        result = worker._extract_thinking(response)

        assert result == ""


class TestParseToolCalls:
    """Tests for _parse_tool_calls method."""

    def test_parse_valid_say_tool_call(self, worker, sample_mud_tools):
        """Should parse a valid say tool call."""
        worker.tool_user = ToolUser(sample_mud_tools)
        response = '{"say": {"message": "Hello, everyone!"}}'

        results = worker._parse_tool_calls(response)

        assert len(results) == 1
        assert results[0].is_valid
        assert results[0].function_name == "say"
        assert results[0].arguments == {"message": "Hello, everyone!"}

    def test_parse_valid_emote_tool_call(self, worker, sample_mud_tools):
        """Should parse a valid emote tool call."""
        worker.tool_user = ToolUser(sample_mud_tools)
        response = '{"emote": {"action": "smiles warmly"}}'

        results = worker._parse_tool_calls(response)

        assert len(results) == 1
        assert results[0].is_valid
        assert results[0].function_name == "emote"
        assert results[0].arguments == {"action": "smiles warmly"}

    def test_parse_tool_call_with_think_tags(self, worker, sample_mud_tools):
        """Should parse tool call after think tags."""
        worker.tool_user = ToolUser(sample_mud_tools)
        response = '<think>Prax greeted me, I should respond.</think>\n{"say": {"message": "Hello, Prax!"}}'

        results = worker._parse_tool_calls(response)

        assert len(results) == 1
        assert results[0].is_valid
        assert results[0].function_name == "say"

    def test_parse_invalid_tool_call(self, worker, sample_mud_tools):
        """Should return empty list for invalid tool calls."""
        worker.tool_user = ToolUser(sample_mud_tools)
        response = "This is not a valid tool call."

        results = worker._parse_tool_calls(response)

        assert len(results) == 0

    def test_parse_tool_call_missing_required_param(self, worker, sample_mud_tools):
        """Should return empty list when required param is missing."""
        worker.tool_user = ToolUser(sample_mud_tools)
        response = '{"say": {}}'  # Missing required "message" param

        results = worker._parse_tool_calls(response)

        assert len(results) == 0


class TestEmitActions:
    """Tests for _emit_actions method."""

    @pytest.mark.asyncio
    async def test_emit_single_action(self, worker, mock_redis):
        """Should emit a single action to Redis stream."""
        action = MUDAction(tool="say", args={"message": "Hello!"})

        await worker._emit_actions([action])

        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        assert call_args[0][0] == worker.config.action_stream

        # Verify the data structure
        data = json.loads(call_args[0][1]["data"])
        assert data["agent_id"] == "test_agent"
        assert data["tool"] == "say"
        assert data["command"] == "say Hello!"

    @pytest.mark.asyncio
    async def test_emit_multiple_actions(self, worker, mock_redis):
        """Should emit multiple actions to Redis stream."""
        actions = [
            MUDAction(tool="emote", args={"action": "smiles"}),
            MUDAction(tool="say", args={"message": "Hello!"}),
        ]

        await worker._emit_actions(actions)

        assert mock_redis.xadd.call_count == 2

    @pytest.mark.asyncio
    async def test_emit_action_handles_redis_error(self, worker, mock_redis):
        """Should handle Redis errors gracefully."""
        import redis.asyncio as redis_async
        mock_redis.xadd.side_effect = redis_async.RedisError("Connection failed")

        action = MUDAction(tool="say", args={"message": "Hello!"})

        # Should not raise, just log the error
        await worker._emit_actions([action])


class TestProcessTurn:
    """Tests for process_turn method."""

    @pytest.fixture
    def fully_mocked_worker(self, worker, sample_mud_tools, sample_room_state, sample_entity_state):
        """Create a worker with all LLM dependencies mocked."""
        # Mock persona
        mock_persona = MagicMock()
        mock_persona.xml_decorator = MagicMock()
        worker.persona = mock_persona

        # Mock tool user
        worker.tool_user = ToolUser(sample_mud_tools)

        # Mock LLM provider
        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(return_value=iter([
            '{"say": {"message": "Hello, Prax!"}}'
        ]))
        worker._llm_provider = mock_llm

        # Mock chat config
        mock_config = MagicMock()
        worker.chat_config = mock_config

        # Set up room and entities
        worker.session.current_room = sample_room_state
        worker.session.entities_present = [sample_entity_state]

        return worker

    @pytest.mark.asyncio
    async def test_process_turn_creates_turn_record(self, fully_mocked_worker, sample_events):
        """Should create a turn record with events and actions."""
        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        assert len(fully_mocked_worker.session.recent_turns) == 1
        turn = fully_mocked_worker.session.recent_turns[0]
        assert len(turn.events_received) == 1
        assert len(turn.actions_taken) == 1
        assert turn.actions_taken[0].tool == "say"

    @pytest.mark.asyncio
    async def test_process_turn_emits_actions_to_redis(self, fully_mocked_worker, sample_events, mock_redis):
        """Should emit parsed actions to Redis stream."""
        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_turn_updates_session_context(self, fully_mocked_worker, sample_events):
        """Should update session context from events."""
        # Add room_state to the event
        sample_events[0].room_state = {
            "room_id": "room_1",
            "name": "New Room",
            "description": "A new room.",
            "exits": {"east": "room_4"},
        }

        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        assert fully_mocked_worker.session.current_room.name == "New Room"

    @pytest.mark.asyncio
    async def test_process_turn_handles_no_tool_calls(self, fully_mocked_worker, sample_events, mock_redis):
        """Should handle response with no valid tool calls."""
        fully_mocked_worker._llm_provider.stream_turns = MagicMock(
            return_value=iter(["I'm not sure what to do."])
        )

        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        # Should still create turn record
        assert len(fully_mocked_worker.session.recent_turns) == 1
        turn = fully_mocked_worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 0

        # Should not emit any actions
        mock_redis.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_turn_handles_llm_error(self, fully_mocked_worker, sample_events, mock_redis):
        """Should handle LLM errors gracefully."""
        fully_mocked_worker._llm_provider.stream_turns = MagicMock(
            side_effect=Exception("LLM API error")
        )

        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        # Should still create turn record with error
        assert len(fully_mocked_worker.session.recent_turns) == 1
        turn = fully_mocked_worker.session.recent_turns[0]
        assert "[ERROR]" in turn.thinking
        assert len(turn.actions_taken) == 0

    @pytest.mark.asyncio
    async def test_process_turn_clears_pending_events(self, fully_mocked_worker, sample_events):
        """Should clear pending events after processing."""
        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        assert len(fully_mocked_worker.session.pending_events) == 0

    @pytest.mark.asyncio
    async def test_process_turn_with_emote_action(self, fully_mocked_worker, sample_events, mock_redis):
        """Should handle emote tool calls correctly."""
        fully_mocked_worker._llm_provider.stream_turns = MagicMock(
            return_value=iter(['{"emote": {"action": "smiles warmly at Prax"}}'])
        )

        with patch.object(fully_mocked_worker, '_build_system_prompt_with_tools', return_value="System prompt"):
            await fully_mocked_worker.process_turn(sample_events)

        turn = fully_mocked_worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 1
        assert turn.actions_taken[0].tool == "emote"
        assert turn.actions_taken[0].args["action"] == "smiles warmly at Prax"


class TestBuildSystemPromptWithTools:
    """Tests for _build_system_prompt_with_tools method."""

    def test_build_system_prompt_includes_tools(self, worker, sample_mud_tools, sample_room_state):
        """Should include tool instructions in system prompt."""
        mock_persona = MagicMock()
        mock_persona.xml_decorator = MagicMock()
        worker.persona = mock_persona
        worker.tool_user = ToolUser(sample_mud_tools)
        worker.session.current_room = sample_room_state

        result = worker._build_system_prompt_with_tools()

        # Should contain XML structure
        assert "<root>" in result
        assert "</root>" in result

        # Should contain MUD instructions
        assert "text-based world" in result or "MUD" in result

    def test_build_system_prompt_without_room(self, worker, sample_mud_tools):
        """Should handle case when no room is set."""
        mock_persona = MagicMock()
        mock_persona.xml_decorator = MagicMock()
        worker.persona = mock_persona
        worker.tool_user = ToolUser(sample_mud_tools)
        worker.session.current_room = None

        # Should not raise
        result = worker._build_system_prompt_with_tools()

        assert result is not None
