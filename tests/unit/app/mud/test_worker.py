# tests/unit/app/mud/test_worker.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUDAgentWorker two-phase action mode."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aim.app.mud.config import MUDConfig
from aim.app.mud.session import MUDSession, RoomState, EntityState
from aim.app.mud.worker import MUDAgentWorker
from aim_mud_types import EventType, MUDEvent, MUDAction, RedisKeys
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser


@pytest.fixture
def mud_config():
    """Create a MUDConfig for testing."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="andi",
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.xadd = AsyncMock(return_value="1234567890-0")
    async def hgetall_side_effect(key):
        if key == RedisKeys.agent_profile("test_agent"):
            return {
                b"room_id": b"room_1",
                b"character_id": b"self_1",
                b"inventory": b"[]",
            }
        if key == RedisKeys.room_profile("room_1"):
            room_state = {
                "room_id": "room_1",
                "name": "The Garden",
                "description": "A serene garden.",
                "exits": {"north": "#2"},
                "tags": [],
            }
            entities = [
                {"entity_id": "p1", "name": "Prax", "entity_type": "player"},
                {
                    "entity_id": "self_1",
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
    redis_mock.hgetall = AsyncMock(side_effect=hgetall_side_effect)
    return redis_mock


@pytest.fixture
def worker(mud_config, mock_redis):
    """Create a MUDAgentWorker with mocked dependencies."""
    worker = MUDAgentWorker(mud_config, mock_redis)
    worker.session = MUDSession(
        agent_id=mud_config.agent_id,
        persona_id=mud_config.persona_id,
    )
    return worker


@pytest.fixture
def decision_tool_user():
    """Load the phase 1 decision tool user (act/move)."""
    loader = ToolLoader("config/tools")
    tools = loader.load_tool_file("config/tools/mud_phase1.yaml")
    return ToolUser(tools)


@pytest.fixture
def mock_decision_strategy():
    """Create a mock MUDDecisionStrategy."""
    from aim.app.mud.strategy import MUDDecisionStrategy

    # Create a mock conversation manager
    mock_conv_manager = MagicMock()
    mock_conv_manager.get_recent_entries = AsyncMock(return_value=[])
    mock_conv_manager.get_history = AsyncMock(return_value=[])

    # Create the strategy
    strategy = MUDDecisionStrategy(mock_conv_manager)
    return strategy


@pytest.fixture
def mock_response_strategy():
    """Create a mock MUDResponseStrategy for phase 2."""
    mock_strategy = AsyncMock()
    mock_strategy.build_turns = AsyncMock(return_value=[
        {"role": "system", "content": "You are Andi"},
        {"role": "user", "content": "Current context"}
    ])
    return mock_strategy


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


class TestNormalizeResponse:
    def test_normalize_response_collapses_blank_lines(self, worker):
        raw = """

[== Andi's Emotional State: +Warmth+ ==]


Hello, Prax!


How are you?

"""
        normalized = worker._normalize_response(raw)
        assert normalized.startswith("[== Andi's Emotional State")
        assert "\n\n\n" not in normalized
        assert "Hello, Prax!" in normalized


class TestEmitActions:
    @pytest.mark.asyncio
    async def test_emit_act_action(self, worker, mock_redis):
        # Use "speak" tool which generates "act" commands
        action = MUDAction(tool="speak", args={"text": "Hello\n\nWorld"})

        await worker._emit_actions([action])

        mock_redis.xadd.assert_called_once()
        call_args = mock_redis.xadd.call_args
        data = json.loads(call_args[0][1]["data"])
        assert data["tool"] == "speak"
        assert data["command"].startswith("act Hello")


class TestProcessTurnTwoPhase:
    @pytest.mark.asyncio
    async def test_process_turn_emits_act(self, worker, sample_events, mock_redis, decision_tool_user, mock_decision_strategy, mock_response_strategy):
        mock_persona = MagicMock()
        mock_persona.system_prompt = MagicMock(return_value="System")
        worker.persona = mock_persona
        worker.chat_config = MagicMock()
        worker.chat_config.max_tokens = 2000
        worker._decision_tool_user = decision_tool_user
        worker._decision_strategy = mock_decision_strategy
        worker._response_strategy = mock_response_strategy
        mock_decision_strategy.set_tool_user(decision_tool_user)

        # Mock the model
        mock_model = MagicMock()
        mock_model.max_tokens = 8000
        worker.model = mock_model

        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=[
            iter(['{"speak": {}}']),
            iter(["[== Andi's Emotional State: +Warmth+ ==]\n\nHello, Prax!"]),
        ])
        worker._llm_provider = mock_llm

        # Set up room and entities
        worker.session.current_room = RoomState(room_id="room_1", name="The Garden")
        worker.session.entities_present = [
            EntityState(entity_id="p1", name="Prax", entity_type="player", is_self=False)
        ]

        await worker.process_turn(sample_events)

        assert len(worker.session.recent_turns) == 1
        turn = worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 1
        assert turn.actions_taken[0].tool == "speak"
        assert mock_llm.stream_turns.call_count == 2
        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_turn_handles_empty_response(self, worker, sample_events, mock_redis, decision_tool_user, mock_decision_strategy, mock_response_strategy):
        mock_persona = MagicMock()
        mock_persona.system_prompt = MagicMock(return_value="System")
        worker.persona = mock_persona
        worker.chat_config = MagicMock()
        worker.chat_config.max_tokens = 2000
        worker._decision_tool_user = decision_tool_user
        worker._decision_strategy = mock_decision_strategy
        worker._response_strategy = mock_response_strategy
        mock_decision_strategy.set_tool_user(decision_tool_user)

        # Mock the model
        mock_model = MagicMock()
        mock_model.max_tokens = 8000
        worker.model = mock_model

        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=[
            iter(['{"speak": {}}']),
            iter(["   "]),
        ])
        worker._llm_provider = mock_llm

        await worker.process_turn(sample_events)

        turn = worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 0
        mock_redis.xadd.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_turn_strips_think_tags(self, worker, sample_events, mock_redis, decision_tool_user, mock_decision_strategy, mock_response_strategy):
        mock_persona = MagicMock()
        mock_persona.system_prompt = MagicMock(return_value="System")
        worker.persona = mock_persona
        worker.chat_config = MagicMock()
        worker.chat_config.max_tokens = 2000
        worker._decision_tool_user = decision_tool_user
        worker._decision_strategy = mock_decision_strategy
        worker._response_strategy = mock_response_strategy
        mock_decision_strategy.set_tool_user(decision_tool_user)

        # Mock the model
        mock_model = MagicMock()
        mock_model.max_tokens = 8000
        worker.model = mock_model

        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=[
            iter(['{"speak": {}}']),
            iter([
                "<think>internal</think>\n[== Andi's Emotional State: +Warmth+ ==]\n\nHello!"
            ]),
        ])
        worker._llm_provider = mock_llm

        await worker.process_turn(sample_events)

        turn = worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 1
        assert "<think>" not in turn.actions_taken[0].args["text"]
        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_turn_move_action(self, worker, sample_events, mock_redis, decision_tool_user, mock_decision_strategy):
        mock_persona = MagicMock()
        mock_persona.system_prompt = MagicMock(return_value="System")
        worker.persona = mock_persona
        worker.chat_config = MagicMock()
        worker._decision_tool_user = decision_tool_user
        worker._decision_strategy = mock_decision_strategy
        mock_decision_strategy.set_tool_user(decision_tool_user)

        mock_llm = MagicMock()
        mock_llm.stream_turns = MagicMock(side_effect=[
            iter(['{"move": {"location": "north"}}']),
        ])
        worker._llm_provider = mock_llm

        worker.session.current_room = RoomState(
            room_id="room_1",
            name="The Garden",
            exits={"north": "#2"},
        )

        await worker.process_turn(sample_events)

        turn = worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 1
        assert turn.actions_taken[0].tool == "move"
        assert mock_llm.stream_turns.call_count == 1
        mock_redis.xadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_turn_take_action(self, worker, sample_events, mock_redis, decision_tool_user, mock_decision_strategy):
        mock_persona = MagicMock()
        mock_persona.system_prompt = MagicMock(return_value="System")
        worker.persona = mock_persona
        worker.chat_config = MagicMock()
        worker._decision_tool_user = decision_tool_user
        worker._decision_strategy = mock_decision_strategy
        mock_decision_strategy.set_tool_user(decision_tool_user)

        # Update the mock_redis hgetall to include the lantern in the room profile
        async def hgetall_with_lantern(key):
            if key == RedisKeys.agent_profile("test_agent"):
                return {
                    b"room_id": b"room_1",
                    b"character_id": b"self_1",
                    b"inventory": b"[]",
                }
            if key == RedisKeys.room_profile("room_1"):
                room_state = {
                    "room_id": "room_1",
                    "name": "The Garden",
                    "description": "A serene garden.",
                    "exits": {"north": "#2"},
                    "tags": [],
                }
                entities = [
                    {"entity_id": "p1", "name": "Prax", "entity_type": "player"},
                    {
                        "entity_id": "self_1",
                        "name": "Andi",
                        "entity_type": "ai",
                        "agent_id": "test_agent",
                    },
                    {
                        "entity_id": "lantern_1",
                        "name": "lantern",
                        "entity_type": "object",
                        "description": "A brass lantern",
                    },
                ]
                return {
                    b"room_state": json.dumps(room_state).encode("utf-8"),
                    b"entities_present": json.dumps(entities).encode("utf-8"),
                }
            return {}

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_with_lantern)

        mock_llm = MagicMock()
        # Single response should work now
        mock_llm.stream_turns = MagicMock(return_value=iter(['{"take": {"object": "lantern"}}']))
        worker._llm_provider = mock_llm

        await worker.process_turn(sample_events)

        turn = worker.session.recent_turns[0]
        assert len(turn.actions_taken) == 1
        assert turn.actions_taken[0].tool == "get"
        assert turn.actions_taken[0].args["object"] == "lantern"
        mock_redis.xadd.assert_called_once()
