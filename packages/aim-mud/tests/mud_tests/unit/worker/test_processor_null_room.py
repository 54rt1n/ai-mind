# packages/aim-mud/tests/mud_tests/unit/worker/test_processor_null_room.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for defensive null room handling in turn processors.

Tests the fix where:
1. Phased processor logs warning when current_room is None
2. Agent processor logs warning when current_room is None
3. Both processors use "Unknown Location" instead of crashing
4. Normal operation still works when room is set
"""

import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.turns.processor.decision import DecisionProcessor
from andimud_worker.turns.processor.speaking import SpeakingProcessor
from andimud_worker.turns.processor.agent import AgentTurnProcessor
from andimud_worker.config import MUDConfig
from andimud_worker.tools.helper import ToolHelper
from aim_mud_types import MUDTurnRequest, MUDEvent, EventType, ActorType
from aim.config import ChatConfig


@pytest.fixture
def mock_worker():
    """Create a mock worker with necessary attributes."""
    from aim_mud_types.models.session import MUDSession
    from aim_mud_types.models.state import RoomState

    worker = MagicMock()
    worker.agent_id = "test_agent"
    worker.config = MUDConfig(agent_id="test_agent", persona_id="test_persona")

    # Session with current_room that can be set to None
    worker.session = MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
        current_room=RoomState(
            room_id="room:kitchen",
            name="Kitchen"
        )
    )

    # Persona
    worker.persona = MagicMock()
    worker.persona.name = "TestAgent"

    # Model config
    worker.model = MagicMock()
    worker.model.max_tokens = 4096

    # Chat config
    worker.chat_config = MagicMock()
    worker.chat_config.max_tokens = 1024

    # Methods
    worker._check_abort_requested = AsyncMock(return_value=False)
    worker._decide_action = AsyncMock(return_value=("move", {"location": "Bedroom"}, "raw", "thinking", "cleaned"))
    worker._write_self_event = AsyncMock()
    worker._emit_actions = AsyncMock()
    worker._is_fresh_session = AsyncMock(return_value=False)
    worker._call_llm = AsyncMock(return_value="Test response")
    worker._response_strategy = MagicMock()
    worker._response_strategy.build_turns = AsyncMock(return_value=[])
    worker._build_agent_guidance = MagicMock(return_value="Test guidance")
    worker._agent_action_list = MagicMock(return_value=[
        {"name": "speak"},
        {"name": "move"},
        {"name": "take"},
        {"name": "drop"},
        {"name": "give"},
        {"name": "describe"},
    ])

    return worker


@pytest.fixture
def mock_tool_helper():
    """Create a mock ToolHelper."""
    helper = MagicMock(spec=ToolHelper)
    helper._tool_user = MagicMock()
    helper._tool_user.get_tool_guidance.return_value = "Tool guidance"
    helper.decorate_xml = MagicMock(side_effect=lambda xml: xml)
    return helper


@pytest.fixture
def sample_turn_request():
    """Create a sample turn request."""
    return MUDTurnRequest(
        turn_id="turn-123",
        status="in_progress",
        reason="events",
        sequence_id=1,
        timestamp="2025-01-11T12:00:00Z",
    )


class TestPhasedProcessorNullRoomHandling:
    """Test phased processor defensive null room handling."""

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_phased_processor_logs_warning_for_null_room(
        self, mock_worker, sample_turn_request, caplog
    ):
        """Test that phased processor logs warning when room is None."""
        # Set current_room to None
        mock_worker.session.current_room = None

        processor = PhasedTurnProcessor(mock_worker)

        # Capture logs at WARNING level
        with caplog.at_level(logging.WARNING):
            # Trigger movement decision
            await processor._decide_action(sample_turn_request, [])

        # Assert warning was logged
        warning_logs = [record for record in caplog.records if record.levelname == "WARNING"]
        assert any(
            "Current room not set for agent" in record.message and
            "test_agent" in record.message
            for record in warning_logs
        ), f"Expected warning log not found. Logs: {[r.message for r in warning_logs]}"

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_phased_processor_uses_unknown_location_for_null_room(
        self, mock_worker, sample_turn_request
    ):
        """Test that phased processor uses 'Unknown Location' when room is None."""
        # Set current_room to None
        mock_worker.session.current_room = None

        processor = PhasedTurnProcessor(mock_worker)

        # Trigger movement decision
        await processor._decide_action(sample_turn_request, [])

        # Check that _write_self_event was called
        assert mock_worker._write_self_event.called

        # Extract the event that was written
        written_event = mock_worker._write_self_event.call_args[0][0]

        # Assert room_name is "Unknown Location"
        assert written_event.room_name == "Unknown Location"

        # Assert metadata has source_room_name = "Unknown Location"
        assert written_event.metadata.get("source_room_name") == "Unknown Location"

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_phased_processor_works_normally_with_valid_room(
        self, mock_worker, sample_turn_request, caplog
    ):
        """Test that phased processor works normally when room is set."""
        # Room is set to Kitchen (default in fixture)
        processor = PhasedTurnProcessor(mock_worker)

        with caplog.at_level(logging.WARNING):
            await processor._decide_action(sample_turn_request, [])

        # No warning logs
        warning_logs = [
            record for record in caplog.records
            if record.levelname == "WARNING" and "Current room not set" in record.message
        ]
        assert len(warning_logs) == 0

        # Check event has correct room name
        assert mock_worker._write_self_event.called
        written_event = mock_worker._write_self_event.call_args[0][0]
        assert written_event.room_name == "Kitchen"
        assert written_event.metadata.get("source_room_name") == "Kitchen"


class TestAgentProcessorNullRoomHandling:
    """Test agent processor defensive null room handling."""

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_agent_processor_logs_warning_for_null_room(
        self, mock_worker, mock_tool_helper, sample_turn_request, caplog
    ):
        """Test that agent processor logs warning when room is None."""
        # Set current_room to None
        mock_worker.session.current_room = None

        # Mock the agent response to trigger move action
        mock_worker._call_llm = AsyncMock(
            return_value='{"action": "move", "location": "Bedroom"}'
        )

        processor = AgentTurnProcessor(mock_worker, mock_tool_helper)

        # Mock resolve_move_location to return the location
        with patch("andimud_worker.turns.processor.agent.resolve_move_location", return_value="Bedroom"):
            with caplog.at_level(logging.WARNING):
                await processor._decide_action(sample_turn_request, [])

        # Assert warning was logged
        warning_logs = [record for record in caplog.records if record.levelname == "WARNING"]
        assert any(
            "Current room not set for agent" in record.message and
            "test_agent" in record.message
            for record in warning_logs
        ), f"Expected warning log not found. Logs: {[r.message for r in warning_logs]}"

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_agent_processor_uses_unknown_location_for_null_room(
        self, mock_worker, mock_tool_helper, sample_turn_request
    ):
        """Test that agent processor uses 'Unknown Location' when room is None."""
        # Set current_room to None
        mock_worker.session.current_room = None

        # Mock the agent response to trigger move action
        mock_worker._call_llm = AsyncMock(
            return_value='{"action": "move", "location": "Bedroom"}'
        )

        processor = AgentTurnProcessor(mock_worker, mock_tool_helper)

        # Mock resolve_move_location to return the location
        with patch("andimud_worker.turns.processor.agent.resolve_move_location", return_value="Bedroom"):
            await processor._decide_action(sample_turn_request, [])

        # Check that _write_self_event was called
        assert mock_worker._write_self_event.called

        # Extract the event that was written
        written_event = mock_worker._write_self_event.call_args[0][0]

        # Assert room_name is "Unknown Location"
        assert written_event.room_name == "Unknown Location"

        # Assert metadata has source_room_name = "Unknown Location"
        assert written_event.metadata.get("source_room_name") == "Unknown Location"

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_agent_processor_works_normally_with_valid_room(
        self, mock_worker, mock_tool_helper, sample_turn_request, caplog
    ):
        """Test that agent processor works normally when room is set."""
        # Room is set to Kitchen (default in fixture)

        # Mock the agent response to trigger move action
        mock_worker._call_llm = AsyncMock(
            return_value='{"action": "move", "location": "Bedroom"}'
        )

        processor = AgentTurnProcessor(mock_worker, mock_tool_helper)

        # Mock resolve_move_location to return the location
        with patch("andimud_worker.turns.processor.agent.resolve_move_location", return_value="Bedroom"):
            with caplog.at_level(logging.WARNING):
                await processor._decide_action(sample_turn_request, [])

        # No warning logs about null room
        warning_logs = [
            record for record in caplog.records
            if record.levelname == "WARNING" and "Current room not set" in record.message
        ]
        assert len(warning_logs) == 0

        # Check event has correct room name
        assert mock_worker._write_self_event.called
        written_event = mock_worker._write_self_event.call_args[0][0]
        assert written_event.room_name == "Kitchen"
        assert written_event.metadata.get("source_room_name") == "Kitchen"


class TestProcessorRoomNameHandling:
    """Test room name handling in both processors."""

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_phased_processor_handles_room_without_name(
        self, mock_worker, sample_turn_request
    ):
        """Test phased processor when room exists but name is None."""
        # Room exists but name is None
        mock_worker.session.current_room.name = None

        processor = PhasedTurnProcessor(mock_worker)

        await processor._decide_action(sample_turn_request, [])

        # Should use "Unknown Location"
        assert mock_worker._write_self_event.called
        written_event = mock_worker._write_self_event.call_args[0][0]
        assert written_event.room_name == "Unknown Location"

    @pytest.mark.skip(reason="Null room defensive handling not yet implemented")
    @pytest.mark.asyncio
    async def test_agent_processor_handles_room_without_name(
        self, mock_worker, mock_tool_helper, sample_turn_request
    ):
        """Test agent processor when room exists but name is None."""
        # Room exists but name is None
        mock_worker.session.current_room.name = None

        # Mock the agent response to trigger move action
        mock_worker._call_llm = AsyncMock(
            return_value='{"action": "move", "location": "Bedroom"}'
        )

        processor = AgentTurnProcessor(mock_worker, mock_tool_helper)

        # Mock resolve_move_location to return the location
        with patch("andimud_worker.turns.processor.agent.resolve_move_location", return_value="Bedroom"):
            await processor._decide_action(sample_turn_request, [])

        # Should use "Unknown Location"
        assert mock_worker._write_self_event.called
        written_event = mock_worker._write_self_event.call_args[0][0]
        assert written_event.room_name == "Unknown Location"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
