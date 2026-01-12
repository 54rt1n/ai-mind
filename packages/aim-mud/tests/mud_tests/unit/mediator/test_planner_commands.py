# tests/unit/mediator/test_planner_commands.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for planner mediator commands.

These commands allow control of the planner feature:
- @planner <agent-id> on/off - toggle planner enabled state
- @plan <agent-id> = <objective> - create a new plan via planner turn
- @update <agent-id> = <guidance> - update plan with guidance
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from andimud_mediator.patterns import PLANNER_PATTERN, PLAN_PATTERN, UPDATE_PATTERN
from aim_mud_types import MUDEvent, EventType, RedisKeys, TurnRequestStatus
from aim_mud_types.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus


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
    redis.set = AsyncMock(return_value=True)
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.xtrim = AsyncMock(return_value=0)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    redis.hkeys = AsyncMock(return_value=[])
    redis.hdel = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)  # Lua script success by default
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def sample_plan():
    """Create a sample active plan for testing."""
    return AgentPlan(
        plan_id="test-123",
        agent_id="andi",
        objective="Test objective",
        summary="Test plan",
        status=PlanStatus.ACTIVE,
        tasks=[PlanTask(
            id=0,
            description="First task description",
            summary="First task",
            context="Why this task matters",
            status=TaskStatus.IN_PROGRESS,
        )],
        current_task_id=0,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestPlannerPattern:
    """Test the PLANNER_PATTERN regex."""

    def test_pattern_on(self):
        """Test pattern matches 'on' command."""
        match = PLANNER_PATTERN.match("@planner andi on")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "on"

    def test_pattern_off(self):
        """Test pattern matches 'off' command."""
        match = PLANNER_PATTERN.match("@planner andi off")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "off"

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = PLANNER_PATTERN.match("@PLANNER ANDI ON")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "ON"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert PLANNER_PATTERN.match("@planner") is None
        assert PLANNER_PATTERN.match("@planner andi") is None
        assert PLANNER_PATTERN.match("@planner andi maybe") is None
        assert PLANNER_PATTERN.match("planner andi on") is None


class TestPlanPattern:
    """Test the PLAN_PATTERN regex."""

    def test_pattern_matches(self):
        """Test pattern matches with objective."""
        match = PLAN_PATTERN.match("@plan andi = Build a feature")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Build a feature"

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = PLAN_PATTERN.match("@PLAN ANDI = Build Feature")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "Build Feature"

    def test_pattern_no_spaces_around_equals(self):
        """Test pattern works without spaces around equals."""
        match = PLAN_PATTERN.match("@plan andi=Build feature")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Build feature"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert PLAN_PATTERN.match("@plan") is None
        assert PLAN_PATTERN.match("@plan andi") is None
        assert PLAN_PATTERN.match("plan andi = objective") is None


class TestUpdatePattern:
    """Test the UPDATE_PATTERN regex."""

    def test_pattern_matches(self):
        """Test pattern matches with guidance."""
        match = UPDATE_PATTERN.match("@update andi = Task completed successfully")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Task completed successfully"

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = UPDATE_PATTERN.match("@UPDATE ANDI = Task done")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "Task done"

    def test_pattern_no_spaces_around_equals(self):
        """Test pattern works without spaces around equals."""
        match = UPDATE_PATTERN.match("@update andi=Task blocked")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Task blocked"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert UPDATE_PATTERN.match("@update") is None
        assert UPDATE_PATTERN.match("@update andi") is None
        assert UPDATE_PATTERN.match("update andi = guidance") is None


class TestHandlePlannerToggle:
    """Test _handle_planner_toggle method."""

    @pytest.mark.asyncio
    async def test_enable_planner(self, mock_redis, mediator_config):
        """Test enabling planner for an agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        result = await mediator._handle_planner_toggle("andi", True)

        assert result is True
        # Verify set was called with the planner enabled key
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args[0]
        assert call_args[0] == RedisKeys.agent_planner_enabled("andi")
        assert call_args[1] == "true"

    @pytest.mark.asyncio
    async def test_disable_planner(self, mock_redis, mediator_config):
        """Test disabling planner for an agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        result = await mediator._handle_planner_toggle("andi", False)

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args[0]
        assert call_args[0] == RedisKeys.agent_planner_enabled("andi")
        assert call_args[1] == "false"

    @pytest.mark.asyncio
    async def test_rejects_unregistered_agent(self, mock_redis, mediator_config):
        """Test that unregistered agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")  # Register andi, not val

        result = await mediator._handle_planner_toggle("val", True)

        assert result is False
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_empty_registered_agents(self, mock_redis, mediator_config):
        """Test that command works when registered_agents is empty (no filtering)."""
        mediator = MediatorService(mock_redis, mediator_config)
        # Don't register any agents

        result = await mediator._handle_planner_toggle("andi", True)

        assert result is True


class TestHandlePlanCommand:
    """Test _handle_plan_command method."""

    @pytest.mark.asyncio
    async def test_creates_planner_turn(self, mock_redis, mediator_config):
        """Test successful planner turn assignment."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })
        mock_redis.eval = AsyncMock(return_value=1)  # CAS success

        result = await mediator._handle_plan_command("andi", "Build a feature")

        assert result is True
        mock_redis.eval.assert_called_once()

        # Verify the update was attempted with correct key
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert args[1] == 1  # One key
        assert args[2] == RedisKeys.agent_turn_request("andi")  # The key

        # Verify fields include metadata with pipeline and objective
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}
        assert "metadata" in fields_dict
        metadata = json.loads(fields_dict["metadata"])
        assert metadata["pipeline"] == "planner"
        assert metadata["objective"] == "Build a feature"
        assert fields_dict["reason"] == "dream"

    @pytest.mark.asyncio
    async def test_rejects_unregistered_agent(self, mock_redis, mediator_config):
        """Test that unregistered agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        result = await mediator._handle_plan_command("val", "Build feature")

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_offline_agent(self, mock_redis, mediator_config):
        """Test that offline agents (no turn_request) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={})  # No turn_request

        result = await mediator._handle_plan_command("andi", "Build feature")

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_crashed_agent(self, mock_redis, mediator_config):
        """Test that crashed agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"crashed",
            b"turn_id": b"old-turn",
            b"sequence_id": b"1",
        })

        result = await mediator._handle_plan_command("andi", "Build feature")

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_busy_agent_in_progress(self, mock_redis, mediator_config):
        """Test that busy agents (in_progress) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"current-turn",
            b"sequence_id": b"1",
        })

        result = await mediator._handle_plan_command("andi", "Build feature")

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_cas_failure(self, mock_redis, mediator_config):
        """Test handling of CAS failure (state changed during assignment)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })
        mock_redis.eval = AsyncMock(return_value=0)  # CAS failure

        result = await mediator._handle_plan_command("andi", "Build feature")

        assert result is False
        mock_redis.expire.assert_not_called()


class TestHandleUpdateCommand:
    """Test _handle_update_command method."""

    @pytest.mark.asyncio
    async def test_update_with_active_plan(self, mock_redis, mediator_config, sample_plan):
        """Test successful update command with active plan."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock get_plan to return sample_plan
        mock_redis.hgetall = AsyncMock(side_effect=[
            # First call: get_plan
            {
                b"plan_id": b"test-123",
                b"agent_id": b"andi",
                b"objective": b"Test objective",
                b"summary": b"Test plan",
                b"status": b"active",
                b"tasks": json.dumps([{
                    "id": 0,
                    "description": "desc",
                    "summary": "sum",
                    "context": "ctx",
                    "status": "in_progress",
                }]).encode(),
                b"current_task_id": b"0",
                b"created_at": datetime.now(timezone.utc).isoformat().encode(),
                b"updated_at": datetime.now(timezone.utc).isoformat().encode(),
            },
            # Second call: get_turn_request
            {
                b"status": b"ready",
                b"turn_id": b"prev-turn",
                b"sequence_id": b"1",
            },
        ])
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_update_command("andi", "Task completed")

        assert result is True
        mock_redis.eval.assert_called_once()

        # Verify metadata includes plan_guidance
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}
        assert "metadata" in fields_dict
        metadata = json.loads(fields_dict["metadata"])
        assert metadata["plan_guidance"] == "Task completed"
        assert fields_dict["reason"] == "idle"

    @pytest.mark.asyncio
    async def test_update_no_plan(self, mock_redis, mediator_config):
        """Test update fails when agent has no plan."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={})  # No plan

        result = await mediator._handle_update_command("andi", "guidance")

        assert result is False

    @pytest.mark.asyncio
    async def test_update_completed_plan(self, mock_redis, mediator_config):
        """Test update fails when plan is completed."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"plan_id": b"test-123",
            b"agent_id": b"andi",
            b"objective": b"Test objective",
            b"summary": b"Test plan",
            b"status": b"completed",  # Not active
            b"tasks": b"[]",
            b"current_task_id": b"0",
            b"created_at": datetime.now(timezone.utc).isoformat().encode(),
            b"updated_at": datetime.now(timezone.utc).isoformat().encode(),
        })

        result = await mediator._handle_update_command("andi", "guidance")

        assert result is False

    @pytest.mark.asyncio
    async def test_update_blocked_plan_allowed(self, mock_redis, mediator_config):
        """Test update works when plan is blocked (can update to unblock)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(side_effect=[
            # First call: get_plan (blocked)
            {
                b"plan_id": b"test-123",
                b"agent_id": b"andi",
                b"objective": b"Test objective",
                b"summary": b"Test plan",
                b"status": b"blocked",
                b"tasks": json.dumps([{
                    "id": 0,
                    "description": "desc",
                    "summary": "sum",
                    "context": "ctx",
                    "status": "blocked",
                }]).encode(),
                b"current_task_id": b"0",
                b"created_at": datetime.now(timezone.utc).isoformat().encode(),
                b"updated_at": datetime.now(timezone.utc).isoformat().encode(),
            },
            # Second call: get_turn_request
            {
                b"status": b"ready",
                b"turn_id": b"prev-turn",
                b"sequence_id": b"1",
            },
        ])
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_update_command("andi", "Blocker resolved")

        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_unregistered_agent(self, mock_redis, mediator_config):
        """Test that unregistered agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        result = await mediator._handle_update_command("val", "guidance")

        assert result is False

    @pytest.mark.asyncio
    async def test_rejects_busy_agent(self, mock_redis, mediator_config):
        """Test that busy agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(side_effect=[
            # First call: get_plan
            {
                b"plan_id": b"test-123",
                b"agent_id": b"andi",
                b"objective": b"Test objective",
                b"summary": b"Test plan",
                b"status": b"active",
                b"tasks": json.dumps([{
                    "id": 0,
                    "description": "desc",
                    "summary": "sum",
                    "context": "ctx",
                    "status": "in_progress",
                }]).encode(),
                b"current_task_id": b"0",
                b"created_at": datetime.now(timezone.utc).isoformat().encode(),
                b"updated_at": datetime.now(timezone.utc).isoformat().encode(),
            },
            # Second call: get_turn_request (busy)
            {
                b"status": b"in_progress",
                b"turn_id": b"current-turn",
                b"sequence_id": b"1",
            },
        ])

        result = await mediator._handle_update_command("andi", "guidance")

        assert result is False


class TestTryHandlePlannerCommand:
    """Test _try_handle_planner_command method."""

    @pytest.mark.asyncio
    async def test_handles_planner_on(self, mock_redis, mediator_config):
        """Test that @planner on command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@planner andi on",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_planner_off(self, mock_redis, mediator_config):
        """Test that @planner off command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@planner andi off",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is True
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_plan_command(self, mock_redis, mediator_config):
        """Test that @plan command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@plan andi = Build a feature",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_update_command(self, mock_redis, mediator_config):
        """Test that @update command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(side_effect=[
            # First call: get_plan
            {
                b"plan_id": b"test-123",
                b"agent_id": b"andi",
                b"objective": b"Test objective",
                b"summary": b"Test plan",
                b"status": b"active",
                b"tasks": json.dumps([{
                    "id": 0,
                    "description": "desc",
                    "summary": "sum",
                    "context": "ctx",
                    "status": "in_progress",
                }]).encode(),
                b"current_task_id": b"0",
                b"created_at": datetime.now(timezone.utc).isoformat().encode(),
                b"updated_at": datetime.now(timezone.utc).isoformat().encode(),
            },
            # Second call: get_turn_request
            {
                b"status": b"ready",
                b"turn_id": b"prev-turn",
                b"sequence_id": b"1",
            },
        ])
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@update andi = Task completed",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_ignores_non_system_events(self, mock_redis, mediator_config):
        """Test that non-SYSTEM events are not processed."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            actor_type="player",
            room_id="#123",
            content="@planner andi on",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is False
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_empty_content(self, mock_redis, mediator_config):
        """Test that empty content is not processed."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_for_non_planner_command(self, mock_redis, mediator_config):
        """Test that non-planner system events return False."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="Server maintenance in 5 minutes.",
        )

        result = await mediator._try_handle_planner_command(event)

        assert result is False


class TestProcessEventWithPlannerCommands:
    """Test that _process_event correctly handles planner commands."""

    @pytest.mark.asyncio
    async def test_planner_command_not_distributed(self, mock_redis, mediator_config):
        """Test that @planner commands are not distributed to agent streams."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        event_data = {
            "type": "system",
            "actor": "system",
            "actor_type": "system",
            "room_id": "#123",
            "content": "@planner andi on",
            "timestamp": "2026-01-01T12:00:00+00:00",
        }
        data = {b"data": json.dumps(event_data).encode()}

        await mediator._process_event("1704096000000-0", data)

        # Event should NOT be added to agent stream (xadd not called)
        mock_redis.xadd.assert_not_called()

        # But it should be marked as processed
        mock_redis.hset.assert_called()
        hset_calls = mock_redis.hset.call_args_list
        # Find the call that marks the event as processed
        processed_call = [c for c in hset_calls if c[0][0] == RedisKeys.EVENTS_PROCESSED]
        assert len(processed_call) == 1

    @pytest.mark.asyncio
    async def test_plan_command_not_distributed(self, mock_redis, mediator_config):
        """Test that @plan commands are not distributed to agent streams."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event_data = {
            "type": "system",
            "actor": "system",
            "actor_type": "system",
            "room_id": "#123",
            "content": "@plan andi = Build feature",
            "timestamp": "2026-01-01T12:00:00+00:00",
        }
        data = {b"data": json.dumps(event_data).encode()}

        await mediator._process_event("1704096000000-0", data)

        # Event should NOT be added to agent stream (xadd not called for distribution)
        mock_redis.xadd.assert_not_called()

        # But it should be marked as processed
        mock_redis.hset.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
