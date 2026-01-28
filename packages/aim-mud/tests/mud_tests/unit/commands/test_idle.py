# packages/aim-mud/tests/mud_tests/unit/commands/test_idle.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for IdleCommand index gap throttle logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta

from andimud_worker.commands.idle import IdleCommand
from aim_mud_types import TurnRequestStatus, TurnReason, MUDTurnRequest
from aim_mud_types.models.coordination import ThoughtState
from aim_mud_types.models.decision import DecisionType, DecisionResult


@pytest.fixture
def mock_worker():
    """Create a mock worker with necessary methods."""
    from andimud_worker.config import MUDConfig

    worker = MagicMock()
    worker.config = MUDConfig(agent_id="test-agent", persona_id="test-persona")

    # Async methods
    worker._check_agent_is_sleeping = AsyncMock(return_value=False)
    worker.ensure_turn_id_current = AsyncMock()
    worker._setup_turn_context = AsyncMock()
    worker.get_plan_guidance = MagicMock(return_value=None)
    worker.get_active_plan = MagicMock(return_value=None)
    worker._should_generate_new_thought = AsyncMock(return_value=False)
    worker.get_new_conversation_entries = AsyncMock(return_value=[])
    worker.collapse_consecutive_entries = MagicMock(return_value=[])
    worker.claim_idle_turn = AsyncMock(return_value="turn-123")
    worker._load_thought_content = AsyncMock()
    worker._clear_thought_content = AsyncMock()
    worker._is_idle_active = AsyncMock(return_value=False)
    worker._is_fresh_session = AsyncMock(return_value=False)
    worker.take_turn = AsyncMock(return_value=None)

    # Attributes
    worker._last_turn_error = None
    worker._last_emitted_action_ids = []

    return worker


@pytest.fixture
def turn_request():
    """Create a turn request fixture."""
    return MUDTurnRequest(
        turn_id="test-turn",
        sequence_id=1000,
        status=TurnRequestStatus.IN_PROGRESS,
        reason=TurnReason.IDLE,
    )


class TestIdleThrottleIndexGap:
    """Tests for index gap throttle logic in IdleCommand."""

    @pytest.mark.asyncio
    async def test_awake_turn_no_index_gap_skips_thought(self, mock_worker, turn_request):
        """Should skip thought when index gap is 0 (no new entries)."""
        # Mock thought with index 42, generated 10 minutes ago
        existing_thought = ThoughtState(
            agent_id="test-agent",
            content="<think>Previous thought</think>",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=10),  # 10 min ago
            last_conversation_index=42
        )

        # Mock the throttle to return False (gap is 0)
        mock_worker._should_generate_new_thought = AsyncMock(return_value=False)
        mock_worker._load_thought_content = AsyncMock()
        mock_worker._is_idle_active = AsyncMock(return_value=False)

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should NOT generate new thought (even though 10 min elapsed)
        mock_worker.claim_idle_turn.assert_not_called()
        mock_worker.get_new_conversation_entries.assert_not_called()

        # Should NOT load thought (throttle blocked, so no thought phase)
        mock_worker._load_thought_content.assert_not_called()

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "Agent awake" in result.message

    @pytest.mark.asyncio
    async def test_awake_turn_large_index_gap_regenerates(self, mock_worker, turn_request, mocker):
        """Should generate thought when index gap >= 5."""
        # Mock thought with index 42, just generated
        existing_thought = ThoughtState(
            agent_id="test-agent",
            content="<think>Previous thought</think>",
            created_at=datetime.now(timezone.utc),  # Just generated
            last_conversation_index=42
        )

        # Mock new entries (gap = 6)
        mock_entries = [MagicMock() for _ in range(6)]

        # Mock throttle returns True (gap >= 5)
        mock_worker._should_generate_new_thought = AsyncMock(return_value=True)
        mock_worker.get_new_conversation_entries = AsyncMock(return_value=mock_entries)
        mock_worker.collapse_consecutive_entries = MagicMock(return_value=mock_entries)
        mock_worker.claim_idle_turn = AsyncMock(return_value="turn-123")
        mock_worker._load_thought_content = AsyncMock()
        mock_worker._is_idle_active = AsyncMock(return_value=False)

        # Mock the _current_turn_entries attribute
        mock_worker._current_turn_entries = None

        # Mock ThinkingTurnProcessor to avoid executing real processor
        mock_thinking_processor = MagicMock()
        mock_thinking_processor.execute = AsyncMock()
        mocker.patch(
            'andimud_worker.commands.idle.ThinkingTurnProcessor',
            return_value=mock_thinking_processor
        )

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should generate thought (gap >= 5, regardless of time)
        mock_worker.claim_idle_turn.assert_awaited_once()
        mock_worker.get_new_conversation_entries.assert_awaited_once()

        # Should execute thinking processor
        mock_thinking_processor.execute.assert_awaited_once()

        # Should reload the thought after generation
        assert mock_worker._load_thought_content.await_count == 1

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "(thought generated)" in result.message

    @pytest.mark.asyncio
    async def test_awake_turn_throttle_passed_but_no_entries(self, mock_worker, turn_request):
        """Should skip thought generation when throttle passes but no new entries exist.

        This is a secondary safety check - throttle might say "yes" based on index,
        but actual entry fetch returns empty list (rare edge case).
        """
        # Mock throttle returns True
        mock_worker._should_generate_new_thought = AsyncMock(return_value=True)

        # But no entries actually exist
        mock_worker.get_new_conversation_entries = AsyncMock(return_value=[])
        mock_worker._load_thought_content = AsyncMock()
        mock_worker._is_idle_active = AsyncMock(return_value=False)

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should NOT claim turn (no entries to process)
        mock_worker.claim_idle_turn.assert_not_called()

        # Should reload existing thought instead
        mock_worker._load_thought_content.assert_awaited_once()

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE

    @pytest.mark.asyncio
    async def test_awake_turn_small_index_gap_time_elapsed(self, mock_worker, turn_request, mocker):
        """Should regenerate when index gap is small (1-4) but time elapsed >= 5 min."""
        # Mock thought with index 42, generated 6 minutes ago
        existing_thought = ThoughtState(
            agent_id="test-agent",
            content="<think>Previous thought</think>",
            created_at=datetime.now(timezone.utc) - timedelta(minutes=6),
            last_conversation_index=42
        )

        # Mock new entries (gap = 2, but time elapsed)
        mock_entries = [MagicMock(), MagicMock()]

        # Mock throttle returns True (time elapsed >= 5 min AND gap > 0)
        mock_worker._should_generate_new_thought = AsyncMock(return_value=True)
        mock_worker.get_new_conversation_entries = AsyncMock(return_value=mock_entries)
        mock_worker.collapse_consecutive_entries = MagicMock(return_value=mock_entries)
        mock_worker.claim_idle_turn = AsyncMock(return_value="turn-124")
        mock_worker._load_thought_content = AsyncMock()
        mock_worker._is_idle_active = AsyncMock(return_value=False)
        mock_worker._current_turn_entries = None

        # Mock ThinkingTurnProcessor to avoid executing real processor
        mock_thinking_processor = MagicMock()
        mock_thinking_processor.execute = AsyncMock()
        mocker.patch(
            'andimud_worker.commands.idle.ThinkingTurnProcessor',
            return_value=mock_thinking_processor
        )

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should generate thought (time elapsed AND gap > 0)
        mock_worker.claim_idle_turn.assert_awaited_once()
        mock_worker.get_new_conversation_entries.assert_awaited_once()
        mock_thinking_processor.execute.assert_awaited_once()

        assert result.complete is True
        assert "(thought generated)" in result.message

    @pytest.mark.asyncio
    async def test_awake_turn_with_active_plan(self, mock_worker, turn_request):
        """Should respect throttle even when plan is active."""
        from aim_mud_types.models.plan import AgentPlan, PlanTask, PlanStatus, TaskStatus

        # Mock active plan
        plan = AgentPlan(
            plan_id="test-plan",
            agent_id="test-agent",
            objective="Test objective",
            summary="Test plan",
            status=PlanStatus.ACTIVE,
            tasks=[
                PlanTask(
                    id=0,
                    description="Task 1",
                    summary="Task 1",
                    context="Context 1",
                    status=TaskStatus.IN_PROGRESS,
                )
            ],
            current_task_id=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_worker.get_active_plan = MagicMock(return_value=plan)
        mock_worker.get_plan_guidance = MagicMock(return_value="Execute task 1")

        # Mock throttle says no (index gap = 0)
        mock_worker._should_generate_new_thought = AsyncMock(return_value=False)
        mock_worker._load_thought_content = AsyncMock()
        mock_worker._is_idle_active = AsyncMock(return_value=False)

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should NOT generate thought (throttle blocked it)
        mock_worker.claim_idle_turn.assert_not_called()

        # Should NOT load thought (throttle blocked, no thought phase)
        mock_worker._load_thought_content.assert_not_called()

        assert result.complete is True

    @pytest.mark.asyncio
    async def test_awake_turn_idle_active_triggers_action(self, mock_worker, turn_request):
        """Should take action when idle_active is True."""
        # Mock throttle says no
        mock_worker._should_generate_new_thought = AsyncMock(return_value=False)
        mock_worker._load_thought_content = AsyncMock()

        # Mock idle_active returns True
        mock_worker._is_idle_active = AsyncMock(return_value=True)

        # Mock successful action
        mock_decision = DecisionResult(
            decision_type=DecisionType.SPEAK,
            args={"target_id": "test", "dialogue": "Hello"},
            thinking="",
            raw_response="",
            cleaned_response="",
        )
        mock_worker.take_turn = AsyncMock(return_value=mock_decision)
        mock_worker._last_emitted_action_ids = ["action-1"]

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should take action
        mock_worker.take_turn.assert_awaited_once()

        # Should clear thought after action
        mock_worker._clear_thought_content.assert_awaited_once()

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "SPEAK" in result.message
        assert result.emitted_action_ids == ["action-1"]

    @pytest.mark.asyncio
    async def test_awake_turn_dual_turn_flow(self, mock_worker, turn_request, mocker):
        """Should execute dual turn (thought + action) when conditions met."""
        # Mock new entries (gap >= 5)
        mock_entries = [MagicMock() for _ in range(6)]

        # Mock throttle returns True
        mock_worker._should_generate_new_thought = AsyncMock(return_value=True)
        mock_worker.get_new_conversation_entries = AsyncMock(return_value=mock_entries)
        mock_worker.collapse_consecutive_entries = MagicMock(return_value=mock_entries)
        mock_worker.claim_idle_turn = AsyncMock(return_value="turn-dual-123")
        mock_worker._load_thought_content = AsyncMock()
        mock_worker._current_turn_entries = None

        # Mock idle_active returns True
        mock_worker._is_idle_active = AsyncMock(return_value=True)

        # Mock successful action
        mock_decision = DecisionResult(
            decision_type=DecisionType.MOVE,
            args={"location": "north"},
            thinking="",
            raw_response="",
            cleaned_response="",
        )
        mock_worker.take_turn = AsyncMock(return_value=mock_decision)
        mock_worker._last_emitted_action_ids = ["action-2"]

        # Mock ThinkingTurnProcessor to avoid executing real processor
        mock_thinking_processor = MagicMock()
        mock_thinking_processor.execute = AsyncMock()
        mocker.patch(
            'andimud_worker.commands.idle.ThinkingTurnProcessor',
            return_value=mock_thinking_processor
        )

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        # Should generate thought
        mock_worker.claim_idle_turn.assert_awaited_once()
        mock_thinking_processor.execute.assert_awaited_once()

        # Should take action
        mock_worker.take_turn.assert_awaited_once()

        assert result.complete is True
        assert "(dual)" in result.message
        assert result.emitted_action_ids == ["action-2"]


class TestIdleCommandSleeping:
    """Tests for IdleCommand when agent is sleeping."""

    @pytest.mark.asyncio
    async def test_sleeping_agent_no_dream(self, mock_worker, turn_request):
        """Should return sleeping message when no dream active."""
        # Mock sleeping
        mock_worker._check_agent_is_sleeping = AsyncMock(return_value=True)
        mock_worker.load_dreaming_state = AsyncMock(return_value=None)
        mock_worker._decide_dream_action = AsyncMock(return_value=None)

        cmd = IdleCommand()
        result = await cmd.execute(mock_worker, turn_request=turn_request, events=[])

        assert result.complete is True
        assert result.status == TurnRequestStatus.DONE
        assert "sleeping" in result.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
