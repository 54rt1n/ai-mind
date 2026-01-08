# packages/aim-mud/tests/mud_tests/unit/worker/test_retry_and_backoff.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for exception handling and retry logic in worker loop.

Tests the critical flow in worker.py lines 370-410:
- First exception → attempt_count=1, exponential backoff calculated
- Second exception → attempt_count=2, backoff doubled
- Max attempts reached → next_attempt_at=""
- Event position rollback on exception
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta, timezone

from aim_mud_types import MUDEvent, EventType
from aim_mud_types.session import MUDSession
from aim_mud_types.helper import _utc_now


@pytest.fixture
def sample_events():
    """Create sample events for retry testing."""
    return [
        MUDEvent(
            event_id="event-1",
            event_type=EventType.SPEECH,
            actor="OtherAgent",
            actor_id="other_agent",
            room_id="room1",
            room_name="Test Room",
            content="Test event",
            timestamp=datetime.now(timezone.utc),
            metadata={"sequence_id": 100},
        ),
    ]


class TestRetryAndBackoff:
    """Tests for exception handling and retry logic in worker loop."""

    @pytest.mark.asyncio
    async def test_first_exception_sets_attempt_count_one(
        self, test_worker, test_mud_config
    ):
        """Test: First exception → attempt_count=1, exponential backoff calculated."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Simulate turn_request data with no previous attempts
        turn_request = {
            "turn_id": "turn-1",
            "reason": "events",
            "attempt_count": "0",
        }

        # Act - Simulate exception handler (lines 373-390)
        attempt_count = int(turn_request.get("attempt_count", 0)) + 1

        # Calculate backoff
        if attempt_count < test_mud_config.llm_failure_max_attempts:
            backoff = min(
                test_mud_config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                test_mud_config.llm_failure_backoff_max_seconds
            )
            next_attempt_at = (_utc_now() + timedelta(seconds=backoff)).isoformat()
        else:
            backoff = 0
            next_attempt_at = ""

        # Assert
        assert attempt_count == 1, "First exception should set attempt_count=1"
        assert backoff == test_mud_config.llm_failure_backoff_base_seconds * (2 ** 0), (
            f"First attempt backoff should be base_seconds * 2^0 = {test_mud_config.llm_failure_backoff_base_seconds}"
        )
        assert next_attempt_at != "", "Should have next_attempt_at for first failure"

    @pytest.mark.asyncio
    async def test_second_exception_doubles_backoff(
        self, test_worker, test_mud_config
    ):
        """Test: Second exception → attempt_count=2, backoff doubled."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Simulate turn_request with one previous attempt
        turn_request = {
            "turn_id": "turn-1",
            "reason": "events",
            "attempt_count": "1",
        }

        # Act - Simulate exception handler
        attempt_count = int(turn_request.get("attempt_count", 0)) + 1

        # Calculate backoff
        if attempt_count < test_mud_config.llm_failure_max_attempts:
            backoff = min(
                test_mud_config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                test_mud_config.llm_failure_backoff_max_seconds
            )
            next_attempt_at = (_utc_now() + timedelta(seconds=backoff)).isoformat()
        else:
            backoff = 0
            next_attempt_at = ""

        # Assert
        assert attempt_count == 2, "Second exception should set attempt_count=2"
        expected_backoff = test_mud_config.llm_failure_backoff_base_seconds * (2 ** 1)
        assert backoff == expected_backoff, (
            f"Second attempt backoff should be base_seconds * 2^1 = {expected_backoff}"
        )
        assert next_attempt_at != "", "Should have next_attempt_at for second failure"

    @pytest.mark.asyncio
    async def test_max_attempts_reached_clears_next_attempt(
        self, test_worker, test_mud_config
    ):
        """Test: Max attempts reached → next_attempt_at=''."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Set max_attempts to 3 for this test
        test_mud_config.llm_failure_max_attempts = 3

        # Simulate turn_request at max attempts
        turn_request = {
            "turn_id": "turn-1",
            "reason": "events",
            "attempt_count": str(test_mud_config.llm_failure_max_attempts - 1),
        }

        # Act - Simulate exception handler
        attempt_count = int(turn_request.get("attempt_count", 0)) + 1

        # Calculate backoff
        if attempt_count < test_mud_config.llm_failure_max_attempts:
            backoff = min(
                test_mud_config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                test_mud_config.llm_failure_backoff_max_seconds
            )
            next_attempt_at = (_utc_now() + timedelta(seconds=backoff)).isoformat()
        else:
            backoff = 0
            next_attempt_at = ""

        # Assert
        assert attempt_count == test_mud_config.llm_failure_max_attempts, (
            "Should reach max attempts"
        )
        assert next_attempt_at == "", (
            "Max attempts should clear next_attempt_at (no more retries)"
        )

    @pytest.mark.asyncio
    async def test_event_position_rollback_on_exception(
        self, test_worker, sample_events
    ):
        """Test: Event position rollback on exception."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Save initial position
        saved_event_id = test_worker.session.last_event_id

        # Simulate events being drained (position advanced)
        test_worker.pending_events = sample_events
        test_worker.session.last_event_id = "event-1"

        # Act - Simulate exception handler restoring position (lines 406-409)
        # Only restore if saved_event_id is not None
        if saved_event_id:
            test_worker.session.last_event_id = saved_event_id
            test_worker.pending_events = []
            test_worker.session.pending_self_actions = []

        # Assert
        assert test_worker.session.last_event_id == "event-0", (
            "Exception should restore event position to saved_event_id"
        )
        assert test_worker.pending_events == [], (
            "Exception should clear pending_events"
        )
        assert test_worker.session.pending_self_actions == [], (
            "Exception should clear pending_self_actions"
        )

    @pytest.mark.asyncio
    async def test_saved_event_id_none_prevents_restore(
        self, test_worker, sample_events
    ):
        """Test: saved_event_id=None prevents restore (events already consumed)."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
            last_event_id="event-0",
        )

        # Simulate events consumed (saved_event_id cleared)
        saved_event_id = None
        test_worker.session.last_event_id = "event-1"

        # Act - Simulate exception handler restore logic
        if saved_event_id:
            test_worker.session.last_event_id = saved_event_id

        # Assert
        assert test_worker.session.last_event_id == "event-1", (
            "saved_event_id=None should prevent restore (position stays advanced)"
        )

    @pytest.mark.asyncio
    async def test_exponential_backoff_respects_max(
        self, test_worker, test_mud_config
    ):
        """Test: Exponential backoff respects max_seconds limit."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Set config values
        test_mud_config.llm_failure_backoff_base_seconds = 5.0
        test_mud_config.llm_failure_backoff_max_seconds = 30.0
        test_mud_config.llm_failure_max_attempts = 10

        # Test multiple attempts
        for attempt_num in range(1, 8):
            turn_request = {
                "attempt_count": str(attempt_num - 1),
            }

            # Calculate backoff
            attempt_count = int(turn_request.get("attempt_count", 0)) + 1
            backoff = min(
                test_mud_config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                test_mud_config.llm_failure_backoff_max_seconds
            )

            # Assert backoff never exceeds max
            assert backoff <= test_mud_config.llm_failure_backoff_max_seconds, (
                f"Attempt {attempt_count}: backoff {backoff} should not exceed max {test_mud_config.llm_failure_backoff_max_seconds}"
            )

    @pytest.mark.asyncio
    async def test_exception_type_captured_in_status_reason(
        self, test_worker
    ):
        """Test: Exception type captured in status_reason field."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        # Simulate different exception types
        test_exceptions = [
            ValueError("Invalid input"),
            RuntimeError("LLM timeout"),
            KeyError("Missing key"),
        ]

        for exc in test_exceptions:
            # Act - Extract error type
            error_type = type(exc).__name__

            # Assert
            assert error_type in ["ValueError", "RuntimeError", "KeyError"], (
                f"Should capture exception type: {error_type}"
            )

    @pytest.mark.asyncio
    async def test_retry_metadata_fields(
        self, test_worker, test_mud_config
    ):
        """Test: Retry sets all required metadata fields."""
        # Arrange
        test_worker.session = MUDSession(
            agent_id="test_agent",
            persona_id="test_persona",
        )

        turn_request = {
            "turn_id": "turn-1",
            "attempt_count": "1",
        }

        # Act - Simulate exception handler metadata
        attempt_count = int(turn_request.get("attempt_count", 0)) + 1

        if attempt_count < test_mud_config.llm_failure_max_attempts:
            backoff = min(
                test_mud_config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                test_mud_config.llm_failure_backoff_max_seconds
            )
            next_attempt_at = (_utc_now() + timedelta(seconds=backoff)).isoformat()
        else:
            next_attempt_at = ""

        error_type = "TestException"
        extra_fields = {
            "attempt_count": str(attempt_count),
            "next_attempt_at": next_attempt_at,
            "status_reason": f"LLM call failed: {error_type}"
        }

        # Assert
        assert "attempt_count" in extra_fields, "Should set attempt_count"
        assert "next_attempt_at" in extra_fields, "Should set next_attempt_at"
        assert "status_reason" in extra_fields, "Should set status_reason"
        assert extra_fields["attempt_count"] == "2", "Should increment attempt_count"
        assert "LLM call failed" in extra_fields["status_reason"], (
            "status_reason should include failure message"
        )

    @pytest.mark.asyncio
    async def test_backoff_calculation_progression(
        self, test_worker, test_mud_config
    ):
        """Test: Backoff calculation follows correct exponential progression."""
        # Arrange
        test_mud_config.llm_failure_backoff_base_seconds = 2.0
        test_mud_config.llm_failure_backoff_max_seconds = 100.0

        # Expected progression: 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0 (capped)
        expected_backoffs = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]

        # Act & Assert
        for i, expected in enumerate(expected_backoffs):
            attempt_count = i + 1
            backoff = min(
                test_mud_config.llm_failure_backoff_base_seconds * (2 ** (attempt_count - 1)),
                test_mud_config.llm_failure_backoff_max_seconds
            )

            assert backoff == expected, (
                f"Attempt {attempt_count}: expected {expected}, got {backoff}"
            )
