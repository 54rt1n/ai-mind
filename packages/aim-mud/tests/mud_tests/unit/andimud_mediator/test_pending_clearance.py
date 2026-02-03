# packages/aim-mud/tests/mud_tests/unit/andimud_mediator/test_pending_clearance.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for PENDING status clearance fix in andimud_mediator.

This test suite validates the critical CAS bug fix where:
1. The _clear_pending_for_echo() method now uses the ORIGINAL turn_id for CAS checks
   (before mutation), not the NEW turn_id after mutation
2. The _update_turn_request() wrapper now returns the bool success value instead of None

Philosophy: Real objects with mocked external services only.
- Mock: Redis (external service), _get_turn_request (dependency)
- Real: EventsMixin logic, MUDTurnRequest, MUDEvent
"""

import pytest
import uuid
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aim_mud_types import (
    ActorType,
    EventType,
    MUDEvent,
    TurnRequestStatus,
    TurnReason,
)
from aim_mud_types.models.coordination import MUDTurnRequest
from aim_mud_types.helper import _utc_now


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_turn_request():
    """Create a sample MUDTurnRequest in PENDING status with pending_action_ids."""
    return MUDTurnRequest(
        turn_id="original-turn-uuid",
        status=TurnRequestStatus.PENDING,
        reason=TurnReason.EVENTS,
        message="Waiting for action echo",
        heartbeat_at=_utc_now(),
        assigned_at=_utc_now(),
        sequence_id=100,
        attempt_count=0,
        metadata={
            "pending_action_ids": ["action-123", "action-456"]
        }
    )


@pytest.fixture
def sample_event():
    """Create a sample MUDEvent with an action_id."""
    return MUDEvent(
        event_id="event-123",
        event_type=EventType.SPEECH,
        actor="TestCharacter",
        actor_id="char-123",
        action_id="action-123",  # Matches one of the pending actions
        actor_type=ActorType.AI,
        room_id="room-1",
        room_name="Test Room",
        content="Test speech content",
        timestamp=_utc_now(),
        metadata={}
    )


@pytest.fixture
def mediator_with_mocks(mock_redis):
    """Create a minimal mediator-like object with EventsMixin for testing.

    Mocks:
    - Redis (external service)
    - _get_turn_request (dependency method - stubbed for testing)

    Real:
    - EventsMixin logic
    - _clear_pending_for_echo implementation
    - _update_turn_request wrapper
    """
    from andimud_mediator.mixins.events import EventsMixin

    class TestMediator(EventsMixin):
        """Minimal mediator for testing EventsMixin methods."""
        def __init__(self, redis):
            self.redis = redis
            self.running = True
            self.registered_agents = ["agent-1", "agent-2"]
            self._turn_index = 0
            self.last_event_id = "0"

            # Mock config
            self.config = MagicMock()
            self.config.event_poll_timeout = 1.0
            self.config.event_stream = "mud:events"
            self.config.events_processed_hash_max = 1000

        async def _get_turn_request(self, agent_id: str):
            """Stub method - will be mocked in tests."""
            return None

    mediator = TestMediator(redis=mock_redis)
    return mediator


# =============================================================================
# Test Cases
# =============================================================================

class TestPendingClearanceCAS:
    """Test suite for PENDING status clearance CAS fix."""

    @pytest.mark.asyncio
    async def test_cas_success_on_full_clear(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test CAS success when all pending actions are cleared.

        Verifies:
        - CAS check uses ORIGINAL turn_id (not mutated)
        - Success log appears
        - No warning log appears
        """
        caplog.set_level(logging.INFO)

        # Mock _get_turn_request to return our sample turn request
        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            # Mock _update_turn_request at the client level to return True (CAS success)
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=True)) as mock_update:
                # Clear only the matching action (leaving action-456)
                sample_turn_request.metadata["pending_action_ids"] = ["action-123"]

                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was called
                assert mock_update.called

                # Extract the call arguments
                call_args = mock_update.call_args
                agent_id = call_args[0][0]
                turn_request = call_args[0][1]
                expected_turn_id = call_args[1].get('expected_turn_id')

                # Verify correct arguments
                assert agent_id == "agent-1"
                assert expected_turn_id == "original-turn-uuid", \
                    "CAS check MUST use ORIGINAL turn_id, not mutated turn_id"

                # Verify the turn_request was transitioned to READY
                assert turn_request.status == TurnRequestStatus.READY
                assert turn_request.turn_id != "original-turn-uuid", \
                    "turn_id should be mutated to a new UUID"

                # Verify success log appears
                assert "Mediator cleared PENDING for agent-1 (all echoes received)" in caplog.text

                # Verify no warning log appears
                assert "CAS FAILED" not in caplog.text

    @pytest.mark.asyncio
    async def test_cas_failure_on_full_clear(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test CAS failure when clearing all pending actions.

        Verifies:
        - Warning log about CAS failure appears
        - Success log does NOT appear
        """
        caplog.set_level(logging.INFO)

        # Mock _get_turn_request to return our sample turn request
        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            # Mock _update_turn_request to return False (CAS failure)
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=False)):
                # Clear only the matching action
                sample_turn_request.metadata["pending_action_ids"] = ["action-123"]

                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify warning log appears
                assert "Mediator CAS FAILED when clearing PENDING for agent-1" in caplog.text

                # Verify success log does NOT appear
                assert "Mediator cleared PENDING for agent-1 (all echoes received)" not in caplog.text

    @pytest.mark.asyncio
    async def test_cas_success_on_partial_match(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test CAS success when partially clearing pending actions.

        Verifies:
        - Only matching action_id is removed
        - Metadata is updated with remaining action_ids
        - Debug log appears
        - No warning log appears
        """
        caplog.set_level(logging.DEBUG)

        # Mock _get_turn_request to return our sample turn request with 2 actions
        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            # Mock _update_turn_request to return True (CAS success)
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=True)) as mock_update:
                # Call the method under test (will match action-123, leave action-456)
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was called
                assert mock_update.called

                # Extract the call arguments
                call_args = mock_update.call_args
                turn_request = call_args[0][1]
                expected_turn_id = call_args[1].get('expected_turn_id')

                # Verify CAS used original turn_id
                assert expected_turn_id == "original-turn-uuid"

                # Verify status remains PENDING
                assert turn_request.status == TurnRequestStatus.PENDING

                # Verify metadata contains only the remaining action_id
                assert turn_request.metadata["pending_action_ids"] == ["action-456"]

                # Verify debug log appears
                assert "Mediator partial match: 1 remaining for agent-1" in caplog.text

                # Verify no warning log appears
                assert "CAS FAILED" not in caplog.text

    @pytest.mark.asyncio
    async def test_cas_failure_on_partial_match(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test CAS failure when partially clearing pending actions.

        Verifies:
        - Warning log about CAS failure appears
        - Debug log does NOT appear
        """
        caplog.set_level(logging.DEBUG)

        # Mock _get_turn_request to return our sample turn request with 2 actions
        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            # Mock _update_turn_request to return False (CAS failure)
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=False)):
                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify warning log appears
                assert "Mediator CAS FAILED on partial match for agent-1" in caplog.text

                # Verify debug log does NOT appear
                assert "Mediator partial match:" not in caplog.text

    @pytest.mark.asyncio
    async def test_original_turn_id_preserved_through_mutation(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test that original turn_id is preserved for CAS even after mutation.

        This is the core bug fix test. Verifies:
        - turn_request.turn_id is mutated to a NEW uuid
        - CAS check uses the ORIGINAL uuid (saved before mutation)
        - The NEW uuid is NOT used for the CAS check
        """
        caplog.set_level(logging.DEBUG)

        original_turn_id = sample_turn_request.turn_id
        assert original_turn_id == "original-turn-uuid"

        # Mock _get_turn_request to return our sample turn request
        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            # Mock _update_turn_request to capture arguments and return True
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=True)) as mock_update:
                # Clear only the matching action (triggers new_turn_id=True)
                sample_turn_request.metadata["pending_action_ids"] = ["action-123"]

                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was called
                assert mock_update.called

                # Extract the call arguments
                call_args = mock_update.call_args
                turn_request = call_args[0][1]
                expected_turn_id = call_args[1].get('expected_turn_id')

                # CRITICAL VERIFICATION: CAS used ORIGINAL turn_id
                assert expected_turn_id == "original-turn-uuid", \
                    "CAS MUST use original turn_id saved BEFORE mutation"

                # Verify turn_id was mutated to a NEW uuid
                assert turn_request.turn_id != "original-turn-uuid", \
                    "turn_id should be mutated to a new UUID by transition_turn_request"

                # Verify the NEW turn_id is a valid UUID
                try:
                    uuid.UUID(turn_request.turn_id)
                except ValueError:
                    pytest.fail(f"Mutated turn_id is not a valid UUID: {turn_request.turn_id}")

    @pytest.mark.asyncio
    async def test_update_turn_request_returns_bool(
        self,
        mediator_with_mocks,
        sample_turn_request
    ):
        """Test that _update_turn_request returns bool, not None.

        Verifies:
        - Returns True when CAS succeeds
        - Returns False when CAS fails
        - Does NOT return None
        """
        # Test CAS success
        with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=True)):
            result = await mediator_with_mocks._update_turn_request(
                agent_id="agent-1",
                turn_request=sample_turn_request,
                expected_turn_id="original-turn-uuid"
            )
            assert result is True, "_update_turn_request should return True on CAS success"

        # Test CAS failure
        with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=False)):
            result = await mediator_with_mocks._update_turn_request(
                agent_id="agent-1",
                turn_request=sample_turn_request,
                expected_turn_id="wrong-turn-uuid"
            )
            assert result is False, "_update_turn_request should return False on CAS failure"

    @pytest.mark.asyncio
    async def test_no_action_when_status_not_pending(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test that no action is taken when status is not PENDING."""
        caplog.set_level(logging.DEBUG)

        # Change status to READY (not PENDING)
        sample_turn_request.status = TurnRequestStatus.READY

        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock()) as mock_update:
                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was NOT called
                assert not mock_update.called, "Should not update when status is not PENDING"

    @pytest.mark.asyncio
    async def test_no_action_when_action_id_not_in_pending_set(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test that no action is taken when action_id doesn't match pending set."""
        caplog.set_level(logging.DEBUG)

        # Change event action_id to something not in pending_action_ids
        sample_event.action_id = "action-999"

        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock()) as mock_update:
                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was NOT called
                assert not mock_update.called, "Should not update when action_id doesn't match"

    @pytest.mark.asyncio
    async def test_handles_empty_pending_action_ids(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test that empty pending_action_ids is handled gracefully."""
        caplog.set_level(logging.DEBUG)

        # Set empty pending_action_ids
        sample_turn_request.metadata["pending_action_ids"] = []

        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock()) as mock_update:
                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was NOT called
                assert not mock_update.called, "Should not update when pending_action_ids is empty"

    @pytest.mark.asyncio
    async def test_handles_missing_pending_action_ids_key(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test that missing pending_action_ids key is handled gracefully."""
        caplog.set_level(logging.DEBUG)

        # Remove pending_action_ids key from metadata
        sample_turn_request.metadata = {}

        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            new=AsyncMock(return_value=sample_turn_request)
        ):
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock()) as mock_update:
                # Call the method under test
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1"]
                )

                # Verify _update_turn_request was NOT called
                assert not mock_update.called, "Should not update when pending_action_ids key is missing"

    @pytest.mark.asyncio
    async def test_processes_multiple_agents(
        self,
        mediator_with_mocks,
        sample_turn_request,
        sample_event,
        caplog
    ):
        """Test that multiple agents are processed independently."""
        caplog.set_level(logging.INFO)

        # Create a second turn request for agent-2
        turn_request_2 = MUDTurnRequest(
            turn_id="agent-2-turn-uuid",
            status=TurnRequestStatus.PENDING,
            reason=TurnReason.EVENTS,
            sequence_id=101,
            metadata={"pending_action_ids": ["action-123"]}
        )

        # Mock _get_turn_request to return different turn requests for each agent
        async def mock_get_turn(agent_id):
            if agent_id == "agent-1":
                return sample_turn_request
            elif agent_id == "agent-2":
                return turn_request_2
            return None

        with patch.object(
            mediator_with_mocks,
            '_get_turn_request',
            side_effect=mock_get_turn
        ):
            with patch('aim_mud_types.client.RedisMUDClient.update_turn_request', new=AsyncMock(return_value=True)) as mock_update:
                # Clear only the matching action for agent-1
                sample_turn_request.metadata["pending_action_ids"] = ["action-123"]

                # Call the method under test with both agents
                await mediator_with_mocks._clear_pending_for_echo(
                    event=sample_event,
                    agents_for_delivery=["agent-1", "agent-2"]
                )

                # Verify _update_turn_request was called twice (once per agent)
                assert mock_update.call_count == 2

                # Verify both agents got success logs
                assert "Mediator cleared PENDING for agent-1 (all echoes received)" in caplog.text
                assert "Mediator cleared PENDING for agent-2 (all echoes received)" in caplog.text
