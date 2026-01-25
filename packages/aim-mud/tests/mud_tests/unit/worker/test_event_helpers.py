# packages/aim-mud/tests/mud_tests/unit/worker/test_event_helpers.py
# Tests for event helper methods
# Philosophy: Test real helper methods with mocked dependencies

import pytest
from unittest.mock import AsyncMock

from aim_mud_types import MUDSession


@pytest.fixture
def initialized_worker(test_worker):
    """Worker with initialized session."""
    test_worker.session = MUDSession(
        agent_id=test_worker.config.agent_id,
        persona_id=test_worker.config.persona_id,
        max_recent_turns=20,
    )
    test_worker.session.last_event_id = "0-0"
    return test_worker


# NOTE: TestRestoreEventPosition class has been removed.
# Event position restoration via _restore_event_position was removed from the codebase.
# Events are now consumed when they're pushed to conversation history, with each
# conversation entry tracking its own last_event_id.


class TestDrainWithSettle:
    """Tests for _drain_with_settle() helper method."""

    @pytest.mark.asyncio
    async def test_drain_returns_empty_list_when_no_events(self, initialized_worker, mock_redis):
        """Test _drain_with_settle returns empty list when drain_events returns no events."""
        from unittest.mock import AsyncMock

        # Mock Redis to return no events (external service)
        mock_redis.xinfo_stream.return_value = {"last-generated-id": "0"}
        mock_redis.xrange.return_value = []

        # Mock _update_agent_profile (internal helper, but we don't care about profile updates in this test)
        initialized_worker._update_agent_profile = AsyncMock()

        # Act - tests REAL _drain_with_settle() and REAL drain_events()
        result = await initialized_worker._drain_with_settle()

        # Assert
        assert result == []
        mock_redis.xinfo_stream.assert_called()  # Verify Redis was accessed

    @pytest.mark.asyncio
    async def test_drain_returns_events_from_single_batch(self, initialized_worker, mock_redis, monkeypatch):
        """Test _drain_with_settle returns events from single batch."""
        import asyncio
        import json
        from unittest.mock import AsyncMock
        from aim_mud_types import EventType, ActorType

        # Mock asyncio.sleep to avoid actual delays
        original_sleep = asyncio.sleep
        async def mock_sleep(seconds):
            await original_sleep(0)
        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Mock Redis stream responses (external service)
        mock_redis.xinfo_stream.return_value = {"last-generated-id": "1-1"}

        # First xrange call returns events, second returns empty
        event1_data = {
            "event_type": EventType.SPEECH.value,
            "actor": "Player1",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "Hello",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 1,
        }
        event2_data = {
            "event_type": EventType.EMOTE.value,
            "actor": "Player2",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "waves",
            "timestamp": "2026-01-08T12:00:01+00:00",
            "sequence_id": 2,
        }

        # First call returns events, second call returns empty (settle complete)
        mock_redis.xrange.side_effect = [
            [("1-0", {"data": json.dumps(event1_data)}), ("1-1", {"data": json.dumps(event2_data)})],
            []
        ]

        # Mock _update_agent_profile
        initialized_worker._update_agent_profile = AsyncMock()

        # Act - tests REAL _drain_with_settle() and REAL drain_events()
        result = await initialized_worker._drain_with_settle()

        # Assert - verify the REAL code processed events correctly
        assert len(result) == 2
        assert result[0].event_type == EventType.SPEECH
        assert result[0].actor == "Player1"
        assert result[1].event_type == EventType.EMOTE
        assert result[1].actor == "Player2"

    @pytest.mark.asyncio
    async def test_drain_accumulates_events_from_multiple_batches(self, initialized_worker, mock_redis, monkeypatch):
        """Test _drain_with_settle accumulates events across multiple drains."""
        import asyncio
        import json
        from unittest.mock import AsyncMock
        from aim_mud_types import EventType, ActorType

        # Mock asyncio.sleep to avoid actual delays
        original_sleep = asyncio.sleep
        async def mock_sleep(seconds):
            await original_sleep(0)
        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Mock Redis stream responses for multiple drains
        event1_data = {
            "event_type": EventType.SPEECH.value,
            "actor": "Player1",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "Hello",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 1,
        }
        event2_data = {
            "event_type": EventType.EMOTE.value,
            "actor": "Player2",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "waves",
            "timestamp": "2026-01-08T12:00:01+00:00",
            "sequence_id": 2,
        }
        event3_data = {
            "event_type": EventType.SPEECH.value,
            "actor": "Player1",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "How are you?",
            "timestamp": "2026-01-08T12:00:02+00:00",
            "sequence_id": 3,
        }

        # Simulate settle behavior: first drain gets 2 events, second drain gets 1 event, third gets none
        # Note: xinfo_stream is called on each drain, xrange follows
        mock_redis.xinfo_stream.side_effect = [
            {"last-generated-id": "1-1"},  # First drain
            {"last-generated-id": "1-2"},  # Second drain (new event arrived)
            {"last-generated-id": "1-2"},  # Third drain (no new events)
        ]

        mock_redis.xrange.side_effect = [
            # First drain: 2 events
            [("1-0", {"data": json.dumps(event1_data)}), ("1-1", {"data": json.dumps(event2_data)})],
            # Second drain: 1 new event
            [("1-2", {"data": json.dumps(event3_data)})],
            # Third drain: no new events (settle complete)
            []
        ]

        # Mock _update_agent_profile
        initialized_worker._update_agent_profile = AsyncMock()

        # Act - tests REAL _drain_with_settle() and REAL drain_events()
        result = await initialized_worker._drain_with_settle()

        # Assert - verify the REAL code accumulated all events
        assert len(result) == 3
        assert result[0].actor == "Player1"
        assert result[0].content == "Hello"
        assert result[1].actor == "Player2"
        assert result[1].content == "waves"
        assert result[2].actor == "Player1"
        assert result[2].content == "How are you?"

    @pytest.mark.asyncio
    async def test_drain_respects_max_sequence_id(self, initialized_worker, mock_redis, monkeypatch):
        """Test _drain_with_settle passes max_sequence_id to drain_events."""
        import asyncio
        import json
        from unittest.mock import AsyncMock
        from aim_mud_types import EventType, ActorType

        # Mock asyncio.sleep to avoid actual delays
        original_sleep = asyncio.sleep
        async def mock_sleep(seconds):
            await original_sleep(0)
        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Mock Redis with events that have different sequence_ids
        event1_data = {
            "event_type": EventType.SPEECH.value,
            "actor": "Player1",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "Before cutoff",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 10,
        }
        event2_data = {
            "event_type": EventType.SPEECH.value,
            "actor": "Player2",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "After cutoff",
            "timestamp": "2026-01-08T12:00:01+00:00",
            "sequence_id": 20,  # Should be filtered out
        }

        mock_redis.xinfo_stream.return_value = {"last-generated-id": "1-1"}
        mock_redis.xrange.side_effect = [
            [("1-0", {"data": json.dumps(event1_data)}), ("1-1", {"data": json.dumps(event2_data)})],
            []
        ]

        # Mock _update_agent_profile
        initialized_worker._update_agent_profile = AsyncMock()

        # Act - pass max_sequence_id=15 (should filter out event with seq=20)
        result = await initialized_worker._drain_with_settle(max_sequence_id=15)

        # Assert - verify REAL code filtered by sequence_id
        assert len(result) == 1
        assert result[0].metadata.get("sequence_id") == 10
        assert result[0].content == "Before cutoff"

    @pytest.mark.asyncio
    async def test_drain_with_settle_waits_between_batches(self, initialized_worker, mock_redis, monkeypatch):
        """Test _drain_with_settle waits settle_seconds between drains."""
        import asyncio
        import json
        from unittest.mock import AsyncMock
        from aim_mud_types import EventType, ActorType

        # Mock Redis to return events on first drain, empty on second
        event_data = {
            "event_type": EventType.SPEECH.value,
            "actor": "Player1",
            "actor_type": ActorType.PLAYER.value,
            "room_id": "#123",
            "content": "Hello",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 1,
        }

        # _drain_with_settle makes 3 drain_events calls:
        # 1. First drain returns events, sleeps settle_time
        # 2. Second drain returns empty, empty_after_events=1, sleeps final_settle_time
        # 3. Third drain returns empty, empty_after_events=2, breaks
        mock_redis.xinfo_stream.side_effect = [
            {"last-generated-id": "1-0"},
            {"last-generated-id": "1-0"},
            {"last-generated-id": "1-0"},
        ]
        mock_redis.xrange.side_effect = [
            [("1-0", {"data": json.dumps(event_data)})],
            [],
            [],
        ]

        # Mock _update_agent_profile
        initialized_worker._update_agent_profile = AsyncMock()

        # Mock asyncio.sleep to track if it's called
        sleep_called = []
        original_sleep = asyncio.sleep

        async def mock_sleep(seconds):
            sleep_called.append(seconds)
            # Don't actually sleep in tests
            await original_sleep(0)

        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Act - tests REAL _drain_with_settle() sleep logic
        await initialized_worker._drain_with_settle()

        # Assert - verify REAL code called sleep with correct duration
        # Implementation sleeps twice:
        # 1. After first batch (events): sleeps settle_time
        # 2. After second empty batch (empty_after_events=1): sleeps final_settle_time (settle_time // 3)
        assert len(sleep_called) == 2
        settle_time = initialized_worker.config.event_settle_seconds
        final_settle_time = settle_time // 3
        assert sleep_called[0] == settle_time
        assert sleep_called[1] == final_settle_time


class TestPendingActionIdTracking:
    """Tests for pending action_id tracking in the PENDING state.

    When a command emits actions, the main loop sets the turn to PENDING status
    with pending_action_ids in metadata. The loop then drains events looking for
    events with matching action_ids. When found, it pushes to conversation and
    transitions to READY.
    """

    @pytest.mark.asyncio
    async def test_drain_finds_matching_action_id(self, initialized_worker, mock_redis, monkeypatch):
        """Test that drain_events returns events with action_id in metadata."""
        import asyncio
        import json
        from aim_mud_types import EventType, ActorType

        # Mock asyncio.sleep to avoid actual delays
        original_sleep = asyncio.sleep
        async def mock_sleep(seconds):
            await original_sleep(0)
        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Setup self-action event with action_id
        action_id = "act_12345_abcd"
        event_data = {
            "event_type": EventType.EMOTE.value,
            "actor": "Andi",
            "actor_type": ActorType.AI.value,
            "room_id": "#123",
            "content": "settles into a comfortable position.",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 100,
            "is_self_action": True,
            "action_id": action_id,
        }

        mock_redis.xinfo_stream.return_value = {"last-generated-id": "100-0"}
        mock_redis.xrange.return_value = [
            ("100-0", {"data": json.dumps(event_data)})
        ]

        # Act
        result = await initialized_worker.drain_events(timeout=0)

        # Assert
        assert len(result) == 1
        # action_id is a field on MUDEvent, not in metadata
        assert result[0].action_id == action_id

    @pytest.mark.asyncio
    async def test_pending_action_ids_match_detection(self, initialized_worker, mock_redis, monkeypatch):
        """Test detection of matching action_ids in pending state."""
        import asyncio
        import json
        from aim_mud_types import EventType, ActorType

        # Mock asyncio.sleep to avoid actual delays
        original_sleep = asyncio.sleep
        async def mock_sleep(seconds):
            await original_sleep(0)
        monkeypatch.setattr(asyncio, 'sleep', mock_sleep)

        # Setup events - one matches, one doesn't
        action_id = "act_12345_abcd"
        pending_action_ids = [action_id, "act_67890_efgh"]

        event_data = {
            "event_type": EventType.EMOTE.value,
            "actor": "Andi",
            "actor_type": ActorType.AI.value,
            "room_id": "#123",
            "content": "stretches.",
            "timestamp": "2026-01-08T12:00:00+00:00",
            "sequence_id": 100,
            "is_self_action": True,
            "action_id": action_id,
        }

        mock_redis.xinfo_stream.return_value = {"last-generated-id": "100-0"}
        mock_redis.xrange.return_value = [
            ("100-0", {"data": json.dumps(event_data)})
        ]

        # Act
        events = await initialized_worker.drain_events(timeout=0)

        # Check if any event matches pending_action_ids
        # action_id is a field on MUDEvent, not in metadata
        matched = False
        for event in events:
            event_action_id = event.action_id
            if event_action_id and event_action_id in pending_action_ids:
                matched = True
                break

        # Assert
        assert matched is True
