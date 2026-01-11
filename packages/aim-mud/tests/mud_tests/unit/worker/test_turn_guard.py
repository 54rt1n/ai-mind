# packages/aim-mud/tests/mud_tests/unit/worker/test_turn_guard.py
# Tests for turn guard functionality (_should_process_turn)
# Philosophy: Test the guard logic with mocked Redis responses

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from aim_mud_types.coordination import MUDTurnRequest, TurnRequestStatus
from aim_mud_types.helper import _utc_now
from andimud_worker.config import MUDConfig
from andimud_worker.mixins.datastore.turn_request import TurnRequestMixin


class TestTurnGuard:
    """Tests for _should_process_turn() turn guard method."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for turn guard tests."""
        redis = MagicMock()
        redis.scan = AsyncMock()
        redis.hgetall = AsyncMock()
        redis.pipeline = MagicMock()
        return redis

    @pytest.fixture
    def worker(self, mock_redis):
        """Worker instance with turn guard method bound."""
        worker = MagicMock()
        worker.redis = mock_redis
        worker.config = MUDConfig(agent_id="andi", persona_id="andi")
        # Bind the method we're testing
        worker._should_process_turn = TurnRequestMixin._should_process_turn.__get__(
            worker, type(worker)
        )
        return worker

    def create_turn_request(
        self,
        turn_id: str = "turn-123",
        status: TurnRequestStatus = TurnRequestStatus.IN_PROGRESS,
        assigned_at: datetime = None,
        heartbeat_at: datetime = None,
        sequence_id: int = 1,
    ) -> MUDTurnRequest:
        """Helper to create test turn request."""
        if assigned_at is None:
            assigned_at = _utc_now()
        if heartbeat_at is None:
            heartbeat_at = _utc_now()

        return MUDTurnRequest(
            turn_id=turn_id,
            status=status,
            assigned_at=assigned_at,
            heartbeat_at=heartbeat_at,
            sequence_id=sequence_id,
            reason="events",
        )

    def setup_redis_scan(
        self, mock_redis, keys: list[str]
    ):
        """Setup mock Redis SCAN to return agent keys."""
        # Convert keys to bytes (Redis returns bytes)
        byte_keys = [k.encode() for k in keys]
        mock_redis.scan.return_value = (0, byte_keys)

    def setup_redis_pipeline(
        self, mock_redis, turn_data_list: list[dict]
    ):
        """Setup mock Redis pipeline to return turn request data."""
        pipeline_mock = MagicMock()
        pipeline_mock.hgetall = MagicMock()

        # Convert data to bytes (Redis returns bytes)
        results = []
        for data in turn_data_list:
            byte_data = {
                k.encode(): v.encode() if isinstance(v, str) else v
                for k, v in data.items()
            }
            results.append(byte_data)

        pipeline_mock.execute = AsyncMock(return_value=results)
        mock_redis.pipeline.return_value = pipeline_mock

    @pytest.mark.asyncio
    async def test_should_process_single_agent(self, worker, mock_redis):
        """Test that single agent always processes (no competition)."""
        # Arrange
        turn_request = self.create_turn_request(turn_id="turn-andi-1")
        self.setup_redis_scan(mock_redis, ["agent:andi:turn_request"])
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": "2026-01-09T12:00:00+00:00",
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                }
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_should_process_oldest_turn(self, worker, mock_redis):
        """Test that oldest turn is allowed to process."""
        # Arrange
        now = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        old_time = now - timedelta(minutes=5)
        new_time = now

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=old_time,
            heartbeat_at=old_time,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": old_time.isoformat(),
                    "heartbeat_at": old_time.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": new_time.isoformat(),
                    "heartbeat_at": new_time.isoformat(),
                },
            ],
        )

        # Act - mock _utc_now to control current time
        with patch('andimud_worker.mixins.datastore.turn_request._utc_now', return_value=now):
            result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_should_not_process_newer_turn(self, worker, mock_redis):
        """Test that newer turn waits for older turn."""
        # Arrange
        now = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        old_time = now - timedelta(minutes=5)
        new_time = now

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=new_time,  # We're newer
            heartbeat_at=new_time,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": new_time.isoformat(),
                    "heartbeat_at": new_time.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": old_time.isoformat(),  # Other is older
                    "heartbeat_at": old_time.isoformat(),
                },
            ],
        )

        # Act - mock _utc_now to control current time
        with patch('andimud_worker.mixins.datastore.turn_request._utc_now', return_value=now):
            result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_ignores_stale_heartbeat(self, worker, mock_redis):
        """Test that stale heartbeats are ignored (>300s old)."""
        # Arrange
        now = _utc_now()
        our_time = now
        stale_time = now - timedelta(seconds=350)  # 350s ago, > 300s threshold

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=our_time,
            heartbeat_at=our_time,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": our_time.isoformat(),
                    "heartbeat_at": our_time.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": stale_time.isoformat(),  # Older assigned_at
                    "heartbeat_at": stale_time.isoformat(),  # But stale heartbeat
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Should ignore stale turn and proceed

    @pytest.mark.asyncio
    async def test_ignores_non_active_status(self, worker, mock_redis):
        """Test that non-active statuses (DONE, FAIL, etc.) are ignored."""
        # Arrange
        now = _utc_now()
        old_time = now - timedelta(minutes=5)

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.DONE.value,  # Not active
                    "assigned_at": old_time.isoformat(),  # Even though older
                    "heartbeat_at": old_time.isoformat(),
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Should ignore DONE status

    @pytest.mark.asyncio
    async def test_timestamp_tie_breaker(self, worker, mock_redis):
        """Test that same assigned_at uses lexicographic turn_id comparison."""
        # Arrange
        now = _utc_now()

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",  # Alphabetically before turn-val-1
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",  # Same timestamp
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # "turn-andi-1" < "turn-val-1" lexicographically

    @pytest.mark.asyncio
    async def test_timestamp_tie_breaker_loses(self, worker, mock_redis):
        """Test that tie-breaker can make us wait."""
        # Arrange
        now = _utc_now()

        turn_request = self.create_turn_request(
            turn_id="turn-val-1",  # Alphabetically after turn-andi-1
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",  # Same timestamp, but alphabetically first
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is False  # "turn-val-1" > "turn-andi-1" lexicographically

    @pytest.mark.asyncio
    async def test_redis_error_fails_safe(self, worker, mock_redis):
        """Test that Redis errors fail safe (return True to avoid deadlock)."""
        # Arrange
        turn_request = self.create_turn_request()
        mock_redis.scan.side_effect = Exception("Redis connection error")

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Fail safe to avoid deadlock

    @pytest.mark.asyncio
    async def test_empty_redis_returns_true(self, worker, mock_redis):
        """Test that empty Redis (no agent keys) returns True."""
        # Arrange
        turn_request = self.create_turn_request()
        self.setup_redis_scan(mock_redis, [])  # No keys

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True

    @pytest.mark.asyncio
    async def test_handles_missing_hash_data(self, worker, mock_redis):
        """Test handling of missing/empty hash data."""
        # Arrange
        turn_request = self.create_turn_request(turn_id="turn-andi-1")
        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )

        # First hash has data, second is empty
        pipeline_mock = MagicMock()
        pipeline_mock.hgetall = MagicMock()
        pipeline_mock.execute = AsyncMock(return_value=[
            {
                b"turn_id": b"turn-andi-1",
                b"status": b"in_progress",
                b"assigned_at": b"2026-01-09T12:00:00+00:00",
                b"heartbeat_at": b"2026-01-09T12:00:30+00:00",
            },
            {},  # Empty hash
        ])
        mock_redis.pipeline.return_value = pipeline_mock

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Only andi's turn exists, should process

    @pytest.mark.asyncio
    async def test_handles_corrupted_timestamp(self, worker, mock_redis):
        """Test handling of corrupted timestamp data."""
        # Arrange
        turn_request = self.create_turn_request(turn_id="turn-andi-1")
        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": "2026-01-09T12:00:00+00:00",
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": "INVALID_TIMESTAMP",  # Corrupted
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Should ignore corrupted entry and process

    @pytest.mark.asyncio
    async def test_handles_missing_assigned_at(self, worker, mock_redis):
        """Test handling of missing assigned_at field."""
        # Arrange
        turn_request = self.create_turn_request(turn_id="turn-andi-1")
        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": "2026-01-09T12:00:00+00:00",
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    # Missing assigned_at
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Should ignore entry without assigned_at

    @pytest.mark.asyncio
    async def test_handles_assigned_status(self, worker, mock_redis):
        """Test that ASSIGNED status is also considered active."""
        # Arrange
        now = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        old_time = now - timedelta(minutes=5)

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.ASSIGNED.value,  # Also active
                    "assigned_at": old_time.isoformat(),
                    "heartbeat_at": old_time.isoformat(),
                },
            ],
        )

        # Act - mock _utc_now to control current time
        with patch('andimud_worker.mixins.datastore.turn_request._utc_now', return_value=now):
            result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is False  # Should wait for older ASSIGNED turn

    @pytest.mark.asyncio
    async def test_multiple_stale_heartbeats(self, worker, mock_redis):
        """Test with multiple agents having stale heartbeats."""
        # Arrange
        now = _utc_now()
        stale_time = now - timedelta(seconds=400)

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            [
                "agent:andi:turn_request",
                "agent:val:turn_request",
                "agent:nova:turn_request",
            ]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": stale_time.isoformat(),
                    "heartbeat_at": stale_time.isoformat(),  # Stale
                },
                {
                    "turn_id": "turn-nova-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": stale_time.isoformat(),
                    "heartbeat_at": stale_time.isoformat(),  # Also stale
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # All others are stale, we should process

    @pytest.mark.asyncio
    async def test_no_active_turns_found(self, worker, mock_redis):
        """Test when no active turns are found (all DONE/FAIL)."""
        # Arrange
        turn_request = self.create_turn_request(turn_id="turn-andi-1")
        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.DONE.value,
                    "assigned_at": "2026-01-09T12:00:00+00:00",
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.FAIL.value,
                    "assigned_at": "2026-01-09T11:55:00+00:00",
                    "heartbeat_at": "2026-01-09T12:00:30+00:00",
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # No active turns, should proceed

    @pytest.mark.asyncio
    async def test_ignores_execute_status(self, worker, mock_redis):
        """Test that EXECUTE status is ignored by turn guard."""
        # Arrange
        now = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        old_time = now - timedelta(minutes=5)

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.EXECUTE.value,  # Should be ignored
                    "assigned_at": old_time.isoformat(),  # Even though older
                    "heartbeat_at": old_time.isoformat(),
                },
            ],
        )

        # Act - mock _utc_now to control current time
        with patch('andimud_worker.mixins.datastore.turn_request._utc_now', return_value=now):
            result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Should ignore EXECUTE status

    @pytest.mark.asyncio
    async def test_ignores_executing_status(self, worker, mock_redis):
        """Test that EXECUTING status is ignored by turn guard."""
        # Arrange
        now = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        old_time = now - timedelta(minutes=5)

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=now,
            heartbeat_at=now,
        )

        self.setup_redis_scan(
            mock_redis,
            ["agent:andi:turn_request", "agent:val:turn_request"]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": now.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.EXECUTING.value,  # Should be ignored
                    "assigned_at": old_time.isoformat(),  # Even though older
                    "heartbeat_at": old_time.isoformat(),
                },
            ],
        )

        # Act - mock _utc_now to control current time
        with patch('andimud_worker.mixins.datastore.turn_request._utc_now', return_value=now):
            result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # Should ignore EXECUTING status

    @pytest.mark.asyncio
    async def test_oldest_assigned_selected_when_execute_present(self, worker, mock_redis):
        """Test that oldest ASSIGNED turn is selected even with EXECUTE turns present."""
        # Arrange
        now = _utc_now()
        old_assigned_time = now - timedelta(minutes=10)
        execute_time = now - timedelta(minutes=15)  # Older than assigned, but should be ignored
        new_assigned_time = now

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=old_assigned_time,  # Oldest ASSIGNED
            heartbeat_at=now,  # Recent heartbeat to avoid stale detection
        )

        self.setup_redis_scan(
            mock_redis,
            [
                "agent:andi:turn_request",
                "agent:val:turn_request",
                "agent:nova:turn_request",
            ]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.ASSIGNED.value,
                    "assigned_at": old_assigned_time.isoformat(),
                    "heartbeat_at": now.isoformat(),  # Recent heartbeat
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.EXECUTE.value,  # Oldest overall, but ignored
                    "assigned_at": execute_time.isoformat(),
                    "heartbeat_at": now.isoformat(),  # Recent heartbeat
                },
                {
                    "turn_id": "turn-nova-1",
                    "status": TurnRequestStatus.ASSIGNED.value,
                    "assigned_at": new_assigned_time.isoformat(),
                    "heartbeat_at": now.isoformat(),  # Recent heartbeat
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # We are the oldest ASSIGNED turn

    @pytest.mark.asyncio
    async def test_waits_for_older_assigned_when_execute_present(self, worker, mock_redis):
        """Test that we wait for older ASSIGNED turn even with EXECUTE turns present."""
        # Arrange
        now = _utc_now()
        old_assigned_time = now - timedelta(minutes=10)
        execute_time = now - timedelta(minutes=15)
        new_assigned_time = now  # We're newer

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=new_assigned_time,  # We're newer
            heartbeat_at=now,  # Recent heartbeat
        )

        self.setup_redis_scan(
            mock_redis,
            [
                "agent:andi:turn_request",
                "agent:val:turn_request",
                "agent:nova:turn_request",
            ]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.ASSIGNED.value,
                    "assigned_at": new_assigned_time.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.EXECUTE.value,  # Should be ignored
                    "assigned_at": execute_time.isoformat(),
                    "heartbeat_at": now.isoformat(),
                },
                {
                    "turn_id": "turn-nova-1",
                    "status": TurnRequestStatus.ASSIGNED.value,  # Older ASSIGNED
                    "assigned_at": old_assigned_time.isoformat(),
                    "heartbeat_at": now.isoformat(),  # Recent heartbeat
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is False  # Should wait for older ASSIGNED turn (nova)

    @pytest.mark.asyncio
    async def test_multiple_execute_statuses_all_ignored(self, worker, mock_redis):
        """Test that multiple EXECUTE/EXECUTING statuses are all ignored."""
        # Arrange
        now = datetime(2026, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
        execute_time_1 = now - timedelta(minutes=15)
        execute_time_2 = now - timedelta(minutes=10)
        our_time = now

        turn_request = self.create_turn_request(
            turn_id="turn-andi-1",
            assigned_at=our_time,
            heartbeat_at=our_time,
        )

        self.setup_redis_scan(
            mock_redis,
            [
                "agent:andi:turn_request",
                "agent:val:turn_request",
                "agent:nova:turn_request",
            ]
        )
        self.setup_redis_pipeline(
            mock_redis,
            [
                {
                    "turn_id": "turn-andi-1",
                    "status": TurnRequestStatus.IN_PROGRESS.value,
                    "assigned_at": our_time.isoformat(),
                    "heartbeat_at": our_time.isoformat(),
                },
                {
                    "turn_id": "turn-val-1",
                    "status": TurnRequestStatus.EXECUTE.value,
                    "assigned_at": execute_time_1.isoformat(),
                    "heartbeat_at": execute_time_1.isoformat(),
                },
                {
                    "turn_id": "turn-nova-1",
                    "status": TurnRequestStatus.EXECUTING.value,
                    "assigned_at": execute_time_2.isoformat(),
                    "heartbeat_at": execute_time_2.isoformat(),
                },
            ],
        )

        # Act
        result = await worker._should_process_turn(turn_request)

        # Assert
        assert result is True  # All EXECUTE/EXECUTING ignored, we're the only active turn
