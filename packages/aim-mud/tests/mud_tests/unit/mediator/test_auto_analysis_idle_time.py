"""Unit tests for stream-based idle time calculation."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import RedisKeys


@pytest.fixture
def mediator_config():
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
        auto_analysis_enabled=True,
        auto_analysis_idle_seconds=60,
        auto_analysis_cooldown_seconds=10,
    )


@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    return redis


class TestStreamBasedIdleDetection:
    """Test stream-based idle time calculation."""

    @pytest.mark.asyncio
    async def test_calculates_idle_from_event_stream(self, mock_redis, mediator_config):
        """Test that idle time is calculated from event stream timestamp."""
        # Mock event stream with message 5 minutes ago
        five_min_ago_ms = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(side_effect=lambda key: {
            "last-generated-id": f"{five_min_ago_ms}-0".encode() if key == RedisKeys.MUD_EVENTS else b"0-0"
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should trigger (5 min > 60s threshold)
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_recent_activity_prevents_trigger(self, mock_redis, mediator_config):
        """Test that recent activity prevents auto-analysis."""
        # Mock action stream with message 30 seconds ago
        thirty_sec_ago_ms = int((datetime.now(timezone.utc) - timedelta(seconds=30)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(side_effect=lambda key: {
            "last-generated-id": f"{thirty_sec_ago_ms}-0".encode()
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should not trigger (30s < 60s threshold)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_most_recent_activity(self, mock_redis, mediator_config):
        """Test that most recent timestamp from either stream is used."""
        now_utc = datetime.now(timezone.utc)
        old_event_ms = int((now_utc - timedelta(minutes=10)).timestamp() * 1000)
        recent_action_ms = int((now_utc - timedelta(seconds=30)).timestamp() * 1000)

        mock_redis.xinfo_stream = AsyncMock(side_effect=lambda key: {
            "last-generated-id": (
                f"{old_event_ms}-0".encode() if key == RedisKeys.MUD_EVENTS
                else f"{recent_action_ms}-0".encode()
            )
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = now_utc - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should not trigger - uses recent action (30s), not old event (10min)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_empty_streams(self, mock_redis, mediator_config):
        """Test that empty streams don't trigger auto-analysis."""
        mock_redis.xinfo_stream = AsyncMock(side_effect=lambda key: {
            "last-generated-id": b"0-0"
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should not trigger - no activity yet
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_stream_query_failure(self, mock_redis, mediator_config):
        """Test that stream query errors are handled gracefully."""
        import redis.exceptions

        # Simulate stream doesn't exist (ResponseError)
        mock_redis.xinfo_stream = AsyncMock(
            side_effect=redis.exceptions.ResponseError("no such key")
        )

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should not trigger - stream doesn't exist yet
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_bytes_encoding(self, mock_redis, mediator_config):
        """Test that bytes-encoded stream IDs are decoded correctly."""
        thirty_sec_ago_ms = int((datetime.now(timezone.utc) - timedelta(seconds=30)).timestamp() * 1000)

        # Return bytes-encoded stream ID (as Redis would)
        mock_redis.xinfo_stream = AsyncMock(side_effect=lambda key: {
            b"last-generated-id": f"{thirty_sec_ago_ms}-0".encode()
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should not trigger - 30s is below 60s threshold
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_one_stream_empty(self, mock_redis, mediator_config):
        """Test that idle time uses the non-empty stream when one is empty."""
        five_min_ago_ms = int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp() * 1000)

        # Events stream has data, actions stream is empty
        mock_redis.xinfo_stream = AsyncMock(side_effect=lambda key: {
            "last-generated-id": (
                f"{five_min_ago_ms}-0".encode() if key == RedisKeys.MUD_EVENTS
                else b"0-0"
            )
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should trigger - uses event timestamp (5 min > 60s)
                mock_scan.assert_called_once()


class TestSleepingAgentsIdleDetection:
    """Test idle detection for sleeping agents when streams don't exist or are empty.

    These tests verify the fix for the case where all agents are sleeping and
    streams don't exist (e.g., after system reset). In this scenario, sleeping
    agents should still receive analysis tasks overnight.
    """

    @pytest.mark.asyncio
    async def test_auto_analysis_triggers_when_streams_missing_and_all_sleeping(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis triggers when streams don't exist but all agents sleeping."""
        import redis.exceptions

        # Simulate stream doesn't exist (ResponseError)
        mock_redis.xinfo_stream = AsyncMock(
            side_effect=redis.exceptions.ResponseError("no such key")
        )

        # Mock sleeping agent - need to mock hget for get_agent_is_sleeping
        # Key format is "agent:{agent_id}" (from RedisKeys.agent_profile)
        def hget_side_effect(key, field):
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str.startswith("agent:") and field == "is_sleeping":
                return b"true"
            return None

        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', new_callable=AsyncMock, return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should trigger - all agents sleeping with no streams = infinitely idle
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_analysis_skips_when_streams_missing_and_some_awake(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis skips when streams don't exist and some agents awake."""
        import redis.exceptions

        # Simulate stream doesn't exist (ResponseError)
        mock_redis.xinfo_stream = AsyncMock(
            side_effect=redis.exceptions.ResponseError("no such key")
        )

        # Mock awake agent - hget returns None for is_sleeping (not sleeping)
        def hget_side_effect(key, field):
            return None  # Not sleeping

        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        # Mock turn_request as ready
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', new_callable=AsyncMock, return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should NOT trigger - awake agents should create activity
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_analysis_triggers_when_streams_empty_and_all_sleeping(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis triggers when streams are empty but all agents sleeping."""
        # Streams exist but are empty (0-0)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": b"0-0"
        })

        # Mock sleeping agent - need to mock hget for get_agent_is_sleeping
        # Key format is "agent:{agent_id}" (from RedisKeys.agent_profile)
        def hget_side_effect(key, field):
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str.startswith("agent:") and field == "is_sleeping":
                return b"true"
            return None

        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', new_callable=AsyncMock, return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should trigger - all agents sleeping with empty streams = infinitely idle
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_analysis_skips_when_streams_empty_and_some_awake(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis skips when streams are empty and some agents awake."""
        # Streams exist but are empty (0-0)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": b"0-0"
        })

        # Mock awake agent - hget returns None for is_sleeping (not sleeping)
        def hget_side_effect(key, field):
            return None  # Not sleeping

        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        # Mock turn_request as ready
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', new_callable=AsyncMock, return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should NOT trigger - awake agents should create activity
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_agents_with_streams_missing(
        self, mock_redis, mediator_config
    ):
        """Test with mix of sleeping and awake agents when streams are missing."""
        import redis.exceptions

        # Simulate stream doesn't exist (ResponseError)
        mock_redis.xinfo_stream = AsyncMock(
            side_effect=redis.exceptions.ResponseError("no such key")
        )

        # andi is sleeping, val is awake
        # Key format is "agent:{agent_id}" (from RedisKeys.agent_profile)
        def hget_side_effect(key, field):
            key_str = key.decode() if isinstance(key, bytes) else key
            if field == "is_sleeping":
                if key_str == "agent:andi":
                    return b"true"  # andi is sleeping
                elif key_str == "agent:val":
                    return None  # val is not sleeping
            return None

        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        # Mock turn_request as ready (for val who is not sleeping)
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.register_agent("val")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', new_callable=AsyncMock, return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should NOT trigger - val is awake, should create activity
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_sleeping_agents_trigger_with_multiple_agents(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis triggers when ALL multiple agents are sleeping."""
        import redis.exceptions

        # Simulate stream doesn't exist (ResponseError)
        mock_redis.xinfo_stream = AsyncMock(
            side_effect=redis.exceptions.ResponseError("no such key")
        )

        # All agents are sleeping - mock hget for get_agent_is_sleeping
        # Key format is "agent:{agent_id}" (from RedisKeys.agent_profile)
        def hget_side_effect(key, field):
            key_str = key.decode() if isinstance(key, bytes) else key
            if field == "is_sleeping" and key_str.startswith("agent:"):
                return b"true"  # All agents sleeping
            return None

        mock_redis.hget = AsyncMock(side_effect=hget_side_effect)

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.register_agent("val")
        mediator.register_agent("nova")
        mediator._last_auto_analysis_check = datetime.now(timezone.utc) - timedelta(seconds=120)

        with patch.object(mediator, '_scan_for_unanalyzed_conversations') as mock_scan:
            with patch.object(mediator, '_is_paused', new_callable=AsyncMock, return_value=False):
                await mediator._check_auto_analysis_trigger()

                # Should trigger - all 3 agents sleeping = infinitely idle
                mock_scan.assert_called_once()
