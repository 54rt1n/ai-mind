# packages/aim-mud/tests/unit/mediator/test_auto_analysis.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for semi-autonomous analysis mode.

Tests the auto-analysis trigger logic that automatically initiates conversation
analysis when the system is idle for a configured duration.
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import RedisKeys, TurnRequestStatus
from aim_mud_types.helper import _utc_now


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration with auto-analysis enabled."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
        auto_analysis_enabled=True,
        auto_analysis_idle_seconds=300,  # 5 minutes
        auto_analysis_cooldown_seconds=60,  # 1 minute
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.expire = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)  # CAS success
    redis.incr = AsyncMock(side_effect=lambda key: 1)  # Sequence counter
    # Default stream info: empty streams (no activity)
    redis.xinfo_stream = AsyncMock(return_value={"last-generated-id": b"0-0"})
    return redis


@pytest.fixture
def ready_turn_request():
    """Fixture for a ready turn request with completed_at timestamp."""
    # Completed 400 seconds ago (past idle threshold of 300)
    completed_time = _utc_now() - timedelta(seconds=400)
    return {
        b"status": b"ready",
        b"turn_id": b"prev-turn-123",
        b"reason": b"events",
        b"heartbeat_at": _utc_now().isoformat().encode(),
        b"sequence_id": b"1",
        b"completed_at": completed_time.isoformat().encode(),
    }


@pytest.fixture
def sample_conversation_report():
    """Fixture for a conversation report with unanalyzed conversations."""
    return {
        "conv_001": {
            "mud-world": 5,
            "mud-agent": 3,
            "analysis": 0,
            "timestamp_max": "2026-01-01T10:00:00+00:00",
        },
        "conv_002": {
            "mud-world": 10,
            "mud-agent": 8,
            "analysis": 0,
            "timestamp_max": "2026-01-01T09:00:00+00:00",  # Older
        },
        "conv_003": {
            "mud-world": 2,
            "mud-agent": 1,
            "analysis": 1,  # Already analyzed
            "timestamp_max": "2026-01-01T11:00:00+00:00",
        },
    }


# =============================================================================
# 1. IDLE DETECTION TESTS
# =============================================================================

class TestIdleDetection:
    """Test idle detection logic for auto-analysis trigger."""

    @pytest.mark.asyncio
    async def test_auto_analysis_disabled_by_config(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis does not trigger when disabled in config."""
        mediator_config.auto_analysis_enabled = False
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        await mediator._check_auto_analysis_trigger()

        # Should not check Redis or scan for conversations
        mock_redis.hgetall.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_analysis_respects_mediator_pause(
        self, mock_redis, mediator_config
    ):
        """Test that auto-analysis does not trigger when mediator is paused."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock mediator pause state
        with patch.object(mediator, '_is_paused', return_value=True):
            await mediator._check_auto_analysis_trigger()

        # Should not proceed with idle detection
        mock_redis.hgetall.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_analysis_respects_sleeping_agents(
        self, mock_redis, mediator_config, ready_turn_request
    ):
        """Test that sleeping agents are excluded from idle check."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.register_agent("val")

        # Mock _is_paused
        with patch.object(mediator, '_is_paused', return_value=False):
            # andi is sleeping, val is ready with completed_at in past
            def hgetall_side_effect(key):
                # Handle both bytes and string keys
                key_str = key.decode() if isinstance(key, bytes) else key
                if "andi:profile" in key_str:
                    return {b"is_sleeping": b"true"}
                elif "val:profile" in key_str:
                    return {b"is_sleeping": b"false"}
                elif "turn_request" in key_str:
                    return ready_turn_request
                return {}

            mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

            # Mock stream with old activity (400 seconds ago, past 300s threshold)
            idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
            mock_redis.xinfo_stream = AsyncMock(return_value={
                "last-generated-id": f"{idle_time_ms}-0".encode()
            })

            # Set cooldown to allow check
            mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

            # Mock _scan_for_unanalyzed_conversations
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should trigger (andi is sleeping, val is idle, streams show old activity)
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_analysis_cooldown_not_elapsed(
        self, mock_redis, mediator_config
    ):
        """Test that cooldown prevents rapid re-triggering."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Set last check to just 30 seconds ago (cooldown is 60 seconds)
        recent_time = _utc_now() - timedelta(seconds=30)
        mediator._last_auto_analysis_check = recent_time

        with patch.object(mediator, '_is_paused', return_value=False):
            await mediator._check_auto_analysis_trigger()

        # Should not proceed due to cooldown
        mock_redis.hgetall.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_analysis_agents_not_idle(
        self, mock_redis, mediator_config
    ):
        """Test that system is not considered idle when agents are busy."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock busy agent
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"current-turn",
        })

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger (agent busy)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_analysis_idle_threshold_not_reached(
        self, mock_redis, mediator_config
    ):
        """Test that trigger does not fire before idle threshold is reached."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Completed just 200 seconds ago (threshold is 300)
        completed_time = _utc_now() - timedelta(seconds=200)
        turn_request_data = {
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"completed_at": completed_time.isoformat().encode(),
        }

        # Mock ready agent and no sleeping state
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}  # Not sleeping
            return turn_request_data

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger yet (only 200s idle, need 300s)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_analysis_idle_threshold_reached(
        self, mock_redis, mediator_config, ready_turn_request
    ):
        """Test that trigger fires when idle threshold is reached."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock ready agent and no sleeping state
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}  # Not sleeping
            return ready_turn_request

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Mock stream with activity 400 seconds ago (past 300s threshold)
        idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": f"{idle_time_ms}-0".encode()
        })

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should trigger (400s idle, threshold is 300s)
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_idle_detection_with_failed_turn_in_backoff(
        self, mock_redis, mediator_config
    ):
        """Test that failed turns in backoff are not considered idle."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock failed turn still in backoff
        future_time = _utc_now() + timedelta(seconds=120)
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"fail",
            b"turn_id": b"failed-turn",
            b"next_attempt_at": future_time.isoformat().encode(),
        })

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger (agent in backoff, not truly idle)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_detection_with_retry_turn_in_backoff(
        self, mock_redis, mediator_config
    ):
        """Test that RETRY turns in backoff are not considered idle."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock retry turn still in backoff
        future_time = _utc_now() + timedelta(seconds=120)
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"retry",
            b"turn_id": b"retry-turn",
            b"next_attempt_at": future_time.isoformat().encode(),
        })

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger (agent in backoff, not truly idle)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_detection_with_retry_turn_past_backoff(
        self, mock_redis, mediator_config
    ):
        """Test that RETRY turns past backoff don't block idle detection."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock retry turn past backoff time
        past_time = _utc_now() - timedelta(seconds=10)
        completed_time = _utc_now() - timedelta(seconds=400)

        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}  # Not sleeping
            return {
                b"status": b"retry",
                b"turn_id": b"retry-turn",
                b"next_attempt_at": past_time.isoformat().encode(),
                b"completed_at": completed_time.isoformat().encode(),
            }

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger - RETRY status blocks idle detection
                # even if past backoff time (agent not truly "ready")
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_idle_detection_with_crashed_agent(
        self, mock_redis, mediator_config
    ):
        """Test that crashed/offline agents block idle detection."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock crashed agent (no turn_request)
        mock_redis.hgetall = AsyncMock(return_value={})

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger (agent offline/crashed)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_sleeping_agents_excluded_from_idle_check(
        self, mock_redis, mediator_config, ready_turn_request
    ):
        """Test that only sleeping agents result in idle state and can trigger auto-analysis."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock sleeping agent with completed_at in past
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {b"is_sleeping": b"true"}
            return ready_turn_request

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Mock stream with activity 400 seconds ago (past 300s threshold)
        idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": f"{idle_time_ms}-0".encode()
        })

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                # Should trigger since only agent is sleeping
                await mediator._check_auto_analysis_trigger()

                # Should trigger (sleeping agents excluded, system considered idle)
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_sleeping_agents_can_receive_analysis_tasks(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that sleeping agents can receive analysis task assignments."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock sleeping agent
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {b"is_sleeping": b"true"}
            return {b"status": b"ready", b"turn_id": b"prev-turn"}

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Mock conversation report
        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        # Mock stream with old activity (to pass idle detection)
        idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": f"{idle_time_ms}-0".encode()
        })

        # Mock _handle_analysis_command
        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should call handler for sleeping agent
            mock_handle.assert_called_once()
            call_args = mock_handle.call_args
            assert call_args[1]["agent_id"] == "andi"


# =============================================================================
# 2. CONVERSATION SCANNING TESTS
# =============================================================================

class TestConversationScanning:
    """Test conversation scanning logic for unanalyzed conversations."""

    @pytest.mark.asyncio
    async def test_scan_finds_unanalyzed_conversation(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that scan correctly identifies unanalyzed conversations."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock conversation report
        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should find and trigger analysis
            mock_handle.assert_called_once()
            call_args = mock_handle.call_args
            assert call_args[1]["agent_id"] == "andi"
            assert call_args[1]["scenario"] == "analysis_dialogue"
            # Should pick oldest unanalyzed (conv_002)
            assert call_args[1]["conversation_id"] == "conv_002"
            assert call_args[1]["guidance"] is None

    @pytest.mark.asyncio
    async def test_scan_skips_already_analyzed(
        self, mock_redis, mediator_config
    ):
        """Test that scan skips conversations with existing analysis."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # All conversations already analyzed
        report = {
            "conv_001": {
                "mud-world": 5,
                "mud-agent": 3,
                "analysis": 2,  # Already analyzed
                "timestamp_max": "2026-01-01T10:00:00+00:00",
            },
        }
        mock_redis.get = AsyncMock(
            return_value=json.dumps(report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command'
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should not trigger analysis
            mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_requires_mud_docs(
        self, mock_redis, mediator_config
    ):
        """Test that scan requires mud-world or mud-agent docs."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Conversation with no MUD docs
        report = {
            "conv_001": {
                "mud-world": 0,
                "mud-agent": 0,
                "analysis": 0,
                "timestamp_max": "2026-01-01T10:00:00+00:00",
            },
        }
        mock_redis.get = AsyncMock(
            return_value=json.dumps(report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command'
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should not trigger analysis (no MUD docs)
            mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_oldest_conversation_first(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that scan prioritizes oldest unanalyzed conversation."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should pick conv_002 (oldest timestamp)
            call_args = mock_handle.call_args
            assert call_args[1]["conversation_id"] == "conv_002"

    @pytest.mark.asyncio
    async def test_scan_handles_empty_report(
        self, mock_redis, mediator_config
    ):
        """Test that scan handles empty conversation report."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Empty report
        mock_redis.get = AsyncMock(return_value=json.dumps({}).encode())

        with patch.object(
            mediator, '_handle_analysis_command'
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should not crash, no analysis triggered
            mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_handles_missing_report(
        self, mock_redis, mediator_config
    ):
        """Test that scan handles missing conversation report."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # No report in Redis
        mock_redis.get = AsyncMock(return_value=None)

        with patch.object(
            mediator, '_handle_analysis_command'
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should not crash, no analysis triggered
            mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_handles_invalid_report_format(
        self, mock_redis, mediator_config
    ):
        """Test that scan handles invalid report format gracefully."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Invalid JSON structure (list instead of dict)
        mock_redis.get = AsyncMock(
            return_value=json.dumps(["invalid", "format"]).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command'
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should not crash, no analysis triggered
            mock_handle.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_handles_non_dict_doc_counts(
        self, mock_redis, mediator_config
    ):
        """Test that scan handles non-dict doc_counts gracefully."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Invalid doc_counts (string instead of dict)
        report = {
            "conv_001": "invalid_format",
            "conv_002": {
                "mud-world": 5,
                "mud-agent": 3,
                "analysis": 0,
                "timestamp_max": "2026-01-01T10:00:00+00:00",
            },
        }
        mock_redis.get = AsyncMock(
            return_value=json.dumps(report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should skip conv_001, process conv_002
            mock_handle.assert_called_once()
            call_args = mock_handle.call_args
            assert call_args[1]["conversation_id"] == "conv_002"

    @pytest.mark.asyncio
    async def test_multiple_unanalyzed_conversations_prioritizes_oldest(
        self, mock_redis, mediator_config
    ):
        """Test that multiple unanalyzed conversations are prioritized by timestamp."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        report = {
            "conv_new": {
                "mud-world": 1,
                "mud-agent": 1,
                "analysis": 0,
                "timestamp_max": "2026-01-09T12:00:00+00:00",  # Newest
            },
            "conv_old": {
                "mud-world": 1,
                "mud-agent": 1,
                "analysis": 0,
                "timestamp_max": "2026-01-01T08:00:00+00:00",  # Oldest
            },
            "conv_mid": {
                "mud-world": 1,
                "mud-agent": 1,
                "analysis": 0,
                "timestamp_max": "2026-01-05T10:00:00+00:00",  # Middle
            },
        }
        mock_redis.get = AsyncMock(
            return_value=json.dumps(report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should pick conv_old (oldest)
            call_args = mock_handle.call_args
            assert call_args[1]["conversation_id"] == "conv_old"


# =============================================================================
# 3. TURN ASSIGNMENT TESTS
# =============================================================================

class TestTurnAssignment:
    """Test turn assignment for analysis tasks."""

    @pytest.mark.asyncio
    async def test_analysis_command_called_correctly(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that _handle_analysis_command is called with correct parameters."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            mock_handle.assert_called_once_with(
                agent_id="andi",
                scenario="analysis_dialogue",
                conversation_id="conv_002",
                guidance=None,
            )

    @pytest.mark.asyncio
    async def test_analysis_assignment_success(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test successful analysis turn assignment."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        # Mock successful assignment
        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should log success (verify by checking call was made)
            mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_analysis_assignment_failure(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test handling of analysis turn assignment failure."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        # Mock failed assignment (agent busy or offline)
        with patch.object(
            mediator, '_handle_analysis_command', return_value=False
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should still complete without crashing
            mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_turn_assignment_cas_failure(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that CAS failure is handled gracefully."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
        })
        mock_redis.eval = AsyncMock(return_value=0)  # CAS failure

        await mediator._scan_for_unanalyzed_conversations()

        # Should complete without crashing
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_round_robin_assigns_one_agent_per_trigger(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that round-robin assigns analysis to only ONE agent per trigger."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.register_agent("val")

        mock_redis.get = AsyncMock(
            return_value=json.dumps(sample_conversation_report).encode()
        )

        with patch.object(
            mediator, '_handle_analysis_command', return_value=True
        ) as mock_handle:
            await mediator._scan_for_unanalyzed_conversations()

            # Should only assign to ONE agent per trigger
            assert mock_handle.call_count == 1

            # First trigger should assign to first agent in round-robin
            first_agent = mock_handle.call_args[1]["agent_id"]
            assert first_agent in ["andi", "val"]

            # Second trigger should assign to next agent
            mock_handle.reset_mock()
            await mediator._scan_for_unanalyzed_conversations()

            assert mock_handle.call_count == 1
            second_agent = mock_handle.call_args[1]["agent_id"]
            assert second_agent in ["andi", "val"]
            assert second_agent != first_agent  # Different agent


# =============================================================================
# 4. STATE MANAGEMENT TESTS
# =============================================================================

class TestStateManagement:
    """Test state management for cooldown and idle detection."""

    @pytest.mark.asyncio
    async def test_idle_detection_uses_completed_at(
        self, mock_redis, mediator_config, ready_turn_request
    ):
        """Test that idle detection uses stream timestamps (not completed_at)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock idle agent
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}  # Not sleeping
            return ready_turn_request

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Mock stream with activity 400 seconds ago (past 300s threshold)
        idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": f"{idle_time_ms}-0".encode()
        })

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should trigger based on stream timestamp
                mock_scan.assert_called_once()

    @pytest.mark.asyncio
    async def test_busy_agent_blocks_idle_detection(
        self, mock_redis, mediator_config
    ):
        """Test that busy agents block idle detection."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Set cooldown to allow check
        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        # Mock busy agent - need to handle profile check first
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}  # Not sleeping
            return {
                b"status": b"in_progress",
                b"turn_id": b"current-turn",
            }

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                await mediator._check_auto_analysis_trigger()

                # Should not trigger (agent busy)
                mock_scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_cooldown_prevents_rapid_retrigger(
        self, mock_redis, mediator_config, ready_turn_request
    ):
        """Test that cooldown prevents rapid re-triggering."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock idle agent with completed_at in past
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}
            return ready_turn_request

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Mock stream with activity 400 seconds ago (past 300s threshold)
        idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": f"{idle_time_ms}-0".encode()
        })

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ) as mock_scan:
                # Set cooldown to allow first check
                mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

                # First trigger should work
                await mediator._check_auto_analysis_trigger()
                assert mock_scan.call_count == 1

                # Second trigger within cooldown should not work
                # (last_auto_analysis_check was just updated)
                await mediator._check_auto_analysis_trigger()
                # Still only 1 call (cooldown prevented second trigger)
                assert mock_scan.call_count == 1

    @pytest.mark.asyncio
    async def test_last_check_timestamp_updated(
        self, mock_redis, mediator_config, ready_turn_request
    ):
        """Test that last check timestamp is updated on trigger."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock idle agent
        def hgetall_side_effect(key):
            key_str = key.decode() if isinstance(key, bytes) else key
            if "profile" in key_str:
                return {}
            return ready_turn_request

        mock_redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        # Mock stream with activity 400 seconds ago (past 300s threshold)
        idle_time_ms = int((_utc_now() - timedelta(seconds=400)).timestamp() * 1000)
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "last-generated-id": f"{idle_time_ms}-0".encode()
        })

        with patch.object(mediator, '_is_paused', return_value=False):
            with patch.object(
                mediator, '_scan_for_unanalyzed_conversations'
            ):
                # Set old last check time to pass cooldown
                old_time = _utc_now() - timedelta(seconds=120)
                mediator._last_auto_analysis_check = old_time

                await mediator._check_auto_analysis_trigger()

                # Last check timestamp should be updated
                assert mediator._last_auto_analysis_check > old_time


# =============================================================================
# 5. REPORT REFRESH TESTS
# =============================================================================

class TestReportRefresh:
    """Test conversation report refresh before scanning."""

    @pytest.mark.asyncio
    async def test_refresh_conversation_reports_called_before_scan(
        self, mock_redis, mediator_config, sample_conversation_report
    ):
        """Test that _refresh_conversation_reports is called before scanning."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        with patch.object(
            mediator, '_refresh_conversation_reports'
        ) as mock_refresh:
            mock_refresh.return_value = None

            # Mock conversation report and analysis command
            mock_redis.get = AsyncMock(
                return_value=json.dumps(sample_conversation_report).encode()
            )
            with patch.object(
                mediator, '_handle_analysis_command', return_value=True
            ):
                await mediator._scan_for_unanalyzed_conversations()

                # Should call refresh before scanning
                mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_uses_agent_profile_persona_id(
        self, mock_redis, mediator_config
    ):
        """Test that refresh uses persona_id from agent profile."""
        from aim_mud_types.profile import AgentProfile

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Track which memory path was used
        memory_paths_used = []

        def mock_conversation_model_init(self, memory_path, **kwargs):
            memory_paths_used.append(memory_path)
            # Create a mock DataFrame for get_conversation_report
            import pandas as pd
            self.get_conversation_report = MagicMock(
                return_value=pd.DataFrame()
            )

        mock_redis.set = AsyncMock(return_value=True)

        # Mock RedisMUDClient.get_agent_profile to return profile with different persona_id
        async def mock_get_agent_profile(agent_id):
            if agent_id == "andi":
                return AgentProfile(agent_id="andi", persona_id="nova")
            return None

        with patch(
            'aim_mud_types.client.RedisMUDClient.get_agent_profile',
            side_effect=mock_get_agent_profile
        ):
            with patch(
                'andimud_mediator.mixins.agents.ConversationModel.__init__',
                mock_conversation_model_init
            ):
                await mediator._refresh_conversation_reports()

                # Should use persona_id from profile
                assert len(memory_paths_used) == 1
                assert memory_paths_used[0] == "memory/nova"

    @pytest.mark.asyncio
    async def test_refresh_handles_missing_profile(
        self, mock_redis, mediator_config
    ):
        """Test that refresh handles missing agent profile gracefully."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock missing profile
        async def mock_get_agent_profile(agent_id):
            return None

        with patch(
            'aim_mud_types.client.RedisMUDClient.get_agent_profile',
            side_effect=mock_get_agent_profile
        ):
            # Should not crash
            await mediator._refresh_conversation_reports()

    @pytest.mark.asyncio
    async def test_refresh_stores_report_in_redis(
        self, mock_redis, mediator_config
    ):
        """Test that refresh stores generated report in Redis."""
        from aim_mud_types.profile import AgentProfile

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock conversation report DataFrame
        import pandas as pd
        mock_df = pd.DataFrame([
            {
                'conversation_id': 'conv_001',
                'mud-world': 5,
                'mud-agent': 3,
                'analysis': 0,
            }
        ])

        mock_redis.set = AsyncMock(return_value=True)

        # Mock RedisMUDClient.get_agent_profile
        async def mock_get_agent_profile(agent_id):
            if agent_id == "andi":
                return AgentProfile(agent_id="andi", persona_id="andi")
            return None

        with patch(
            'aim_mud_types.client.RedisMUDClient.get_agent_profile',
            side_effect=mock_get_agent_profile
        ):
            with patch(
                'andimud_mediator.mixins.agents.ConversationModel'
            ) as mock_cvm_class:
                mock_cvm = MagicMock()
                mock_cvm.get_conversation_report.return_value = mock_df
                mock_cvm_class.return_value = mock_cvm

                await mediator._refresh_conversation_reports()

                # Should store report in Redis
                mock_redis.set.assert_called_once()
                call_args = mock_redis.set.call_args
                assert call_args[0][0] == RedisKeys.agent_conversation_report("andi")
                # Verify JSON structure
                stored_json = call_args[0][1]
                stored_data = json.loads(stored_json)
                assert 'conv_001' in stored_data

    @pytest.mark.asyncio
    async def test_refresh_handles_empty_dataframe(
        self, mock_redis, mediator_config
    ):
        """Test that refresh handles empty DataFrame correctly."""
        from aim_mud_types.profile import AgentProfile

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock empty DataFrame
        import pandas as pd
        mock_df = pd.DataFrame()

        mock_redis.set = AsyncMock(return_value=True)

        # Mock RedisMUDClient.get_agent_profile
        async def mock_get_agent_profile(agent_id):
            if agent_id == "andi":
                return AgentProfile(agent_id="andi", persona_id="andi")
            return None

        with patch(
            'aim_mud_types.client.RedisMUDClient.get_agent_profile',
            side_effect=mock_get_agent_profile
        ):
            with patch(
                'andimud_mediator.mixins.agents.ConversationModel'
            ) as mock_cvm_class:
                mock_cvm = MagicMock()
                mock_cvm.get_conversation_report.return_value = mock_df
                mock_cvm_class.return_value = mock_cvm

                await mediator._refresh_conversation_reports()

                # Should store empty dict
                mock_redis.set.assert_called_once()
                call_args = mock_redis.set.call_args
                stored_json = call_args[0][1]
                stored_data = json.loads(stored_json)
                assert stored_data == {}

    @pytest.mark.asyncio
    async def test_refresh_handles_multiple_agents(
        self, mock_redis, mediator_config
    ):
        """Test that refresh processes multiple agents."""
        from aim_mud_types.profile import AgentProfile

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mediator.register_agent("val")

        # Mock agent profiles via RedisMUDClient
        async def mock_get_agent_profile(agent_id):
            if agent_id == "andi":
                return AgentProfile(agent_id="andi", persona_id="andi")
            elif agent_id == "val":
                return AgentProfile(agent_id="val", persona_id="val")
            return None

        mock_redis.set = AsyncMock(return_value=True)

        import pandas as pd
        mock_df = pd.DataFrame()

        with patch(
            'aim_mud_types.client.RedisMUDClient.get_agent_profile',
            side_effect=mock_get_agent_profile
        ):
            with patch(
                'andimud_mediator.mixins.agents.ConversationModel'
            ) as mock_cvm_class:
                mock_cvm = MagicMock()
                mock_cvm.get_conversation_report.return_value = mock_df
                mock_cvm_class.return_value = mock_cvm

                await mediator._refresh_conversation_reports()

                # Should create ConversationModel for both agents
                assert mock_cvm_class.call_count == 2

    @pytest.mark.asyncio
    async def test_refresh_handles_exception_gracefully(
        self, mock_redis, mediator_config
    ):
        """Test that refresh handles exceptions without crashing."""
        from aim_mud_types.profile import AgentProfile

        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock RedisMUDClient.get_agent_profile
        async def mock_get_agent_profile(agent_id):
            if agent_id == "andi":
                return AgentProfile(agent_id="andi", persona_id="andi")
            return None

        # Mock ConversationModel to raise exception
        with patch(
            'aim_mud_types.client.RedisMUDClient.get_agent_profile',
            side_effect=mock_get_agent_profile
        ):
            with patch(
                'andimud_mediator.mixins.agents.ConversationModel'
            ) as mock_cvm_class:
                mock_cvm_class.side_effect = Exception("Test error")

                # Should not crash
                await mediator._refresh_conversation_reports()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
