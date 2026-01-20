# packages/aim-mud/tests/mud_tests/unit/mediator/test_retry_metadata.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for metadata preservation during retry operations.

Tests that verify the core bug fix: when a turn request is retried,
the metadata field is preserved from the original request to the new request.
This ensures that scenario, conversation_id, query, and guidance context
are not lost during retry.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types import RedisKeys, TurnRequestStatus, TurnReason


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
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.set = AsyncMock(return_value=True)
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


class TestRetryPreservesMetadataAnalysis:
    """Test that retry preserves metadata for analysis dreams."""

    @pytest.mark.asyncio
    async def test_retry_preserves_analysis_metadata(self, mock_redis, mediator_config):
        """Test that retry preserves metadata for analysis turns."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Original metadata from an analysis command
        original_metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_123",
            "guidance": "Focus on emotional patterns"
        }

        # Setup: Mock Redis to return a turn in RETRY status with metadata
        # next_attempt_at is in the past so retry is ready
        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"old-turn-id",
            b"status": b"retry",
            b"reason": b"dream",
            b"sequence_id": b"5",
            b"attempt_count": b"1",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)  # CAS success

        # Call: _maybe_assign_turn with RETRY reason
        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True
        mock_redis.eval.assert_called_once()

        # Verify: New turn has same metadata
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]

        # Extract fields from Lua script arguments
        field_args = args[5:]  # Fields start after key and CAS params
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata
        assert new_metadata["scenario"] == "analysis_dialogue"
        assert new_metadata["conversation_id"] == "conv_123"
        assert new_metadata["guidance"] == "Focus on emotional patterns"

        # Verify attempt_count is preserved and incremented
        assert "attempt_count" in fields_dict
        # Note: attempt_count is preserved, not incremented in _maybe_assign_turn
        # Incrementing happens in worker when it detects retry
        assert fields_dict["attempt_count"] == "1"

    @pytest.mark.asyncio
    async def test_retry_preserves_analysis_metadata_no_guidance(self, mock_redis, mediator_config):
        """Test retry preserves metadata without guidance field."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        original_metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_456"
            # No guidance field
        }

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"old-turn-id-2",
            b"status": b"retry",
            b"sequence_id": b"10",
            b"attempt_count": b"2",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata
        assert "guidance" not in new_metadata


class TestRetryPreservesMetadataCreative:
    """Test that retry preserves metadata for creative dreams."""

    @pytest.mark.asyncio
    async def test_retry_preserves_journal_metadata(self, mock_redis, mediator_config):
        """Test that retry preserves metadata for journal turns."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        original_metadata = {
            "scenario": "journaler_dialogue",
            "query": "What did I learn today?",
            "guidance": "Focus on emotional growth"
        }

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"journal-turn-id",
            b"status": b"retry",
            b"sequence_id": b"15",
            b"attempt_count": b"1",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata
        assert new_metadata["scenario"] == "journaler_dialogue"
        assert new_metadata["query"] == "What did I learn today?"
        assert new_metadata["guidance"] == "Focus on emotional growth"

    @pytest.mark.asyncio
    async def test_retry_preserves_ponder_metadata_query_only(self, mock_redis, mediator_config):
        """Test retry preserves metadata with query but no guidance."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        original_metadata = {
            "scenario": "philosopher_dialogue",
            "query": "What is the meaning of existence?"
        }

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"ponder-turn-id",
            b"status": b"retry",
            b"sequence_id": b"20",
            b"attempt_count": b"3",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata
        assert new_metadata["query"] == "What is the meaning of existence?"
        assert "guidance" not in new_metadata

    @pytest.mark.asyncio
    async def test_retry_preserves_daydream_metadata_no_params(self, mock_redis, mediator_config):
        """Test retry preserves metadata with scenario only (no query/guidance)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        original_metadata = {
            "scenario": "daydream_dialogue"
        }

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"daydream-turn-id",
            b"status": b"retry",
            b"sequence_id": b"25",
            b"attempt_count": b"1",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata
        assert new_metadata["scenario"] == "daydream_dialogue"
        assert "query" not in new_metadata
        assert "guidance" not in new_metadata


class TestRetryMetadataEdgeCases:
    """Test edge cases for retry metadata preservation."""

    @pytest.mark.asyncio
    async def test_retry_with_none_metadata(self, mock_redis, mediator_config):
        """Test that retry works when original had no metadata."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        # Turn with no metadata field (e.g., from event-driven turn)
        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"event-turn-id",
            b"status": b"retry",
            b"reason": b"events",
            b"sequence_id": b"30",
            b"attempt_count": b"1",
            b"next_attempt_at": past_time.encode(),
            # No metadata field
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        # Metadata should not be in fields (None is skipped during serialization)
        assert "metadata" not in fields_dict

    @pytest.mark.asyncio
    async def test_retry_with_complex_metadata(self, mock_redis, mediator_config):
        """Test retry preserves complex nested metadata structures."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        original_metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_complex",
            "guidance": "Deep analysis",
            "config": {
                "model": "claude-opus-4",
                "temperature": 0.7,
                "tags": ["emotional", "relational"]
            }
        }

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"complex-turn-id",
            b"status": b"retry",
            b"sequence_id": b"35",
            b"attempt_count": b"2",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata
        assert new_metadata["config"]["model"] == "claude-opus-4"
        assert new_metadata["config"]["tags"] == ["emotional", "relational"]

    @pytest.mark.asyncio
    async def test_retry_from_fail_status(self, mock_redis, mediator_config):
        """Test that retry from FAIL status also preserves metadata."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        original_metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_fail"
        }

        past_time = str(int((datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()))

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"fail-turn-id",
            b"status": b"fail",  # FAIL status instead of RETRY
            b"sequence_id": b"40",
            b"attempt_count": b"5",
            b"next_attempt_at": past_time.encode(),
            b"metadata": json.dumps(original_metadata).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.RETRY)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        assert "metadata" in fields_dict
        new_metadata = json.loads(fields_dict["metadata"])
        assert new_metadata == original_metadata


class TestNonRetryMetadata:
    """Test that non-retry turns don't inherit metadata."""

    @pytest.mark.asyncio
    async def test_new_event_turn_has_no_metadata(self, mock_redis, mediator_config):
        """Test that new event-driven turn doesn't inherit old metadata."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Previous turn had metadata, but it was completed
        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"completed-turn",
            b"status": b"done",
            b"sequence_id": b"45",
            b"metadata": json.dumps({"scenario": "old_scenario"}).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        # Assign new turn with EVENTS reason (not RETRY)
        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.EVENTS)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        # New turn should NOT have metadata (not a retry)
        assert "metadata" not in fields_dict

    @pytest.mark.asyncio
    async def test_ready_to_retry_turn_has_no_metadata(self, mock_redis, mediator_config):
        """Test that READY status doesn't inherit metadata from old turn."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"turn_id": b"ready-turn",
            b"status": b"ready",
            b"sequence_id": b"50",
            b"metadata": json.dumps({"scenario": "previous_scenario"}).encode(),
        })
        mock_redis.eval = AsyncMock(return_value=1)

        # Assign new turn with EVENTS reason
        success = await mediator._maybe_assign_turn("andi", reason=TurnReason.EVENTS)

        assert success is True

        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        field_args = args[5:]
        fields_dict = {field_args[i]: field_args[i+1] for i in range(0, len(field_args), 2)}

        # New turn should NOT inherit old metadata
        assert "metadata" not in fields_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
