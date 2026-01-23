# packages/aim-mud/tests/mud_tests/unit/test_redis_client_metadata.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for metadata field serialization and deserialization.

Tests the metadata field_validator and Redis serialization to ensure:
1. Metadata dicts are serialized to JSON strings for Redis
2. JSON strings are deserialized back to dicts via field_validator
3. Invalid JSON raises appropriate errors
4. Roundtrip serialization preserves metadata exactly
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from aim_mud_types.models.coordination import MUDTurnRequest, TurnRequestStatus, TurnReason
from aim_mud_types.client import RedisMUDClient


class TestMetadataFieldValidator:
    """Test the metadata field_validator for JSON deserialization."""

    def test_valid_json_string_to_dict(self):
        """Test that valid JSON string is converted to dict."""
        metadata_json = json.dumps({
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_123",
            "guidance": "Focus on emotions"
        })

        turn = MUDTurnRequest(
            turn_id="turn_1",
            sequence_id=1,
            metadata=metadata_json
        )

        assert isinstance(turn.metadata, dict)
        assert turn.metadata["scenario"] == "analysis_dialogue"
        assert turn.metadata["conversation_id"] == "conv_123"
        assert turn.metadata["guidance"] == "Focus on emotions"

    def test_dict_passthrough(self):
        """Test that dict values pass through unchanged."""
        metadata_dict = {
            "scenario": "journaler_dialogue",
            "query": "What did I learn today?"
        }

        turn = MUDTurnRequest(
            turn_id="turn_2",
            sequence_id=2,
            metadata=metadata_dict
        )

        assert isinstance(turn.metadata, dict)
        assert turn.metadata == metadata_dict

    def test_none_passthrough(self):
        """Test that None values pass through unchanged."""
        turn = MUDTurnRequest(
            turn_id="turn_3",
            sequence_id=3,
            metadata=None
        )

        assert turn.metadata is None

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises ValueError."""
        invalid_json = "{not valid json"

        with pytest.raises(ValueError, match="Invalid JSON in metadata field"):
            MUDTurnRequest(
                turn_id="turn_4",
                sequence_id=4,
                metadata=invalid_json
            )

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON in metadata field"):
            MUDTurnRequest(
                turn_id="turn_5",
                sequence_id=5,
                metadata=""
            )

    def test_nested_metadata_structure(self):
        """Test that nested dict structures are preserved."""
        metadata_json = json.dumps({
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_456",
            "config": {
                "depth": 3,
                "topics": ["emotions", "relationships"]
            }
        })

        turn = MUDTurnRequest(
            turn_id="turn_6",
            sequence_id=6,
            metadata=metadata_json
        )

        assert turn.metadata["config"]["depth"] == 3
        assert turn.metadata["config"]["topics"] == ["emotions", "relationships"]


class TestRedisMetadataSerialization:
    """Test Redis serialization/deserialization of metadata."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock async Redis client."""
        redis = AsyncMock()
        redis.hgetall = AsyncMock(return_value={})
        redis.eval = AsyncMock(return_value=1)
        redis.expire = AsyncMock(return_value=True)
        return redis

    @pytest.fixture
    def client(self, mock_redis):
        """Create RedisMUDClient with mocked Redis."""
        return RedisMUDClient(mock_redis)

    def test_serialize_dict_to_json(self, client):
        """Test that dict fields are serialized as JSON."""
        metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_789"
        }

        result = client._serialize_value(metadata)

        assert isinstance(result, str)
        assert result == json.dumps(metadata)
        # Verify it's valid JSON
        parsed = json.loads(result)
        assert parsed == metadata

    def test_serialize_none_metadata(self, client):
        """Test that None metadata is serialized as None (skipped)."""
        result = client._serialize_value(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_deserialize_json_to_dict(self, client, mock_redis):
        """Test that JSON strings are deserialized to dicts."""
        # Simulate Redis returning metadata as JSON string
        metadata_dict = {
            "scenario": "journaler_dialogue",
            "query": "What happened today?"
        }
        mock_redis.hgetall.return_value = {
            b"turn_id": b"turn_abc",
            b"status": b"assigned",
            b"reason": b"dream",
            b"sequence_id": b"10",
            b"attempt_count": b"0",
            b"metadata": json.dumps(metadata_dict).encode()
        }

        turn = await client.get_turn_request("test_agent")

        assert turn is not None
        assert isinstance(turn.metadata, dict)
        assert turn.metadata == metadata_dict

    @pytest.mark.asyncio
    async def test_roundtrip_metadata(self, client, mock_redis):
        """Test serialize → deserialize preserves metadata."""
        # Create turn with metadata
        original_metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_roundtrip",
            "guidance": "Be thorough"
        }

        turn_request = MUDTurnRequest(
            turn_id="turn_roundtrip",
            sequence_id=20,
            status=TurnRequestStatus.ASSIGNED,
            reason=TurnReason.DREAM,
            metadata=original_metadata
        )

        # Serialize
        serialized = client._serialize_object(turn_request)
        assert "metadata" in serialized
        assert isinstance(serialized["metadata"], str)

        # Simulate Redis storage and retrieval
        mock_redis.hgetall.return_value = {
            k.encode(): v.encode() if isinstance(v, str) else v
            for k, v in serialized.items()
        }

        # Deserialize
        retrieved = await client.get_turn_request("test_agent")

        assert retrieved is not None
        assert retrieved.metadata == original_metadata

    @pytest.mark.asyncio
    async def test_roundtrip_none_metadata(self, client, mock_redis):
        """Test that None metadata survives roundtrip."""
        turn_request = MUDTurnRequest(
            turn_id="turn_none",
            sequence_id=21,
            status=TurnRequestStatus.READY,
            metadata=None
        )

        serialized = client._serialize_object(turn_request)
        # None values should be skipped
        assert "metadata" not in serialized

        # Simulate Redis storage (no metadata field)
        mock_redis.hgetall.return_value = {
            b"turn_id": b"turn_none",
            b"status": b"ready",
            b"reason": b"events",
            b"sequence_id": b"21",
            b"attempt_count": b"0",
        }

        retrieved = await client.get_turn_request("test_agent")

        assert retrieved is not None
        assert retrieved.metadata is None

    @pytest.mark.asyncio
    async def test_complex_metadata_roundtrip(self, client, mock_redis):
        """Test roundtrip with complex nested metadata."""
        complex_metadata = {
            "scenario": "analysis_dialogue",
            "conversation_id": "conv_complex",
            "guidance": "Deep analysis",
            "config": {
                "model": "claude-opus-4",
                "temperature": 0.7,
                "tags": ["emotional", "relational"]
            }
        }

        turn_request = MUDTurnRequest(
            turn_id="turn_complex",
            sequence_id=22,
            metadata=complex_metadata
        )

        serialized = client._serialize_object(turn_request)

        # Simulate Redis roundtrip
        mock_redis.hgetall.return_value = {
            k.encode(): v.encode() if isinstance(v, str) else v
            for k, v in serialized.items()
        }

        retrieved = await client.get_turn_request("test_agent")

        assert retrieved is not None
        assert retrieved.metadata == complex_metadata
        assert retrieved.metadata["config"]["model"] == "claude-opus-4"
        assert retrieved.metadata["config"]["tags"] == ["emotional", "relational"]


class TestMetadataSerializationEdgeCases:
    """Test edge cases for metadata serialization."""

    @pytest.fixture
    def client(self):
        """Create RedisMUDClient with mocked Redis."""
        return RedisMUDClient(AsyncMock())

    def test_empty_dict_metadata(self, client):
        """Test that empty dict is serialized correctly."""
        result = client._serialize_value({})
        assert result == "{}"

    def test_unicode_in_metadata(self, client):
        """Test that unicode characters are preserved."""
        metadata = {
            "scenario": "analysis_dialogue",
            "guidance": "Focus on emotions 😊 and feelings ♥"
        }

        result = client._serialize_value(metadata)
        parsed = json.loads(result)
        assert parsed == metadata
        assert "😊" in parsed["guidance"]

    def test_special_characters_in_metadata(self, client):
        """Test that special characters are escaped properly."""
        metadata = {
            "query": "What about \"quotes\" and 'apostrophes'?",
            "guidance": "Handle\nnewlines\tand\ttabs"
        }

        result = client._serialize_value(metadata)
        parsed = json.loads(result)
        assert parsed == metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
