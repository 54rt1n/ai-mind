# packages/aim-mud/tests/mud_tests/unit/test_redis_client.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for RedisMUDClient base functionality.

Tests serialization, deserialization, and CAS operations.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

from aim_mud_types.models.coordination import TurnRequestStatus, TurnReason
from aim_mud_types.client import BaseRedisMUDClient


class TestSerializeValue:
    """Tests for _serialize_value helper method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    def test_serialize_none_returns_none(self, client):
        """None values should return None (will be skipped)."""
        assert client._serialize_value(None) is None

    def test_serialize_datetime_to_isoformat(self, client):
        """datetime should be converted to Unix timestamp string."""
        dt = datetime(2026, 1, 10, 12, 30, 45, tzinfo=timezone.utc)
        result = client._serialize_value(dt)
        # Should return Unix timestamp as string
        expected_timestamp = int(dt.timestamp())
        assert result == str(expected_timestamp)

    def test_serialize_enum_to_value(self, client):
        """Enum should be converted to .value."""
        result = client._serialize_value(TurnRequestStatus.IN_PROGRESS)
        assert result == "in_progress"

    def test_serialize_bool_true(self, client):
        """True should become lowercase 'true'."""
        result = client._serialize_value(True)
        assert result == "true"

    def test_serialize_bool_false(self, client):
        """False should become lowercase 'false'."""
        result = client._serialize_value(False)
        assert result == "false"

    def test_serialize_int(self, client):
        """int should be converted to string."""
        result = client._serialize_value(42)
        assert result == "42"

    def test_serialize_float(self, client):
        """float should be converted to string."""
        result = client._serialize_value(3.14)
        assert result == "3.14"

    def test_serialize_str_unchanged(self, client):
        """str should remain unchanged."""
        result = client._serialize_value("hello")
        assert result == "hello"


class TestSerializeObject:
    """Tests for _serialize_object method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    def test_serialize_simple_model(self, client):
        """Should serialize Pydantic model to dict of strings."""
        from pydantic import BaseModel

        class SimpleModel(BaseModel):
            name: str
            age: int
            active: bool

        obj = SimpleModel(name="test", age=25, active=True)
        result = client._serialize_object(obj)

        assert result == {
            "name": "test",
            "age": "25",
            "active": "true"
        }

    def test_serialize_skips_none_values(self, client):
        """None values should be skipped in output."""
        from pydantic import BaseModel
        from typing import Optional

        class ModelWithOptional(BaseModel):
            required: str
            optional: Optional[str] = None

        obj = ModelWithOptional(required="value")
        result = client._serialize_object(obj)

        assert result == {"required": "value"}
        assert "optional" not in result

    def test_serialize_with_datetime_and_enum(self, client):
        """Should handle datetime and enum fields."""
        from pydantic import BaseModel
        from typing import Optional

        class ComplexModel(BaseModel):
            status: TurnRequestStatus
            timestamp: datetime
            count: int
            message: Optional[str] = None

        dt = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        obj = ComplexModel(
            status=TurnRequestStatus.ASSIGNED,
            timestamp=dt,
            count=5
        )
        result = client._serialize_object(obj)

        # datetime should be serialized as Unix timestamp string
        expected_timestamp = str(int(dt.timestamp()))
        assert result == {
            "status": "assigned",
            "timestamp": expected_timestamp,
            "count": "5"
        }


class TestGetHash:
    """Tests for _get_hash method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    @pytest.mark.asyncio
    async def test_get_hash_returns_none_when_missing(self, client):
        """Should return None when key doesn't exist."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        client.redis.hgetall.return_value = {}
        result = await client._get_hash(TestModel, "test:key")

        assert result is None
        client.redis.hgetall.assert_called_once_with("test:key")

    @pytest.mark.asyncio
    async def test_get_hash_deserializes_bytes(self, client):
        """Should decode bytes and deserialize to model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            count: int

        client.redis.hgetall.return_value = {
            b"name": b"test",
            b"count": b"42"
        }

        result = await client._get_hash(TestModel, "test:key")

        assert result is not None
        assert result.name == "test"
        assert result.count == 42

    @pytest.mark.asyncio
    async def test_get_hash_handles_mixed_types(self, client):
        """Should handle mix of bytes and strings."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field1: str
            field2: int

        client.redis.hgetall.return_value = {
            "field1": "value",  # Already string
            b"field2": b"99"    # Bytes
        }

        result = await client._get_hash(TestModel, "test:key")

        assert result is not None
        assert result.field1 == "value"
        assert result.field2 == 99


class TestCreateHash:
    """Tests for _create_hash method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    @pytest.mark.asyncio
    async def test_create_hash_success(self, client):
        """Should create hash when key doesn't exist."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            count: int

        obj = TestModel(name="test", count=5)
        client.redis.eval.return_value = 1

        result = await client._create_hash("test:key", obj, exists_ok=False)

        assert result is True
        # Verify Lua script was called
        assert client.redis.eval.called
        call_args = client.redis.eval.call_args
        # Should have key and field-value pairs
        assert "test:key" in call_args[0]

    @pytest.mark.asyncio
    async def test_create_hash_fails_when_exists(self, client):
        """Should return False when key exists and exists_ok=False."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        obj = TestModel(field="value")
        client.redis.eval.return_value = 0

        result = await client._create_hash("test:key", obj, exists_ok=False)

        assert result is False

    @pytest.mark.asyncio
    async def test_create_hash_with_exists_ok(self, client):
        """Should succeed when exists_ok=True."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        obj = TestModel(field="value")
        client.redis.eval.return_value = 1

        result = await client._create_hash("test:key", obj, exists_ok=True)

        assert result is True


class TestUpdateHash:
    """Tests for _update_hash method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    @pytest.mark.asyncio
    async def test_update_hash_without_cas(self, client):
        """Should update hash without CAS check."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            field: str

        obj = TestModel(field="newvalue")
        client.redis.eval.return_value = 1

        result = await client._update_hash("test:key", obj)

        assert result is True

    @pytest.mark.asyncio
    async def test_update_hash_with_cas_success(self, client):
        """Should update when CAS check passes."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            turn_id: str
            status: str

        obj = TestModel(turn_id="turn123", status="done")
        client.redis.eval.return_value = 1

        result = await client._update_hash(
            "test:key", obj,
            cas_field="turn_id",
            cas_value="turn123"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_hash_with_cas_failure(self, client):
        """Should return False when CAS check fails."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            turn_id: str
            status: str

        obj = TestModel(turn_id="turn123", status="done")
        client.redis.eval.return_value = 0

        result = await client._update_hash(
            "test:key", obj,
            cas_field="turn_id",
            cas_value="turn123"
        )

        assert result is False


class TestUpdateFields:
    """Tests for _update_fields method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    @pytest.mark.asyncio
    async def test_update_fields_simple(self, client):
        """Should update specific fields."""
        client.redis.eval.return_value = 1

        result = await client._update_fields(
            "test:key",
            {"status": "active", "count": 10}
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_fields_with_cas(self, client):
        """Should support CAS on partial update."""
        client.redis.eval.return_value = 1

        result = await client._update_fields(
            "test:key",
            {"heartbeat_at": "2026-01-10T12:00:00"},
            cas_field="status",
            cas_value="in_progress"
        )

        assert result is True


class TestGetField:
    """Tests for _get_field method."""

    @pytest.fixture
    def client(self):
        """Create a BaseRedisMUDClient instance with mocked Redis."""
        mock_redis = AsyncMock()
        return BaseRedisMUDClient(mock_redis)

    @pytest.mark.asyncio
    async def test_get_field_returns_string(self, client):
        """Should return field value as string."""
        client.redis.hget.return_value = b"value"

        result = await client._get_field("test:key", "field")

        assert result == "value"
        client.redis.hget.assert_called_once_with("test:key", "field")

    @pytest.mark.asyncio
    async def test_get_field_returns_none_when_missing(self, client):
        """Should return None when field doesn't exist."""
        client.redis.hget.return_value = None

        result = await client._get_field("test:key", "field")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_field_with_converter(self, client):
        """Should apply converter function."""
        client.redis.hget.return_value = b"42"

        result = await client._get_field("test:key", "count", convert=int)

        assert result == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
