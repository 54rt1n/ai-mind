# tests/unit/mud/test_worker_persona_id.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD agent worker persona_id fix."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from aim_mud_types import RedisKeys


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="andi",
        persona_id="Andi",
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock(return_value=1)
    return redis


class TestPersonaIdFix:
    """Test persona_id is written to Redis agent profile on startup."""

    @pytest.mark.asyncio
    async def test_update_agent_profile_includes_persona_id(self, mud_config, mock_redis):
        """Test _update_agent_profile writes persona_id when provided."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        await worker._update_agent_profile(
            persona_id="Andi",
            agent_id="andi"
        )

        # Verify hset was called with persona_id in the mapping
        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        key = call_args[0][0]
        mapping = call_args[1]["mapping"]

        assert key == RedisKeys.agent_profile("andi")
        assert "persona_id" in mapping
        assert mapping["persona_id"] == "Andi"
        assert "agent_id" in mapping
        assert mapping["agent_id"] == "andi"
        assert "updated_at" in mapping

    @pytest.mark.asyncio
    async def test_update_agent_profile_skips_none_persona_id(self, mud_config, mock_redis):
        """Test _update_agent_profile doesn't write persona_id when None."""
        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        await worker._update_agent_profile(agent_id="andi")

        # Verify persona_id is not in the mapping when None
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        assert "persona_id" not in mapping
        assert "agent_id" in mapping
        assert "updated_at" in mapping

    @pytest.mark.asyncio
    async def test_load_agent_profile_writes_persona_id_on_empty_profile(
        self, mud_config, mock_redis
    ):
        """Test _load_agent_profile writes persona_id when initializing empty profile."""
        # Mock empty profile (no data in Redis)
        mock_redis.hgetall = AsyncMock(return_value={})

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MagicMock()

        # Mock persona with correct persona_id
        mock_persona = MagicMock()
        mock_persona.persona_id = "Andi"
        worker.persona = mock_persona

        await worker._load_agent_profile()

        # Verify hset was called with persona_id in the mapping
        mock_redis.hset.assert_called_once()
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]

        assert "persona_id" in mapping
        assert mapping["persona_id"] == "Andi"
        assert "agent_id" in mapping
        assert mapping["agent_id"] == "andi"

    @pytest.mark.asyncio
    async def test_load_agent_profile_preserves_existing_data(
        self, mud_config, mock_redis
    ):
        """Test _load_agent_profile doesn't overwrite when profile exists."""
        # Mock existing profile data
        existing_profile = {
            b"persona_id": b"Andi",
            b"agent_id": b"andi",
            b"last_event_id": b"1234567890-0",
        }
        mock_redis.hgetall = AsyncMock(return_value=existing_profile)

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MagicMock()
        worker.session.last_event_id = None

        mock_persona = MagicMock()
        mock_persona.persona_id = "Andi"
        worker.persona = mock_persona

        await worker._load_agent_profile()

        # Verify hset was NOT called (profile already exists)
        mock_redis.hset.assert_not_called()

        # Verify session was updated from existing profile
        assert worker.session.last_event_id == "1234567890-0"

    @pytest.mark.asyncio
    async def test_load_agent_profile_uses_persona_from_roster(
        self, mud_config, mock_redis
    ):
        """Test that worker uses persona loaded from roster for persona_id."""
        mock_redis.hgetall = AsyncMock(return_value={})

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MagicMock()

        # Simulate persona loaded from Roster with correct capitalization
        mock_persona = MagicMock()
        mock_persona.persona_id = "Andi"  # Capital A from config/persona/Andi.json
        worker.persona = mock_persona

        await worker._load_agent_profile()

        # Verify persona_id from Roster is used, not lowercased agent_id
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        assert mapping["persona_id"] == "Andi"  # Not "andi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
