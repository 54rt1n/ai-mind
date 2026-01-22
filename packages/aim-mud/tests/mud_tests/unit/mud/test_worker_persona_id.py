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
        from unittest.mock import patch, AsyncMock

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Mock the client's update method - patch where it's imported in the profile module
        with patch('andimud_worker.mixins.datastore.profile.RedisMUDClient') as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.update_agent_profile_fields = AsyncMock(return_value=True)

            await worker._update_agent_profile(persona_id="Andi")

            # Verify update_agent_profile_fields was called with persona_id
            mock_client.update_agent_profile_fields.assert_called_once()
            call_kwargs = mock_client.update_agent_profile_fields.call_args[1]
            assert "persona_id" in call_kwargs
            assert call_kwargs["persona_id"] == "Andi"
            assert call_kwargs["touch_updated_at"] is True

    @pytest.mark.asyncio
    async def test_update_agent_profile_skips_none_persona_id(self, mud_config, mock_redis):
        """Test _update_agent_profile doesn't write persona_id when None."""
        from unittest.mock import patch, AsyncMock

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)

        # Mock the client's update method - patch where it's imported in the profile module
        with patch('andimud_worker.mixins.datastore.profile.RedisMUDClient') as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.update_agent_profile_fields = AsyncMock(return_value=True)

            await worker._update_agent_profile()

            # Verify persona_id is not in kwargs when None
            mock_client.update_agent_profile_fields.assert_called_once()
            call_kwargs = mock_client.update_agent_profile_fields.call_args[1]
            assert "persona_id" not in call_kwargs
            assert call_kwargs["touch_updated_at"] is True

    @pytest.mark.asyncio
    async def test_load_agent_profile_writes_persona_id_on_empty_profile(
        self, mud_config, mock_redis
    ):
        """Test _load_agent_profile writes persona_id when initializing empty profile."""
        from unittest.mock import patch, AsyncMock

        # Mock empty profile (no data in Redis)
        mock_redis.hgetall = AsyncMock(return_value={})

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MagicMock()

        # Mock persona with correct persona_id
        mock_persona = MagicMock()
        mock_persona.persona_id = "Andi"
        worker.persona = mock_persona

        # Mock the client's methods - patch where it's imported in the profile module
        with patch('andimud_worker.mixins.datastore.profile.RedisMUDClient') as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.get_agent_profile_raw = AsyncMock(return_value=None)  # Empty profile
            mock_client.update_agent_profile_fields = AsyncMock(return_value=True)

            await worker._load_agent_profile()

            # Verify update_agent_profile_fields was called with persona_id
            mock_client.update_agent_profile_fields.assert_called_once()
            call_kwargs = mock_client.update_agent_profile_fields.call_args[1]
            assert "persona_id" in call_kwargs
            assert call_kwargs["persona_id"] == "Andi"

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
        from unittest.mock import patch, AsyncMock

        mock_redis.hgetall = AsyncMock(return_value={})

        worker = MUDAgentWorker(config=mud_config, redis_client=mock_redis)
        worker.session = MagicMock()

        # Simulate persona loaded from Roster with correct capitalization
        mock_persona = MagicMock()
        mock_persona.persona_id = "Andi"  # Capital A from config/persona/Andi.json
        worker.persona = mock_persona

        # Mock the client's methods - patch where it's imported in the profile module
        with patch('andimud_worker.mixins.datastore.profile.RedisMUDClient') as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.get_agent_profile_raw = AsyncMock(return_value=None)  # Empty profile
            mock_client.update_agent_profile_fields = AsyncMock(return_value=True)

            await worker._load_agent_profile()

            # Verify persona_id from Roster is used, not lowercased agent_id
            mock_client.update_agent_profile_fields.assert_called_once()
            call_kwargs = mock_client.update_agent_profile_fields.call_args[1]
            assert call_kwargs["persona_id"] == "Andi"  # Not "andi"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
