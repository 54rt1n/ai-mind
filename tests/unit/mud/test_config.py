# tests/unit/mud/test_config.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD configuration."""

import pytest

from aim.app.mud.config import MUDConfig


class TestMUDConfig:
    """Tests for MUDConfig dataclass."""

    def test_config_required_fields(self):
        """Test MUDConfig with only required fields."""
        config = MUDConfig(agent_id="andi", persona_id="andi")

        assert config.agent_id == "andi"
        assert config.persona_id == "andi"

    def test_config_defaults(self):
        """Test MUDConfig default values."""
        config = MUDConfig(agent_id="andi", persona_id="andi")

        # Redis defaults
        assert config.redis_url == "redis://localhost:6379"
        assert config.action_stream == "mud:actions"

        # Timing defaults
        assert config.spontaneous_check_interval == 60.0
        assert config.spontaneous_action_interval == 300.0

        # Memory defaults
        assert config.top_n_memories == 10
        assert config.max_recent_turns == 20

        # LLM defaults
        assert config.llm_provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_config_computed_agent_stream(self):
        """Test agent_stream is computed from agent_id."""
        config = MUDConfig(agent_id="andi", persona_id="andi")

        assert config.agent_stream == "agent:andi:events"

    def test_config_computed_memory_path(self):
        """Test memory_path is computed from persona_id."""
        config = MUDConfig(agent_id="andi", persona_id="andi")

        assert config.memory_path == "memory/andi"

    def test_config_computed_pause_key(self):
        """Test pause_key is computed from agent_id."""
        config = MUDConfig(agent_id="andi", persona_id="andi")

        assert config.pause_key == "mud:agent:andi:paused"

    def test_config_explicit_agent_stream(self):
        """Test explicit agent_stream overrides computed default."""
        config = MUDConfig(
            agent_id="andi",
            persona_id="andi",
            agent_stream="custom:stream:events",
        )

        assert config.agent_stream == "custom:stream:events"

    def test_config_explicit_memory_path(self):
        """Test explicit memory_path overrides computed default."""
        config = MUDConfig(
            agent_id="andi",
            persona_id="andi",
            memory_path="/custom/memory/path",
        )

        assert config.memory_path == "/custom/memory/path"

    def test_config_custom_values(self):
        """Test MUDConfig with custom values."""
        config = MUDConfig(
            agent_id="roommate",
            persona_id="roommate",
            redis_url="redis://custom:6380",
            action_stream="custom:actions",
            spontaneous_check_interval=30.0,
            spontaneous_action_interval=120.0,
            top_n_memories=20,
            max_recent_turns=50,
            llm_provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=4096,
        )

        assert config.agent_id == "roommate"
        assert config.persona_id == "roommate"
        assert config.redis_url == "redis://custom:6380"
        assert config.action_stream == "custom:actions"
        assert config.spontaneous_check_interval == 30.0
        assert config.spontaneous_action_interval == 120.0
        assert config.top_n_memories == 20
        assert config.max_recent_turns == 50
        assert config.llm_provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 4096

    def test_config_different_agent_and_persona(self):
        """Test config where agent_id differs from persona_id."""
        config = MUDConfig(agent_id="andi_mud", persona_id="andi")

        assert config.agent_id == "andi_mud"
        assert config.persona_id == "andi"
        assert config.agent_stream == "agent:andi_mud:events"
        assert config.memory_path == "memory/andi"
        assert config.pause_key == "mud:agent:andi_mud:paused"
