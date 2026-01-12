# tests/unit/llm/test_model_set.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""Unit tests for ModelSet class."""

import pytest
from unittest.mock import MagicMock, patch

from aim.config import ChatConfig
from aim.agents.persona import Persona
from aim.llm.model_set import ModelSet


@pytest.fixture
def mock_config():
    """Create a mock ChatConfig."""
    config = MagicMock(spec=ChatConfig)
    config.default_model = "gpt-4o"
    config.thought_model = "o1-mini"
    config.codex_model = "gpt-4o-mini"
    config.decision_model = "gpt-3.5-turbo"
    config.agent_model = "gpt-4-turbo"
    config.model_config_path = "config/models.yaml"
    return config


@pytest.fixture
def mock_persona():
    """Create a mock Persona with model overrides."""
    persona = MagicMock(spec=Persona)
    persona.models = {
        "default": "claude-3.5-sonnet",
        "thought": "claude-3.5-sonnet",
        "tool": "claude-3.5-haiku"
    }
    return persona


class TestModelSetCreation:
    """Test ModelSet creation from config."""

    def test_from_config_without_persona(self, mock_config):
        """Test ModelSet initialization from config only."""
        model_set = ModelSet.from_config(mock_config)

        assert model_set.default_model == "gpt-4o"
        assert model_set.thought_model == "o1-mini"
        assert model_set.codex_model == "gpt-4o-mini"
        assert model_set.decision_model == "gpt-3.5-turbo"
        assert model_set.agent_model == "gpt-4-turbo"
        # Roles without env variables should fallback to default
        assert model_set.chat_model == "gpt-4o"
        assert model_set.tool_model == "gpt-4o"
        assert model_set.code_model == "gpt-4o"

    def test_from_config_with_persona_override(self, mock_config, mock_persona):
        """Test persona models override config."""
        model_set = ModelSet.from_config(mock_config, mock_persona)

        # Persona overrides should be used
        assert model_set.default_model == "claude-3.5-sonnet"
        assert model_set.thought_model == "claude-3.5-sonnet"
        assert model_set.tool_model == "claude-3.5-haiku"

        # Non-overridden roles should fallback to default (from persona override)
        assert model_set.chat_model == "claude-3.5-sonnet"
        assert model_set.code_model == "claude-3.5-sonnet"

    def test_from_config_partial_persona_override(self, mock_config, mock_persona):
        """Test partial persona override with env fallback."""
        # Only override tool, let codex use env variable
        mock_persona.models = {"tool": "claude-3.5-haiku"}

        model_set = ModelSet.from_config(mock_config, mock_persona)

        # Tool uses persona override
        assert model_set.tool_model == "claude-3.5-haiku"
        # Codex uses env variable
        assert model_set.codex_model == "gpt-4o-mini"
        # Thought uses env variable
        assert model_set.thought_model == "o1-mini"
        # Default is from config
        assert model_set.default_model == "gpt-4o"

    def test_from_config_missing_default_model(self):
        """Test that missing DEFAULT_MODEL raises ValueError."""
        config = MagicMock(spec=ChatConfig)
        config.default_model = None

        with pytest.raises(ValueError, match="DEFAULT_MODEL must be specified"):
            ModelSet.from_config(config)


class TestModelSelection:
    """Test model selection by role."""

    def test_get_model_name_for_all_roles(self, mock_config):
        """Test get_model_name returns correct model for each role."""
        model_set = ModelSet.from_config(mock_config)

        assert model_set.get_model_name("default") == "gpt-4o"
        assert model_set.get_model_name("chat") == "gpt-4o"
        assert model_set.get_model_name("thought") == "o1-mini"
        assert model_set.get_model_name("codex") == "gpt-4o-mini"
        assert model_set.get_model_name("decision") == "gpt-3.5-turbo"
        assert model_set.get_model_name("agent") == "gpt-4-turbo"
        assert model_set.get_model_name("tool") == "gpt-4o"
        assert model_set.get_model_name("code") == "gpt-4o"
        assert model_set.get_model_name("analysis") == "gpt-4o"
        assert model_set.get_model_name("writing") == "gpt-4o"
        assert model_set.get_model_name("research") == "gpt-4o"
        assert model_set.get_model_name("planning") == "gpt-4o"

    def test_get_model_name_unknown_role_uses_default(self, mock_config):
        """Test unknown role falls back to default."""
        model_set = ModelSet.from_config(mock_config)

        assert model_set.get_model_name("unknown_role") == "gpt-4o"


class TestProviderCaching:
    """Test LLM provider caching."""

    def test_get_provider_creates_provider(self, mock_config):
        """Test get_provider creates and caches provider."""
        model_set = ModelSet.from_config(mock_config)

        # Mock the provider creation
        mock_provider = MagicMock()
        with patch.object(model_set, '_create_provider', return_value=mock_provider):
            provider = model_set.get_provider("default")

            assert provider == mock_provider
            # Verify provider is cached
            assert "gpt-4o" in model_set._providers
            assert model_set._providers["gpt-4o"] == mock_provider

    def test_get_provider_returns_cached(self, mock_config):
        """Test get_provider returns cached provider on subsequent calls."""
        model_set = ModelSet.from_config(mock_config)

        mock_provider = MagicMock()
        with patch.object(model_set, '_create_provider', return_value=mock_provider) as mock_create:
            provider1 = model_set.get_provider("default")
            provider2 = model_set.get_provider("default")

            # Should only create once
            mock_create.assert_called_once()
            # Should return same instance
            assert provider1 is provider2

    def test_multiple_roles_share_provider(self, mock_config):
        """Test multiple roles using same model share provider instance."""
        model_set = ModelSet.from_config(mock_config)

        mock_provider = MagicMock()
        with patch.object(model_set, '_create_provider', return_value=mock_provider) as mock_create:
            # Both chat and code use default model (gpt-4o)
            chat_provider = model_set.get_provider("chat")
            code_provider = model_set.get_provider("code")

            # Should only create provider once
            mock_create.assert_called_once()
            # Both should be the same instance
            assert chat_provider is code_provider


class TestPriorityOrder:
    """Test model selection priority: persona > env > default."""

    def test_priority_persona_overrides_env(self, mock_config, mock_persona):
        """Test persona override takes precedence over env variable."""
        # Persona has thought override, config has thought_model
        mock_persona.models = {"thought": "claude-3.5-sonnet"}
        mock_config.thought_model = "o1-mini"

        model_set = ModelSet.from_config(mock_config, mock_persona)

        # Persona override should win
        assert model_set.thought_model == "claude-3.5-sonnet"

    def test_priority_env_overrides_default(self, mock_config):
        """Test env variable takes precedence over default."""
        mock_config.default_model = "gpt-4o"
        mock_config.thought_model = "o1-mini"

        model_set = ModelSet.from_config(mock_config)

        # Env variable should win for thought
        assert model_set.thought_model == "o1-mini"
        # Default used for others
        assert model_set.chat_model == "gpt-4o"

    def test_priority_default_fallback(self, mock_config):
        """Test default is used when no overrides exist."""
        mock_config.default_model = "gpt-4o"
        mock_config.thought_model = None
        mock_config.codex_model = None

        model_set = ModelSet.from_config(mock_config)

        # All should use default
        assert model_set.thought_model == "gpt-4o"
        assert model_set.codex_model == "gpt-4o"
        assert model_set.chat_model == "gpt-4o"
