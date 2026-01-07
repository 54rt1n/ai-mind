# packages/aim-mud/tests/mud_tests/unit/worker/test_phase1_modelset.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for Phase 1 ModelSet configuration and persona overrides."""

import pytest
from unittest.mock import Mock, patch

from aim.config import ChatConfig
from aim.llm.model_set import ModelSet
from aim.agents.persona import Persona


@pytest.fixture
def base_chat_config():
    """Create a base ChatConfig for testing."""
    config = ChatConfig()
    config.default_model = "anthropic/claude-sonnet-4-5-20250929"
    config.thought_model = "anthropic/claude-opus-4-5-20251101"
    config.codex_model = "anthropic/claude-opus-4-5-20251101"
    return config


@pytest.fixture
def persona_with_tool_model():
    """Create a test Persona with tool model configured."""
    persona = Mock(spec=Persona)
    persona.persona_id = "test_persona"
    persona.models = {"tool": "deepseek-ai/DeepSeek-V3-0324"}
    return persona


@pytest.fixture
def persona_without_tool_model():
    """Create a test Persona without tool model configured."""
    persona = Mock(spec=Persona)
    persona.persona_id = "test_persona_no_tool"
    persona.models = {}
    return persona


@pytest.fixture
def persona_with_multiple_models():
    """Create a test Persona with multiple model overrides."""
    persona = Mock(spec=Persona)
    persona.persona_id = "test_persona_multi"
    persona.models = {
        "tool": "deepseek-ai/DeepSeek-V3-0324",
        "chat": "anthropic/claude-opus-4-5-20251101",
        "default": "groq/llama-3.3-70b-versatile"
    }
    return persona


def test_modelset_uses_tool_model_from_persona(base_chat_config, persona_with_tool_model):
    """Test that ModelSet correctly uses persona's tool model override."""
    # When: ModelSet is created with persona that has tool model configured
    model_set = ModelSet.from_config(base_chat_config, persona=persona_with_tool_model)

    # Then: tool role resolves to persona's tool model
    assert model_set.get_model_name("tool") == "deepseek-ai/DeepSeek-V3-0324"
    assert model_set.tool_model == "deepseek-ai/DeepSeek-V3-0324"


def test_modelset_falls_back_to_default_when_no_tool_model(base_chat_config, persona_without_tool_model):
    """Test that ModelSet falls back to default model when persona has no tool model."""
    # When: ModelSet is created with persona that has no tool model
    model_set = ModelSet.from_config(base_chat_config, persona=persona_without_tool_model)

    # Then: tool role falls back to default model
    assert model_set.get_model_name("tool") == base_chat_config.default_model
    assert model_set.tool_model == base_chat_config.default_model


def test_modelset_without_persona_uses_default(base_chat_config):
    """Test that ModelSet uses default model when no persona is provided."""
    # When: ModelSet is created without persona
    model_set = ModelSet.from_config(base_chat_config, persona=None)

    # Then: tool role uses default model
    assert model_set.get_model_name("tool") == base_chat_config.default_model
    assert model_set.tool_model == base_chat_config.default_model


def test_modelset_persona_overrides_take_priority(base_chat_config, persona_with_multiple_models):
    """Test that persona overrides take priority over config defaults."""
    # When: ModelSet is created with persona that overrides multiple models
    model_set = ModelSet.from_config(base_chat_config, persona=persona_with_multiple_models)

    # Then: all overridden roles use persona's models
    assert model_set.get_model_name("tool") == "deepseek-ai/DeepSeek-V3-0324"
    assert model_set.get_model_name("chat") == "anthropic/claude-opus-4-5-20251101"
    assert model_set.get_model_name("default") == "groq/llama-3.3-70b-versatile"

    # And: non-overridden roles fall back to the overridden default
    assert model_set.get_model_name("analysis") == "groq/llama-3.3-70b-versatile"
    assert model_set.get_model_name("writing") == "groq/llama-3.3-70b-versatile"


def test_modelset_thought_model_respects_env_variable(base_chat_config, persona_with_tool_model):
    """Test that thought model respects env variable before falling back to default."""
    # When: ModelSet is created with persona (but persona has no thought model override)
    model_set = ModelSet.from_config(base_chat_config, persona=persona_with_tool_model)

    # Then: thought model uses the env variable (config.thought_model), not default
    assert model_set.get_model_name("thought") == base_chat_config.thought_model
    assert model_set.thought_model == "anthropic/claude-opus-4-5-20251101"


def test_modelset_codex_model_respects_env_variable(base_chat_config, persona_with_tool_model):
    """Test that codex model respects env variable before falling back to default."""
    # When: ModelSet is created with persona (but persona has no codex model override)
    model_set = ModelSet.from_config(base_chat_config, persona=persona_with_tool_model)

    # Then: codex model uses the env variable (config.codex_model), not default
    assert model_set.get_model_name("codex") == base_chat_config.codex_model
    assert model_set.codex_model == "anthropic/claude-opus-4-5-20251101"


def test_get_provider_returns_provider_for_tool_role(base_chat_config, persona_with_tool_model):
    """Test that get_provider returns a provider instance for the tool role."""
    # When: ModelSet is created and provider is requested for tool role
    model_set = ModelSet.from_config(base_chat_config, persona=persona_with_tool_model)

    with patch.object(model_set, '_create_provider') as mock_create_provider:
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Get provider (should create it since cache is empty)
        provider = model_set.get_provider("tool")

        # Then: provider is created with the correct model name
        mock_create_provider.assert_called_once_with("deepseek-ai/DeepSeek-V3-0324")
        assert provider == mock_provider


def test_get_provider_caches_providers(base_chat_config, persona_with_tool_model):
    """Test that get_provider caches providers by model name."""
    # When: ModelSet is created and provider is requested multiple times
    model_set = ModelSet.from_config(base_chat_config, persona=persona_with_tool_model)

    with patch.object(model_set, '_create_provider') as mock_create_provider:
        mock_provider = Mock()
        mock_create_provider.return_value = mock_provider

        # Get provider twice for the same role
        provider1 = model_set.get_provider("tool")
        provider2 = model_set.get_provider("tool")

        # Then: provider is created only once (cached)
        mock_create_provider.assert_called_once_with("deepseek-ai/DeepSeek-V3-0324")
        assert provider1 == provider2 == mock_provider


def test_modelset_resolution_priority_order(base_chat_config):
    """Test the full priority order: persona > env variable > default."""
    # Test persona with all overrides
    persona_full = Mock(spec=Persona)
    persona_full.persona_id = "test_full"
    persona_full.models = {
        "tool": "persona-tool-model",
        "thought": "persona-thought-model",
        "default": "persona-default-model"
    }

    model_set = ModelSet.from_config(base_chat_config, persona=persona_full)

    # Persona overrides take highest priority
    assert model_set.get_model_name("tool") == "persona-tool-model"
    assert model_set.get_model_name("thought") == "persona-thought-model"
    assert model_set.get_model_name("default") == "persona-default-model"

    # Test persona with partial overrides
    persona_partial = Mock(spec=Persona)
    persona_partial.persona_id = "test_partial"
    persona_partial.models = {"tool": "persona-tool-model"}

    model_set = ModelSet.from_config(base_chat_config, persona=persona_partial)

    # Tool uses persona override
    assert model_set.get_model_name("tool") == "persona-tool-model"
    # Thought falls back to env variable
    assert model_set.get_model_name("thought") == base_chat_config.thought_model
    # Others fall back to default
    assert model_set.get_model_name("chat") == base_chat_config.default_model
