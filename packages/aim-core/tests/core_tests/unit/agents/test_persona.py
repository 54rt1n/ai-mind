# tests/unit/agents/test_persona.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

"""Unit tests for Persona class, specifically model validation."""

import pytest

from aim.agents.persona import Persona


@pytest.fixture
def minimal_persona_data():
    """Minimal valid persona data for testing."""
    return {
        "persona_id": "Test",
        "name": "Test",
        "full_name": "Test Persona",
        "notes": "Test notes",
        "chat_strategy": "xmlmemory",
        "aspects": {},
        "attributes": {},
        "features": {},
        "wakeup": ["Hello"],
        "base_thoughts": [],
        "pif": {},
        "nshot": {},
        "default_location": "Test Location",
        "wardrobe": {"default": {}},
        "current_outfit": "default",
    }


class TestPersonaModelValidation:
    """Test Persona model validation in from_dict."""

    def test_valid_string_model(self, minimal_persona_data):
        """Test that single string model is valid."""
        minimal_persona_data["models"] = {"chat": "claude-3.5-sonnet"}

        persona = Persona.from_dict(minimal_persona_data)

        assert persona.models == {"chat": "claude-3.5-sonnet"}

    def test_valid_list_model(self, minimal_persona_data):
        """Test that list of string models is valid."""
        minimal_persona_data["models"] = {
            "chat": ["claude-3.5-sonnet", "gpt-4o", "gemini-pro"]
        }

        persona = Persona.from_dict(minimal_persona_data)

        assert persona.models == {"chat": ["claude-3.5-sonnet", "gpt-4o", "gemini-pro"]}

    def test_valid_mixed_models(self, minimal_persona_data):
        """Test that mix of string and list models is valid."""
        minimal_persona_data["models"] = {
            "default": "base-model",
            "chat": ["chat-a", "chat-b"],
            "tool": "tool-model"
        }

        persona = Persona.from_dict(minimal_persona_data)

        assert persona.models["default"] == "base-model"
        assert persona.models["chat"] == ["chat-a", "chat-b"]
        assert persona.models["tool"] == "tool-model"

    def test_models_none_is_valid(self, minimal_persona_data):
        """Test that missing models field is valid."""
        # No models key at all
        persona = Persona.from_dict(minimal_persona_data)

        assert persona.models is None

    def test_models_not_dict_raises(self, minimal_persona_data):
        """Test that non-dict models raises ValueError."""
        minimal_persona_data["models"] = "not-a-dict"

        with pytest.raises(ValueError, match="Persona models must be a dict"):
            Persona.from_dict(minimal_persona_data)

    def test_model_value_invalid_type_raises(self, minimal_persona_data):
        """Test that model value of wrong type raises ValueError."""
        minimal_persona_data["models"] = {"chat": 123}

        with pytest.raises(ValueError, match="Persona model 'chat' must be str or list"):
            Persona.from_dict(minimal_persona_data)

    def test_model_value_dict_raises(self, minimal_persona_data):
        """Test that model value of dict type raises ValueError."""
        minimal_persona_data["models"] = {"chat": {"nested": "dict"}}

        with pytest.raises(ValueError, match="Persona model 'chat' must be str or list"):
            Persona.from_dict(minimal_persona_data)

    def test_empty_list_raises(self, minimal_persona_data):
        """Test that empty list model raises ValueError."""
        minimal_persona_data["models"] = {"chat": []}

        with pytest.raises(ValueError, match="Persona model list 'chat' cannot be empty"):
            Persona.from_dict(minimal_persona_data)

    def test_list_with_non_string_raises(self, minimal_persona_data):
        """Test that list with non-string items raises ValueError."""
        minimal_persona_data["models"] = {"chat": ["model-a", 123, "model-b"]}

        with pytest.raises(ValueError, match="Persona model list 'chat' must contain only strings"):
            Persona.from_dict(minimal_persona_data)

    def test_list_with_none_raises(self, minimal_persona_data):
        """Test that list with None item raises ValueError."""
        minimal_persona_data["models"] = {"chat": ["model-a", None]}

        with pytest.raises(ValueError, match="Persona model list 'chat' must contain only strings"):
            Persona.from_dict(minimal_persona_data)

    def test_list_with_nested_list_raises(self, minimal_persona_data):
        """Test that list with nested list raises ValueError."""
        minimal_persona_data["models"] = {"chat": ["model-a", ["nested", "list"]]}

        with pytest.raises(ValueError, match="Persona model list 'chat' must contain only strings"):
            Persona.from_dict(minimal_persona_data)

    def test_single_item_list_is_valid(self, minimal_persona_data):
        """Test that single-item list is valid."""
        minimal_persona_data["models"] = {"chat": ["only-one-model"]}

        persona = Persona.from_dict(minimal_persona_data)

        assert persona.models == {"chat": ["only-one-model"]}


class TestPersonaToDict:
    """Test that Persona.to_dict preserves model lists."""

    def test_to_dict_preserves_string_models(self, minimal_persona_data):
        """Test that to_dict preserves string models."""
        minimal_persona_data["models"] = {"chat": "claude-3.5-sonnet"}

        persona = Persona.from_dict(minimal_persona_data)
        result = persona.to_dict()

        assert result["models"] == {"chat": "claude-3.5-sonnet"}

    def test_to_dict_preserves_list_models(self, minimal_persona_data):
        """Test that to_dict preserves list models."""
        minimal_persona_data["models"] = {
            "chat": ["claude-3.5-sonnet", "gpt-4o"]
        }

        persona = Persona.from_dict(minimal_persona_data)
        result = persona.to_dict()

        assert result["models"] == {"chat": ["claude-3.5-sonnet", "gpt-4o"]}

    def test_to_dict_preserves_mixed_models(self, minimal_persona_data):
        """Test that to_dict preserves mixed string/list models."""
        minimal_persona_data["models"] = {
            "default": "base-model",
            "chat": ["chat-a", "chat-b"],
            "tool": "tool-model"
        }

        persona = Persona.from_dict(minimal_persona_data)
        result = persona.to_dict()

        assert result["models"]["default"] == "base-model"
        assert result["models"]["chat"] == ["chat-a", "chat-b"]
        assert result["models"]["tool"] == "tool-model"
