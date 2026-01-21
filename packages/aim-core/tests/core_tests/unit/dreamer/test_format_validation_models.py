# tests/core_tests/unit/dreamer/test_format_validation_models.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for FormatValidation Pydantic models."""

import pytest
from pydantic import ValidationError

from aim.dreamer.core.models import (
    FormatValidation,
    StepConfig,
    StepOutput,
    StandardStepDefinition,
)


class TestFormatValidation:
    """Tests for FormatValidation model."""

    def test_default_values(self):
        """Test FormatValidation default values - validation enabled by default."""
        config = FormatValidation()

        assert config.require_emotional_header is True  # Enabled by default
        assert config.max_retries == 3
        assert config.fallback_model == "fallback"
        assert config.persona_name_override is None

    def test_custom_values(self):
        """Test FormatValidation with custom values."""
        config = FormatValidation(
            require_emotional_header=True,
            max_retries=5,
            fallback_model="haiku",
            persona_name_override="Custom Name"
        )

        assert config.require_emotional_header is True
        assert config.max_retries == 5
        assert config.fallback_model == "haiku"
        assert config.persona_name_override == "Custom Name"

    def test_fallback_model_can_be_none(self):
        """Test fallback_model can be set to None to disable fallback."""
        config = FormatValidation(fallback_model=None)

        assert config.fallback_model is None

    def test_require_emotional_header_enabled(self):
        """Test enabling emotional header requirement."""
        config = FormatValidation(require_emotional_header=True)

        assert config.require_emotional_header is True

    def test_max_retries_custom_value(self):
        """Test setting custom max_retries value."""
        config = FormatValidation(max_retries=10)

        assert config.max_retries == 10

    def test_persona_name_override_custom(self):
        """Test setting custom persona name override."""
        config = FormatValidation(persona_name_override="Test Persona")

        assert config.persona_name_override == "Test Persona"

    def test_serialization_round_trip(self):
        """Test FormatValidation can be serialized and deserialized."""
        original = FormatValidation(
            require_emotional_header=True,
            max_retries=7,
            fallback_model="opus",
            persona_name_override="Andi"
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = FormatValidation(**data)

        assert restored.require_emotional_header == original.require_emotional_header
        assert restored.max_retries == original.max_retries
        assert restored.fallback_model == original.fallback_model
        assert restored.persona_name_override == original.persona_name_override

    def test_minimal_config(self):
        """Test FormatValidation with minimal required configuration."""
        config = FormatValidation(require_emotional_header=True)

        # Other fields should have defaults
        assert config.max_retries == 3
        assert config.fallback_model == "fallback"
        assert config.persona_name_override is None


class TestStepConfigWithFormatValidation:
    """Tests for StepConfig.format_validation field."""

    def test_step_config_default_has_format_validation(self):
        """Test StepConfig has format_validation enabled by default."""
        config = StepConfig()

        assert config.format_validation is not None
        assert config.format_validation.require_emotional_header is True
        assert config.format_validation.max_retries == 3
        assert config.format_validation.fallback_model == "fallback"

    def test_step_config_with_format_validation(self):
        """Test StepConfig with format_validation."""
        format_val = FormatValidation(
            require_emotional_header=True,
            max_retries=5
        )
        config = StepConfig(format_validation=format_val)

        assert config.format_validation is not None
        assert config.format_validation.require_emotional_header is True
        assert config.format_validation.max_retries == 5

    def test_step_config_inline_format_validation(self):
        """Test StepConfig with inline format_validation dict."""
        config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                fallback_model="haiku"
            )
        )

        assert config.format_validation.require_emotional_header is True
        assert config.format_validation.fallback_model == "haiku"

    def test_step_config_serialization_with_format_validation(self):
        """Test StepConfig serialization includes format_validation."""
        format_val = FormatValidation(require_emotional_header=True)
        config = StepConfig(
            max_tokens=2048,
            temperature=0.8,
            format_validation=format_val
        )

        data = config.model_dump()

        assert "format_validation" in data
        assert data["format_validation"]["require_emotional_header"] is True

    def test_step_config_deserialization_with_format_validation(self):
        """Test StepConfig can be deserialized with format_validation."""
        data = {
            "max_tokens": 1024,
            "format_validation": {
                "require_emotional_header": True,
                "max_retries": 5,
                "fallback_model": "haiku",
                "persona_name_override": "Test"
            }
        }

        config = StepConfig(**data)

        assert config.format_validation is not None
        assert config.format_validation.require_emotional_header is True
        assert config.format_validation.max_retries == 5
        assert config.format_validation.fallback_model == "haiku"
        assert config.format_validation.persona_name_override == "Test"


class TestStepDefinitionWithFormatValidation:
    """Tests for step definitions with format_validation."""

    def test_standard_step_with_format_validation(self):
        """Test StandardStepDefinition with format_validation in config."""
        step = StandardStepDefinition(
            id="test_step",
            prompt="Test prompt",
            config=StepConfig(
                format_validation=FormatValidation(
                    require_emotional_header=True,
                    max_retries=4
                )
            ),
            output=StepOutput(document_type="test"),
            next=["end"]
        )

        assert step.config.format_validation is not None
        assert step.config.format_validation.require_emotional_header is True
        assert step.config.format_validation.max_retries == 4

    def test_standard_step_default_format_validation(self):
        """Test StandardStepDefinition has format_validation enabled by default."""
        step = StandardStepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
            next=["end"]
        )

        # Format validation is enabled by default
        assert step.config.format_validation is not None
        assert step.config.format_validation.require_emotional_header is True

    def test_standard_step_serialization_round_trip(self):
        """Test StandardStepDefinition serialization with format_validation."""
        original = StandardStepDefinition(
            id="test_step",
            prompt="Test prompt",
            config=StepConfig(
                max_tokens=2048,
                format_validation=FormatValidation(
                    require_emotional_header=True,
                    max_retries=6,
                    fallback_model="opus"
                )
            ),
            output=StepOutput(document_type="test", weight=0.8),
            next=["next_step"]
        )

        # Serialize
        data = original.model_dump()

        # Deserialize
        restored = StandardStepDefinition(**data)

        assert restored.config.format_validation.require_emotional_header is True
        assert restored.config.format_validation.max_retries == 6
        assert restored.config.format_validation.fallback_model == "opus"

    def test_format_validation_enabled_by_default(self):
        """Test that format_validation is enabled by default."""
        step = StandardStepDefinition(
            id="test",
            prompt="Test",
            output=StepOutput(document_type="test"),
            next=["end"]
        )

        # Default StepConfig has format_validation with require_emotional_header=True
        assert step.config.format_validation is not None
        assert step.config.format_validation.require_emotional_header is True

        # To disable validation, set require_emotional_header=False
        step_disabled = StandardStepDefinition(
            id="test2",
            prompt="Test",
            config=StepConfig(
                format_validation=FormatValidation(require_emotional_header=False)
            ),
            output=StepOutput(document_type="test"),
            next=["end"]
        )
        assert step_disabled.config.format_validation.require_emotional_header is False
