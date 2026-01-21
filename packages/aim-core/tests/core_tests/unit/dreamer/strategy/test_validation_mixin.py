# tests/core_tests/unit/dreamer/strategy/test_validation_mixin.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for FormatValidationMixin."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aim.dreamer.core.strategy.validation_mixin import FormatValidationMixin
from aim.dreamer.core.models import FormatValidation, StepConfig, StandardStepDefinition, StepOutput


# Test implementation of FormatValidationMixin
class TestStrategy(FormatValidationMixin):
    """Concrete test strategy implementing FormatValidationMixin."""

    def __init__(self, executor, step_def, llm_responses=None):
        """Initialize test strategy.

        Args:
            executor: Mock executor with persona and model_set
            step_def: Step definition
            llm_responses: List of responses to return from LLM (for testing)
        """
        self.executor = executor
        self.step_def = step_def
        self._llm_responses = llm_responses or []
        self._response_index = 0

    async def _stream_response_inner(self, turns, system_message):
        """Mock LLM response streaming."""
        if self._response_index < len(self._llm_responses):
            response = self._llm_responses[self._response_index]
            self._response_index += 1
            return response
        return "Default response"

    async def _stream_response_with_model(self, turns, system_message, model_role):
        """Mock LLM response streaming with specific model."""
        # For fallback testing, use the last response in the list
        if self._llm_responses:
            return self._llm_responses[-1]
        return "Fallback response"

    def _get_model(self):
        """Get primary model."""
        return self.executor.model_set.default_model

    def _get_model_by_role(self, role):
        """Get model by role."""
        if role == "fallback":
            fallback_model = MagicMock()
            fallback_model.name = "fallback-model"
            return fallback_model
        return None


# --- Fixtures ---

@pytest.fixture
def mock_persona():
    """Create a mock Persona."""
    persona = MagicMock()
    persona.full_name = "Test Persona"
    return persona


@pytest.fixture
def mock_model_set():
    """Create a mock ModelSet."""
    model_set = MagicMock()
    primary_model = MagicMock()
    primary_model.name = "primary-model"
    model_set.default_model = primary_model
    return model_set


@pytest.fixture
def mock_executor(mock_persona, mock_model_set):
    """Create a mock executor."""
    executor = MagicMock()
    executor.persona = mock_persona
    executor.model_set = mock_model_set
    return executor


@pytest.fixture
def base_step_def():
    """Create a base step definition without format validation."""
    return StandardStepDefinition(
        id="test_step",
        prompt="Test prompt",
        output=StepOutput(document_type="test"),
        next=["end"]
    )


# --- Test Cases ---

class TestFormatValidationMixinNoValidation:
    """Tests for mixin behavior when validation is not configured."""

    @pytest.mark.asyncio
    async def test_validation_disabled_via_flag_single_pass(self, mock_executor, base_step_def):
        """Test single pass when require_emotional_header is explicitly False."""
        # Validation is on by default, so we must explicitly disable it
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=False)
        )

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=["Response without header"]
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == "Response without header"
        assert used_fallback is False
        assert strategy._response_index == 1  # Called once

    @pytest.mark.asyncio
    async def test_validation_disabled_single_pass(self, mock_executor, base_step_def):
        """Test single pass when require_emotional_header is False."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=False)
        )

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=["Response without header"]
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == "Response without header"
        assert used_fallback is False
        assert strategy._response_index == 1


class TestFormatValidationMixinSuccess:
    """Tests for successful format validation scenarios."""

    @pytest.mark.asyncio
    async def test_validation_passes_first_try(self, mock_executor, base_step_def):
        """Test validation passes on first attempt."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=True, max_retries=3)
        )

        valid_response = "[== Test Persona's Emotional State: +Joy+ ==]\n\nResponse content"

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=[valid_response]
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == valid_response
        assert used_fallback is False
        assert strategy._response_index == 1  # Only called once

    @pytest.mark.asyncio
    async def test_validation_passes_with_think_block(self, mock_executor, base_step_def):
        """Test validation passes when header follows think block."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=True)
        )

        valid_response = (
            "<think>Internal reasoning here</think>\n"
            "[== Test Persona's Emotional State: +Curiosity+ ==]\n\n"
            "Response content"
        )

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=[valid_response]
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == valid_response
        assert used_fallback is False


class TestFormatValidationMixinRetry:
    """Tests for retry logic when validation fails."""

    @pytest.mark.asyncio
    async def test_validation_fails_succeeds_on_retry(self, mock_executor, base_step_def):
        """Test validation fails first, succeeds on second attempt."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=True, max_retries=3)
        )

        invalid_response = "Response without header"
        valid_response = "[== Test Persona's Emotional State: +Relief+ ==]\n\nFixed response"

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=[invalid_response, valid_response]
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == valid_response
        assert used_fallback is False
        assert strategy._response_index == 2  # Called twice

    @pytest.mark.asyncio
    async def test_validation_fails_multiple_times_then_succeeds(self, mock_executor, base_step_def):
        """Test multiple failures before success."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=True, max_retries=5)
        )

        responses = [
            "No header 1",
            "No header 2",
            "No header 3",
            "[== Test Persona's Emotional State: +Determination+ ==]\n\nSuccess!"
        ]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == responses[3]
        assert used_fallback is False
        assert strategy._response_index == 4

    @pytest.mark.asyncio
    async def test_all_retries_fail(self, mock_executor, base_step_def):
        """Test all retries exhausted returns last response."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=2,
                fallback_model=None  # Disable fallback
            )
        )

        responses = [
            "No header 1",
            "No header 2",
            "No header 3",  # max_retries=2 means 2 retries + fallback (disabled)
        ]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        # Should return last attempt's response
        assert response == "No header 2"  # Index 1 (second attempt before fallback)
        assert used_fallback is False
        assert strategy._response_index == 2

    @pytest.mark.asyncio
    async def test_guidance_injection_on_failure(self, mock_executor, base_step_def):
        """Test that guidance is injected into turns on validation failure."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=True, max_retries=2)
        )

        # Track turns passed to _stream_response_inner
        captured_turns = []

        async def capture_turns(turns, system_message):
            captured_turns.append([dict(t) for t in turns])  # Deep copy
            if len(captured_turns) == 1:
                return "No header"
            return "[== Test Persona's Emotional State: +Joy+ ==]\n\nFixed"

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )
        strategy._stream_response_inner = capture_turns

        initial_turns = [{"role": "user", "content": "test"}]

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=initial_turns,
            system_message="system"
        )

        # First call: original turns
        assert len(captured_turns[0]) == 1
        assert "Gentle reminder" not in captured_turns[0][0]["content"]

        # Second call: turns with guidance injected
        assert len(captured_turns[1]) == 1
        assert "Gentle reminder" in captured_turns[1][0]["content"]
        assert "Emotional State" in captured_turns[1][0]["content"]


class TestFormatValidationMixinFallback:
    """Tests for fallback model behavior."""

    @pytest.mark.asyncio
    async def test_fallback_model_used_after_retries(self, mock_executor, base_step_def):
        """Test fallback model is used after max_retries exhausted."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=2,
                fallback_model="fallback"
            )
        )

        responses = [
            "No header 1",
            "No header 2",
            "[== Test Persona's Emotional State: +Success+ ==]\n\nFallback worked!"
        ]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == responses[2]
        assert used_fallback is True

    @pytest.mark.asyncio
    async def test_fallback_disabled(self, mock_executor, base_step_def):
        """Test no fallback when fallback_model is None."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=2,
                fallback_model=None
            )
        )

        responses = ["No header 1", "No header 2"]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        # Should return last retry response
        assert response == "No header 2"
        assert used_fallback is False

    @pytest.mark.asyncio
    async def test_fallback_same_as_primary(self, mock_executor, base_step_def):
        """Test fallback skipped when it's the same as primary model."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=1,
                fallback_model="fallback"
            )
        )

        # Make fallback same as primary
        primary_model = MagicMock()
        primary_model.name = "same-model"
        mock_executor.model_set.default_model = primary_model

        fallback_model = MagicMock()
        fallback_model.name = "same-model"  # Same name as primary

        def get_model_by_role(role):
            if role == "fallback":
                return fallback_model
            return None

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=["No header"]
        )
        strategy._get_model_by_role = get_model_by_role

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        # Should not use fallback (same model)
        assert response == "No header"
        assert used_fallback is False

    @pytest.mark.asyncio
    async def test_fallback_model_not_found(self, mock_executor, base_step_def):
        """Test fallback skipped when model role not found."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=1,
                fallback_model="nonexistent"
            )
        )

        def get_model_by_role(role):
            return None  # Model not found

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=["No header"]
        )
        strategy._get_model_by_role = get_model_by_role

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        assert response == "No header"
        assert used_fallback is False

    @pytest.mark.asyncio
    async def test_fallback_also_fails_validation(self, mock_executor, base_step_def):
        """Test fallback model also fails validation."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=1,
                fallback_model="fallback"
            )
        )

        responses = [
            "No header from primary",
            "No header from fallback either"
        ]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        # Should return fallback's response even though it failed
        assert response == "No header from fallback either"
        assert used_fallback is True


class TestFormatValidationMixinGuidanceInjection:
    """Tests for guidance injection behavior."""

    def test_inject_format_guidance_early_attempt(self, mock_executor, base_step_def):
        """Test gentle guidance for early attempts."""
        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )

        turns = [{"role": "user", "content": "original"}]
        strategy._inject_format_guidance(turns, "Andi", attempt=1, max_retries=3)

        # Should append guidance to last user turn
        assert len(turns) == 1
        assert "Gentle reminder" in turns[0]["content"]
        assert "Andi's Emotional State" in turns[0]["content"]
        assert "CRITICAL" not in turns[0]["content"]

    def test_inject_format_guidance_final_retry(self, mock_executor, base_step_def):
        """Test critical guidance before fallback."""
        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )

        turns = [{"role": "user", "content": "original"}]
        strategy._inject_format_guidance(turns, "Nova", attempt=3, max_retries=3)

        # Should use critical wording
        assert "CRITICAL FORMAT REQUIREMENT" in turns[0]["content"]
        assert "MUST begin" in turns[0]["content"]
        assert "Nova's Emotional State" in turns[0]["content"]

    def test_inject_format_guidance_creates_new_turn(self, mock_executor, base_step_def):
        """Test guidance creates new turn if last turn is not user."""
        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )

        turns = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"}
        ]
        strategy._inject_format_guidance(turns, "Tiberius", attempt=1, max_retries=3)

        # Should create new user turn
        assert len(turns) == 3
        assert turns[2]["role"] == "user"
        assert "Gentle reminder" in turns[2]["content"]
        assert "Tiberius's Emotional State" in turns[2]["content"]

    def test_inject_format_guidance_preserves_original_turns(self, mock_executor, base_step_def):
        """Test that original turns list is not mutated."""
        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )

        original_turns = [{"role": "user", "content": "test"}]
        working_turns = [dict(t) for t in original_turns]  # Copy

        strategy._inject_format_guidance(working_turns, "Andi", attempt=1, max_retries=3)

        # Original should be unchanged
        assert len(original_turns) == 1
        assert "Gentle reminder" not in original_turns[0]["content"]

        # Working should be modified
        assert "Gentle reminder" in working_turns[0]["content"]


class TestFormatValidationMixinPersonaName:
    """Tests for persona name resolution."""

    def test_get_persona_name_from_executor(self, mock_executor, base_step_def):
        """Test persona name comes from executor.persona by default."""
        mock_executor.persona.full_name = "Andi Valentine"

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )

        name = strategy._get_persona_name()
        assert name == "Andi Valentine"

    def test_get_persona_name_override(self, mock_executor, base_step_def):
        """Test persona_name_override takes precedence."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                persona_name_override="Custom Name"
            )
        )

        mock_executor.persona.full_name = "Original Name"

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def
        )

        name = strategy._get_persona_name()
        assert name == "Custom Name"

    def test_get_persona_name_fallback(self, mock_executor, base_step_def):
        """Test fallback to 'Agent' when persona not available."""
        executor_without_persona = MagicMock(spec=[])  # Empty spec = no attributes
        # Don't set executor.persona

        strategy = TestStrategy(
            executor=executor_without_persona,
            step_def=base_step_def
        )

        name = strategy._get_persona_name()
        assert name == "Agent"


class TestFormatValidationMixinEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_turns_list(self, mock_executor, base_step_def):
        """Test validation with empty turns list."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(require_emotional_header=True)
        )

        valid_response = "[== Test Persona's Emotional State: +Joy+ ==]\n\nResponse"

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=[valid_response]
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[],
            system_message="system"
        )

        assert response == valid_response
        assert used_fallback is False

    @pytest.mark.asyncio
    async def test_max_retries_zero(self, mock_executor, base_step_def):
        """Test max_retries=0 means one attempt only."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=0,
                fallback_model="fallback"
            )
        )

        responses = [
            "No header",
            "[== Test Persona's Emotional State: +Joy+ ==]\n\nFallback"
        ]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        response, used_fallback = await strategy._stream_with_format_validation(
            turns=[{"role": "user", "content": "test"}],
            system_message="system"
        )

        # Should go straight to fallback after first failure
        assert response == responses[1]
        assert used_fallback is True

    @pytest.mark.asyncio
    async def test_turns_list_not_mutated(self, mock_executor, base_step_def):
        """Test that original turns list is not mutated."""
        base_step_def.config = StepConfig(
            format_validation=FormatValidation(
                require_emotional_header=True,
                max_retries=2
            )
        )

        original_turns = [{"role": "user", "content": "original"}]
        original_content = original_turns[0]["content"]

        responses = [
            "No header",
            "[== Test Persona's Emotional State: +Joy+ ==]\n\nFixed"
        ]

        strategy = TestStrategy(
            executor=mock_executor,
            step_def=base_step_def,
            llm_responses=responses
        )

        await strategy._stream_with_format_validation(
            turns=original_turns,
            system_message="system"
        )

        # Original turns should be unchanged
        assert len(original_turns) == 1
        assert original_turns[0]["content"] == original_content
        assert "Gentle reminder" not in original_turns[0]["content"]
