# aim/dreamer/core/strategy/validation_mixin.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""FormatValidationMixin - Provides retry logic for response format validation.

This mixin enables strategies to automatically retry LLM calls when
response format requirements are not met. It supports:
- Emotional state header validation
- Progressive guidance injection on failures
- Fallback model support after retries exhausted
"""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from aim.utils.format_validation import has_emotional_state_header
from aim.utils.think import extract_think_tags

if TYPE_CHECKING:
    from ..models import FormatValidation


logger = logging.getLogger(__name__)


class FormatValidationMixin:
    """Mixin providing format validation and retry logic for LLM responses.

    Strategies that inherit this mixin must implement:
    - _stream_response_inner(turns, system_message) -> str
    - _stream_response_with_model(turns, system_message, model) -> str
    - _get_model() -> model object
    - _get_model_by_role(role: str) -> model object

    The mixin provides:
    - _stream_with_format_validation() - Main entry point for validated streaming
    - _inject_format_guidance() - Append guidance to turns on format failure

    Usage:
        class MyStrategy(FormatValidationMixin, BaseStepStrategy):
            async def execute(self):
                ...
                response, used_fallback = await self._stream_with_format_validation(
                    turns, system_message
                )
    """

    @abstractmethod
    async def _stream_response_inner(
        self,
        turns: list[dict],
        system_message: str,
    ) -> str:
        """Stream LLM response without validation. Must be implemented by strategy.

        Args:
            turns: Conversation turns
            system_message: System prompt

        Returns:
            Complete response string
        """
        pass

    @abstractmethod
    async def _stream_response_with_model(
        self,
        turns: list[dict],
        system_message: str,
        model_role: str,
    ) -> str:
        """Stream LLM response using a specific model role. Must be implemented by strategy.

        Args:
            turns: Conversation turns
            system_message: System prompt
            model_role: Model role name (e.g., "fallback")

        Returns:
            Complete response string
        """
        pass

    @abstractmethod
    def _get_model(self):
        """Get the language model for this step. Must be implemented by strategy."""
        pass

    @abstractmethod
    def _get_model_by_role(self, role: str):
        """Get a language model by role name. Must be implemented by strategy."""
        pass

    def _get_format_validation(self) -> Optional["FormatValidation"]:
        """Get format validation config from step definition.

        Returns:
            FormatValidation config or None if not configured
        """
        if hasattr(self, 'step_def') and hasattr(self.step_def, 'config'):
            return getattr(self.step_def.config, 'format_validation', None)
        return None

    def _get_persona_name(self) -> str:
        """Get the name to use for emotional state header validation.

        Resolution order:
        1. format_validation.persona_name_override if set
        2. For dialogue steps with aspect speaker: aspect.name
        3. For dialogue steps with persona speaker: persona.name
        4. Default: persona.name (not full_name)
        5. Fallback: "Agent"

        Returns:
            Name string for validation
        """
        from aim.agents.aspects import get_aspect_or_default
        from ..models import SpeakerType

        # Check for explicit override first
        format_config = self._get_format_validation()
        if format_config and format_config.persona_name_override:
            return format_config.persona_name_override

        # Check if this is a dialogue step with speaker info
        if hasattr(self, 'step_def') and hasattr(self.step_def, 'speaker'):
            speaker = self.step_def.speaker
            if speaker and hasattr(self, 'executor') and self.executor.persona:
                if speaker.type == SpeakerType.ASPECT:
                    # Resolve aspect name
                    aspect_name = speaker.aspect_name
                    if not aspect_name and self.executor.framework and self.executor.framework.dialogue:
                        aspect_name = self.executor.framework.dialogue.primary_aspect
                    if aspect_name:
                        aspect = get_aspect_or_default(self.executor.persona, aspect_name)
                        return aspect.name
                # For persona speaker, fall through to use persona.name

        # Default: use persona.name (not full_name)
        if hasattr(self, 'executor') and hasattr(self.executor, 'persona') and self.executor.persona:
            return self.executor.persona.name

        return "Agent"

    async def _stream_with_format_validation(
        self,
        turns: list[dict],
        system_message: str,
    ) -> tuple[str, bool]:
        """Stream LLM response with format validation and retry logic.

        If format_validation is not configured or require_emotional_header
        is False, performs a single pass without validation.

        If configured, runs retry loop up to max_retries times, injecting
        progressive format guidance on failures. If all retries fail and
        fallback_model is configured, attempts one final call with the
        fallback model.

        Args:
            turns: Conversation turns (will be copied, not mutated)
            system_message: System prompt

        Returns:
            Tuple of (response_string, used_fallback: bool)
        """
        format_config = self._get_format_validation()

        # No validation configured - single pass
        if not format_config or not format_config.require_emotional_header:
            response = await self._stream_response_inner(turns, system_message)
            return response, False

        # Validation enabled - run retry loop
        max_retries = format_config.max_retries
        fallback_model = format_config.fallback_model
        persona_name = self._get_persona_name()

        # Work with a copy of turns to avoid mutating the original
        working_turns = [dict(t) for t in turns]
        used_fallback = False
        response = ""

        for attempt in range(max_retries + 1):  # +1 for fallback attempt
            # Determine if this is a fallback attempt
            is_fallback = attempt >= max_retries

            if is_fallback:
                # Check if fallback is configured and different from primary
                if not fallback_model:
                    logger.warning("Fallback model not configured; giving up after %d retries", max_retries)
                    break

                primary_model = self._get_model()
                fallback_model_obj = self._get_model_by_role(fallback_model)

                if fallback_model_obj is None:
                    logger.warning("Fallback model role '%s' not found; giving up", fallback_model)
                    break

                primary_name = getattr(primary_model, 'name', str(primary_model)) if primary_model else 'unknown'
                fallback_name = getattr(fallback_model_obj, 'name', str(fallback_model_obj))

                if primary_name == fallback_name:
                    logger.warning("Fallback model same as primary; giving up after %d retries", max_retries)
                    break

                logger.info(
                    "Attempting fallback model (%s) after %d failures",
                    fallback_name, max_retries
                )
                used_fallback = True
                response = await self._stream_response_with_model(
                    working_turns, system_message, fallback_model
                )
            else:
                # Regular attempt
                attempt_label = f"{attempt + 1}/{max_retries}"
                logger.debug("Format validation attempt %s", attempt_label)
                response = await self._stream_response_inner(working_turns, system_message)

            # Extract think tags and validate format
            cleaned_response, _ = extract_think_tags(response)

            if has_emotional_state_header(cleaned_response, persona_name):
                if used_fallback:
                    logger.info("Fallback model succeeded with format validation")
                elif attempt > 0:
                    logger.info("Format validation succeeded on attempt %d", attempt + 1)
                return response, used_fallback

            # Format validation failed
            if is_fallback:
                logger.warning("Fallback model also failed format validation")
                break

            logger.warning(
                "Response missing Emotional State header (attempt %d/%d)",
                attempt + 1, max_retries
            )

            # Inject guidance for next attempt
            self._inject_format_guidance(
                working_turns,
                persona_name,
                attempt + 1,
                max_retries,
            )

        # All attempts failed - return last response anyway
        logger.error(
            "Format validation failed after %d attempts%s",
            max_retries,
            " and fallback" if used_fallback else ""
        )
        return response, used_fallback

    def _inject_format_guidance(
        self,
        turns: list[dict],
        persona_name: str,
        attempt: int,
        max_retries: int,
    ) -> None:
        """Inject format guidance into turns for next retry attempt.

        Uses progressive wording: gentle on early attempts, critical
        before fallback model is tried.

        Args:
            turns: Conversation turns (will be mutated)
            persona_name: Name to use in guidance example
            attempt: Current attempt number (1-indexed)
            max_retries: Maximum retry attempts
        """
        is_final_retry = attempt >= max_retries

        if is_final_retry:
            # Critical wording before fallback
            guidance = (
                f"\n\n[CRITICAL FORMAT REQUIREMENT: You MUST begin your response with "
                f"[== {persona_name}'s Emotional State: <your emotions> ==] followed by prose.]"
            )
        else:
            # Gentle wording for early attempts
            guidance = (
                f"\n\n[Gentle reminder: Please begin with your emotional state, "
                f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotions+> ==] "
                f"then continue with prose.]"
            )

        # Append to last user turn, or add new user turn
        if turns and turns[-1]["role"] == "user":
            turns[-1]["content"] += guidance
        else:
            turns.append({"role": "user", "content": guidance.strip()})
