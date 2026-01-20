# packages/aim-mud/tests/mud_tests/unit/worker/test_dreamer.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for DreamerMixin.

NOTE: As of 2026-01-19, the legacy DreamerRunner was removed in favor of the
strategy-based ScenarioFramework system. Dreams are now executed using
`initialize_scenario_dream()` and `execute_scenario_step()` methods.

The inline scheduler has been replaced with a direct execution model where
steps are executed one-by-one through idle turns.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from andimud_worker.mixins.dreamer import (
    DreamResult,
    DreamDecision,
    CONVERSATION_ANALYSIS_SCENARIOS,
)
from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from aim.config import ChatConfig


class TestConversationAnalysisScenarios:
    """Test the CONVERSATION_ANALYSIS_SCENARIOS constant."""

    def test_expected_scenarios_present(self):
        """Test all expected scenarios are in the set."""
        expected = {"analysis_dialogue", "summarizer"}
        assert CONVERSATION_ANALYSIS_SCENARIOS == expected


class TestDreamResult:
    """Test DreamResult dataclass."""

    def test_create_success_result(self):
        """Test creating successful DreamResult."""
        result = DreamResult(
            success=True,
            pipeline_id="pipeline_123",
            scenario="journaler_dialogue",
            duration_seconds=45.2,
        )
        assert result.success is True
        assert result.pipeline_id == "pipeline_123"
        assert result.scenario == "journaler_dialogue"
        assert result.error is None
        assert result.duration_seconds == 45.2

    def test_create_failure_result(self):
        """Test creating failed DreamResult."""
        result = DreamResult(
            success=False,
            scenario="analysis_dialogue",
            error="Pipeline failed",
            duration_seconds=10.5,
        )
        assert result.success is False
        assert result.pipeline_id is None
        assert result.scenario == "analysis_dialogue"
        assert result.error == "Pipeline failed"
        assert result.duration_seconds == 10.5

    def test_create_with_defaults(self):
        """Test creating DreamResult with default values."""
        result = DreamResult(success=True)
        assert result.success is True
        assert result.pipeline_id is None
        assert result.scenario is None
        assert result.error is None
        assert result.duration_seconds == 0.0


class TestDreamDecision:
    """Test DreamDecision dataclass."""

    def test_create_with_all_fields(self):
        """Test creating DreamDecision with all fields."""
        decision = DreamDecision(
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            query="What happened?",
            guidance="Focus on emotions",
        )
        assert decision.scenario == "analysis_dialogue"
        assert decision.conversation_id == "conv_123"
        assert decision.query == "What happened?"
        assert decision.guidance == "Focus on emotions"

    def test_create_with_defaults(self):
        """Test creating DreamDecision with default values."""
        decision = DreamDecision(
            scenario="summarizer",
            conversation_id="conv_456",
        )
        assert decision.scenario == "summarizer"
        assert decision.conversation_id == "conv_456"
        assert decision.query is None
        assert decision.guidance is None


class TestConversationIdSelection:
    """Test conversation ID selection logic used by process_dream_turn.

    Analysis scenarios (analysis_dialogue, summarizer) use the MUD conversation ID.
    Creative scenarios generate standalone conversation IDs.
    """

    def test_analysis_scenarios_in_constant(self):
        """Verify analysis scenarios are correctly identified."""
        assert "analysis_dialogue" in CONVERSATION_ANALYSIS_SCENARIOS
        assert "summarizer" in CONVERSATION_ANALYSIS_SCENARIOS
        assert "journaler_dialogue" not in CONVERSATION_ANALYSIS_SCENARIOS
        assert "daydream_dialogue" not in CONVERSATION_ANALYSIS_SCENARIOS


def test_chatconfig_fixture_has_no_fake_attributes(test_config):
    """Verify our test fixtures don't add nonexistent attributes to ChatConfig.

    This test exists because we previously corrupted ChatConfig with fake
    attributes (model_name) that don't exist on the real class, causing
    production crashes. NEVER AGAIN.

    Production crashed on 2026-01-08 because:
    - Test fixture added config.model_name (FAKE attribute)
    - Tests passed with fake data
    - Production code used config.model_name at runner.py:224
    - Production crashed: 'ChatConfig' object has no attribute 'model_name'

    The correct attribute is default_model, not model_name.
    """
    # These are REAL attributes that should exist on ChatConfig
    base_config = ChatConfig()
    assert hasattr(base_config, 'default_model'), "ChatConfig should have default_model"
    assert hasattr(base_config, 'thought_model'), "ChatConfig should have thought_model"
    assert hasattr(base_config, 'codex_model'), "ChatConfig should have codex_model"

    # This is a FAKE attribute that should NOT exist
    assert not hasattr(base_config, 'model_name'), \
        "ChatConfig should NOT have model_name - use default_model instead"

    # Verify our fixture doesn't corrupt the object
    # Fixture should only set attributes that exist on real ChatConfig
    # Get all non-dunder attributes from both objects
    real_attrs = {attr for attr in dir(base_config) if not attr.startswith('_')}
    fixture_attrs = {attr for attr in dir(test_config) if not attr.startswith('_')}

    # Find attributes that exist on fixture but not on real class
    fake_attrs = fixture_attrs - real_attrs
    assert fake_attrs == set(), \
        f"Fixture adds fake attributes: {fake_attrs}. Use only real ChatConfig attributes."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
