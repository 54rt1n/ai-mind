# tests/unit/dreamer/test_executor.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for aim/dreamer/executor.py"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch

from aim.dreamer.core.executor import (
    select_model_name,
    build_turns,
    format_memories_xml,
    extract_think_tags,
    create_message,
    RetryableError,
    execute_step,
)
from aim.dreamer.core.models import (
    PipelineState,
    StepDefinition,
    StepConfig,
    StepOutput,
    StepResult,
    Scenario,
    ScenarioContext,
    MemoryAction,
)


@pytest.fixture
def mock_model_set():
    """Create a mock ModelSet for testing."""
    model_set = MagicMock()
    model_set.default_model = "default-model"
    model_set.analysis_model = "analysis-model"
    model_set.codex_model = "codex-model"
    model_set.get_model_name = MagicMock(side_effect=lambda role: f"{role}-model")
    return model_set


class TestSelectModelName:
    """Test model selection logic."""

    def test_analyst_scenario_codex_step_uses_codex_model(self, mock_model_set):
        """Verify the analyst scenario's codex step has is_codex=True and uses codex_model.

        This is an integration test that loads the actual analyst.yaml scenario
        and verifies the codex step is properly configured to use the codex_model.
        """
        from aim.dreamer.core.scenario import load_scenario

        # Load the actual analyst scenario
        scenario = load_scenario("analyst")

        # Verify the codex step exists and has is_codex=True
        assert "codex" in scenario.steps, "Analyst scenario should have a 'codex' step"
        codex_step = scenario.steps["codex"]
        assert codex_step.config.is_codex is True, (
            f"Codex step should have is_codex=True, got {codex_step.config.is_codex}"
        )

        # Create a state with codex_model set
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="analyst",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            codex_model="codex-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Verify select_model_name returns the codex_model for this step
        result = select_model_name(state, codex_step.config, mock_model_set)
        assert result == "codex-model", (
            f"Expected codex-model for codex step, got {result}"
        )

    def test_analyst_codex_step_falls_back_when_codex_model_not_set(self, mock_model_set):
        """Verify codex step falls back to default model when codex_model is not configured.

        This tests the common misconfiguration where is_codex=True in the scenario
        but codex_model is not set in the config/state - the step will still run
        but use the default model instead of a specialized codex model.
        """
        from aim.dreamer.core.scenario import load_scenario

        scenario = load_scenario("analyst")
        codex_step = scenario.steps["codex"]

        # State WITHOUT codex_model set (None)
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="analyst",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            codex_model=None,  # Not set!
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Should fall back to default model
        result = select_model_name(state, codex_step.config, mock_model_set)
        assert result == "default-model", (
            f"Expected default-model when codex_model is None, got {result}"
        )

    def test_model_override_takes_precedence(self, mock_model_set):
        """Model override should be used if set."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            thought_model="thought-model",
            codex_model="codex-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        step_config = StepConfig(model_override="override-model")

        result = select_model_name(state, step_config, mock_model_set)
        assert result == "override-model"

    def test_codex_model_when_is_codex(self, mock_model_set):
        """Codex model should be used when is_codex is True."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            codex_model="codex-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        step_config = StepConfig(is_codex=True)

        result = select_model_name(state, step_config, mock_model_set)
        assert result == "codex-model"

    def test_thought_model_when_is_thought(self, mock_model_set):
        """Thought model should be used when is_thought is True."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            thought_model="thought-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        step_config = StepConfig(is_thought=True)

        result = select_model_name(state, step_config, mock_model_set)
        assert result == "thought-model"

    def test_default_model_when_no_overrides(self, mock_model_set):
        """Default model should be used when no overrides are set."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        step_config = StepConfig()

        result = select_model_name(state, step_config, mock_model_set)
        assert result == "default-model"

    def test_codex_without_codex_model_falls_back_to_default(self, mock_model_set):
        """When is_codex is True but no codex_model, fall back to default."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            codex_model=None,
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        step_config = StepConfig(is_codex=True)

        result = select_model_name(state, step_config, mock_model_set)
        assert result == "default-model"


class TestBuildTurns:
    """Test turns list construction."""

    def test_basic_turns_structure(self):
        """Basic turns should include system, prompt, and wakeup if memories present."""
        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI assistant."
        mock_persona.get_wakeup.return_value = "Hello!"

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            persona_mood="cheerful",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        prompt = "Please analyze this."
        memories = []
        prior_outputs = []

        turns, system_message = build_turns(
            state, prompt, memories, prior_outputs, mock_persona,
            max_context_tokens=32768, max_output_tokens=4096
        )

        # System message is returned separately (for config.system_message)
        assert system_message == "You are an AI assistant."

        # Turns should only have user prompt (no system in turns)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Please analyze this."

        # Verify system_prompt was called with correct parameters
        mock_persona.system_prompt.assert_called_once_with(
            mood="cheerful",
            location=None,
            user_id="user",
            system_message=None,
        )

    def test_turns_with_memories(self):
        """Turns with memories should include memory XML and wakeup."""
        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI assistant."
        mock_persona.get_wakeup.return_value = "Hello!"

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        prompt = "Please analyze this."
        memories = [
            {"content": "Previous conversation.", "timestamp": 123456},
        ]
        prior_outputs = []

        turns, system_message = build_turns(
            state, prompt, memories, prior_outputs, mock_persona,
            max_context_tokens=32768, max_output_tokens=4096
        )

        # System message is returned separately
        assert system_message == "You are an AI assistant."

        # Should have memory context, wakeup, and prompt (no system in turns)
        assert len(turns) == 3
        assert turns[0]["role"] == "user"
        assert "<memories>" in turns[0]["content"]
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hello!"
        assert turns[2]["role"] == "user"
        assert turns[2]["content"] == "Please analyze this."

    def test_turns_with_prior_steps(self):
        """Turns should include prior step outputs loaded from CVM."""
        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI assistant."
        mock_persona.get_wakeup.return_value = "Hello!"

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        prompt = "Continue the analysis."
        memories = []
        prior_outputs = [
            {"content": "First step output.", "step_id": "step1"},
            {"content": "Second step output.", "step_id": "step2"},
        ]

        turns, system_message = build_turns(
            state, prompt, memories, prior_outputs, mock_persona,
            max_context_tokens=32768, max_output_tokens=4096
        )

        # System message is returned separately
        assert system_message == "You are an AI assistant."

        # Should have 2 prior outputs and current prompt (no system in turns)
        assert len(turns) == 3
        assert turns[0]["role"] == "assistant"
        assert turns[0]["content"] == "First step output."
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Second step output."
        assert turns[2]["role"] == "user"
        assert turns[2]["content"] == "Continue the analysis."

    def test_memory_trimming(self):
        """Memories should be trimmed when exceeding token budget."""
        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "System."
        mock_persona.get_wakeup.return_value = "Hi!"

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        prompt = "Analyze."
        # Create many memories that would exceed a small budget
        memories = [
            {"content": f"Memory {i} with some content." * 10}
            for i in range(100)
        ]
        prior_outputs = []

        # With very small context, should trim memories
        turns, system_message = build_turns(
            state, prompt, memories, prior_outputs, mock_persona,
            max_context_tokens=2048, max_output_tokens=512
        )

        # Should have some turns but not all 100 memories
        assert len(turns) >= 1  # At least the prompt
        if len(turns) > 1:
            # If memories included, check they were trimmed
            memory_turn = turns[0]["content"]
            memory_count = memory_turn.count("<memory>")
            assert memory_count < 100, f"Expected fewer than 100 memories, got {memory_count}"


class TestFormatMemoriesXML:
    """Test XML formatting for memories."""

    def test_empty_memories(self):
        """Empty memories list should return empty memories tag."""
        result = format_memories_xml([])
        assert result == "<memories>\n</memories>"

    def test_single_memory(self):
        """Single memory should be properly formatted."""
        memories = [
            {"content": "This is a memory.", "timestamp": 123456}
        ]

        result = format_memories_xml(memories)

        assert "<memories>" in result
        assert "<memory>" in result
        assert "This is a memory." in result
        assert "</memory>" in result
        assert "</memories>" in result

    def test_multiple_memories(self):
        """Multiple memories should be properly formatted."""
        memories = [
            {"content": "First memory.", "timestamp": 123456},
            {"content": "Second memory.", "timestamp": 123457},
            {"content": "Third memory.", "timestamp": 123458},
        ]

        result = format_memories_xml(memories)

        assert result.count("<memory>") == 3
        assert result.count("</memory>") == 3
        assert "First memory." in result
        assert "Second memory." in result
        assert "Third memory." in result


class TestExtractThinkTags:
    """Test think tag extraction."""

    def test_no_think_tags(self):
        """Response without think tags should return unchanged content."""
        response = "This is a normal response."

        content, think = extract_think_tags(response)

        assert content == "This is a normal response."
        assert think is None

    def test_with_think_tags(self):
        """Response with think tags should extract them."""
        response = "<think>Internal reasoning here.</think>This is the visible response."

        content, think = extract_think_tags(response)

        assert content == "This is the visible response."
        assert think == "Internal reasoning here."

    def test_multiple_think_tags(self):
        """Multiple think tags should all be extracted."""
        response = "<think>First thought.</think>Some text.<think>Second thought.</think>More text."

        content, think = extract_think_tags(response)

        # Content should have think tags removed
        assert "<think>" not in content
        assert "</think>" not in content
        # Think should contain the extracted content (implementation may vary)
        assert think is not None

    def test_think_tags_with_newlines(self):
        """Think tags spanning multiple lines should work."""
        response = """<think>
This is a multi-line
internal thought.
</think>
Final response here."""

        content, think = extract_think_tags(response)

        assert "Final response here." in content
        assert "multi-line" in think


class TestCreateMessage:
    """Test ConversationMessage creation."""

    def test_creates_valid_message(self):
        """Should create a valid ConversationMessage."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            step_counter=3,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        step_def = StepDefinition(
            id="analyze",
            prompt="Analyze this.",
            output=StepOutput(document_type="analysis", weight=0.8),
        )

        result = StepResult(
            step_id="analyze",
            response="Analysis complete.",
            think="Internal reasoning.",
            doc_id="doc-123",
            document_type="analysis",
            document_weight=0.8,
            tokens_used=50,
            timestamp=datetime.now(timezone.utc),
        )

        message = create_message(state, step_def, result)

        assert message.doc_id == "doc-123"
        assert message.conversation_id == "conv-1"
        assert message.user_id == "user"
        assert message.persona_id == "assistant"
        assert message.sequence_no == 6  # step_counter * 2
        assert message.branch == 0
        assert message.role == "assistant"
        assert message.content == "Analysis complete."
        assert message.think == "Internal reasoning."
        assert message.document_type == "analysis"
        assert message.weight == 0.8
        assert message.speaker_id == "assistant"
        assert message.inference_model == "default-model"


class TestExecuteStep:
    """Test full step execution (integration test with mocks)."""

    @pytest.mark.asyncio
    async def test_execute_step_basic_flow(self, mock_model_set):
        """Test basic step execution flow."""
        from aim.config import ChatConfig

        # Create mock objects
        mock_cvm = Mock()
        mock_cvm.query.return_value = Mock(empty=True, to_dict=Mock(return_value=[]))

        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI."
        mock_persona.get_wakeup.return_value = "Hello!"
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}

        # Use real ChatConfig instead of Mock for dataclass compatibility
        mock_config = ChatConfig(temperature=0.7)

        # Create state
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Create scenario
        scenario = Scenario(
            name="test",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={},
        )

        # Create step definition
        step_def = StepDefinition(
            id="test_step",
            prompt="Test prompt: {{ step_num }}",
            config=StepConfig(max_tokens=100),
            output=StepOutput(document_type="test", weight=1.0),
        )

        # Mock the LLM provider
        with patch("aim.dreamer.core.executor.LanguageModelV2") as mock_model_v2:
            mock_model_instance = Mock()
            mock_model_instance.max_output_tokens = 4096
            mock_model_instance.max_tokens = 32768
            mock_provider = Mock()
            mock_provider.stream_turns.return_value = iter(["Test", " response", "."])
            mock_model_instance.llm_factory.return_value = mock_provider

            mock_models_dict = {"gpt-4": mock_model_instance}
            mock_model_v2.index_models.return_value = mock_models_dict

            # Execute step
            result, context_doc_ids, is_initial_context = await execute_step(
                state=state,
                scenario=scenario,
                step_def=step_def,
                cvm=mock_cvm,
                persona=mock_persona,
                config=mock_config,
                model_set=mock_model_set,
            )

        # Verify result
        assert result.step_id == "test_step"
        assert result.response == "Test response."
        assert result.document_type == "test"
        assert result.document_weight == 1.0
        assert result.tokens_used > 0
        assert isinstance(context_doc_ids, list)
        assert isinstance(is_initial_context, bool)

    @pytest.mark.asyncio
    async def test_execute_step_with_think_tags(self, mock_model_set):
        """Test step execution with think tags in response."""
        from aim.config import ChatConfig

        mock_cvm = Mock()
        mock_cvm.query.return_value = Mock(empty=True, to_dict=Mock(return_value=[]))

        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI."
        mock_persona.get_wakeup.return_value = "Hello!"
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}

        mock_config = ChatConfig(temperature=0.7)

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        scenario = Scenario(
            name="test",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[]),
            steps={},
        )

        step_def = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            config=StepConfig(max_tokens=100),
            output=StepOutput(document_type="test", weight=1.0),
        )

        with patch("aim.dreamer.core.executor.LanguageModelV2") as mock_model_v2:
            mock_model_instance = Mock()
            mock_model_instance.max_output_tokens = 4096
            mock_model_instance.max_tokens = 32768
            mock_provider = Mock()
            mock_provider.stream_turns.return_value = iter([
                "<think>Internal reasoning</think>",
                "Visible response."
            ])
            mock_model_instance.llm_factory.return_value = mock_provider

            mock_models_dict = {"gpt-4": mock_model_instance}
            mock_model_v2.index_models.return_value = mock_models_dict

            result, context_doc_ids, is_initial_context = await execute_step(
                state=state,
                scenario=scenario,
                step_def=step_def,
                cvm=mock_cvm,
                persona=mock_persona,
                config=mock_config,
                model_set=mock_model_set,
            )

        assert result.response == "Visible response."
        assert result.think == "Internal reasoning"
        assert isinstance(context_doc_ids, list)
        assert isinstance(is_initial_context, bool)

    @pytest.mark.asyncio
    async def test_execute_step_with_memories(self, mock_model_set):
        """Test step execution with memory query."""
        import pandas as pd
        from aim.config import ChatConfig

        mock_cvm = Mock()
        mock_memories_df = pd.DataFrame([
            {"doc_id": "doc1", "content": "Previous memory 1", "timestamp": 123456},
            {"doc_id": "doc2", "content": "Previous memory 2", "timestamp": 123457},
        ])
        mock_cvm.query.return_value = mock_memories_df
        # Mock get_by_doc_id for context loading
        mock_cvm.get_by_doc_id.side_effect = lambda doc_id: {
            "doc1": {"content": "Previous memory 1", "timestamp": 123456},
            "doc2": {"content": "Previous memory 2", "timestamp": 123457},
        }.get(doc_id)

        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI."
        mock_persona.get_wakeup.return_value = "Hello!"
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}

        mock_config = ChatConfig(temperature=0.7)

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        scenario = Scenario(
            name="test",
            description="Test scenario",
            context=ScenarioContext(required_aspects=[], core_documents=["summary"]),
            steps={},
        )

        step_def = StepDefinition(
            id="test_step",
            prompt="Analyze memories",
            config=StepConfig(max_tokens=100),
            output=StepOutput(document_type="analysis", weight=1.0),
            context=[
                MemoryAction(
                    action="search_memories",
                    query_text="test search query",
                    top_n=5,
                    document_types=["conversation"],
                ),
            ],
        )

        with patch("aim.dreamer.core.executor.LanguageModelV2") as mock_model_v2:
            mock_model_instance = Mock()
            mock_model_instance.max_output_tokens = 4096
            mock_model_instance.max_tokens = 32768
            mock_provider = Mock()
            mock_provider.stream_turns.return_value = iter(["Memory analysis complete."])
            mock_model_instance.llm_factory.return_value = mock_provider

            mock_models_dict = {"gpt-4": mock_model_instance}
            mock_model_v2.index_models.return_value = mock_models_dict

            result, context_doc_ids, is_initial_context = await execute_step(
                state=state,
                scenario=scenario,
                step_def=step_def,
                cvm=mock_cvm,
                persona=mock_persona,
                config=mock_config,
                model_set=mock_model_set,
            )

        # Verify memory query was called
        mock_cvm.query.assert_called_once()
        assert result.response == "Memory analysis complete."
        assert isinstance(context_doc_ids, list)
        assert isinstance(is_initial_context, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
