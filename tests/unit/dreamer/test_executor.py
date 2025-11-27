# tests/unit/dreamer/test_executor.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for aim/dreamer/executor.py"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import tiktoken

from aim.dreamer.executor import (
    select_model_name,
    build_turns,
    format_memories_xml,
    extract_think_tags,
    create_message,
    RetryableError,
    execute_step,
)
from aim.dreamer.models import (
    PipelineState,
    StepDefinition,
    StepConfig,
    StepOutput,
    StepMemory,
    StepResult,
    Scenario,
    ScenarioContext,
)


class TestSelectModelName:
    """Test model selection logic."""

    def test_model_override_takes_precedence(self):
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        step_config = StepConfig(model_override="override-model")

        result = select_model_name(state, step_config)
        assert result == "override-model"

    def test_codex_model_when_is_codex(self):
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        step_config = StepConfig(is_codex=True)

        result = select_model_name(state, step_config)
        assert result == "codex-model"

    def test_thought_model_when_is_thought(self):
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        step_config = StepConfig(is_thought=True)

        result = select_model_name(state, step_config)
        assert result == "thought-model"

    def test_default_model_when_no_overrides(self):
        """Default model should be used when no overrides are set."""
        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        step_config = StepConfig()

        result = select_model_name(state, step_config)
        assert result == "default-model"

    def test_codex_without_codex_model_falls_back_to_default(self):
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        step_config = StepConfig(is_codex=True)

        result = select_model_name(state, step_config)
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        prompt = "Please analyze this."
        memories = []
        prior_outputs = []

        turns, system_message = build_turns(state, prompt, memories, prior_outputs, mock_persona)

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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        prompt = "Please analyze this."
        memories = [
            {"content": "Previous conversation.", "timestamp": 123456},
        ]
        prior_outputs = []

        turns, system_message = build_turns(state, prompt, memories, prior_outputs, mock_persona)

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

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="default-model",
            branch=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        prompt = "Continue the analysis."
        memories = []
        prior_outputs = [
            {"content": "First step output.", "step_id": "step1"},
            {"content": "Second step output.", "step_id": "step2"},
        ]

        turns, system_message = build_turns(state, prompt, memories, prior_outputs, mock_persona)

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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            timestamp=datetime.utcnow(),
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
    async def test_execute_step_basic_flow(self):
        """Test basic step execution flow."""
        from aim.config import ChatConfig

        # Create mock objects
        mock_cvm = Mock()
        mock_cvm.query.return_value = Mock(empty=True, to_dict=Mock(return_value=[]))

        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI."
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}

        # Use real ChatConfig instead of Mock for dataclass compatibility
        mock_config = ChatConfig(temperature=0.7)

        encoder = tiktoken.get_encoding("cl100k_base")

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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            memory=StepMemory(top_n=0),
        )

        # Mock the LLM provider
        with patch("aim.dreamer.executor.LanguageModelV2") as mock_model_v2:
            mock_model_instance = Mock()
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
                encoder=encoder,
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
    async def test_execute_step_with_think_tags(self):
        """Test step execution with think tags in response."""
        from aim.config import ChatConfig

        mock_cvm = Mock()
        mock_cvm.query.return_value = Mock(empty=True, to_dict=Mock(return_value=[]))

        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI."
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}

        mock_config = ChatConfig(temperature=0.7)

        encoder = tiktoken.get_encoding("cl100k_base")

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            memory=StepMemory(top_n=0),
        )

        with patch("aim.dreamer.executor.LanguageModelV2") as mock_model_v2:
            mock_model_instance = Mock()
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
                encoder=encoder,
            )

        assert result.response == "Visible response."
        assert result.think == "Internal reasoning"
        assert isinstance(context_doc_ids, list)
        assert isinstance(is_initial_context, bool)

    @pytest.mark.asyncio
    async def test_execute_step_with_memories(self):
        """Test step execution with memory query."""
        import pandas as pd
        from aim.config import ChatConfig

        mock_cvm = Mock()
        mock_memories_df = pd.DataFrame([
            {"content": "Previous memory 1", "timestamp": 123456},
            {"content": "Previous memory 2", "timestamp": 123457},
        ])
        mock_cvm.query.return_value = mock_memories_df

        mock_persona = Mock()
        mock_persona.system_prompt.return_value = "You are an AI."
        mock_persona.get_wakeup.return_value = "Hello!"
        mock_persona.pronouns = {"subj": "they", "obj": "them", "poss": "their"}
        mock_persona.aspects = {}

        mock_config = ChatConfig(temperature=0.7)

        encoder = tiktoken.get_encoding("cl100k_base")

        state = PipelineState(
            pipeline_id="test-123",
            scenario_name="test",
            conversation_id="conv-1",
            persona_id="assistant",
            user_id="user",
            model="gpt-4",
            branch=0,
            step_counter=1,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            memory=StepMemory(top_n=5, document_type=["conversation"], sort_by="relevance"),
        )

        with patch("aim.dreamer.executor.LanguageModelV2") as mock_model_v2:
            mock_model_instance = Mock()
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
                encoder=encoder,
            )

        # Verify memory query was called
        mock_cvm.query.assert_called_once()
        assert result.response == "Memory analysis complete."
        assert isinstance(context_doc_ids, list)
        assert isinstance(is_initial_context, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
