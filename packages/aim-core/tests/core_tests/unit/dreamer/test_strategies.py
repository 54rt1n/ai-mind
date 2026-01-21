# tests/unit/dreamer/test_strategies.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for step execution strategies."""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from aim.dreamer.core.strategy import (
    BaseStepStrategy,
    ScenarioExecutor,
    ScenarioStepResult,
    StepFactory,
    ContextOnlyStrategy,
    StandardStrategy,
    ToolCallingStrategy,
    RenderingStrategy,
)
from aim.dreamer.core.models import (
    ContextOnlyStepDefinition,
    StandardStepDefinition,
    ToolCallingStepDefinition,
    RenderingStepDefinition,
    StepConfig,
    StepOutput,
    Condition,
    ScenarioTool,
    MemoryAction,
    StepResult,
    FormatValidation,
)
from aim.dreamer.core.framework import ScenarioFramework
from aim.dreamer.core.state import ScenarioState


# --- Fixtures ---

@pytest.fixture
def mock_cvm():
    """Create a mock ConversationModel."""
    cvm = MagicMock()
    cvm.get_by_doc_id.return_value = {"content": "test content", "document_type": "test"}
    cvm.insert.return_value = None
    return cvm


@pytest.fixture
def mock_persona():
    """Create a REAL Persona fixture - NOT A MOCK."""
    from aim.agents.persona import Persona, Aspect

    return Persona(
        persona_id="test-persona",
        chat_strategy="standard",
        name="Test",
        full_name="Test Persona",
        notes="Test persona for unit tests",
        aspects={},
        attributes={"sex": "neutral"},
        features={},
        wakeup=["Test wakeup"],
        base_thoughts=["Test thought"],
        pif={},
        nshot={},
        default_location="Test Location",
        wardrobe={"default": {}},
        current_outfit="default",
        system_header="Test system header",
    )


@pytest.fixture
def mock_config():
    """Create a mock ChatConfig."""
    from aim.config import ChatConfig
    return ChatConfig(temperature=0.7)


@pytest.fixture
def mock_model_set():
    """Create a mock ModelSet."""
    model_set = MagicMock()
    model_set.default_model = "default-model"
    model_set.thought_model = "thought-model"
    model_set.codex_model = "codex-model"
    model_set.get_model_name.return_value = "default-model"
    return model_set


@pytest.fixture
def sample_framework():
    """Create a sample ScenarioFramework."""
    return ScenarioFramework(
        name="test_scenario",
        first_step="gather",
        steps={
            "gather": ContextOnlyStepDefinition(
                id="gather",
                context=[MemoryAction(action="search_memories", top_n=10)],
                next=["process"]
            ),
            "process": StandardStepDefinition(
                id="process",
                prompt="Process the context",
                output=StepOutput(document_type="processed"),
                next=["end"]
            )
        }
    )


@pytest.fixture
def sample_state():
    """Create a sample ScenarioState."""
    return ScenarioState.initial(
        first_step="gather",
        conversation_id="conv123",
        guidance="Test guidance",
        query_text="Test query"
    )


@pytest.fixture
def executor(mock_cvm, mock_persona, mock_config, mock_model_set, sample_framework, sample_state):
    """Create a ScenarioExecutor with mocked dependencies."""
    return ScenarioExecutor(
        state=sample_state,
        framework=sample_framework,
        config=mock_config,
        cvm=mock_cvm,
        persona=mock_persona,
        model_set=mock_model_set,
    )


# --- ScenarioStepResult Tests ---

class TestScenarioStepResult:
    """Tests for ScenarioStepResult dataclass."""

    def test_step_result_success(self):
        """Test successful step result."""
        result = ScenarioStepResult(
            success=True,
            next_step="next_step",
            state_changed=True,
            doc_created=True
        )
        assert result.success is True
        assert result.next_step == "next_step"
        assert result.state_changed is True
        assert result.doc_created is True
        assert result.error is None

    def test_step_result_failure(self):
        """Test failed step result."""
        result = ScenarioStepResult(
            success=False,
            next_step="abort",
            error="Something went wrong"
        )
        assert result.success is False
        assert result.next_step == "abort"
        assert result.error == "Something went wrong"


# --- ScenarioExecutor Tests ---

class TestScenarioExecutor:
    """Tests for ScenarioExecutor."""

    def test_executor_creation(self, mock_cvm, mock_persona, mock_config, mock_model_set, sample_framework, sample_state):
        """Test executor creation."""
        executor = ScenarioExecutor(
            state=sample_state,
            framework=sample_framework,
            config=mock_config,
            cvm=mock_cvm,
            persona=mock_persona,
            model_set=mock_model_set,
        )
        assert executor.state is sample_state
        assert executor.framework is sample_framework
        assert executor.cvm is mock_cvm
        assert executor.persona is mock_persona

    def test_executor_with_model_set(self, mock_cvm, mock_persona, mock_config, mock_model_set, sample_framework, sample_state):
        """Test executor with explicit model_set."""
        executor = ScenarioExecutor(
            state=sample_state,
            framework=sample_framework,
            config=mock_config,
            cvm=mock_cvm,
            persona=mock_persona,
            model_set=mock_model_set,
        )
        assert executor.model_set is mock_model_set

    @pytest.mark.asyncio
    async def test_execute_updates_current_step(self, executor):
        """Test that execute() updates state.current_step from result."""
        mock_strategy = MagicMock(spec=BaseStepStrategy)
        mock_strategy.execute = AsyncMock(return_value=ScenarioStepResult(
            success=True,
            next_step="new_step"
        ))

        result = await executor.execute(mock_strategy)

        assert result.next_step == "new_step"
        assert executor.state.current_step == "new_step"


# --- StepFactory Tests ---

class TestStepFactory:
    """Tests for StepFactory."""

    def test_factory_creates_context_only_strategy(self, executor):
        """Test factory creates ContextOnlyStrategy for context_only steps."""
        step_def = ContextOnlyStepDefinition(
            id="test",
            next=["end"]
        )

        strategy = StepFactory.create(executor, step_def)

        assert isinstance(strategy, ContextOnlyStrategy)
        assert strategy.step_def is step_def
        assert strategy.executor is executor

    def test_factory_creates_standard_strategy(self, executor):
        """Test factory creates StandardStrategy for standard steps."""
        step_def = StandardStepDefinition(
            id="test",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
            next=["end"]
        )

        strategy = StepFactory.create(executor, step_def)

        assert isinstance(strategy, StandardStrategy)

    def test_factory_creates_tool_calling_strategy(self, executor):
        """Test factory creates ToolCallingStrategy for tool_calling steps."""
        step_def = ToolCallingStepDefinition(
            id="test",
            prompt="Test prompt",
            tools=["test_tool"],
            next_conditions=[Condition(default="end")]
        )

        strategy = StepFactory.create(executor, step_def)

        assert isinstance(strategy, ToolCallingStrategy)

    def test_factory_creates_rendering_strategy(self, executor):
        """Test factory creates RenderingStrategy for rendering steps."""
        step_def = RenderingStepDefinition(
            id="test",
            template="# {{ title }}",
            output=StepOutput(document_type="rendered"),
            next=["end"]
        )

        strategy = StepFactory.create(executor, step_def)

        assert isinstance(strategy, RenderingStrategy)

    def test_factory_unknown_type_raises(self, executor):
        """Test factory raises for unknown step type."""
        # Create a mock step with unknown type
        step_def = MagicMock()
        step_def.type = "unknown_type"

        with pytest.raises(ValueError) as exc_info:
            StepFactory.create(executor, step_def)
        assert "unknown_type" in str(exc_info.value).lower()

    def test_get_strategy_class(self):
        """Test get_strategy_class returns correct class."""
        assert StepFactory.get_strategy_class("context_only") is ContextOnlyStrategy
        assert StepFactory.get_strategy_class("standard") is StandardStrategy
        assert StepFactory.get_strategy_class("tool_calling") is ToolCallingStrategy
        assert StepFactory.get_strategy_class("rendering") is RenderingStrategy

    def test_register_custom_strategy(self, executor):
        """Test registering a custom strategy."""
        class CustomStrategy(BaseStepStrategy):
            async def execute(self):
                return ScenarioStepResult(success=True, next_step="end")

        StepFactory.register("custom", CustomStrategy)

        assert StepFactory.get_strategy_class("custom") is CustomStrategy

        # Clean up
        del StepFactory._strategy_map["custom"]

    def test_register_invalid_strategy_raises(self):
        """Test registering non-strategy class raises."""
        class NotAStrategy:
            pass

        with pytest.raises(TypeError):
            StepFactory.register("invalid", NotAStrategy)


# --- Strategy Base Tests ---

class TestBaseStepStrategy:
    """Tests for BaseStepStrategy behavior through concrete implementations."""

    def test_get_next_step(self, executor):
        """Test _get_next_step returns first item from next list."""
        step_def = ContextOnlyStepDefinition(
            id="test",
            next=["step2", "step3"]
        )
        strategy = ContextOnlyStrategy(executor, step_def)

        assert strategy._get_next_step() == "step2"

    def test_get_next_step_empty(self, executor):
        """Test _get_next_step returns 'end' for empty next list."""
        step_def = ContextOnlyStepDefinition(
            id="test",
            next=[]
        )
        strategy = ContextOnlyStrategy(executor, step_def)

        assert strategy._get_next_step() == "end"


# --- ToolCallingStrategy Condition Tests ---

class TestToolCallingConditions:
    """Tests for ToolCallingStrategy condition evaluation."""

    def test_condition_default_matches(self, executor):
        """Test default condition always matches."""
        step_def = ToolCallingStepDefinition(
            id="test",
            prompt="Test",
            tools=["tool"],
            next_conditions=[Condition(default="fallback")]
        )
        strategy = ToolCallingStrategy(executor, step_def)

        step_result = StepResult(
            step_id="test",
            response="",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=10,
            timestamp=datetime.now(timezone.utc),
            tool_name="tool",
            tool_result={"value": "anything"}
        )

        condition = Condition(default="fallback")
        assert strategy._condition_matches(condition, step_result) is True

    def test_condition_equals_matches(self, executor):
        """Test == condition matching."""
        step_def = ToolCallingStepDefinition(
            id="test",
            prompt="Test",
            tools=["tool"],
            next_conditions=[Condition(default="end")]
        )
        strategy = ToolCallingStrategy(executor, step_def)

        step_result = StepResult(
            step_id="test",
            response="",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=10,
            timestamp=datetime.now(timezone.utc),
            tool_name="accept",
            tool_result={"accept": True}
        )

        condition = Condition(
            source="tool_result.accept",
            condition="==",
            target="True",
            goto="accepted"
        )
        assert strategy._condition_matches(condition, step_result) is True

    def test_condition_in_matches(self, executor):
        """Test 'in' condition matching."""
        step_def = ToolCallingStepDefinition(
            id="test",
            prompt="Test",
            tools=["tool"],
            next_conditions=[Condition(default="end")]
        )
        strategy = ToolCallingStrategy(executor, step_def)

        step_result = StepResult(
            step_id="test",
            response="",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=10,
            timestamp=datetime.now(timezone.utc),
            tool_name="categorize",
            tool_result={"category": "urgent"}
        )

        condition = Condition(
            source="tool_result.category",
            condition="in",
            target=["urgent", "critical"],
            goto="high_priority"
        )
        assert strategy._condition_matches(condition, step_result) is True

    def test_resolve_field_nested(self, executor):
        """Test field resolution for nested paths."""
        step_def = ToolCallingStepDefinition(
            id="test",
            prompt="Test",
            tools=["tool"],
            next_conditions=[Condition(default="end")]
        )
        strategy = ToolCallingStrategy(executor, step_def)

        step_result = StepResult(
            step_id="test",
            response="",
            doc_id="doc123",
            document_type="test",
            document_weight=1.0,
            tokens_used=10,
            timestamp=datetime.now(timezone.utc),
            tool_name="complex",
            tool_result={"nested": {"deep": "value"}}
        )

        # Test simple field
        assert strategy._resolve_field("tool_name", step_result) == "complex"

        # Test nested field
        assert strategy._resolve_field("tool_result.nested", step_result) == {"deep": "value"}


# --- Import Verification ---

class TestStrategyImports:
    """Verify all strategy classes can be imported."""

    def test_all_strategies_importable(self):
        """Test all strategy classes are importable from package."""
        from aim.dreamer.core.strategy import (
            BaseStepStrategy,
            ScenarioExecutor,
            ScenarioStepResult,
            StepFactory,
            ContextOnlyStrategy,
            StandardStrategy,
            ToolCallingStrategy,
            RenderingStrategy,
        )

        # Verify they are classes
        assert isinstance(BaseStepStrategy, type)
        assert isinstance(ContextOnlyStrategy, type)
        assert isinstance(StandardStrategy, type)
        assert isinstance(ToolCallingStrategy, type)
        assert isinstance(RenderingStrategy, type)

        # Verify inheritance
        assert issubclass(ContextOnlyStrategy, BaseStepStrategy)
        assert issubclass(StandardStrategy, BaseStepStrategy)
        assert issubclass(ToolCallingStrategy, BaseStepStrategy)
        assert issubclass(RenderingStrategy, BaseStepStrategy)


# --- Strategy Execution Tests ---

class TestContextOnlyStrategyExecution:
    """Tests for ContextOnlyStrategy execute() method."""

    @pytest.mark.asyncio
    async def test_execute_with_context_actions(self, executor):
        """Test execute with context DSL actions updates memory_refs."""
        step_def = ContextOnlyStepDefinition(
            id="gather",
            context=[MemoryAction(action="search_memories", top_n=5)],
            next=["process"]
        )
        strategy = ContextOnlyStrategy(executor, step_def)

        # Mock execute_memory_actions to return doc IDs
        with patch('aim.dreamer.core.memory_dsl.execute_memory_actions') as mock_exec:
            mock_exec.return_value = ["doc1", "doc2", "doc3"]

            # Mock cvm.get_by_doc_id to return documents
            executor.cvm.get_by_doc_id.side_effect = [
                {"doc_id": "doc1", "document_type": "memory", "content": "mem1"},
                {"doc_id": "doc2", "document_type": "memory", "content": "mem2"},
                {"doc_id": "doc3", "document_type": "memory", "content": "mem3"},
            ]

            result = await strategy.execute()

            # Verify memory_dsl was called
            mock_exec.assert_called_once()
            call_args = mock_exec.call_args
            assert call_args[1]['actions'] == step_def.context
            assert call_args[1]['cvm'] is executor.cvm
            assert call_args[1]['query_text'] == executor.state.query_text
            assert call_args[1]['state'] is executor.state

            # Verify state.memory_refs was populated
            assert len(executor.state.memory_refs) == 3
            assert executor.state.memory_refs[0].doc_id == "doc1"
            assert executor.state.memory_refs[1].doc_id == "doc2"
            assert executor.state.memory_refs[2].doc_id == "doc3"

            # Verify result
            assert result.success is True
            assert result.next_step == "process"
            assert result.state_changed is True
            assert result.doc_created is False

    @pytest.mark.asyncio
    async def test_execute_without_context(self, executor):
        """Test execute without context DSL just advances to next step."""
        step_def = ContextOnlyStepDefinition(
            id="noop",
            next=["end"]
        )
        strategy = ContextOnlyStrategy(executor, step_def)

        result = await strategy.execute()

        # Verify no memory_refs added
        assert len(executor.state.memory_refs) == 0

        # Verify result
        assert result.success is True
        assert result.next_step == "end"
        assert result.state_changed is True
        assert result.doc_created is False


class TestStandardStrategyExecution:
    """Tests for StandardStrategy execute() method."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.stream_turns.return_value = iter(["This is ", "a test ", "response."])
        return provider

    @pytest.fixture
    def mock_model(self, mock_llm_provider):
        """Create a mock LanguageModel."""
        model = MagicMock()
        model.name = "test-model"
        model.max_tokens = 32768
        model.max_output_tokens = 4096
        model.llm_factory.return_value = mock_llm_provider
        return model

    @pytest.mark.asyncio
    async def test_execute_creates_document(self, executor, mock_model, mock_llm_provider):
        """Test execute creates document in CVM."""
        step_def = StandardStepDefinition(
            id="analyze",
            prompt="Analyze the data",
            config=StepConfig(format_validation=FormatValidation(require_emotional_header=False)),
            output=StepOutput(document_type="analysis", weight=1.0),
            next=["end"]
        )
        strategy = StandardStrategy(executor, step_def, llm_provider=mock_llm_provider)

        # Mock _get_model to return our mock model
        # Mock _build_turns to avoid complex build_turns logic
        with patch.object(strategy, '_get_model', return_value=mock_model):
            with patch.object(strategy, '_build_turns', return_value=([{'role': 'user', 'content': 'test'}], 'system prompt')):
                result = await strategy.execute()

                # Verify LLM was called
                mock_llm_provider.stream_turns.assert_called_once()

                # Verify document was created
                executor.cvm.insert.assert_called_once()
                created_msg = executor.cvm.insert.call_args[0][0]
                assert created_msg.content == "This is a test response."
                assert created_msg.document_type == "analysis"
                assert created_msg.weight == 1.0
                assert created_msg.scenario_name == "test_scenario"
                assert created_msg.step_name == "analyze"

                # Verify state was updated
                assert len(executor.state.step_doc_ids) == 1
                assert len(executor.state.step_results) == 1

                # Verify result
                assert result.success is True
                assert result.next_step == "end"
                assert result.state_changed is True
                assert result.doc_created is True

    @pytest.mark.asyncio
    async def test_execute_extracts_think_tags(self, executor, mock_model):
        """Test execute extracts and stores think tags."""
        step_def = StandardStepDefinition(
            id="think",
            prompt="Think deeply",
            config=StepConfig(format_validation=FormatValidation(require_emotional_header=False)),
            output=StepOutput(document_type="thought", weight=0.5),
            next=["end"]
        )

        # Mock provider that returns response with think tags
        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([
            "<think>Internal reasoning</think>",
            "Visible response"
        ])

        # Update the mock model to use this provider
        mock_model.llm_factory.return_value = mock_provider

        strategy = StandardStrategy(executor, step_def, llm_provider=mock_provider)

        with patch.object(strategy, '_get_model', return_value=mock_model):
            with patch.object(strategy, '_build_turns', return_value=([{'role': 'user', 'content': 'test'}], 'system prompt')):
                result = await strategy.execute()

                # Verify think was extracted
                created_msg = executor.cvm.insert.call_args[0][0]
                assert created_msg.content == "Visible response"
                assert created_msg.think == "Internal reasoning"

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_context_actions(self, executor, mock_model, mock_llm_provider):
        """Test execute with context DSL loads memories."""
        step_def = StandardStepDefinition(
            id="summarize",
            prompt="Summarize these memories",
            context=[MemoryAction(action="search_memories", top_n=3)],
            output=StepOutput(document_type="summary"),
            next=["end"]
        )
        strategy = StandardStrategy(executor, step_def, llm_provider=mock_llm_provider)

        # Mock execute_memory_actions
        with patch('aim.dreamer.core.memory_dsl.execute_memory_actions') as mock_exec:
            mock_exec.return_value = ["mem1", "mem2"]

            # Mock get_by_doc_id to always return a document (called multiple times)
            def get_doc(doc_id):
                return {"doc_id": doc_id, "document_type": "memory", "content": f"Content for {doc_id}"}

            executor.cvm.get_by_doc_id.side_effect = get_doc

            with patch.object(strategy, '_get_model', return_value=mock_model):
                with patch.object(strategy, '_build_turns', return_value=([{'role': 'user', 'content': 'test'}], 'system prompt')):
                    result = await strategy.execute()

                    # Verify context was executed
                    mock_exec.assert_called_once()

                    # Verify memory_refs were populated
                    assert len(executor.state.memory_refs) == 2

                    assert result.success is True


class TestRenderingStrategyExecution:
    """Tests for RenderingStrategy execute() method."""

    @pytest.mark.asyncio
    async def test_execute_renders_template(self, executor):
        """Test execute renders template with context variables."""
        step_def = RenderingStepDefinition(
            id="render",
            template="# Report\n\nGuidance: {{ guidance }}\nQuery: {{ query_text }}",
            output=StepOutput(document_type="report"),
            next=["end"]
        )
        strategy = RenderingStrategy(executor, step_def)

        result = await strategy.execute()

        # Verify document was created
        executor.cvm.insert.assert_called_once()
        created_msg = executor.cvm.insert.call_args[0][0]
        assert "Guidance: Test guidance" in created_msg.content
        assert "Query: Test query" in created_msg.content
        assert created_msg.document_type == "report"

        # Verify state was updated
        assert len(executor.state.step_doc_ids) == 1

        # Verify result
        assert result.success is True
        assert result.next_step == "end"
        assert result.state_changed is True
        assert result.doc_created is True

    @pytest.mark.asyncio
    async def test_execute_with_persona_variables(self, executor):
        """Test execute includes persona in template context."""
        step_def = RenderingStepDefinition(
            id="greet",
            template="Hello, I am {{ persona.name }} ({{ pronouns.subj }}/{{ pronouns.obj }})",
            output=StepOutput(document_type="greeting"),
            next=["end"]
        )
        strategy = RenderingStrategy(executor, step_def)

        result = await strategy.execute()

        # Verify persona was in context
        created_msg = executor.cvm.insert.call_args[0][0]
        assert "I am Test" in created_msg.content
        assert "(they/them)" in created_msg.content

        assert result.success is True


class TestToolCallingStrategyExecution:
    """Tests for ToolCallingStrategy execute() method."""

    @pytest.fixture
    def mock_tool_user(self):
        """Create a mock ToolUser."""
        tool_user = MagicMock()
        tool_user.xml_decorator.return_value = MagicMock(render=lambda: "tool xml")
        tool_user.get_tool_guidance.return_value = "Use these tools"
        return tool_user

    @pytest.fixture
    def mock_tool_call(self):
        """Create a valid mock tool call result."""
        call = MagicMock()
        call.is_valid = True
        call.function_name = "categorize"
        call.arguments = {"category": "urgent", "priority": 1}
        return call

    @pytest.mark.asyncio
    async def test_execute_with_valid_tool_call(self, executor, mock_tool_call):
        """Test execute with valid tool call."""
        step_def = ToolCallingStepDefinition(
            id="categorize",
            prompt="Categorize this item",
            tools=["categorize"],
            next_conditions=[
                Condition(
                    source="tool_result.category",
                    condition="==",
                    target="urgent",
                    goto="handle_urgent"
                ),
                Condition(default="handle_normal")
            ]
        )

        # Create a mock LLM provider
        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter(["Tool response"])

        # Create a mock model
        mock_model = MagicMock()
        mock_model.name = "tool-model"
        mock_model.max_output_tokens = 1024
        mock_model.llm_factory.return_value = mock_provider

        strategy = ToolCallingStrategy(executor, step_def, llm_provider=mock_provider)

        # Mock dependencies
        # Mock framework.get_tools to avoid needing to define tools
        with patch.object(type(executor.framework), 'get_tools', return_value=[MagicMock()]):
            with patch.object(strategy, '_get_model', return_value=mock_model):
                with patch('aim.tool.formatting.ToolUser') as MockToolUser:
                    mock_tool_user = MagicMock()
                    mock_tool_user.process_response.return_value = mock_tool_call
                    mock_tool_user.xml_decorator.return_value = MagicMock(render=lambda: "system")
                    mock_tool_user.get_tool_guidance.return_value = "Use tool"
                    MockToolUser.return_value = mock_tool_user

                    result = await strategy.execute()

                    # Verify LLM was called
                    mock_provider.stream_turns.assert_called_once()

                    # Verify tool was processed
                    mock_tool_user.process_response.assert_called_once_with("Tool response")

                    # Verify step result was recorded
                    assert len(executor.state.step_results) == 1
                    step_result = executor.state.step_results["categorize"]
                    assert step_result.tool_name == "categorize"
                    assert step_result.tool_result == {"category": "urgent", "priority": 1}

                    # Verify condition evaluation
                    assert result.success is True
                    assert result.next_step == "handle_urgent"  # Matched first condition
                    assert result.state_changed is True
                    assert result.doc_created is False

    @pytest.mark.asyncio
    async def test_execute_with_retry_on_invalid_tool_call(self, executor):
        """Test execute retries when LLM doesn't call tool."""
        step_def = ToolCallingStepDefinition(
            id="retry_test",
            prompt="Use a tool",
            tools=["test_tool"],
            config=StepConfig(tool_retries=2),
            next_conditions=[Condition(default="end")]
        )

        # Create invalid tool call
        invalid_call = MagicMock()
        invalid_call.is_valid = False

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter(["Invalid response"])

        mock_model = MagicMock()
        mock_model.name = "tool-model"
        mock_model.max_output_tokens = 1024
        mock_model.llm_factory.return_value = mock_provider

        strategy = ToolCallingStrategy(executor, step_def, llm_provider=mock_provider)

        with patch.object(type(executor.framework), 'get_tools', return_value=[MagicMock()]):
            with patch.object(strategy, '_get_model', return_value=mock_model):
                with patch('aim.tool.formatting.ToolUser') as MockToolUser:
                    mock_tool_user = MagicMock()
                    mock_tool_user.process_response.return_value = invalid_call
                    mock_tool_user.xml_decorator.return_value = MagicMock(render=lambda: "system")
                    mock_tool_user.get_tool_guidance.return_value = "Use tool"
                    MockToolUser.return_value = mock_tool_user

                    result = await strategy.execute()

                    # Verify LLM was called tool_retries times
                    assert mock_provider.stream_turns.call_count == 2

                    # Verify failure result
                    assert result.success is False
                    assert result.next_step == "abort"
                    assert "No valid tool call" in result.error

    @pytest.mark.asyncio
    async def test_execute_evaluates_conditions(self, executor, mock_tool_call):
        """Test execute evaluates conditions to determine next step."""
        step_def = ToolCallingStepDefinition(
            id="router",
            prompt="Route this",
            tools=["categorize"],
            next_conditions=[
                Condition(
                    source="tool_result.category",
                    condition="in",
                    target=["urgent", "critical"],
                    goto="high_priority"
                ),
                Condition(default="low_priority")
            ]
        )

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter(["Response"])

        mock_model = MagicMock()
        mock_model.name = "tool-model"
        mock_model.max_output_tokens = 1024
        mock_model.llm_factory.return_value = mock_provider

        strategy = ToolCallingStrategy(executor, step_def, llm_provider=mock_provider)

        with patch.object(type(executor.framework), 'get_tools', return_value=[MagicMock()]):
            with patch.object(strategy, '_get_model', return_value=mock_model):
                with patch('aim.tool.formatting.ToolUser') as MockToolUser:
                    mock_tool_user = MagicMock()
                    mock_tool_user.process_response.return_value = mock_tool_call
                    mock_tool_user.xml_decorator.return_value = MagicMock(render=lambda: "system")
                    mock_tool_user.get_tool_guidance.return_value = "Use tool"
                    MockToolUser.return_value = mock_tool_user

                    result = await strategy.execute()

                    # Verify "in" condition matched
                    assert result.next_step == "high_priority"

    @pytest.mark.asyncio
    async def test_execute_handles_max_iterations(self, executor, mock_tool_call):
        """Test execute respects max_iterations and redirects to on_limit."""
        step_def = ToolCallingStepDefinition(
            id="loop",
            prompt="Loop this",
            tools=["test"],
            config=StepConfig(max_iterations=2, on_limit="break_out"),
            next_conditions=[Condition(default="loop")]  # Would normally loop
        )

        # Simulate already having 2 iterations
        executor.state.step_iterations["loop"] = 1  # Will increment to 2

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter(["Response"])

        mock_model = MagicMock()
        mock_model.name = "tool-model"
        mock_model.max_output_tokens = 1024
        mock_model.llm_factory.return_value = mock_provider

        strategy = ToolCallingStrategy(executor, step_def, llm_provider=mock_provider)

        with patch.object(type(executor.framework), 'get_tools', return_value=[MagicMock()]):
            with patch.object(strategy, '_get_model', return_value=mock_model):
                with patch('aim.tool.formatting.ToolUser') as MockToolUser:
                    mock_tool_user = MagicMock()
                    mock_tool_user.process_response.return_value = mock_tool_call
                    mock_tool_user.xml_decorator.return_value = MagicMock(render=lambda: "system")
                    mock_tool_user.get_tool_guidance.return_value = "Use tool"
                    MockToolUser.return_value = mock_tool_user

                    result = await strategy.execute()

                    # Verify max_iterations kicked in
                    assert executor.state.step_iterations["loop"] == 2
                    assert result.next_step == "break_out"  # Not "loop"


# --- Memory Refs Lifecycle Tests ---

class TestMemoryRefsLifecycle:
    """Tests for memory_refs accumulation and explicit flush behavior.

    These tests verify that:
    1. memory_refs persists across steps without explicit flush/clear
    2. DSL flush action clears state.memory_refs
    3. DSL clear action clears state.memory_refs
    4. Steps without context DSL preserve existing memory_refs
    """

    @pytest.mark.asyncio
    async def test_memory_refs_persist_across_steps_without_flush(self, executor):
        """memory_refs should persist when second step has no context DSL.

        Bug: execute_context_actions() auto-clears memory_refs every time.
        Expected: memory_refs should only clear on explicit flush/clear.
        """
        # Step 1: Load some documents
        step1_def = ContextOnlyStepDefinition(
            id="load_context",
            context=[MemoryAction(action="search_memories", top_n=3)],
            next=["process"]
        )
        strategy1 = ContextOnlyStrategy(executor, step1_def)

        with patch('aim.dreamer.core.memory_dsl.execute_memory_actions') as mock_exec:
            mock_exec.return_value = ["doc1", "doc2", "doc3"]
            executor.cvm.get_by_doc_id.side_effect = [
                {"doc_id": "doc1", "document_type": "memory", "content": "mem1"},
                {"doc_id": "doc2", "document_type": "memory", "content": "mem2"},
                {"doc_id": "doc3", "document_type": "memory", "content": "mem3"},
            ]

            await strategy1.execute()

        # Verify step 1 loaded docs
        assert len(executor.state.memory_refs) == 3
        assert [ref.doc_id for ref in executor.state.memory_refs] == ["doc1", "doc2", "doc3"]

        # Step 2: No context DSL - memory_refs should PERSIST
        step2_def = ContextOnlyStepDefinition(
            id="process",
            context=None,  # No context DSL
            next=["end"]
        )
        strategy2 = ContextOnlyStrategy(executor, step2_def)

        await strategy2.execute()

        # CRITICAL: memory_refs should still have the docs from step 1
        assert len(executor.state.memory_refs) == 3, \
            "memory_refs was cleared but should persist across steps"
        assert [ref.doc_id for ref in executor.state.memory_refs] == ["doc1", "doc2", "doc3"]

    @pytest.mark.asyncio
    async def test_memory_refs_persist_when_second_step_loads_more(self, executor):
        """memory_refs should accumulate when steps load without flush.

        Step 1 loads A, B, C
        Step 2 loads D, E (without flush)
        Result should be A, B, C, D, E
        """
        # Step 1: Load docs A, B, C
        step1_def = ContextOnlyStepDefinition(
            id="load_first",
            context=[MemoryAction(action="search_memories", top_n=3)],
            next=["load_more"]
        )
        strategy1 = ContextOnlyStrategy(executor, step1_def)

        with patch('aim.dreamer.core.memory_dsl.execute_memory_actions') as mock_exec:
            mock_exec.return_value = ["docA", "docB", "docC"]
            executor.cvm.get_by_doc_id.side_effect = [
                {"doc_id": "docA", "document_type": "memory", "content": "A"},
                {"doc_id": "docB", "document_type": "memory", "content": "B"},
                {"doc_id": "docC", "document_type": "memory", "content": "C"},
            ]
            await strategy1.execute()

        assert len(executor.state.memory_refs) == 3

        # Step 2: Load docs D, E (NO flush - should accumulate)
        step2_def = ContextOnlyStepDefinition(
            id="load_more",
            context=[MemoryAction(action="search_memories", top_n=2)],
            next=["end"]
        )
        strategy2 = ContextOnlyStrategy(executor, step2_def)

        with patch('aim.dreamer.core.memory_dsl.execute_memory_actions') as mock_exec:
            mock_exec.return_value = ["docD", "docE"]
            executor.cvm.get_by_doc_id.side_effect = [
                {"doc_id": "docD", "document_type": "memory", "content": "D"},
                {"doc_id": "docE", "document_type": "memory", "content": "E"},
            ]
            await strategy2.execute()

        # Should have ALL 5 docs accumulated
        assert len(executor.state.memory_refs) == 5, \
            f"Expected 5 docs (A,B,C + D,E), got {len(executor.state.memory_refs)}"
        doc_ids = [ref.doc_id for ref in executor.state.memory_refs]
        assert doc_ids == ["docA", "docB", "docC", "docD", "docE"]

    @pytest.mark.asyncio
    async def test_dsl_flush_clears_memory_refs(self, executor):
        """DSL flush action should clear state.memory_refs.

        Tests the actual DSL execution (no mocking execute_memory_actions)
        to verify flush clears both accumulated_doc_ids and state.memory_refs.
        """
        from aim.dreamer.core.state import DocRef
        from aim.dreamer.core.memory_dsl import execute_memory_actions

        # Pre-populate memory_refs to simulate prior step's work
        executor.state.memory_refs = [
            DocRef(doc_id="docA", document_type="memory"),
            DocRef(doc_id="docB", document_type="memory"),
            DocRef(doc_id="docC", document_type="memory"),
        ]
        assert len(executor.state.memory_refs) == 3

        # Execute DSL with flush - should clear memory_refs
        actions = [MemoryAction(action="flush")]
        doc_ids = execute_memory_actions(
            actions=actions,
            state=executor.state,
            cvm=executor.cvm,
        )

        # flush returns empty (nothing loaded after flush)
        assert doc_ids == []

        # CRITICAL: state.memory_refs should be cleared by flush
        assert len(executor.state.memory_refs) == 0, \
            f"Expected 0 docs after flush, got {len(executor.state.memory_refs)}"

    @pytest.mark.asyncio
    async def test_dsl_clear_clears_memory_refs(self, executor):
        """DSL clear action should clear state.memory_refs (same as flush)."""
        from aim.dreamer.core.state import DocRef
        from aim.dreamer.core.memory_dsl import execute_memory_actions

        # Pre-populate memory_refs
        executor.state.memory_refs = [
            DocRef(doc_id="docX", document_type="memory"),
            DocRef(doc_id="docY", document_type="memory"),
        ]
        assert len(executor.state.memory_refs) == 2

        # Execute DSL with clear
        actions = [MemoryAction(action="clear")]
        doc_ids = execute_memory_actions(
            actions=actions,
            state=executor.state,
            cvm=executor.cvm,
        )

        # clear returns empty
        assert doc_ids == []

        # state.memory_refs should be cleared
        assert len(executor.state.memory_refs) == 0

    @pytest.mark.asyncio
    async def test_dsl_flush_then_load_replaces_memory_refs(self, executor):
        """DSL flush followed by load should replace (not accumulate) memory_refs.

        This tests the full flow through execute_context_actions:
        1. Pre-populate memory_refs (simulating step 1)
        2. Step 2 has flush then load_conversation
        3. Result should be ONLY the new docs from load_conversation
        """
        from aim.dreamer.core.state import DocRef

        # Pre-populate memory_refs to simulate prior step
        executor.state.memory_refs = [
            DocRef(doc_id="old1", document_type="memory"),
            DocRef(doc_id="old2", document_type="memory"),
        ]

        # Mock CVM to return conversation history for load_conversation
        import pandas as pd
        executor.cvm.get_conversation_history.return_value = pd.DataFrame({
            'doc_id': ['new1', 'new2', 'new3'],
            'document_type': ['conversation', 'conversation', 'conversation'],
        })
        executor.cvm.get_by_doc_id.side_effect = [
            {"doc_id": "new1", "document_type": "conversation", "content": "N1"},
            {"doc_id": "new2", "document_type": "conversation", "content": "N2"},
            {"doc_id": "new3", "document_type": "conversation", "content": "N3"},
        ]

        # Step with flush then load_conversation
        step_def = ContextOnlyStepDefinition(
            id="flush_and_load",
            context=[
                MemoryAction(action="flush"),
                MemoryAction(action="load_conversation", target="current"),
            ],
            next=["end"]
        )
        strategy = ContextOnlyStrategy(executor, step_def)

        await strategy.execute()

        # Should have ONLY the new docs (flush cleared old1, old2)
        assert len(executor.state.memory_refs) == 3, \
            f"Expected 3 new docs, got {len(executor.state.memory_refs)}"
        doc_ids = [ref.doc_id for ref in executor.state.memory_refs]
        assert doc_ids == ["new1", "new2", "new3"]


# --- Dialogue Query Text Tests ---

class TestDialogueQueryTextFallback:
    """Tests for using dialogue turn content as search query text.

    When search_memories is called without explicit query_text, and
    state has dialogue_turns, the dialogue content should be used
    as the query for semantic memory search.
    """

    @pytest.mark.asyncio
    async def test_search_memories_uses_dialogue_turns_as_query(self, executor):
        """search_memories should use dialogue turn content when no query_text."""
        from aim.dreamer.core.memory_dsl import _search_memories
        from aim.dreamer.core.models import DialogueTurn

        # Pre-populate dialogue_turns (simulating a dialogue scenario)
        executor.state.dialogue_turns = [
            DialogueTurn(
                speaker_id="aspect:coder",
                content="Let's analyze the authentication system and its security implications.",
                think=None,
                step_id="step1",
                doc_id="turn1",
                document_type="dialogue-coder",
            ),
            DialogueTurn(
                speaker_id="persona:andi",
                content="I've been thinking about how the session tokens are managed.",
                think=None,
                step_id="step2",
                doc_id="turn2",
                document_type="dialogue",
            ),
        ]

        # Clear query_text so fallback is used
        executor.state.query_text = None

        # Mock CVM query to capture the query text used
        import pandas as pd
        captured_query = []

        def mock_query(query_texts, **kwargs):
            captured_query.extend(query_texts)
            return pd.DataFrame({'doc_id': ['mem1', 'mem2']})

        executor.cvm.query = mock_query

        # Create action without explicit query_text
        action = MemoryAction(action="search_memories", top_n=5)

        # Execute search
        doc_ids = _search_memories(
            action=action,
            state=executor.state,
            cvm=executor.cvm,
            accumulated_doc_ids=[],
        )

        # Verify dialogue content was used as query
        assert len(captured_query) == 1
        query_used = captured_query[0]
        assert "authentication system" in query_used
        assert "session tokens" in query_used

        # Verify results returned
        assert doc_ids == ['mem1', 'mem2']

    @pytest.mark.asyncio
    async def test_search_memories_prefers_explicit_query_over_dialogue(self, executor):
        """Explicit query_text should take precedence over dialogue turns."""
        from aim.dreamer.core.memory_dsl import _search_memories
        from aim.dreamer.core.models import DialogueTurn

        # Pre-populate dialogue_turns
        executor.state.dialogue_turns = [
            DialogueTurn(
                speaker_id="aspect:coder",
                content="Dialogue content that should NOT be used.",
                think=None,
                step_id="step1",
                doc_id="turn1",
                document_type="dialogue-coder",
            ),
        ]

        # Set explicit query_text
        executor.state.query_text = "explicit query about memory management"

        # Mock CVM query to capture the query text used
        import pandas as pd
        captured_query = []

        def mock_query(query_texts, **kwargs):
            captured_query.extend(query_texts)
            return pd.DataFrame({'doc_id': ['mem1']})

        executor.cvm.query = mock_query

        action = MemoryAction(action="search_memories", top_n=5)

        _search_memories(
            action=action,
            state=executor.state,
            cvm=executor.cvm,
            accumulated_doc_ids=[],
        )

        # Verify explicit query was used, NOT dialogue content
        assert len(captured_query) == 1
        assert "explicit query" in captured_query[0]
        assert "Dialogue content" not in captured_query[0]

    @pytest.mark.asyncio
    async def test_search_memories_empty_dialogue_returns_empty(self, executor):
        """Empty dialogue_turns with no query_text should return empty."""
        from aim.dreamer.core.memory_dsl import _search_memories

        # Empty dialogue_turns
        executor.state.dialogue_turns = []
        executor.state.query_text = None

        action = MemoryAction(action="search_memories", top_n=5)

        doc_ids = _search_memories(
            action=action,
            state=executor.state,
            cvm=executor.cvm,
            accumulated_doc_ids=[],
        )

        # Should return empty (warning logged)
        assert doc_ids == []
