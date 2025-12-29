# tests/unit/refiner/test_engine.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for the ExplorationEngine with 3-step agentic flow."""

import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

import pandas as pd

from aim.config import ChatConfig
from aim.refiner.context import GatheredContext
from aim.refiner.paradigm import Paradigm


@dataclass
class MockPipelineResult:
    """Mock result from DreamerClient.start()."""
    success: bool
    pipeline_id: str = None
    error: str = None


class TestExplorationEngine:
    """Tests for the ExplorationEngine class with 3-step flow."""

    @pytest.fixture
    def mock_config(self):
        """Real ChatConfig with test values."""
        return ChatConfig(
            default_model="test-default-model",
            thought_model="test-thought-model",
            persona_id="test-persona",
            tools_path="config/tools",
            persona_path="config/persona",
        )

    @pytest.fixture
    def mock_cvm(self):
        """Mock ConversationModel."""
        cvm = MagicMock()
        cvm.next_conversation_id.return_value = "new-conv-123"
        cvm.query.return_value = pd.DataFrame([
            {"doc_id": "1", "content": "test content about consciousness", "document_type": "codex", "date": "2025-01-01"},
            {"doc_id": "2", "content": "journal about emotions", "document_type": "journal", "date": "2025-01-02"},
        ])
        return cvm

    @pytest.fixture
    def mock_dreamer_client(self):
        """Mock DreamerClient."""
        client = AsyncMock()
        client.start.return_value = MockPipelineResult(
            success=True,
            pipeline_id="pipeline-123"
        )
        return client

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona."""
        persona = MagicMock()
        persona.persona_id = "test-persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def mock_roster(self, mock_persona):
        """Mock Roster."""
        roster = MagicMock()
        roster.personas = {"test-persona": mock_persona}
        return roster

    @pytest.fixture
    def mock_redis_cache_idle(self):
        """Mock RedisCache that reports API as idle."""
        cache = MagicMock()
        cache.get_api_last_activity.return_value = time.time() - 600
        return cache

    @pytest.fixture
    def mock_redis_cache_active(self):
        """Mock RedisCache that reports API as active."""
        cache = MagicMock()
        cache.get_api_last_activity.return_value = time.time() - 60
        return cache

    @pytest.fixture
    def mock_context_gatherer(self):
        """Mock ContextGatherer with broad_gather and targeted_gather."""
        gatherer = MagicMock()

        # Mock broad_gather to return documents (async method)
        broad_docs = [
            {"content": "brainstorm idea about creativity", "doc_id": "1", "document_type": "brainstorm"},
            {"content": "thoughts on consciousness", "doc_id": "2", "document_type": "pondering"},
        ]
        broad_context = GatheredContext(
            documents=broad_docs,
            paradigm="brainstorm",
            tokens_used=500,
            conversation_count=0,
            other_count=2,
        )
        gatherer.broad_gather = AsyncMock(return_value=broad_context)

        # Mock targeted_gather to return focused documents (async method)
        targeted_docs = [
            {"content": "detailed doc about creativity", "doc_id": "3", "document_type": "codex"},
        ]
        targeted_context = GatheredContext(
            documents=targeted_docs,
            paradigm="philosopher",
            tokens_used=200,
            conversation_count=0,
            other_count=1,
        )
        gatherer.targeted_gather = AsyncMock(return_value=targeted_context)

        return gatherer

    @pytest.fixture
    def mock_llm_select_topic(self):
        """Mock LLM that returns <think> + tool call for topic selection."""
        provider = MagicMock()
        response = '''<think>
Analyzing these brainstorm ideas...
The creativity theme is most compelling.
</think>

{"select_topic": {"topic": "nature of creativity", "approach": "philosopher", "reasoning": "Underexplored"}}'''
        provider.stream_turns.return_value = iter([response])
        return provider

    @pytest.fixture
    def mock_llm_accept(self):
        """Mock LLM that accepts exploration."""
        provider = MagicMock()
        response = '''<think>
These documents reveal new connections...
</think>

{"validate_exploration": {"accept": true, "reasoning": "Novel insights", "query_text": "Explore creativity"}}'''
        provider.stream_turns.return_value = iter([response])
        return provider

    @pytest.fixture
    def mock_llm_reject(self):
        """Mock LLM that rejects exploration."""
        provider = MagicMock()
        response = '''<think>
This territory has been well-explored...
</think>

{"validate_exploration": {"accept": false, "reasoning": "Already covered"}}'''
        provider.stream_turns.return_value = iter([response])
        return provider

    @pytest.fixture
    def engine(self, mock_config, mock_cvm, mock_dreamer_client, mock_persona):
        """Create an ExplorationEngine with mocked dependencies."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                return ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                    idle_threshold_seconds=300,
                )

    # Test initialization
    def test_engine_initializes_with_config(self, mock_config, mock_cvm, mock_dreamer_client, mock_persona):
        """Engine should initialize with provided config."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                engine = ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                )

        assert engine.config is mock_config
        assert engine.cvm is mock_cvm
        assert engine.dreamer_client is mock_dreamer_client

    def test_engine_uses_model_name_parameter(self, mock_config, mock_cvm, mock_dreamer_client, mock_persona):
        """Engine should use provided model_name parameter."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                engine = ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                    model_name="test-custom-model",
                )

        assert engine.model_name == "test-custom-model"

    # Test is_api_idle
    @pytest.mark.asyncio
    async def test_is_api_idle_true_when_inactive(self, engine, mock_redis_cache_idle):
        """is_api_idle should return True when API inactive for threshold."""
        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            result = await engine.is_api_idle()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_api_idle_false_when_active(self, engine, mock_redis_cache_active):
        """is_api_idle should return False when API recently active."""
        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_active):
            result = await engine.is_api_idle()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_api_idle_true_when_no_activity_recorded(self, engine):
        """is_api_idle should return True when no API activity recorded."""
        cache = MagicMock()
        cache.get_api_last_activity.return_value = None

        with patch.object(engine, '_get_redis_cache', return_value=cache):
            result = await engine.is_api_idle()

        assert result is True

    # Test run_exploration - returns None when not idle
    @pytest.mark.asyncio
    async def test_run_exploration_returns_none_when_not_idle(self, engine, mock_redis_cache_active):
        """run_exploration should return None when API is not idle."""
        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_active):
            pipeline_id, suggested = await engine.run_exploration()

        assert pipeline_id is None

    # Test 3-step flow - full acceptance
    @pytest.mark.asyncio
    async def test_run_exploration_full_flow_accept(
        self, engine, mock_redis_cache_idle, mock_context_gatherer
    ):
        """Test complete 3-step flow with acceptance."""
        # Step 1: Topic selection response
        step1_response = '''<think>Looking at docs...</think>
{"select_topic": {"topic": "consciousness", "approach": "philosopher", "reasoning": "Interesting"}}'''

        # Step 2: Validation acceptance response
        step2_response = '''<think>Validating...</think>
{"validate_exploration": {"accept": true, "reasoning": "Good topic", "query_text": "What is consciousness?", "guidance": "Be deep"}}'''

        call_count = [0]
        def mock_stream(turns, config):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([step1_response])
            return iter([step2_response])

        mock_provider = MagicMock()
        mock_provider.stream_turns.side_effect = mock_stream

        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
                engine.context_gatherer = mock_context_gatherer
                pipeline_id, suggested = await engine.run_exploration()

        # Should have called LLM twice (step 1 and step 2)
        assert mock_provider.stream_turns.call_count == 2
        # Should return pipeline ID
        assert pipeline_id == "pipeline-123"

    # Test 3-step flow - rejection
    @pytest.mark.asyncio
    async def test_run_exploration_rejects_when_validation_fails(
        self, engine, mock_redis_cache_idle, mock_context_gatherer, mock_dreamer_client
    ):
        """Test that exploration stops when validation rejects."""
        step1_response = '''<think>Looking...</think>
{"select_topic": {"topic": "boring topic", "approach": "philosopher", "reasoning": "Maybe interesting"}}'''

        step2_response = '''<think>Actually no...</think>
{"validate_exploration": {"accept": false, "reasoning": "Not compelling enough"}}'''

        call_count = [0]
        def mock_stream(turns, config):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([step1_response])
            return iter([step2_response])

        mock_provider = MagicMock()
        mock_provider.stream_turns.side_effect = mock_stream

        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
                engine.context_gatherer = mock_context_gatherer
                pipeline_id, suggested = await engine.run_exploration()

        # Should return None (no pipeline)
        assert pipeline_id is None
        # Pipeline should not have been triggered
        mock_dreamer_client.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_exploration_returns_none_on_empty_broad_context(
        self, engine, mock_redis_cache_idle
    ):
        """run_exploration should return None when broad_gather returns empty."""
        mock_gatherer = MagicMock()
        mock_gatherer.broad_gather = AsyncMock(return_value=GatheredContext(
            documents=[],
            paradigm="brainstorm",
        ))

        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            engine.context_gatherer = mock_gatherer
            pipeline_id, suggested = await engine.run_exploration()

        assert pipeline_id is None

    @pytest.mark.asyncio
    async def test_run_exploration_returns_none_on_empty_targeted_context(
        self, engine, mock_redis_cache_idle
    ):
        """run_exploration should return None when targeted_gather returns empty."""
        mock_gatherer = MagicMock()

        # Broad gather returns docs (async method)
        mock_gatherer.broad_gather = AsyncMock(return_value=GatheredContext(
            documents=[{"content": "test", "doc_id": "1"}],
            paradigm="brainstorm",
        ))

        # Targeted gather returns empty (async method)
        mock_gatherer.targeted_gather = AsyncMock(return_value=GatheredContext(
            documents=[],
            paradigm="philosopher",
        ))

        step1_response = '''{"select_topic": {"topic": "test", "approach": "philosopher", "reasoning": "test"}}'''
        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([step1_response])

        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
                engine.context_gatherer = mock_gatherer
                pipeline_id, suggested = await engine.run_exploration()

        assert pipeline_id is None

    @pytest.mark.asyncio
    async def test_run_exploration_passes_context_documents_to_pipeline(
        self, engine, mock_redis_cache_idle, mock_context_gatherer, mock_dreamer_client
    ):
        """run_exploration should pass context_documents to the pipeline."""
        step1_response = '''<think>...</think>
{"select_topic": {"topic": "test", "approach": "philosopher", "reasoning": "test"}}'''

        step2_response = '''<think>...</think>
{"validate_exploration": {"accept": true, "reasoning": "good", "query_text": "test query", "guidance": "test"}}'''

        call_count = [0]
        def mock_stream(turns, config):
            call_count[0] += 1
            if call_count[0] == 1:
                return iter([step1_response])
            return iter([step2_response])

        mock_provider = MagicMock()
        mock_provider.stream_turns.side_effect = mock_stream

        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
                engine.context_gatherer = mock_context_gatherer
                await engine.run_exploration()

        # Check that start was called with context_documents
        mock_dreamer_client.start.assert_called_once()
        call_kwargs = mock_dreamer_client.start.call_args[1]
        assert "context_documents" in call_kwargs
        assert isinstance(call_kwargs["context_documents"], list)

    @pytest.mark.asyncio
    async def test_run_exploration_returns_none_when_topic_selection_fails(
        self, engine, mock_redis_cache_idle, mock_context_gatherer
    ):
        """run_exploration should return None when topic selection returns invalid."""
        # Invalid response - missing required fields
        invalid_response = '''{"select_topic": {"topic": "test"}}'''  # missing approach and reasoning

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([invalid_response])

        with patch.object(engine, '_get_redis_cache', return_value=mock_redis_cache_idle):
            with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
                engine.context_gatherer = mock_context_gatherer
                pipeline_id, suggested = await engine.run_exploration()

        assert pipeline_id is None


class TestExplorationEngineToolValidation:
    """Tests for tool validation using ToolUser in ExplorationEngine."""

    @pytest.fixture
    def mock_config(self):
        """Real ChatConfig with test values."""
        return ChatConfig(
            default_model="test-model",
            thought_model="test-model",
            persona_id="test-persona",
            tools_path="config/tools",
            persona_path="config/persona",
        )

    @pytest.fixture
    def mock_cvm(self):
        """Mock ConversationModel with documents."""
        cvm = MagicMock()
        cvm.query.return_value = pd.DataFrame([
            {"doc_id": "1", "content": "test", "document_type": "codex", "date": "2025-01-01"},
        ])
        cvm.next_conversation_id.return_value = "conv-123"
        return cvm

    @pytest.fixture
    def mock_dreamer_client(self):
        """Mock DreamerClient."""
        client = AsyncMock()
        client.start.return_value = MockPipelineResult(success=True, pipeline_id="pipe-123")
        return client

    @pytest.fixture
    def mock_persona(self):
        """Mock Persona."""
        persona = MagicMock()
        persona.persona_id = "test-persona"
        persona.name = "Test"
        persona.full_name = "Test Persona"
        persona.pronouns = {"subj": "she", "poss": "her", "obj": "her"}
        persona.aspects = {}
        persona.xml_decorator.return_value = MagicMock()
        persona.xml_decorator.return_value.render.return_value = "<persona>Test</persona>"
        return persona

    @pytest.fixture
    def sample_documents(self):
        """Sample documents list for testing."""
        return [{"content": "test doc", "doc_id": "1", "document_type": "codex"}]

    @pytest.mark.asyncio
    async def test_select_topic_uses_tool_user_for_validation(
        self, mock_config, mock_cvm, mock_dreamer_client, mock_persona, sample_documents
    ):
        """Engine should use ToolUser.process_response() for validation."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                engine = ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                )

        # Response with valid tool call
        valid_response = '{"select_topic": {"topic": "test", "approach": "philosopher", "reasoning": "test reason"}}'

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([valid_response])

        # Load paradigm for the test
        paradigm = Paradigm.load("brainstorm")

        with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
            # This should parse and validate the tool call
            result = await engine._select_topic(paradigm, sample_documents)

        # Result should be the parsed tool call arguments
        assert result is not None
        assert result.get("topic") == "test"
        assert result.get("approach") == "philosopher"

    @pytest.mark.asyncio
    async def test_select_topic_handles_invalid_tool_call(
        self, mock_config, mock_cvm, mock_dreamer_client, mock_persona, sample_documents
    ):
        """Engine should handle invalid tool calls gracefully."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                engine = ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                )

        # Response with invalid tool call (missing required field)
        invalid_response = '{"select_topic": {"topic": "test"}}'

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([invalid_response])

        # Load paradigm for the test
        paradigm = Paradigm.load("brainstorm")

        with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
            result = await engine._select_topic(paradigm, sample_documents)

        # Should return None for invalid call
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_exploration_accepts_valid_response(
        self, mock_config, mock_cvm, mock_dreamer_client, mock_persona, sample_documents
    ):
        """Engine should accept valid validation responses."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                engine = ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                )

        valid_response = '''{"validate_exploration": {"accept": true, "reasoning": "Good", "query_text": "Explore this"}}'''

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([valid_response])

        topic_result = {"topic": "test", "approach": "philosopher", "reasoning": "test reason"}

        # Load paradigm for the test
        paradigm = Paradigm.load("brainstorm")

        with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
            result = await engine._validate_exploration(
                paradigm=paradigm,
                topic_result=topic_result,
                documents=sample_documents,
            )

        assert result is not None
        assert result.get("accept") is True
        assert result.get("query_text") == "Explore this"

    @pytest.mark.asyncio
    async def test_validate_exploration_handles_rejection(
        self, mock_config, mock_cvm, mock_dreamer_client, mock_persona, sample_documents
    ):
        """Engine should handle rejection responses."""
        from aim.refiner.engine import ExplorationEngine

        with patch('aim.refiner.engine.Persona') as MockPersona:
            MockPersona.from_config.return_value = mock_persona
            with patch('aim.refiner.engine.ContextGatherer'):
                engine = ExplorationEngine(
                    config=mock_config,
                    cvm=mock_cvm,
                    dreamer_client=mock_dreamer_client,
                )

        reject_response = '''{"validate_exploration": {"accept": false, "reasoning": "Not interesting"}}'''

        mock_provider = MagicMock()
        mock_provider.stream_turns.return_value = iter([reject_response])

        topic_result = {"topic": "test", "approach": "philosopher", "reasoning": "test reason"}

        # Load paradigm for the test
        paradigm = Paradigm.load("brainstorm")

        with patch.object(engine, '_get_llm_provider', return_value=mock_provider):
            result = await engine._validate_exploration(
                paradigm=paradigm,
                topic_result=topic_result,
                documents=sample_documents,
            )

        assert result is not None
        assert result.get("accept") is False
