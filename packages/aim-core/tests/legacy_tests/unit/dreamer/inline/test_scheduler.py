# tests/unit/dreamer/inline/test_scheduler.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for inline pipeline scheduler."""

import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import uuid

from aim_legacy.dreamer.inline.scheduler import execute_pipeline_inline, _execute_standard_inline, _execute_dialogue_inline
from aim.dreamer.core.models import Scenario, ScenarioContext, StepDefinition, StepOutput, StepConfig, StepResult
from aim.conversation.message import ConversationMessage


@pytest.fixture
def mock_config():
    """Mock ChatConfig."""
    config = MagicMock()
    config.default_model = "gpt-4"
    config.thought_model = "gpt-4-mini"
    config.codex_model = "claude-opus-4"
    config.temperature = 0.7
    config.max_tokens = 2048
    return config


@pytest.fixture
def mock_persona():
    """Mock Persona."""
    persona = MagicMock()
    persona.persona_id = "andi"
    persona.full_name = "Andi"
    persona.name = "Andi"
    persona.pronouns = {"subject": "she", "object": "her", "possessive": "her"}
    persona.aspects = {}
    persona.models = {}
    persona.system_prompt = MagicMock(return_value="You are Andi, a helpful AI assistant.")
    persona.get_wakeup = MagicMock(return_value="I'm listening.")
    return persona


@pytest.fixture
def mock_roster(mock_persona):
    """Mock Roster."""
    roster = MagicMock()
    roster.get_persona = MagicMock(return_value=mock_persona)
    return roster


@pytest.fixture
def mock_cvm():
    """Mock ConversationModel."""
    cvm = MagicMock()
    cvm.get_next_branch = MagicMock(return_value=0)
    cvm.insert = MagicMock()
    cvm.get_by_doc_id = MagicMock(return_value=None)
    cvm.query = MagicMock(return_value=MagicMock(empty=True))
    cvm.get_conversation_history = MagicMock(return_value=MagicMock(empty=True))
    cvm.index = MagicMock()
    cvm.index.search = MagicMock(return_value=MagicMock(empty=True))
    return cvm


@pytest.fixture
def simple_scenario():
    """Simple two-step scenario for testing."""
    return Scenario(
        name="test_scenario",
        version=2,
        flow="standard",
        description="Test scenario",
        requires_conversation=False,
        context=ScenarioContext(
            required_aspects=[],
            core_documents=[],
            enhancement_documents=[],
            location="test location",
            thoughts=[],
        ),
        seed=[],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Test prompt 1",
                config=StepConfig(max_tokens=512),
                output=StepOutput(document_type="test", weight=1.0),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Test prompt 2",
                config=StepConfig(max_tokens=512),
                output=StepOutput(document_type="test", weight=1.0),
                next=[],
                depends_on=["step1"],
            ),
        },
    )


@pytest.mark.asyncio
async def test_execute_pipeline_inline_standard(
    mock_config, mock_roster, mock_cvm, mock_persona, simple_scenario
):
    """Test inline execution of a standard scenario."""
    # Mock scenario loading
    with patch('aim_legacy.dreamer.inline.scheduler.load_scenario', return_value=simple_scenario):
        # Mock execute_step to return results
        mock_result_1 = StepResult(
            step_id="step1",
            response="Test response 1",
            think=None,
            doc_id=ConversationMessage.next_doc_id(),
            document_type="test",
            document_weight=1.0,
            tokens_used=100,
            timestamp=datetime.now(timezone.utc),
        )
        mock_result_2 = StepResult(
            step_id="step2",
            response="Test response 2",
            think=None,
            doc_id=ConversationMessage.next_doc_id(),
            document_type="test",
            document_weight=1.0,
            tokens_used=150,
            timestamp=datetime.now(timezone.utc),
        )

        with patch('aim_legacy.dreamer.inline.scheduler.execute_step', new_callable=AsyncMock) as mock_execute:
            # Return different results for each call
            mock_execute.side_effect = [
                (mock_result_1, [], False),
                (mock_result_2, [], False),
            ]

            # Execute pipeline
            pipeline_id = await execute_pipeline_inline(
                scenario_name="test_scenario",
                config=mock_config,
                cvm=mock_cvm,
                roster=mock_roster,
                persona_id="andi",
                query_text="Test query",
            )

            # Verify pipeline_id is a valid UUID
            assert uuid.UUID(pipeline_id)

            # Verify execute_step was called twice (once per step)
            assert mock_execute.call_count == 2

            # Verify CVM insert was called twice
            assert mock_cvm.insert.call_count == 2


@pytest.mark.asyncio
async def test_execute_pipeline_inline_with_heartbeat(
    mock_config, mock_roster, mock_cvm, mock_persona, simple_scenario
):
    """Test inline execution with heartbeat callback."""
    heartbeat_calls = []

    async def heartbeat_callback(pipeline_id, step_id):
        heartbeat_calls.append((pipeline_id, step_id))

    # Mock scenario loading
    with patch('aim_legacy.dreamer.inline.scheduler.load_scenario', return_value=simple_scenario):
        # Mock execute_step
        mock_result = StepResult(
            step_id="test",
            response="Test response",
            think=None,
            doc_id=ConversationMessage.next_doc_id(),
            document_type="test",
            document_weight=1.0,
            tokens_used=100,
            timestamp=datetime.now(timezone.utc),
        )

        with patch('aim_legacy.dreamer.inline.scheduler.execute_step', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = (mock_result, [], False)

            # Execute pipeline with heartbeat
            pipeline_id = await execute_pipeline_inline(
                scenario_name="test_scenario",
                config=mock_config,
                cvm=mock_cvm,
                roster=mock_roster,
                persona_id="andi",
                heartbeat_callback=heartbeat_callback,
            )

            # Verify heartbeat was called for each step
            assert len(heartbeat_calls) == 2
            assert heartbeat_calls[0][1] == "step1"
            assert heartbeat_calls[1][1] == "step2"
            # Both should have the same pipeline_id
            assert heartbeat_calls[0][0] == heartbeat_calls[1][0] == pipeline_id


@pytest.mark.asyncio
async def test_execute_pipeline_inline_persona_not_found(mock_config, mock_cvm):
    """Test error handling when persona is not found."""
    roster = MagicMock()
    roster.get_persona = MagicMock(return_value=None)

    scenario = Scenario(
        name="test",
        version=2,
        flow="standard",
        description="Test",
        requires_conversation=False,
        context=ScenarioContext(required_aspects=[]),
        seed=[],
        steps={},
    )

    with patch('aim_legacy.dreamer.inline.scheduler.load_scenario', return_value=scenario):
        with pytest.raises(ValueError, match="Persona 'unknown' not found"):
            await execute_pipeline_inline(
                scenario_name="test",
                config=mock_config,
                cvm=mock_cvm,
                roster=roster,
                persona_id="unknown",
            )


@pytest.mark.asyncio
async def test_execute_pipeline_inline_missing_conversation(
    mock_config, mock_roster, mock_cvm
):
    """Test error handling when required conversation is missing."""
    scenario = Scenario(
        name="analyst",
        version=2,
        flow="standard",
        description="Analyst scenario",
        requires_conversation=True,  # Requires conversation
        context=ScenarioContext(required_aspects=[]),
        seed=[],
        steps={},
    )

    with patch('aim_legacy.dreamer.inline.scheduler.load_scenario', return_value=scenario):
        with pytest.raises(ValueError, match="requires a conversation_id"):
            await execute_pipeline_inline(
                scenario_name="analyst",
                config=mock_config,
                cvm=mock_cvm,
                roster=mock_roster,
                persona_id="andi",
                conversation_id=None,  # Missing required conversation_id
            )


@pytest.mark.asyncio
async def test_execute_pipeline_inline_seed_actions(
    mock_config, mock_roster, mock_cvm, mock_persona
):
    """Test that seed actions are executed and results flow into context."""
    from aim.dreamer.core.models import MemoryAction

    scenario = Scenario(
        name="test_with_seed",
        version=2,
        flow="standard",
        description="Test with seed",
        requires_conversation=False,
        context=ScenarioContext(required_aspects=[]),
        seed=[
            MemoryAction(
                action="get_memory",
                document_types=["codex"],
                top_n=5,
            )
        ],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Test prompt",
                config=StepConfig(max_tokens=512),
                output=StepOutput(document_type="test", weight=1.0),
                next=[],
            ),
        },
    )

    with patch('aim_legacy.dreamer.inline.scheduler.load_scenario', return_value=scenario):
        with patch('aim_legacy.dreamer.inline.scheduler.execute_memory_actions') as mock_seed:
            # Mock seed returning some doc_ids
            mock_seed.return_value = ["doc1", "doc2", "doc3"]

            mock_result = StepResult(
                step_id="step1",
                response="Test response",
                think=None,
                doc_id=ConversationMessage.next_doc_id(),
                document_type="test",
                document_weight=1.0,
                tokens_used=100,
                timestamp=datetime.now(timezone.utc),
            )

            with patch('aim_legacy.dreamer.inline.scheduler.execute_step', new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (mock_result, [], False)

                # Execute pipeline
                pipeline_id = await execute_pipeline_inline(
                    scenario_name="test_with_seed",
                    config=mock_config,
                    cvm=mock_cvm,
                    roster=mock_roster,
                    persona_id="andi",
                )

                # Verify seed actions were executed
                assert mock_seed.call_count == 1

                # Verify execute_step was called
                assert mock_execute.call_count == 1

                # Verify the state passed to execute_step has the seed context
                call_args = mock_execute.call_args
                state = call_args.kwargs['state']
                # Seed context should be in the initial state
                assert state.context_doc_ids[:3] == ["doc1", "doc2", "doc3"]


@pytest.mark.asyncio
async def test_execute_dialogue_inline(mock_config, mock_roster, mock_cvm, mock_persona):
    """Test inline execution of a dialogue scenario."""
    dialogue_scenario = Scenario(
        name="test_dialogue",
        version=2,
        flow="dialogue",
        description="Test dialogue",
        requires_conversation=False,
        context=ScenarioContext(required_aspects=["coder"]),
        seed=[],
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Test prompt 1",
                config=StepConfig(max_tokens=512),
                output=StepOutput(document_type="dialogue", weight=1.0),
                next=[],
            ),
        },
    )

    # Create a mock DialogueStrategy
    mock_strategy = MagicMock()
    mock_strategy.name = "test_dialogue"
    mock_strategy.get_execution_order = MagicMock(return_value=["step1"])

    # Create a mock DialogueState
    mock_state = MagicMock()
    mock_state.pipeline_id = str(uuid.uuid4())
    mock_state.turns = []

    # Create a mock DialogueTurn
    mock_turn = MagicMock()
    mock_turn.speaker_id = "aspect:coder"
    mock_turn.doc_id = ConversationMessage.next_doc_id()

    # Mock the entire _execute_dialogue_inline function to avoid deep mocking
    expected_pipeline_id = str(uuid.uuid4())

    with patch('aim_legacy.dreamer.inline.scheduler.load_scenario', return_value=dialogue_scenario):
        with patch('aim_legacy.dreamer.inline.scheduler._execute_dialogue_inline', new_callable=AsyncMock) as mock_execute_dialogue:
            mock_execute_dialogue.return_value = expected_pipeline_id

            # Execute dialogue pipeline
            pipeline_id = await execute_pipeline_inline(
                scenario_name="test_dialogue",
                config=mock_config,
                cvm=mock_cvm,
                roster=mock_roster,
                persona_id="andi",
            )

            # Verify pipeline_id matches
            assert pipeline_id == expected_pipeline_id

            # Verify _execute_dialogue_inline was called with correct params
            assert mock_execute_dialogue.call_count == 1
            call_kwargs = mock_execute_dialogue.call_args.kwargs
            assert call_kwargs['scenario'] == dialogue_scenario
            assert call_kwargs['persona'] == mock_persona
            assert call_kwargs['config'] == mock_config
            assert call_kwargs['cvm'] == mock_cvm
