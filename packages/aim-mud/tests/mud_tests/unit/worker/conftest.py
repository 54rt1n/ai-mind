# packages/aim-mud/tests/mud_tests/unit/worker/conftest.py
# Worker-specific fixtures for MUD agent worker tests
# Philosophy: Real objects with mocked external services only

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from andimud_worker.config import MUDConfig
from andimud_worker.conversation import MUDConversationManager
from andimud_worker.conversation.memory import MUDDecisionStrategy, MUDResponseStrategy
from andimud_worker.worker import MUDAgentWorker
# NOTE: Commented out 2026-01-09 - No longer needed after inline scheduler migration
# from aim.dreamer.state import StateStore
# from aim.dreamer.scheduler import Scheduler
# from aim.dreamer.models import (
#     Scenario, ScenarioContext, StepDefinition, StepConfig, StepOutput,
#     PipelineState, StepJob, StepResult
# )


@pytest.fixture
def test_mud_config() -> MUDConfig:
    """A minimal real MUDConfig for testing.

    No mocks. This is a real MUDConfig object with sensible defaults
    for worker tests.
    """
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
        memory_path="memory/test_persona",
        spontaneous_check_interval=60.0,
        spontaneous_action_interval=300.0,
        top_n_memories=10,
        max_recent_turns=20,
        bucket_max_tokens=28000,
        conversation_max_tokens=50000,
    )


@pytest.fixture
def test_conversation_manager(mock_redis, test_mud_config):
    """A real MUDConversationManager with mocked Redis.

    Mocks: Redis (external service)
    Real: MUDConversationManager, all its logic

    This manager handles the Redis conversation list for agent turns.
    """
    return MUDConversationManager(
        redis=mock_redis,
        agent_id=test_mud_config.agent_id,
        persona_id=test_mud_config.persona_id,
        max_tokens=test_mud_config.conversation_max_tokens,
    )


@pytest.fixture
def test_decision_strategy(test_chat_manager):
    """A real MUDDecisionStrategy for Phase 1 testing.

    No mocks. This is a real strategy that uses the real ChatManager.
    The strategy itself contains no external dependencies - it just
    builds turns using chat manager's components.
    """
    return MUDDecisionStrategy(chat=test_chat_manager)


@pytest.fixture
def test_response_strategy(test_chat_manager):
    """A real MUDResponseStrategy for Phase 2 testing.

    No mocks. This is a real strategy that uses the real ChatManager.
    The strategy extends XMLMemoryTurnStrategy for full memory-augmented
    responses.
    """
    return MUDResponseStrategy(chat=test_chat_manager)


@pytest.fixture
def test_worker(test_mud_config, mock_redis, test_config):
    """A real MUDAgentWorker for testing.

    Mocks: Redis (external service)
    Real: MUDAgentWorker, all its initialization logic

    The worker is created but not started (start() would load persona,
    CVM, etc.). Use this for testing worker methods that don't require
    full initialization.

    For tests that need a fully initialized worker, use monkeypatch
    to mock specific initialization steps.
    """
    return MUDAgentWorker(
        config=test_mud_config,
        redis_client=mock_redis,
        chat_config=test_config,
    )


# =============================================================================
# Dreamer Test Fixtures
# =============================================================================
# NOTE: Commented out 2026-01-09 - No longer needed after inline scheduler migration
# Integration tests that used these fixtures have been skipped and need updating

# @pytest.fixture
# def test_state_store(mock_redis):
#     """Real StateStore with mocked Redis (external service).

#     Mocks: Redis (external)
#     Real: StateStore logic
#     """
#     return StateStore(
#         redis_client=mock_redis,
#         key_prefix="test:dreamer",
#     )


# @pytest.fixture
# def test_scheduler(mock_redis, test_state_store):
#     """Real Scheduler with mocked Redis (external service).

#     Mocks: Redis (external)
#     Real: Scheduler logic, queue management
#     """
#     return Scheduler(
#         redis_client=mock_redis,
#         state_store=test_state_store,
#     )


# @pytest.fixture
# def minimal_standard_scenario():
#     """Minimal standard flow scenario for testing.

#     Two-step pipeline:
#     - step1: No dependencies (root)
#     - step2: Depends on step1
#     """
#     return Scenario(
#         name="test_standard",
#         version=2,
#         flow="standard",
#         description="Minimal test scenario",
#         requires_conversation=False,
#         context=ScenarioContext(
#             required_aspects=[],
#             core_documents=[],
#             enhancement_documents=[],
#             location="",
#             thoughts=[],
#         ),
#         seed=[],
#         steps={
#             "step1": StepDefinition(
#                 id="step1",
#                 prompt="Test step 1: {{ query_text }}",
#                 config=StepConfig(max_tokens=100),
#                 output=StepOutput(
#                     document_type="test-step1",
#                     weight=1.0,
#                 ),
#                 next=["step2"],
#             ),
#             "step2": StepDefinition(
#                 id="step2",
#                 prompt="Test step 2: Build on previous results.",
#                 config=StepConfig(max_tokens=100),
#                 output=StepOutput(
#                     document_type="test-step2",
#                     weight=1.0,
#                 ),
#                 next=[],
#                 depends_on=["step1"],
#             ),
#         },
#     )


# @pytest.fixture
# def test_pipeline_state(test_config):
#     """Real PipelineState for testing.

#     Represents a pipeline mid-execution.
#     """
#     return PipelineState(
#         pipeline_id="test_pipeline_123",
#         scenario_name="test_standard",
#         conversation_id="test_conversation",
#         persona_id="test_persona",
#         user_id="test_user",
#         model=test_config.default_model,
#         thought_model=test_config.thought_model,
#         codex_model=test_config.codex_model,
#         guidance=None,
#         query_text="Test query",
#         persona_mood=None,
#         branch=0,
#         step_counter=1,
#         completed_steps=[],
#         step_doc_ids={},
#         seed_doc_ids={},
#         context_doc_ids=[],
#         context_documents=None,
#         created_at=datetime.now(timezone.utc),
#         updated_at=datetime.now(timezone.utc),
#     )


# @pytest.fixture
# def test_step_job():
#     """Real StepJob for testing.

#     Represents a job in the scheduler queue.
#     """
#     return StepJob(
#         pipeline_id="test_pipeline_123",
#         step_id="step1",
#         attempt=1,
#         max_attempts=3,
#         enqueued_at=datetime.now(timezone.utc),
#         priority=0,
#     )


# @pytest.fixture
# def mock_execute_step():
#     """Mock for execute_step function (contains LLM calls - external).

#     This mocks the LLM execution, NOT the internal pipeline logic.
#     Use this when testing pipeline orchestration without actual LLM calls.
#     """
#     async def fake_execute_step(state, scenario, step_def, cvm, persona, config, model_set):
#         """Return a fake successful step result."""
#         return (
#             StepResult(
#                 step_id=step_def.id,
#                 response=f"Test response for {step_def.id}",
#                 think=None,
#                 doc_id=f"doc_{step_def.id}_123",
#                 document_type=step_def.output.document_type,
#                 document_weight=step_def.output.weight,
#                 tokens_used=50,
#                 timestamp=datetime.now(timezone.utc),
#             ),
#             [],  # context_doc_ids
#             False,  # is_initial_context
#         )

#     return fake_execute_step


# @pytest.fixture
# def mock_load_scenario(minimal_standard_scenario):
#     """Mock load_scenario to return test scenario.

#     Mocks: File system access (external)
#     Real: Scenario objects and logic
#     """
#     def load_test_scenario(scenario_name):
#         return minimal_standard_scenario

#     return load_test_scenario
