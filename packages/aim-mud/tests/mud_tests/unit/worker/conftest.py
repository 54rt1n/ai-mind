# packages/aim-mud/tests/mud_tests/unit/worker/conftest.py
# Worker-specific fixtures for MUD agent worker tests
# Philosophy: Real objects with mocked external services only

import pytest
from unittest.mock import AsyncMock

from andimud_worker.config import MUDConfig
from andimud_worker.conversation import MUDConversationManager
from andimud_worker.conversation.memory import MUDDecisionStrategy, MUDResponseStrategy
from andimud_worker.worker import MUDAgentWorker


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
