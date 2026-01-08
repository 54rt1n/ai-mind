# packages/aim-mud/tests/conftest.py
# Central fixture repository for aim-mud tests
# Philosophy: Real objects with mocked external services only

import pytest
from unittest.mock import MagicMock, AsyncMock
from aim.agents.persona import Persona, Aspect
from aim.agents.roster import Roster
from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.chat.manager import ChatManager


@pytest.fixture
def mock_redis():
    """Mock for Redis (external service).

    This mocks the Redis client. All Redis operations should use this fixture.
    """
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hgetall = AsyncMock(return_value={})
    redis.hdel = AsyncMock(return_value=1)
    redis.xadd = AsyncMock(return_value=b"stream-id-123")
    redis.xread = AsyncMock(return_value=[])
    redis.xack = AsyncMock(return_value=1)
    redis.close = AsyncMock()
    return redis


@pytest.fixture
def test_config() -> ChatConfig:
    """A minimal real ChatConfig for testing.

    No mocks. This is a real ChatConfig object with sensible defaults.
    """
    config = ChatConfig(
        persona_path="config/persona",
        tools_path="config/tools",
        memory_path="memory",
        user_id="test_user",
        persona_id="test_persona",
        conversation_id="test_conversation"
    )
    # Set model defaults that would normally come from .env
    config.default_model = "anthropic/claude-sonnet-4-5-20250929"
    config.thought_model = "anthropic/claude-opus-4-5-20251101"
    config.codex_model = "anthropic/claude-opus-4-5-20251101"
    config.llm_provider = "anthropic"
    config.anthropic_api_key = "test-key-for-testing"
    return config


@pytest.fixture
def test_persona() -> Persona:
    """A minimal real Persona for testing.

    No mocks. This is a real Persona object that can be used in tests
    to validate actual Persona behavior.
    """
    data = {
        "persona_id": "test_persona",
        "persona_version": "1.0.0",
        "chat_strategy": "xmlmemory",
        "notes": "A test persona for unit tests",
        "name": "Test",
        "full_name": "Test Persona",
        "aspects": {},
        "base_thoughts": ["I am a test persona"],
        "pif": {
            "Identity": "You are a test persona used in unit tests.",
            "Purpose": "To validate test behavior with real objects."
        },
        "nshot": {},
        "system_header": "You are a helpful test assistant.",
        "wakeup": ["Test persona online."],
        "attributes": {
            "species": "AI",
            "sex": "neutral",
            "age": "0"
        },
        "features": {
            "personality": "Helpful and straightforward for testing."
        },
        "default_location": "Test Environment",
        "wardrobe": {
            "default": {
                "outfit": "Standard test configuration"
            }
        },
        "current_outfit": "default",
        "persona_tools": {},
        "wardrobe_tools": {}
    }

    return Persona.from_dict(data)


@pytest.fixture
def test_roster(test_config: ChatConfig, test_persona: Persona) -> Roster:
    """A real Roster with test personas.

    No mocks. This is a real Roster object containing real Persona objects.
    Includes both "test_persona" and "andi" for compatibility with tests.
    """
    # Create a second persona with persona_id="andi" for tests that use that ID
    andi_data = {
        "persona_id": "andi",
        "persona_version": "1.0.0",
        "chat_strategy": "xmlmemory",
        "notes": "Test persona for andi",
        "name": "Andi",
        "full_name": "Andi Test Persona",
        "aspects": {},
        "base_thoughts": ["I am andi"],
        "pif": {
            "Identity": "You are andi, a test persona.",
            "Purpose": "To validate test behavior."
        },
        "nshot": {},
        "system_header": "You are andi, a test assistant.",
        "wakeup": ["Andi online."],
        "attributes": {"species": "AI", "sex": "neutral", "age": "0"},
        "features": {"personality": "Helpful for testing."},
        "default_location": "Test Environment",
        "wardrobe": {"default": {"outfit": "Standard test configuration"}},
        "current_outfit": "default",
        "persona_tools": {},
        "wardrobe_tools": {},
        "models": {}  # Empty models dict - will use defaults
    }
    andi_persona = Persona.from_dict(andi_data)

    # Create similar persona for "val"
    val_data = andi_data.copy()
    val_data["persona_id"] = "val"
    val_data["name"] = "Val"
    val_data["full_name"] = "Val Test Persona"
    val_persona = Persona.from_dict(val_data)

    personas = {
        test_persona.persona_id: test_persona,
        "andi": andi_persona,
        "val": val_persona,
    }
    return Roster(personas=personas, config=test_config)


@pytest.fixture
def mock_search_index():
    """Mock for SearchIndex (external Tantivy service).

    This mocks only the external Tantivy search engine.
    ConversationModel itself remains real.
    """
    mock_index = MagicMock()
    mock_index.search.return_value = []
    mock_index.insert_document.return_value = None
    mock_index.delete_document.return_value = None
    return mock_index


@pytest.fixture
def test_cvm(tmp_path, mock_search_index, monkeypatch):
    """A real ConversationModel with mocked SearchIndex.

    Mocks: SearchIndex (external Tantivy)
    Real: ConversationModel, ConversationLoader

    Uses tmp_path for file system operations.
    """
    memory_path = str(tmp_path / "memory")
    ConversationModel.maybe_init_folders(memory_path)

    # Patch SearchIndex creation to return our mock
    def mock_search_index_init(self, *args, **kwargs):
        return None

    monkeypatch.setattr("aim.conversation.index.SearchIndex.__init__", mock_search_index_init)

    cvm = ConversationModel(
        memory_path=memory_path,
        embedding_model="test-embedding-model",
        user_timezone="UTC"
    )

    # Replace the index with our mock
    cvm.index = mock_search_index

    return cvm


@pytest.fixture
def test_chat_manager(test_cvm: ConversationModel, test_config: ChatConfig, test_roster: Roster) -> ChatManager:
    """A real ChatManager with real components.

    No mocks. Uses real CVM (with mocked SearchIndex), Config, and Roster.
    """
    return ChatManager(cvm=test_cvm, config=test_config, roster=test_roster)
