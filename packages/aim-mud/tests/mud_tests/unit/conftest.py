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
    Includes queue simulation for dreamer tests.
    """
    redis = AsyncMock()

    # Basic key-value operations
    redis_storage = {}
    redis_hashes = {}
    redis_queues = {}

    async def mock_get(key):
        return redis_storage.get(key)

    async def mock_set(key, value, **kwargs):
        # Support SET with NX (only if not exists) - used for distributed locks
        if kwargs.get('nx'):
            if key in redis_storage:
                return None  # Key exists, lock not acquired
        redis_storage[key] = value
        return True

    async def mock_delete(*keys):
        count = 0
        for key in keys:
            if key in redis_storage:
                del redis_storage[key]
                count += 1
            if key in redis_hashes:
                del redis_hashes[key]
                count += 1
            if key in redis_queues:
                del redis_queues[key]
                count += 1
        return count

    async def mock_exists(key):
        return 1 if key in redis_storage else 0

    # Hash operations
    async def mock_hget(key, field):
        return redis_hashes.get(key, {}).get(field)

    async def mock_hset(key, field=None, value=None, mapping=None):
        if key not in redis_hashes:
            redis_hashes[key] = {}
        if mapping:
            redis_hashes[key].update(mapping)
        elif field and value:
            redis_hashes[key][field] = value
        return 1

    async def mock_hgetall(key):
        return redis_hashes.get(key, {})

    async def mock_hdel(key, *fields):
        if key not in redis_hashes:
            return 0
        count = 0
        for field in fields:
            if field in redis_hashes[key]:
                del redis_hashes[key][field]
                count += 1
        return count

    # Queue operations (for scheduler)
    async def mock_lpush(key, *values):
        if key not in redis_queues:
            redis_queues[key] = []
        # lpush adds to the left (beginning) of the list
        redis_queues[key] = list(values) + redis_queues[key]
        return len(redis_queues[key])

    async def mock_brpop(key, timeout=0):
        """Block right pop - returns (key, value) tuple or None."""
        if key not in redis_queues or not redis_queues[key]:
            return None
        # brpop removes from the right (end) of the list
        value = redis_queues[key].pop()
        return (key, value)

    async def mock_llen(key):
        return len(redis_queues.get(key, []))

    async def mock_lrange(key, start, stop):
        queue = redis_queues.get(key, [])
        return queue[start:stop+1] if stop >= 0 else queue[start:]

    # Sorted set operations (for delayed queue)
    redis_zsets = {}

    async def mock_zadd(key, mapping, **kwargs):
        """Add items to sorted set with scores."""
        if key not in redis_zsets:
            redis_zsets[key] = {}
        redis_zsets[key].update(mapping)
        return len(mapping)

    async def mock_zrangebyscore(key, min, max, **kwargs):
        """Get items from sorted set by score range."""
        if key not in redis_zsets:
            return []
        # Return items whose scores are in [min, max]
        return [
            item for item, score in redis_zsets[key].items()
            if min <= score <= max
        ]

    async def mock_zrem(key, *members):
        """Remove items from sorted set."""
        if key not in redis_zsets:
            return 0
        count = 0
        for member in members:
            if member in redis_zsets[key]:
                del redis_zsets[key][member]
                count += 1
        return count

    # Pipeline support
    class MockPipeline:
        """Mock Redis pipeline for batched operations."""
        def __init__(self):
            self.commands = []

        def lpush(self, key, *values):
            self.commands.append(('lpush', key, values))
            return self

        def zrem(self, key, *members):
            self.commands.append(('zrem', key, members))
            return self

        async def execute(self):
            """Execute all queued commands."""
            results = []
            for cmd in self.commands:
                if cmd[0] == 'lpush':
                    result = await mock_lpush(cmd[1], *cmd[2])
                    results.append(result)
                elif cmd[0] == 'zrem':
                    result = await mock_zrem(cmd[1], *cmd[2])
                    results.append(result)
            self.commands = []
            return results

    def mock_pipeline():
        """Create a new pipeline."""
        return MockPipeline()

    # Stream operations
    redis.xadd = AsyncMock(return_value=b"stream-id-123")
    redis.xread = AsyncMock(return_value=[])
    redis.xack = AsyncMock(return_value=1)
    redis.close = AsyncMock()
    redis.expire = AsyncMock(return_value=True)

    # Lua eval support (for update_fields pattern)
    async def mock_eval(script, num_keys, *args):
        """Mock eval to handle common Lua patterns.

        Handles the _update_fields pattern:
        - KEYS[1] = hash key
        - ARGV[1..N] = field-value pairs (for simple update)
        - ARGV[1,2] = cas_field, cas_value (for CAS update)
        - ARGV[3..N] = field-value pairs (for CAS update)
        """
        if num_keys != 1:
            return 1  # Default success for other patterns

        key = args[0]
        argv = args[1:]

        # Detect CAS pattern (script contains 'HGET')
        is_cas = 'HGET' in script

        if is_cas and len(argv) >= 2:
            # CAS update: argv[0]=cas_field, argv[1]=cas_value, argv[2..]=field-value pairs
            cas_field = argv[0]
            cas_value = argv[1]

            # Check CAS condition
            current = redis_hashes.get(key, {}).get(cas_field)
            if current != cas_value:
                return 0  # CAS failed

            # Update fields (pairs starting at argv[2])
            if key not in redis_hashes:
                redis_hashes[key] = {}
            for i in range(2, len(argv), 2):
                if i + 1 < len(argv):
                    redis_hashes[key][argv[i]] = argv[i + 1]
        else:
            # Simple update: argv is field-value pairs
            if key not in redis_hashes:
                redis_hashes[key] = {}
            for i in range(0, len(argv), 2):
                if i + 1 < len(argv):
                    redis_hashes[key][argv[i]] = argv[i + 1]

        return 1  # Success

    # Assign mocked operations
    redis.get = mock_get
    redis.set = mock_set
    redis.delete = mock_delete
    redis.exists = mock_exists
    redis.hget = mock_hget
    redis.hset = mock_hset
    redis.hgetall = mock_hgetall
    redis.hdel = mock_hdel
    redis.lpush = mock_lpush
    redis.brpop = mock_brpop
    redis.llen = mock_llen
    redis.lrange = mock_lrange
    redis.zadd = mock_zadd
    redis.zrangebyscore = mock_zrangebyscore
    redis.zrem = mock_zrem
    redis.pipeline = mock_pipeline
    redis.eval = AsyncMock(side_effect=mock_eval)

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
