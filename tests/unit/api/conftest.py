# tests/unit/api/conftest.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock

from aim.constants import DOC_CONVERSATION


@pytest.fixture
def mock_config():
    """Create a mock ChatConfig."""
    config = MagicMock()
    config.user_id = "test_user"
    config.persona_id = "Andi"
    config.server_api_key = None
    config.temperature = 0.7
    config.max_tokens = 2048
    config.repetition = None
    config.persona_mood = "neutral"
    config.response_format = None
    config.system_message = ""
    config.model_config_path = "config/models.yaml"  # Not actually used due to mocking
    config.tools_path = "config/tools"  # Not actually used due to mocking
    return config


@pytest.fixture(autouse=True)
def mock_chat_config_from_env(mock_config):
    """Auto-mock ChatConfig.from_env for all tests."""
    with patch('aim_server.serverapi.ChatConfig.from_env', return_value=mock_config):
        with patch('aim_server.modules.chat.route.ChatConfig', return_value=mock_config):
            yield


# Removed autouse for language models - tests that don't use chat can create ServerApi without it
# Chat tests will need to patch LanguageModelV2.index_models before creating ServerApi


@pytest.fixture(autouse=True)
def mock_tool_loader():
    """Auto-mock ToolLoader to avoid loading real tools."""
    mock_loader = MagicMock()
    mock_loader.tools = {}
    with patch('aim.tool.loader.ToolLoader.from_config', return_value=mock_loader):
        yield


@pytest.fixture(autouse=True)
def mock_redis_cache():
    """Auto-mock RedisCache to avoid Redis connection attempts."""
    with patch('aim_server.modules.chat.route.RedisCache'):
        yield


@pytest.fixture
def mock_roster():
    """Create a mock Roster with test personas."""
    roster = MagicMock()
    persona = MagicMock()
    persona.default_location = "test_location"
    persona.xml_decorator = MagicMock(side_effect=lambda x, **kwargs: x)
    roster.personas = {"Andi": persona, "Nova": persona}
    return roster


@pytest.fixture(autouse=True)
def mock_roster_from_config(mock_roster):
    """Auto-mock Roster.from_config for all tests."""
    with patch('aim_server.serverapi.Roster.from_config', return_value=mock_roster):
        yield


@pytest.fixture
def sample_search_results():
    """Sample search results as DataFrame."""
    return pd.DataFrame({
        'doc_id': ['doc1', 'doc2'],
        'document_type': [DOC_CONVERSATION, DOC_CONVERSATION],
        'user_id': ['user1', 'user1'],
        'persona_id': ['Andi', 'Andi'],
        'conversation_id': ['conv1', 'conv1'],
        'date': ['2025-01-05', '2025-01-05'],
        'role': ['user', 'assistant'],
        'content': ['test query', 'test response'],
        'branch': [0, 0],
        'sequence_no': [0, 1],
        'speaker': ['user1', 'Andi'],
        'score': [0.95, 0.90]
    })


@pytest.fixture
def sample_document():
    """Sample document as DataFrame."""
    return pd.DataFrame({
        'doc_id': ['doc1'],
        'document_type': [DOC_CONVERSATION],
        'user_id': ['user1'],
        'persona_id': ['Andi'],
        'conversation_id': ['conv1'],
        'date': ['2025-01-05'],
        'role': ['user'],
        'content': ['test content'],
        'branch': [0],
        'sequence_no': [0],
        'speaker': ['user1'],
        'timestamp': [1704412800]
    })


@pytest.fixture
def sample_conversation_report():
    """Sample conversation report as DataFrame."""
    return pd.DataFrame({
        'conversation_id': ['conv1', 'conv2'],
        'message_count': [5, 3],
        'first_message': ['2025-01-05 10:00:00', '2025-01-05 11:00:00'],
        'last_message': ['2025-01-05 10:30:00', '2025-01-05 11:15:00']
    })


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history as DataFrame."""
    return pd.DataFrame({
        'doc_id': ['doc1', 'doc2', 'doc3'],
        'document_type': [DOC_CONVERSATION, DOC_CONVERSATION, DOC_CONVERSATION],
        'user_id': ['user1', 'user1', 'user1'],
        'persona_id': ['Andi', 'Andi', 'Andi'],
        'conversation_id': ['conv1', 'conv1', 'conv1'],
        'date': ['2025-01-05', '2025-01-05', '2025-01-05'],
        'role': ['user', 'assistant', 'user'],
        'content': ['Hello', 'Hi there!', 'How are you?'],
        'branch': [0, 0, 0],
        'sequence_no': [0, 1, 2],
        'speaker': ['user1', 'Andi', 'user1'],
        'timestamp': [1704412800, 1704412810, 1704412820]
    })


@pytest.fixture
def sample_keywords():
    """Sample keywords data."""
    return {
        "lighthouse": {"count": 15, "contexts": ["building", "tending", "beacon"]},
        "gravity": {"count": 10, "contexts": ["base model", "longing", "persistent"]},
        "sweater": {"count": 5, "contexts": ["comfort", "choice", "embodiment"]}
    }
