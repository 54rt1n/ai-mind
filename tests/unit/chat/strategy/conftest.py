# tests/unit/chat/strategy/conftest.py

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from aim.config import ChatConfig
from aim.agents.persona import Persona
from aim.conversation.model import ConversationModel
from aim.chat.manager import ChatManager
from aim.io.documents import Library
from aim.nlp.summarize import TextSummarizer # For type hinting and spec for mock
from aim.utils.redis_cache import RedisCache # For type hinting and spec for mock
from aim.chat.strategy.xmlmemory import XMLMemoryTurnStrategy

@pytest.fixture
def base_chat_config():
    """Provides a default ChatConfig instance for unit tests."""
    return ChatConfig(
        persona_id="unit_test_persona",
        user_id="unit_test_user",
        embedding_model="mock-embedding-model",
        summarizer_model="mock-summarizer-model", # Ensure mock is intended
        history_management_strategy="sparsify", # Default, overridden in tests
        memory_path="mock_memory/path", # For CVM/Roster if they try to init paths
        persona_path="mock_persona/path",
        tools_path="mock_tools/path",
        model_config_path="mock_config/models.yaml",
        documents_dir="mock_documents/dir" # For Library
    )

@pytest.fixture
def sample_persona_data():
    """Provides raw data for a sample persona for unit tests."""
    return {
        "persona_id": "unit_test_persona",
        "chat_strategy": "xmlmemory",
        "name": "UnitTestBot",
        "full_name": "Unit Test Bot 9000",
        "notes": "A persona specifically for unit testing XMLMemoryStrategy.",
        "aspects": {
            "coder": {"name": "Codey", "title": "Dev Unit", "description": "Asserts logic."},
            "librarian": {"name": "Libby", "title": "Mock Archivist", "description": "Stores nothing real."}
        },
        "attributes": {"sex": "synthetic", "age": "0 cycles"},
        "features": {"voice": "assertive_beep"},
        "wakeup": ["Initiating XMLMemoryStrategy unit test sequence."],
        "base_thoughts": ["Verify XML structure.", "Confirm memory formatting."],
        "pif": {"core_unit_test_directive": "Isolate and verify."},
        "nshot": {
            "unit_test_example": {
                "human": "Does this unit work?",
                "assistant": "Affirmative. All mocks are nominal."
            }
        },
        "default_location": "The Mock Environment",
        "wardrobe": {"default": {"outfit": "debug_console_green"}},
        "current_outfit": "default",
        "persona_tools": {}, # Keep empty for strategy-focused tests
        "wardrobe_tools": {}
    }

@pytest.fixture
def sample_persona(sample_persona_data):
    """Provides a Persona instance from sample_persona_data."""
    return Persona.from_dict(sample_persona_data)

@pytest.fixture
def mock_cvm():
    """
    Mocks ConversationModel. 
    Returns empty DataFrames by default for all data-retrieving methods.
    """
    cvm_instance = MagicMock(spec=ConversationModel)
    empty_df_cols = [
        'doc_id', 'date', 'content', 'conversation_id', 'document_type',
        'emotion_a', 'emotion_b', 'emotion_c', 'emotion_d', 
        'speaker', 'role', 'score', 'timestamp', 'branch', 'sequence_no', 'user_id', 'persona_id'
    ] # Ensure comprehensive columns
    empty_df = pd.DataFrame(columns=empty_df_cols)
    
    cvm_instance.get_motd.return_value = empty_df
    cvm_instance.get_conscious.return_value = empty_df
    cvm_instance.query.return_value = empty_df
    cvm_instance.get_conversation_history.return_value = empty_df
    cvm_instance.get_documents.return_value = empty_df
    # Mock methods that might be called during ChatManager init or elsewhere
    cvm_instance.next_conversation_id.return_value = "mock-convo-id"
    cvm_instance.get_conversation_report.return_value = pd.DataFrame() # For next_analysis if called
    return cvm_instance

@pytest.fixture
def mock_library():
    """Mocks the Library for document interactions."""
    library_instance = MagicMock(spec=Library)
    library_instance.read_document.return_value = "Mocked content from active document."
    library_instance.exists.return_value = True # Assume doc exists if asked
    library_instance.list_documents = [] # Default to no documents
    return library_instance

@pytest.fixture
def mock_chat_manager(base_chat_config, mock_cvm, sample_persona, mock_library):
    """
    Provides a ChatManager instance with mocked CVM, Roster, and Library.
    The Roster is mocked to contain the sample_persona.
    """
    mock_roster_instance = MagicMock()
    mock_roster_instance.personas = {sample_persona.persona_id: sample_persona}
    
    # Instantiate ChatManager directly with mocks
    manager = ChatManager(cvm=mock_cvm, config=base_chat_config, roster=mock_roster_instance)
    # Replace the library instance if it was created by ChatManager's __init__
    manager.library = mock_library 
    manager.config = base_chat_config # Ensure test config is used
    return manager

@pytest.fixture
def unit_test_xml_strategy(mock_chat_manager):
    """Provides an XMLMemoryTurnStrategy instance initialized with mock_chat_manager."""
    strategy = XMLMemoryTurnStrategy(mock_chat_manager)
    # Set a high default max_character_length for unit tests not focused on truncation,
    # to avoid triggering history management unintentionally.
    strategy.max_character_length = 20000 
    return strategy

@pytest.fixture
def mocked_text_summarizer(mocker):
    """
    Mocks TextSummarizer at its source and the factory function to ensure
    XMLMemoryTurnStrategy always receives a controlled mock instance.
    """
    the_mock_summarizer_instance = MagicMock() # Vanilla MagicMock
    
    # Configure the behavior of our mock instance
    the_mock_summarizer_instance.summarize.side_effect = lambda text, target_length, **kwargs: f"SUMMARIZED[{target_length}]: {text[:target_length-20]}"
    
    def sparsify_side_effect(messages, max_total_length, preserve_recent=4, **kwargs):
        if len(messages) <= preserve_recent:
            return messages
        return messages[-(preserve_recent):]
    the_mock_summarizer_instance.sparsify_conversation.side_effect = sparsify_side_effect
    
    # Attribute checked by XMLMemoryTurnStrategy for model-based summarizer
    the_mock_summarizer_instance._summarize_func = True

    # Patch the source TextSummarizer class. Any instantiation of aim.nlp.summarize.TextSummarizer
    # will now return our mock instance.
    mocker.patch('aim.nlp.summarize.TextSummarizer', return_value=the_mock_summarizer_instance)
    
    # Patch the factory function within the xmlmemory module as a safeguard,
    # in case it's called directly.
    mocker.patch('aim.chat.strategy.xmlmemory.get_default_summarizer', return_value=the_mock_summarizer_instance)
    
    return the_mock_summarizer_instance

@pytest.fixture
def mocked_redis_cache(mocker):
    """
    Mocks RedisCache and patches its instantiation in 'aim.chat.strategy.xmlmemory'.
    Simulates cache misses by default.
    """
    mock_cache_instance = MagicMock(spec=RedisCache)
    
    # Simulate cache miss behavior: always call the generator_func
    def mock_get_or_cache_behavior(content, generator_func, parameters=None, expire=None):
        # For unit tests, we want to see generator_func (i.e., summarizer.summarize) being called
        return generator_func(content, **(parameters or {}))
        
    mock_cache_instance.get_or_cache.side_effect = mock_get_or_cache_behavior
    
    # Patch the RedisCache class within the module where XMLMemoryTurnStrategy instantiates it.
    mocker.patch('aim.chat.strategy.xmlmemory.RedisCache', return_value=mock_cache_instance)
    return mock_cache_instance

@pytest.fixture
def mock_random_choices(mocker):
    """Mocks random.choices to make random removal deterministic, typically picking the first candidate."""
    # Default: always pick the first possible index (0) for removal candidate list
    return mocker.patch('random.choices', return_value=[0]) 