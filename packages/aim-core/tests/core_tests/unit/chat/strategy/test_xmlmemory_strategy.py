# tests/unit/chat/strategy/test_xmlmemory_strategy.py

import pytest
import pandas as pd
from datetime import datetime, timedelta
import re
from unittest.mock import call, ANY, MagicMock

from aim.chat.strategy.xmlmemory import XMLMemoryTurnStrategy
from aim.constants import TOKEN_CHARS # For reference
from aim.agents.persona import Persona

# Fixtures from conftest.py are automatically available and used by tests below.

def test_xml_strategy_user_turn_for(unit_test_xml_strategy, sample_persona):
    """Test the basic user_turn_for method."""
    user_input = "This is a test input from the user."
    turn = unit_test_xml_strategy.user_turn_for(sample_persona, user_input)
    assert turn == {"role": "user", "content": user_input}, \
        "user_turn_for should simply wrap the input in a user role dictionary."

def test_extract_memory_metadata_logic(unit_test_xml_strategy):
    """Test the emotion and keyword extraction logic directly."""
    row_data = {
        'emotion_a': 'Elated', 'emotion_b': 'Curious', 'emotion_c': None, 'emotion_d': 'Focused',
        'content': "A **Remarkable Experience** with **Multiple Keywords**. The **Remarkable Experience** was also **Quite Unique**.",
        # Add other necessary columns for pd.Series creation
        'doc_id': 'doc1', 'date': '2024-01-01 00:00:00', 'conversation_id': 'conv1', 
        'document_type': 'journal', 'speaker': 'persona', 'role': 'assistant', 'score': 1.0,
        'timestamp': int(datetime.now().timestamp()), 'branch':0, 'sequence_no':0, 'user_id':'u1', 'persona_id':'p1'
    }
    series = pd.Series(row_data)
    
    emotions, keywords = unit_test_xml_strategy.extract_memory_metadata(series, top_n_keywords=3)
    
    assert emotions == ['Elated', 'Curious', None, 'Focused']
    assert "**Remarkable Experience**" in keywords
    assert "**Multiple Keywords**" in keywords
    assert "**Quite Unique**" in keywords
    assert len(keywords) == 3, "Should return top_n_keywords if available."
    # Check order by frequency: "**Remarkable Experience**" appears twice.
    assert keywords.index("**Remarkable Experience**") == 0 

    # Test with fewer keywords than top_n
    row_data_less_keywords = {**row_data, 'content': "One **Keyword**."}
    series_less = pd.Series(row_data_less_keywords)
    _, keywords_less = unit_test_xml_strategy.extract_memory_metadata(series_less, top_n_keywords=3)
    assert keywords_less == ["**Keyword**"]

class TestGetConsciousMemoryUnit:
    """Unit tests for the get_conscious_memory method."""

    def test_structure_with_no_dynamic_content(self, unit_test_xml_strategy, sample_persona):
        """Test basic XML structure when CVM returns empty and no extras are set."""
        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)
        
        assert "<PraxOS>--== PraxOS Conscious Memory **Online** ==--</PraxOS>" in xml_output
        assert f"<{unit_test_xml_strategy.hud_name}>" in xml_output
        for thought in sample_persona.thoughts: # From sample_persona fixture
            assert f"<thought>{thought}</thought>" in xml_output
        # No dynamic content means no emotion/keyword aggregates from CVM data
        assert "<emotions>" not in xml_output 
        assert "<keywords>" not in xml_output
        assert "<document" not in xml_output # No current_document
        assert "<workspace" not in xml_output # No current_workspace
        assert "<scratchpad" not in xml_output # No scratch_pad

    def test_motd_inclusion_and_filtering(self, unit_test_xml_strategy, sample_persona, mock_chat_manager):
        """Test MOTD data inclusion and age-based filtering."""
        now_ts = int(datetime.now().timestamp())
        one_day_ago_str = datetime.fromtimestamp(now_ts - 86400).strftime('%Y-%m-%d %H:%M:%S')

        motd_df = pd.DataFrame([
            {'doc_id': 'motd_fresh', 'date': one_day_ago_str,
             'content': 'Fresh MOTD: **DailyUpdate**', 'conversation_id': 'm_c1', 'document_type': 'motd',
             'emotion_a': 'Informative', 'emotion_b': None, 'emotion_c': None, 'emotion_d': None,
             'speaker': 'sys', 'role': 'system', 'score': 1.0, 'timestamp': now_ts - 86400,
             'branch':0, 'sequence_no':0, 'user_id':'sys', 'persona_id':'sys'}  # Add missing for Series
        ])
        mock_chat_manager.cvm.get_motd.return_value = motd_df

        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        # Check for the presence of the core MOTD content, allowing for newlines/indentation
        assert "xoxo MOTD:" in xml_output
        assert "Fresh MOTD: **DailyUpdate**" in xml_output
        assert "<emotions>Informative</emotions>" in xml_output # From fresh MOTD
        assert "<keywords>**DailyUpdate**</keywords>" in xml_output

    def test_journal_inclusion(self, unit_test_xml_strategy, sample_persona, mock_chat_manager):
        """Test inclusion of journal entries from get_conscious."""
        now_ts = int(datetime.now().timestamp())
        journal_df = pd.DataFrame([
            {'doc_id': 'journal1', 'date': datetime.fromtimestamp(now_ts).strftime('%Y-%m-%d %H:%M:%S'), 
             'content': 'My thoughts on **UnitTesting** and **Mocks**.', 
             'conversation_id': 'j_c1', 'document_type': 'journal',
             'emotion_a': 'Analytical', 'emotion_b': 'Pleased', 'emotion_c': None, 'emotion_d': None, 
             'speaker': 'persona', 'role': 'assistant', 'score': 1.0, 'timestamp': now_ts,
             'branch':0, 'sequence_no':0, 'user_id':sample_persona.persona_id, 'persona_id':sample_persona.persona_id}
        ])
        mock_chat_manager.cvm.get_conscious.return_value = journal_df
        
        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        assert "<Journal date=" in xml_output
        assert "My thoughts on **UnitTesting** and **Mocks**." in xml_output
        # Check for emotions and keywords content, allowing for newlines/indentation
        expected_emotions_content = "Analytical, Pleased"
        assert f">{expected_emotions_content}</emotions>" in xml_output or f">\\n    {expected_emotions_content}\\n  </emotions>" in xml_output
        expected_keywords_content = "**UnitTesting**, **Mocks**"
        assert f">{expected_keywords_content}</keywords>" in xml_output or f">\\n    {expected_keywords_content}\\n  </keywords>" in xml_output

    def test_active_document_workspace_and_scratchpad(self, unit_test_xml_strategy, sample_persona, mock_chat_manager):
        """Test inclusion of active document, workspace, and scratchpad."""
        mock_chat_manager.current_document = "unit_test_doc.txt" # Mock library will handle read
        mock_chat_manager.current_workspace = "Unit test workspace content."
        unit_test_xml_strategy.scratch_pad = "Unit test scratchpad." # Set directly on strategy
        
        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)
        
        # Original assertion for nowrap=True:
        # assert '<document metadata="name=&quot;unit_test_doc.txt&quot; length=&quot;5&quot;>Mocked content from active document.</document>' in xml_output
        # Adjusted for metadata as stringified dict and content on newline (nowrap=False default)
        assert "<document metadata=\"{'name': 'unit_test_doc.txt', 'length': 5}\">" in xml_output # linter fix
        # The content might be on a new line and indented. We match for the content itself, allowing for surrounding whitespace.
        assert re.search(r">\s*Mocked content from active document.\s*</document>", xml_output), \
            f"Document content not found or not formatted as expected in: {xml_output}"
        
        # Workspace is now a single element with combined content.
        # Check for metadata first
        assert "<workspace metadata=\"{'length': 4}\">" in xml_output, \
            f"Workspace metadata not found or not formatted as expected in: {xml_output}"
        
        # Check for the combined content, allowing for newlines/indentation introduced by the formatter
        # The combined content is "\n*The user is sharing a workspace with you.*\nUnit test workspace content."
        expected_workspace_content_pattern = r">\n\*The user is sharing a workspace with you\.\*\nUnit test workspace content\.\s*</workspace>"
        assert re.search(expected_workspace_content_pattern, xml_output), \
            f"Combined workspace content not found or not formatted as expected in: {xml_output}"
        
        # Scratchpad is now a single element with combined content.
        # Check for metadata first
        assert "<scratchpad metadata=\"{'length': 3}\">" in xml_output, \
            f"Scratchpad metadata not found or not formatted as expected in: {xml_output}"

        # Check for the combined content, allowing for newlines/indentation
        # Combined content: "\nYou are sharing a scratchpad with yourself.\nUnit test scratchpad."
        expected_scratchpad_content_pattern = r">\n\*You are sharing a scratchpad with yourself\.\*\nUnit test scratchpad\.\s*</scratchpad>"
        assert re.search(expected_scratchpad_content_pattern, xml_output), \
            f"Combined scratchpad content not found or not formatted as expected in: {xml_output}"
        
        # Cleanup for other tests that might share the strategy instance
        mock_chat_manager.current_document = None
        mock_chat_manager.current_workspace = None
        unit_test_xml_strategy.scratch_pad = None

    def test_pinned_message_inclusion(self, unit_test_xml_strategy, sample_persona, mock_chat_manager):
        """Test inclusion and processing of pinned messages."""
        now_ts = int(datetime.now().timestamp())
        pinned_doc_id = "pinned_doc_123"
        pinned_date_str = datetime.fromtimestamp(now_ts - 2*86400).strftime('%Y-%m-%d %H:%M:%S')

        pinned_doc_data = {
            'doc_id': pinned_doc_id, 
            'date': pinned_date_str, 
            'content': 'This is **CriticallyImportant** pinned content with a **SpecialKeyword**.',
            'conversation_id': 'p_c1', 
            'document_type': 'pinned_note',
            'emotion_a': 'Focused', 
            'emotion_b': 'Determined', 
            'emotion_c': None, 
            'emotion_d': None, 
            'speaker': 'persona', 
            'role': 'assistant', 
            'score': 1.0, 
            'timestamp': now_ts - 2*86400,
            'branch': 0, 
            'sequence_no': 0, 
            'user_id': sample_persona.persona_id, 
            'persona_id': sample_persona.persona_id
        }
        pinned_doc_series = pd.Series(pinned_doc_data)

        # Mock the get_doc_by_id method on cvm
        mock_chat_manager.cvm.get_documents = MagicMock(return_value=pd.DataFrame([pinned_doc_series]))

        # Pin the message
        unit_test_xml_strategy.pin_message(pinned_doc_id)
        
        xml_output, _ = unit_test_xml_strategy.get_conscious_memory(sample_persona)

        # Assert Pinned section exists and contains the content (doc_id is not included in the tag)
        assert f'<memory_pinned date="{pinned_date_str}" type="pinned_note">' in xml_output
        assert ">This is **CriticallyImportant** pinned content with a **SpecialKeyword**.</memory_pinned>" in xml_output or \
               re.search(r">\s*This is \*\*CriticallyImportant\*\* pinned content with a \*\*SpecialKeyword\*\*\.\s*</memory_pinned>", xml_output)

        # Assert emotions and keywords from pinned message are in global aggregates
        # Assuming nowrap=True for global tags as per user's update
        assert "<emotions>Focused, Determined</emotions>" in xml_output
        assert "<keywords>**CriticallyImportant**, **SpecialKeyword**</keywords>" in xml_output

        # Clean up for other tests
        unit_test_xml_strategy.clear_pinned()
        # Reset the mock if it's specific to this test and might interfere
        mock_chat_manager.cvm.get_documents = MagicMock(return_value=pd.DataFrame(columns=pinned_doc_series.index.tolist() + ['date', 'speaker'])) # Ensure columns match for consistency


class TestChatTurnsForStructureAndContent:
    """Unit tests for the chat_turns_for method's turn construction logic."""

    def test_empty_history_turn_structure(self, unit_test_xml_strategy, sample_persona):
        """Test turn structure with no prior history."""
        user_input = "A fresh start!"
        turns = unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, [])
        
        assert len(turns) == 3, "Expected Consciousness, Wakeup, UserInput turns."
        assert turns[0]['role'] == 'user', "Consciousness block should be role: user."
        assert "<PraxOS>" in turns[0]['content'], "Consciousness XML missing."
        assert turns[1]['role'] == 'assistant', "Wakeup message should be role: assistant."
        assert turns[1]['content'] == sample_persona.get_wakeup()
        assert turns[2]['role'] == 'user', "Final turn must be current user input."
        assert turns[2]['content'] == user_input + "\n\n" # Check for added newlines

    def test_history_alternation_and_content(self, unit_test_xml_strategy, sample_persona):
        """Test turn structure with existing history."""
        history = [
            {"role": "user", "content": "Earlier question."},
            {"role": "assistant", "content": "Earlier answer."}
        ]
        user_input = "Follow-up question."
        turns = unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, history)
        
        # Expected: Consciousness, Wakeup, History[0], History[1], UserInput
        assert len(turns) == 5
        assert "<PraxOS>" in turns[0]['content'] # Consciousness
        assert turns[1]['content'] == sample_persona.get_wakeup() # Wakeup
        assert turns[2] == history[0] # First history item
        assert turns[3] == history[1] # Second history item
        assert turns[4]['content'] == user_input + "\n\n" # Current user input

    def test_thought_content_injection_logic(self, unit_test_xml_strategy, sample_persona):
        """Test how thought_content is injected into the history using insert_at_fold."""
        unit_test_xml_strategy.thought_content = "<think>My brilliant unit test thought.</think>"
        history = [
            {"role": "user", "content": "User asks something."},
            {"role": "assistant", "content": "Assistant replies."}
        ]
        user_input = "User asks again."

        turns = unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, history)

        # insert_at_fold with fold_depth=4 looks for the 4th user turn
        # turns structure: [consciousness, wakeup, history[0], history[1], user_input]
        # Since there are only 3 user turns total (indices 2, 4), fold_depth=4 won't be reached
        # So it will insert at the last user turn before the end
        assert len(turns) == 5
        # Check that thought_content appears somewhere in the turns
        all_content = ''.join(turn['content'] for turn in turns)
        assert "<think>My brilliant unit test thought.</think>" in all_content

        # Test with history ending in user turn
        history_ends_user = [{"role": "user", "content": "A single user utterance."}]
        turns_ends_user = unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, history_ends_user)
        # Expected: Consciousness, Wakeup, history_ends_user[0], user_input (with thought inserted)
        assert len(turns_ends_user) == 4
        all_content_ends_user = ''.join(turn['content'] for turn in turns_ends_user)
        assert "<think>My brilliant unit test thought.</think>" in all_content_ends_user

        unit_test_xml_strategy.thought_content = None # Cleanup


class TestHistoryManagementStrategiesUnit:
    """Unit tests for history management logic, ensuring correct delegation."""

    # This auto-use fixture ensures mocks are set up for all tests in this class
    @pytest.fixture(autouse=True)
    def _ensure_summarizer_and_cache_mocks(self, mocked_text_summarizer, mocked_redis_cache):
        pass

    def _create_long_history(self, num_pairs=5, content_char_len=300):
        history = []
        for i in range(num_pairs):
            history.append({"role": "user", "content": f"U{i}: " + ("X" * (content_char_len - 5))})
            history.append({"role": "assistant", "content": f"A{i}: " + ("Y" * (content_char_len - 5))})
        return history

    def test_sparsification_strategy_call(self, unit_test_xml_strategy, sample_persona, base_chat_config, mocked_text_summarizer):
        """Verify that _apply_sparsification_strategy (via sparsify_conversation) is called."""
        base_chat_config.history_management_strategy = "sparsify"

        # Create long history to force overage
        # Each message ~300 chars = ~75 tokens (rough estimate 4 chars/token)
        long_history = self._create_long_history(num_pairs=2, content_char_len=300)
        user_input = "Query"

        # Use small context/output tokens to force history overage (>50% threshold)
        # With 2 pairs (4 messages) of ~75 tokens each = ~300 tokens
        # usable_context = max_context - max_output - 1024
        # So with max_context=2000 and max_output=500, usable = 2000-500-1024 = 476 tokens
        # History of ~300 tokens / 476 usable = 63% > 50% threshold
        max_context_tokens = 2000
        max_output_tokens = 500
        initial_content_len = 0  # in tokens

        # Call the method that triggers history management
        unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, long_history,
                                               content_len=initial_content_len,
                                               max_context_tokens=max_context_tokens,
                                               max_output_tokens=max_output_tokens)

        mocked_text_summarizer.sparsify_conversation.assert_called_once()
        # Check some key arguments passed to sparsify_conversation
        args, kwargs = mocked_text_summarizer.sparsify_conversation.call_args
        assert not args  # Expect no positional arguments
        assert kwargs['messages'] == long_history # The original history
        assert 'max_total_length' in kwargs
        # max_total_length should be less than original, indicating reduction goal
        assert kwargs['max_total_length'] < sum(len(h['content']) for h in long_history)
        assert kwargs['preserve_recent'] == 4


    def test_ai_summarization_strategy_call(self, unit_test_xml_strategy, sample_persona, base_chat_config, mocked_text_summarizer, mocked_redis_cache):
        """Verify that _apply_ai_summarization_strategy leads to summarize calls."""
        base_chat_config.history_management_strategy = "ai_summarize"
        base_chat_config.summarizer_model = "mock-summarizer-model" # Ensure correct mock path

        # History with messages long enough to be summarized (>100 tokens)
        # preserve_recent=4, so the first 2 messages are candidates
        # Use varied text that tokenizes more realistically (~1 token per 4 chars)
        long_history = [
            {"role": "user", "content": "User question: " + " ".join([f"word{i}" for i in range(100)])},          # Candidate 1 (~100 tokens)
            {"role": "assistant", "content": "Assistant response: " + " ".join([f"answer{i}" for i in range(100)])},  # Candidate 2 (~100 tokens)
            {"role": "user", "content": "User short " + "u"*50},         # Preserved
            {"role": "assistant", "content": "Assistant short " + "a"*50}, # Preserved
            {"role": "user", "content": "User short 2 " + "x"*50},       # Preserved
            {"role": "assistant", "content": "Assi short 2 " + "y"*50},  # Preserved
        ]
        user_input = "Query"

        # Use small max_context_tokens and max_output_tokens to force overage
        max_context_tokens = 2000
        max_output_tokens = 500
        initial_content_len = 0  # in tokens

        unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, long_history,
                                               content_len=initial_content_len,
                                               max_context_tokens=max_context_tokens,
                                               max_output_tokens=max_output_tokens)
        
        # summarize is called via cache.get_or_cache's generator_func
        # Expect it to be called for the two long messages
        assert mocked_text_summarizer.summarize.call_count >= 1 # Could be 2 if both are processed
        mocked_redis_cache.get_or_cache.assert_called()

        # Verify the content passed to summarize (via cache)
        summarize_call_args_list = mocked_text_summarizer.summarize.call_args_list
        called_contents = [c[0][0] for c in summarize_call_args_list] # Get the 'text' arg from each call
        
        assert long_history[0]['content'] in called_contents
        assert long_history[1]['content'] in called_contents
        
        # Verify parameters passed to summarize for one of the calls
        for i, call_args_tuple in enumerate(summarize_call_args_list):
            params_dict = call_args_tuple[1] # kwargs of summarize
            original_content = call_args_tuple[0][0]  # The text being summarized
            assert 'target_length' in params_dict
            assert params_dict['target_length'] < len(original_content), \
                f"Target length {params_dict['target_length']} should be less than original {len(original_content)}"
            assert params_dict['method'] == 'model'


    def test_random_removal_strategy_call_and_effect(self, unit_test_xml_strategy, sample_persona, base_chat_config, mock_random_choices):
        """Verify that _apply_random_removal_strategy is called and reduces history."""
        base_chat_config.history_management_strategy = "random_removal"

        # Create history with enough content to exceed 50% threshold
        # With usable_context=476, we need >238 tokens
        # Using 5 pairs (10 turns) with 150 chars each gives ~315 tokens (66%)
        # This provides enough overage to trigger random removal with sufficient choices
        long_history = self._create_long_history(num_pairs=5, content_char_len=150) # 10 turns
        original_history_length = len(long_history)
        user_input = "Query"

        # Use small max_context_tokens and max_output_tokens to force overage
        max_context_tokens = 2000
        max_output_tokens = 500
        initial_content_len = 0  # in tokens

        # mock_random_choices is configured to always pick the first candidate for removal

        final_turns = unit_test_xml_strategy.chat_turns_for(sample_persona, user_input, long_history,
                                                             content_len=initial_content_len,
                                                             max_context_tokens=max_context_tokens,
                                                             max_output_tokens=max_output_tokens)

        mock_random_choices.assert_called() # Check that our deterministic mock was used.
        
        # Calculate the length of the history part in the final_turns
        # final_turns = [Consciousness, Wakeup, ...processed_history..., UserInput]
        processed_history_part_len = len(final_turns) - 3 
        
        assert processed_history_part_len < original_history_length, \
            "Random removal should have reduced the number of history turns."
        # The exact number can be tricky due to the "take latter half" logic,
        # but it must be shorter. For 10 turns -> candidates (0-5)
        # mock removes index 0 (turn 0&1). History is 8. Latter half is 4.
        # if it ran again (if overage still high), removes from remaining candidates.
        # This test primarily ensures the path is taken and *some* reduction happens.
        # A specific count like `assert processed_history_part_len == 4` would be too brittle. 