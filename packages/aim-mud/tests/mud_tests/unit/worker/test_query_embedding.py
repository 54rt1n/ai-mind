# tests/mud_tests/unit/worker/test_query_embedding.py
# AI-Mind (C) 2026 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for pre-computed embedding passthrough in memory retrieval.

These tests verify that pre-computed embeddings from conversation entries
are passed through to CVM queries for FAISS reranking, ensuring the same
embedding is used for both indexing and retrieval.
"""

import base64
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from aim_mud_types import MUDConversationEntry


class TestGetCurrentTurnEmbedding:
    """Tests for get_current_turn_embedding() helper method."""

    def test_returns_none_when_no_entries(self):
        """get_current_turn_embedding returns None when no entries stored."""
        # Create mock worker without entries
        worker = MagicMock()
        worker._current_turn_entries = []

        # Import and call the method (simulate via attribute access)
        from andimud_worker.mixins.turns import TurnsMixin
        result = TurnsMixin.get_current_turn_embedding(worker)

        assert result is None

    def test_returns_none_when_entries_not_set(self):
        """get_current_turn_embedding returns None when attribute not set."""
        worker = MagicMock(spec=[])  # No _current_turn_entries attribute

        from andimud_worker.mixins.turns import TurnsMixin
        result = TurnsMixin.get_current_turn_embedding(worker)

        assert result is None

    def test_returns_last_user_entry_embedding(self):
        """get_current_turn_embedding returns embedding from last user entry."""
        # Create test embedding
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        encoded = base64.b64encode(test_embedding.tobytes()).decode('ascii')

        # Create entries with user entry last
        entries = [
            MUDConversationEntry(
                role="user",
                content="first user message",
                tokens=10,
                document_type="DOC_MUD_WORLD",
                conversation_id="test",
                sequence_no=1,
                embedding=None,  # First entry has no embedding
            ),
            MUDConversationEntry(
                role="assistant",
                content="assistant response",
                tokens=15,
                document_type="DOC_MUD_AGENT",
                conversation_id="test",
                sequence_no=2,
                embedding=None,  # Assistant entries don't have embeddings
            ),
            MUDConversationEntry(
                role="user",
                content="second user message",
                tokens=12,
                document_type="DOC_MUD_WORLD",
                conversation_id="test",
                sequence_no=3,
                embedding=encoded,  # Last user entry has embedding
            ),
        ]

        worker = MagicMock()
        worker._current_turn_entries = entries

        from andimud_worker.mixins.turns import TurnsMixin
        result = TurnsMixin.get_current_turn_embedding(worker)

        assert result is not None
        np.testing.assert_array_almost_equal(result, test_embedding)

    def test_returns_none_when_only_assistant_entries(self):
        """get_current_turn_embedding returns None when no user entries."""
        entries = [
            MUDConversationEntry(
                role="assistant",
                content="assistant response",
                tokens=15,
                document_type="DOC_MUD_AGENT",
                conversation_id="test",
                sequence_no=1,
                embedding=None,
            ),
        ]

        worker = MagicMock()
        worker._current_turn_entries = entries

        from andimud_worker.mixins.turns import TurnsMixin
        result = TurnsMixin.get_current_turn_embedding(worker)

        assert result is None

    def test_returns_none_when_user_entry_has_no_embedding(self):
        """get_current_turn_embedding returns None when user entry lacks embedding."""
        entries = [
            MUDConversationEntry(
                role="user",
                content="user message",
                tokens=10,
                document_type="DOC_MUD_WORLD",
                conversation_id="test",
                sequence_no=1,
                embedding=None,  # No embedding
            ),
        ]

        worker = MagicMock()
        worker._current_turn_entries = entries

        from andimud_worker.mixins.turns import TurnsMixin
        result = TurnsMixin.get_current_turn_embedding(worker)

        assert result is None


class TestSpeakingProcessorEmbeddingPassthrough:
    """Tests for embedding passthrough in SpeakingProcessor."""

    @pytest.mark.asyncio
    async def test_passes_embedding_to_build_turns(self):
        """SpeakingProcessor passes embedding to build_turns."""
        # Create test embedding
        test_embedding = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        encoded = base64.b64encode(test_embedding.tobytes()).decode('ascii')

        # Create entry with embedding
        entry = MUDConversationEntry(
            role="user",
            content="test message",
            tokens=10,
            document_type="DOC_MUD_WORLD",
            conversation_id="test",
            sequence_no=1,
            embedding=encoded,
        )

        # Create mock worker
        worker = MagicMock()
        worker._current_turn_entries = [entry]
        worker.persona = MagicMock()
        # Create proper session mock with None world_state to skip formatting
        worker.session = MagicMock()
        worker.session.world_state = None
        worker.session.persona_id = "test"
        worker.model = MagicMock()
        worker.model.max_tokens = 8192
        worker.chat_config = MagicMock()
        worker.chat_config.max_tokens = 1024
        worker._is_fresh_session = AsyncMock(return_value=False)
        worker._check_abort_requested = AsyncMock(return_value=False)
        worker._call_llm = AsyncMock(return_value="[== Andi's Emotional State: +Test+ ==]\n\nTest response")
        worker._emit_actions = AsyncMock()
        worker.model_set = MagicMock()
        worker.model_set.get_model_name = MagicMock(return_value="test-model")

        # Mock response strategy
        worker._response_strategy = MagicMock()
        worker._response_strategy.build_turns = AsyncMock(return_value=[
            {"role": "user", "content": "test"}
        ])

        # Import and call
        from andimud_worker.turns.processor.speaking import SpeakingProcessor
        from andimud_worker.mixins.turns import TurnsMixin

        # Patch get_current_turn_embedding to use actual implementation
        worker.get_current_turn_embedding = lambda: TurnsMixin.get_current_turn_embedding(worker)

        processor = SpeakingProcessor(worker)
        processor.memory_query = "test query"

        from aim_mud_types import MUDTurnRequest, MUDEvent
        turn_request = MUDTurnRequest(turn_id="test-turn", reason="events", sequence_id=1)

        await processor._decide_action(turn_request, [])

        # Verify build_turns was called with query_embedding
        worker._response_strategy.build_turns.assert_called_once()
        call_kwargs = worker._response_strategy.build_turns.call_args[1]
        assert "query_embedding" in call_kwargs
        np.testing.assert_array_almost_equal(
            call_kwargs["query_embedding"], test_embedding
        )


class TestThinkingProcessorEmbeddingPassthrough:
    """Tests for embedding passthrough in ThinkingTurnProcessor."""

    @pytest.mark.asyncio
    async def test_passes_embedding_to_build_turns(self):
        """ThinkingTurnProcessor passes embedding to build_turns."""
        # Create test embedding
        test_embedding = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        encoded = base64.b64encode(test_embedding.tobytes()).decode('ascii')

        # Create entry with embedding
        entry = MUDConversationEntry(
            role="user",
            content="test message",
            tokens=10,
            document_type="DOC_MUD_WORLD",
            conversation_id="test",
            sequence_no=1,
            embedding=encoded,
        )

        # Create mock worker
        worker = MagicMock()
        worker._current_turn_entries = [entry]
        worker.config = MagicMock()
        worker.config.agent_id = "test-agent"
        worker.persona = MagicMock()
        worker.session = MagicMock()
        worker.model = MagicMock()
        worker.model.max_tokens = 8192
        worker.chat_config = MagicMock()
        worker.chat_config.max_tokens = 1024
        # Mock redis with async methods
        worker.redis = AsyncMock()
        worker.redis.llen = AsyncMock(return_value=10)
        worker.redis.hset = AsyncMock()
        worker.redis.expire = AsyncMock()
        worker._is_fresh_session = AsyncMock(return_value=False)
        worker._check_abort_requested = AsyncMock(return_value=False)
        worker._load_thought_content = AsyncMock()
        worker._call_llm = AsyncMock(return_value="<reasoning>\n<inspiration>test</inspiration>\n</reasoning>")
        worker._emit_actions = AsyncMock()
        worker.model_set = MagicMock()
        worker.model_set.get_model_name = MagicMock(return_value="test-model")

        # Mock response strategy
        worker._response_strategy = MagicMock()
        worker._response_strategy.build_turns = AsyncMock(return_value=[
            {"role": "user", "content": "test"}
        ])

        # Import and call
        from andimud_worker.turns.processor.thinking import ThinkingTurnProcessor
        from andimud_worker.mixins.turns import TurnsMixin

        # Patch get_current_turn_embedding
        worker.get_current_turn_embedding = lambda: TurnsMixin.get_current_turn_embedding(worker)

        processor = ThinkingTurnProcessor(worker)

        from aim_mud_types import MUDTurnRequest
        turn_request = MUDTurnRequest(turn_id="test-turn", reason="idle", sequence_id=1)

        await processor._decide_action(turn_request, [])

        # Verify build_turns was called with query_embedding
        worker._response_strategy.build_turns.assert_called_once()
        call_kwargs = worker._response_strategy.build_turns.call_args[1]
        assert "query_embedding" in call_kwargs
        np.testing.assert_array_almost_equal(
            call_kwargs["query_embedding"], test_embedding
        )
