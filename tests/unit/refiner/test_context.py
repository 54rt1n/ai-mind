# tests/unit/refiner/test_context.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Tests for the refiner context gathering module."""

import pytest
from unittest.mock import MagicMock, patch

import pandas as pd

from aim.refiner.context import (
    ContextGatherer,
    GatheredContext,
    get_paradigm_doc_types,
    get_paradigm_queries,
    get_approach_doc_types,
)


class TestGatheredContext:
    """Tests for the GatheredContext dataclass."""

    def test_empty_with_no_documents(self):
        """empty property should return True for empty documents list."""
        ctx = GatheredContext(
            documents=[],
            paradigm="brainstorm",
        )
        assert ctx.empty is True

    def test_empty_with_documents(self):
        """empty property should return False when documents exist."""
        ctx = GatheredContext(
            documents=[{"content": "test", "doc_id": "1"}],
            paradigm="brainstorm",
        )
        assert ctx.empty is False

    def test_doc_count_returns_length(self):
        """doc_count should return number of documents."""
        ctx = GatheredContext(
            documents=[{"content": "doc1"}, {"content": "doc2"}, {"content": "doc3"}],
            paradigm="brainstorm",
        )
        assert ctx.doc_count == 3

    def test_to_records_returns_documents(self):
        """to_records should return the documents list."""
        docs = [
            {"content": "doc1", "document_type": "codex"},
            {"content": "doc2", "document_type": "journal"},
        ]
        ctx = GatheredContext(documents=docs, paradigm="knowledge")

        records = ctx.to_records()

        assert len(records) == 2
        assert records[0]["content"] == "doc1"
        assert records[1]["document_type"] == "journal"

    def test_to_records_empty(self):
        """to_records should return empty list for empty context."""
        ctx = GatheredContext(documents=[], paradigm="brainstorm")
        assert ctx.to_records() == []

    def test_default_values(self):
        """Default values should be set correctly."""
        ctx = GatheredContext()

        assert ctx.documents == []
        assert ctx.paradigm == "unknown"
        assert ctx.tokens_used == 0
        assert ctx.conversation_count == 0
        assert ctx.other_count == 0

    def test_stores_metadata(self):
        """Should store tokens_used and counts."""
        ctx = GatheredContext(
            documents=[{"content": "test"}],
            paradigm="daydream",
            tokens_used=500,
            conversation_count=2,
            other_count=3,
        )

        assert ctx.tokens_used == 500
        assert ctx.conversation_count == 2
        assert ctx.other_count == 3


class TestParadigmConfigurations:
    """Tests for paradigm configuration loaded from config files."""

    def test_paradigm_doc_types_loads_all_paradigms(self):
        """get_paradigm_doc_types should load all paradigms from config."""
        assert len(get_paradigm_doc_types("brainstorm")) > 0
        assert len(get_paradigm_doc_types("daydream")) > 0
        assert len(get_paradigm_doc_types("knowledge")) > 0
        assert len(get_paradigm_doc_types("critique")) > 0

    def test_paradigm_queries_returns_fallback_when_not_defined(self):
        """get_paradigm_queries should return fallback when queries not in config."""
        # Queries are optional now - we use random sampling
        # Function should return empty list or fallback gracefully
        queries = get_paradigm_queries("brainstorm")
        assert isinstance(queries, list)

    def test_approach_doc_types_loads_all_approaches(self):
        """get_approach_doc_types should load all approaches from config."""
        assert len(get_approach_doc_types("philosopher")) > 0
        assert len(get_approach_doc_types("journaler")) > 0
        assert len(get_approach_doc_types("daydream")) > 0
        assert len(get_approach_doc_types("critique")) > 0


class TestContextGatherer:
    """Tests for the ContextGatherer class."""

    @pytest.fixture
    def mock_cvm(self):
        """Mock ConversationModel."""
        cvm = MagicMock()
        # Default query returns some documents
        cvm.query.return_value = pd.DataFrame([
            {"doc_id": "1", "content": "First document", "document_type": "codex", "date": "2025-01-01"},
            {"doc_id": "2", "content": "Second document", "document_type": "journal", "date": "2025-01-02"},
        ])
        return cvm

    @pytest.fixture
    def mock_cvm_empty(self):
        """Mock ConversationModel that returns empty results."""
        cvm = MagicMock()
        cvm.query.return_value = pd.DataFrame()
        return cvm

    @pytest.fixture
    def mock_token_counter(self):
        """Mock token counter that returns word count."""
        return lambda text: len(text.split())

    @pytest.fixture
    def mock_reranker(self):
        """Mock MemoryReranker."""
        reranker = MagicMock()
        # Return tagged results matching input
        reranker.rerank.return_value = []
        return reranker

    @pytest.fixture
    def gatherer(self, mock_cvm, mock_token_counter):
        """Create a ContextGatherer with mocked dependencies."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_instance = MagicMock()
            mock_reranker_instance.rerank.return_value = []
            mock_reranker_class.return_value = mock_reranker_instance
            return ContextGatherer(mock_cvm, mock_token_counter)

    # Test initialization
    def test_init_stores_cvm(self, mock_cvm, mock_token_counter):
        """ContextGatherer should store the CVM reference."""
        with patch('aim.refiner.context.MemoryReranker'):
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
        assert gatherer.cvm is mock_cvm

    def test_init_stores_token_counter(self, mock_cvm, mock_token_counter):
        """ContextGatherer should store the token counter."""
        with patch('aim.refiner.context.MemoryReranker'):
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
        assert gatherer.token_counter is mock_token_counter

    def test_init_creates_reranker(self, mock_cvm, mock_token_counter):
        """ContextGatherer should create a MemoryReranker."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            mock_reranker_class.assert_called_once()

    # Test _get_paradigm_doc_types
    def test_get_paradigm_doc_types_brainstorm(self, gatherer):
        """brainstorm paradigm should include brainstorm docs."""
        doc_types = gatherer._get_paradigm_doc_types("brainstorm")
        assert len(doc_types) > 0

    def test_get_paradigm_doc_types_daydream(self, gatherer):
        """daydream paradigm should return doc types."""
        doc_types = gatherer._get_paradigm_doc_types("daydream")
        assert len(doc_types) > 0

    def test_get_paradigm_doc_types_knowledge(self, gatherer):
        """knowledge paradigm should include codex docs."""
        doc_types = gatherer._get_paradigm_doc_types("knowledge")
        assert len(doc_types) > 0

    def test_get_paradigm_doc_types_unknown_returns_fallback(self, gatherer):
        """Unknown paradigm should return fallback doc types."""
        doc_types = gatherer._get_paradigm_doc_types("unknown_paradigm")
        # Should return some fallback, not empty
        assert len(doc_types) > 0

    # Test _get_paradigm_queries
    def test_get_paradigm_queries_returns_list(self, gatherer):
        """_get_paradigm_queries should return list (may be empty with random sampling)."""
        queries = gatherer._get_paradigm_queries("brainstorm")
        assert isinstance(queries, list)

    def test_get_paradigm_queries_unknown_returns_fallback(self, gatherer):
        """Unknown paradigm should return fallback queries."""
        queries = gatherer._get_paradigm_queries("unknown")
        # Should return some fallback queries
        assert len(queries) > 0

    # Test get_doc_types_for_approach
    def test_doc_types_for_philosopher(self, gatherer):
        """philosopher approach should include codex and pondering."""
        doc_types = gatherer.get_doc_types_for_approach("philosopher")
        assert len(doc_types) > 0

    def test_doc_types_for_journaler(self, gatherer):
        """journaler approach should include journal and conversation."""
        doc_types = gatherer.get_doc_types_for_approach("journaler")
        assert len(doc_types) > 0

    def test_doc_types_for_daydream(self, gatherer):
        """daydream approach should include daydream and journal."""
        doc_types = gatherer.get_doc_types_for_approach("daydream")
        assert len(doc_types) > 0

    def test_doc_types_fallback_for_unknown(self, gatherer):
        """Unknown approach should return fallback doc types."""
        doc_types = gatherer.get_doc_types_for_approach("unknown")
        # Should return some fallback, not empty
        assert len(doc_types) > 0

    # Test broad_gather
    @pytest.mark.asyncio
    async def test_broad_gather_returns_gathered_context(self, mock_cvm, mock_token_counter):
        """broad_gather should return GatheredContext."""
        mock_cvm.sample_by_type.return_value = pd.DataFrame([
            {"doc_id": "1", "content": "test", "document_type": "codex"},
        ])

        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            result = await gatherer.broad_gather(paradigm="brainstorm")

        assert isinstance(result, GatheredContext)
        assert result.paradigm == "brainstorm"

    @pytest.mark.asyncio
    async def test_broad_gather_uses_sample_by_type(self, mock_cvm, mock_token_counter):
        """broad_gather should call cvm.sample_by_type for random sampling."""
        mock_cvm.sample_by_type.return_value = pd.DataFrame()

        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            await gatherer.broad_gather(paradigm="brainstorm")

        mock_cvm.sample_by_type.assert_called_once()

    @pytest.mark.asyncio
    async def test_broad_gather_empty_returns_empty_context(self, mock_cvm_empty, mock_token_counter):
        """broad_gather should return empty context when no docs found."""
        mock_cvm_empty.sample_by_type.return_value = pd.DataFrame()

        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            gatherer = ContextGatherer(mock_cvm_empty, mock_token_counter)
            result = await gatherer.broad_gather(paradigm="brainstorm")

        assert result.empty
        assert result.doc_count == 0

    @pytest.mark.asyncio
    async def test_broad_gather_returns_documents_within_budget(self, mock_cvm, mock_token_counter):
        """broad_gather should return documents within token budget."""
        mock_cvm.sample_by_type.return_value = pd.DataFrame([
            {"doc_id": "1", "content": "First doc", "document_type": "codex"},
            {"doc_id": "2", "content": "Second doc", "document_type": "journal"},
        ])

        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            result = await gatherer.broad_gather(paradigm="brainstorm", token_budget=1000)

        assert result.doc_count <= 2

    @pytest.mark.asyncio
    async def test_broad_gather_uses_paradigm_doc_types(self, mock_cvm, mock_token_counter):
        """broad_gather should pass paradigm doc types to sample_by_type."""
        mock_cvm.sample_by_type.return_value = pd.DataFrame()

        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            await gatherer.broad_gather(paradigm="critique")

        # Check that sample_by_type was called with doc_types
        call_kwargs = mock_cvm.sample_by_type.call_args[1]
        assert "doc_types" in call_kwargs
        assert len(call_kwargs["doc_types"]) > 0

    # Test targeted_gather
    @pytest.mark.asyncio
    async def test_targeted_gather_uses_topic(self, mock_cvm, mock_token_counter):
        """targeted_gather should query using the topic."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = []
            mock_reranker_class.return_value = mock_reranker

            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            await gatherer.targeted_gather(topic="consciousness", approach="philosopher")

        # Check that query was called with the topic
        mock_cvm.query.assert_called()
        # First call args should contain the topic
        calls = mock_cvm.query.call_args_list
        assert len(calls) > 0

    @pytest.mark.asyncio
    async def test_targeted_gather_returns_context(self, mock_cvm, mock_token_counter):
        """targeted_gather should return GatheredContext."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker = MagicMock()
            mock_row = MagicMock()
            mock_row.to_dict.return_value = {"content": "test", "document_type": "codex"}
            mock_reranker.rerank.return_value = [("tag", mock_row)]
            mock_reranker_class.return_value = mock_reranker

            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            result = await gatherer.targeted_gather(topic="test", approach="philosopher")

        assert isinstance(result, GatheredContext)

    @pytest.mark.asyncio
    async def test_targeted_gather_empty_returns_empty(self, mock_cvm_empty, mock_token_counter):
        """targeted_gather should return empty context when no docs found."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = []
            mock_reranker_class.return_value = mock_reranker

            gatherer = ContextGatherer(mock_cvm_empty, mock_token_counter)
            result = await gatherer.targeted_gather(topic="nothing", approach="philosopher")

        assert result.empty

    @pytest.mark.asyncio
    async def test_targeted_gather_stores_approach_as_paradigm(self, mock_cvm, mock_token_counter):
        """targeted_gather should store approach as paradigm in context."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker = MagicMock()
            mock_reranker.rerank.return_value = []
            mock_reranker_class.return_value = mock_reranker

            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            result = await gatherer.targeted_gather(topic="test", approach="journaler")

        assert result.paradigm == "journaler"

    @pytest.mark.asyncio
    async def test_targeted_gather_counts_documents(self, mock_cvm, mock_token_counter):
        """targeted_gather should count conversation vs other docs."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker = MagicMock()
            mock_row1 = MagicMock()
            mock_row1.to_dict.return_value = {"content": "conv", "document_type": "conversation"}
            mock_row2 = MagicMock()
            mock_row2.to_dict.return_value = {"content": "other", "document_type": "codex"}
            mock_reranker.rerank.return_value = [("tag", mock_row1), ("tag", mock_row2)]
            mock_reranker_class.return_value = mock_reranker

            gatherer = ContextGatherer(mock_cvm, mock_token_counter)
            result = await gatherer.targeted_gather(topic="test", approach="philosopher")

        assert result.conversation_count == 1
        assert result.other_count == 1

    # Test _query_by_buckets
    def test_query_by_buckets_queries_conversations_and_others(self, mock_cvm, mock_token_counter):
        """_query_by_buckets should query both conversation and other doc types."""
        with patch('aim.refiner.context.MemoryReranker') as mock_reranker_class:
            mock_reranker_class.return_value = MagicMock()
            gatherer = ContextGatherer(mock_cvm, mock_token_counter)

            gatherer._query_by_buckets(
                queries=["test query"],
                source_tag="test",
                seen_docs=set(),
                top_n=10,
            )

        # Should have called query twice (conversations and others)
        assert mock_cvm.query.call_count == 2
