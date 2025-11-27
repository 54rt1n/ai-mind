# tests/unit/dreamer/test_context.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for context DSL."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pandas as pd

from aim.dreamer.models import (
    ContextAction,
    StepDefinition,
    StepConfig,
    StepOutput,
    PipelineState,
)
from aim.dreamer.context import (
    prepare_step_context,
    _load_conversation,
    _query_documents,
    _sort_doc_ids,
    _filter_doc_ids,
)


class TestContextAction:
    """Tests for ContextAction model."""

    def test_load_conversation_action(self):
        """Test ContextAction for load_conversation."""
        action = ContextAction(
            action="load_conversation",
            target="current",
            document_types=["conversation"],
        )
        assert action.action == "load_conversation"
        assert action.target == "current"
        assert action.document_types == ["conversation"]

    def test_query_action(self):
        """Test ContextAction for query."""
        action = ContextAction(
            action="query",
            document_types=["reflection", "pondering"],
            top_n=10,
        )
        assert action.action == "query"
        assert action.document_types == ["reflection", "pondering"]
        assert action.top_n == 10

    def test_sort_action(self):
        """Test ContextAction for sort."""
        action = ContextAction(
            action="sort",
            by="timestamp",
            direction="ascending",
        )
        assert action.action == "sort"
        assert action.by == "timestamp"
        assert action.direction == "ascending"

    def test_filter_action(self):
        """Test ContextAction for filter."""
        action = ContextAction(
            action="filter",
            match="*",
        )
        assert action.action == "filter"
        assert action.match == "*"


class TestStepDefinitionWithContext:
    """Tests for StepDefinition with context DSL."""

    def test_step_definition_with_context(self):
        """Test StepDefinition with context DSL."""
        step = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
            context=[
                ContextAction(action="load_conversation", document_types=["conversation"]),
                ContextAction(action="sort", by="timestamp", direction="ascending"),
            ],
        )
        assert step.context is not None
        assert len(step.context) == 2
        assert step.context[0].action == "load_conversation"
        assert step.context[1].action == "sort"

    def test_step_definition_without_context(self):
        """Test StepDefinition without context DSL."""
        step = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
        )
        assert step.context is None


class TestPipelineStateContextDocIds:
    """Tests for PipelineState context_doc_ids field."""

    def test_context_doc_ids_default(self):
        """Test context_doc_ids defaults to empty list."""
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert state.context_doc_ids == []

    def test_context_doc_ids_accumulation(self):
        """Test context_doc_ids accumulates properly."""
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        state.context_doc_ids.append("doc1")
        state.context_doc_ids.append("doc2")
        assert state.context_doc_ids == ["doc1", "doc2"]

    def test_context_doc_ids_initial_set(self):
        """Test setting context_doc_ids initially."""
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            context_doc_ids=["initial1", "initial2"],
        )
        assert state.context_doc_ids == ["initial1", "initial2"]


class TestPrepareStepContext:
    """Tests for prepare_step_context function."""

    def test_no_context_dsl_uses_accumulated(self):
        """Test step without context DSL uses accumulated context."""
        step = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
        )
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            context_doc_ids=["prior1", "prior2", "prior3"],
        )
        cvm = MagicMock()

        doc_ids, is_initial = prepare_step_context(step, state, cvm)

        assert doc_ids == ["prior1", "prior2", "prior3"]
        assert is_initial is False

    def test_no_context_dsl_empty_accumulated(self):
        """Test step without context DSL and no accumulated context."""
        step = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
        )
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        cvm = MagicMock()

        doc_ids, is_initial = prepare_step_context(step, state, cvm)

        assert doc_ids == []
        assert is_initial is False

    def test_with_context_dsl_returns_initial(self):
        """Test step with context DSL returns is_initial=True."""
        step = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
            context=[
                ContextAction(action="load_conversation", document_types=["conversation"]),
            ],
        )
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test_conv",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Mock CVM
        cvm = MagicMock()
        history_df = pd.DataFrame({
            'doc_id': ['doc1', 'doc2', 'doc3'],
        })
        cvm.get_conversation_history.return_value = history_df

        doc_ids, is_initial = prepare_step_context(step, state, cvm)

        assert doc_ids == ['doc1', 'doc2', 'doc3']
        assert is_initial is True

    def test_deduplicates_doc_ids(self):
        """Test that prepare_step_context deduplicates doc_ids."""
        step = StepDefinition(
            id="test_step",
            prompt="Test prompt",
            output=StepOutput(document_type="test"),
            context=[
                ContextAction(action="load_conversation", document_types=["conversation"]),
                ContextAction(action="load_conversation", document_types=["summary"]),
            ],
        )
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="test_conv",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        # Mock CVM to return overlapping doc_ids
        cvm = MagicMock()
        cvm.get_conversation_history.side_effect = [
            pd.DataFrame({'doc_id': ['doc1', 'doc2']}),
            pd.DataFrame({'doc_id': ['doc2', 'doc3']}),
        ]

        doc_ids, is_initial = prepare_step_context(step, state, cvm)

        # Should deduplicate, preserving order
        assert doc_ids == ['doc1', 'doc2', 'doc3']


class TestLoadConversation:
    """Tests for _load_conversation helper."""

    def test_load_conversation_current_target(self):
        """Test load_conversation with target='current'."""
        action = ContextAction(
            action="load_conversation",
            target="current",
            document_types=["conversation"],
        )
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="my_conversation",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        cvm = MagicMock()
        cvm.get_conversation_history.return_value = pd.DataFrame({
            'doc_id': ['doc1', 'doc2'],
        })

        doc_ids = _load_conversation(action, state, cvm)

        cvm.get_conversation_history.assert_called_once_with(
            "my_conversation",
            query_document_type=["conversation"],
            filter_document_type=None,
        )
        assert doc_ids == ['doc1', 'doc2']

    def test_load_conversation_exclude_types(self):
        """Test load_conversation with exclude_types."""
        action = ContextAction(
            action="load_conversation",
            exclude_types=["ner", "step"],
        )
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="my_conversation",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        cvm = MagicMock()
        cvm.get_conversation_history.return_value = pd.DataFrame({
            'doc_id': ['doc1', 'doc2'],
        })

        doc_ids = _load_conversation(action, state, cvm)

        cvm.get_conversation_history.assert_called_once_with(
            "my_conversation",
            query_document_type=None,
            filter_document_type=["ner", "step"],
        )

    def test_load_conversation_empty_result(self):
        """Test load_conversation with empty result."""
        action = ContextAction(action="load_conversation")
        state = PipelineState(
            pipeline_id="test",
            scenario_name="test",
            conversation_id="my_conversation",
            persona_id="test",
            user_id="test",
            model="test",
            branch=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        cvm = MagicMock()
        cvm.get_conversation_history.return_value = pd.DataFrame()

        doc_ids = _load_conversation(action, state, cvm)

        assert doc_ids == []


class TestSortDocIds:
    """Tests for _sort_doc_ids helper."""

    def test_sort_by_timestamp_ascending(self):
        """Test sorting by timestamp ascending (oldest first)."""
        action = ContextAction(
            action="sort",
            by="timestamp",
            direction="ascending",
        )
        doc_ids = ["doc1", "doc2", "doc3"]

        cvm = MagicMock()
        cvm.get_by_doc_id.side_effect = [
            {"timestamp": 300},  # doc1 - newest
            {"timestamp": 100},  # doc2 - oldest
            {"timestamp": 200},  # doc3 - middle
        ]

        sorted_ids = _sort_doc_ids(action, doc_ids, cvm)

        assert sorted_ids == ["doc2", "doc3", "doc1"]

    def test_sort_by_timestamp_descending(self):
        """Test sorting by timestamp descending (newest first)."""
        action = ContextAction(
            action="sort",
            by="timestamp",
            direction="descending",
        )
        doc_ids = ["doc1", "doc2", "doc3"]

        cvm = MagicMock()
        cvm.get_by_doc_id.side_effect = [
            {"timestamp": 300},  # doc1 - newest
            {"timestamp": 100},  # doc2 - oldest
            {"timestamp": 200},  # doc3 - middle
        ]

        sorted_ids = _sort_doc_ids(action, doc_ids, cvm)

        assert sorted_ids == ["doc1", "doc3", "doc2"]

    def test_sort_empty_list(self):
        """Test sorting empty list."""
        action = ContextAction(action="sort", by="timestamp", direction="ascending")
        cvm = MagicMock()

        sorted_ids = _sort_doc_ids(action, [], cvm)

        assert sorted_ids == []
