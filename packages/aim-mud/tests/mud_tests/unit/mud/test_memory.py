# tests/unit/mud/test_memory.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUD memory bucket and persistence."""

from datetime import datetime, timezone, timedelta
import re
from unittest.mock import MagicMock

import pandas as pd

from andimud_worker.memory import (
    MUDMemoryBucket,
    MUDMemoryPersister,
    MUDMemoryRetriever,
    generate_conversation_id,
)
from andimud_worker.session import MUDEvent, EventType
from aim.constants import DOC_MUD_WORLD, DOC_MUD_AGENT


def _sample_event(event_id: str = "1") -> MUDEvent:
    return MUDEvent(
        event_id=event_id,
        event_type=EventType.SPEECH,
        actor="Prax",
        room_id="#123",
        room_name="The Garden",
        content="Hello there",
        timestamp=datetime.now(timezone.utc),
    )


class TestMUDMemoryBucket:
    def test_bucket_flush_token_threshold(self):
        bucket = MUDMemoryBucket()
        bucket.append_events([_sample_event("1")])
        bucket.append_response("x" * 500, actions=[])

        # Force a low token threshold to trigger flush
        assert bucket.should_flush(max_tokens=1, idle_flush_seconds=600) is True

    def test_bucket_flush_idle_threshold(self):
        bucket = MUDMemoryBucket()
        old_time = datetime.now(timezone.utc) - timedelta(minutes=11)
        event = _sample_event("2")
        event.timestamp = old_time
        bucket.append_events([event])

        assert bucket.should_flush(max_tokens=100000, idle_flush_seconds=600) is True


class TestMUDMemoryPersister:
    def test_persist_bucket_creates_two_docs(self):
        bucket = MUDMemoryBucket()
        bucket.append_events([_sample_event("3")])
        bucket.append_response("Andi says hello", actions=[])

        mock_cvm = MagicMock()
        persister = MUDMemoryPersister()

        conversation_id = persister.persist_bucket(
            bucket=bucket,
            cvm=mock_cvm,
            persona_id="Andi",
            user_id="mud",
            inference_model="test-model",
        )

        assert conversation_id is not None
        assert mock_cvm.insert.call_count == 2

        world_msg = mock_cvm.insert.call_args_list[0][0][0]
        agent_msg = mock_cvm.insert.call_args_list[1][0][0]

        assert world_msg.document_type == DOC_MUD_WORLD
        assert agent_msg.document_type == DOC_MUD_AGENT
        assert world_msg.conversation_id == conversation_id
        assert agent_msg.conversation_id == conversation_id

    def test_conversation_id_format(self):
        conv_id = generate_conversation_id("andimud")
        assert conv_id.startswith("andimud_")
        parts = conv_id.split("_")
        assert len(parts) == 3
        assert re.fullmatch(r"\d+", parts[1])
        assert re.fullmatch(r"[a-f0-9]{9}", parts[2])


class TestMUDMemoryRetriever:
    """Tests for MUDMemoryRetriever (deprecated class kept for compatibility)."""

    def test_retriever_emits_deprecation_warning(self):
        """Test that MUDMemoryRetriever emits a deprecation warning on init."""
        import warnings

        mock_cvm = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MUDMemoryRetriever(mock_cvm, top_n=2)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "MUDMemoryRetriever is deprecated" in str(w[0].message)

    def test_retriever_build_memory_block(self):
        """Test that build_memory_block still works (deprecated but functional)."""
        import warnings

        mock_cvm = MagicMock()
        df = pd.DataFrame([
            {
                "content": "Remember the fountain",
                "document_type": DOC_MUD_WORLD,
                "date": "2026-01-01",
            }
        ])
        mock_cvm.query.return_value = df

        session = MagicMock()
        session.current_room = MagicMock(name="Room")
        session.current_room.name = "The Garden"
        session.current_room.description = "A serene place"
        session.entities_present = []
        session.pending_events = []

        # Suppress deprecation warning for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            retriever = MUDMemoryRetriever(mock_cvm, top_n=2)

        block, token_count = retriever.build_memory_block(session=session, token_budget=500)

        assert "MUDMemory" in block
        assert "Remember the fountain" in block
        assert token_count > 0
