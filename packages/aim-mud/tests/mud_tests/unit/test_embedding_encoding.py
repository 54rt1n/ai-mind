# packages/aim-mud/tests/mud_tests/unit/test_embedding_encoding.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for MUDConversationEntry embedding encoding/decoding.

Tests the base64 roundtrip for float32 embeddings used in MUDLOGIC V2.
"""

import numpy as np
import pytest

from aim_mud_types.models.conversation import MUDConversationEntry


class TestEmbeddingRoundtrip:
    """Tests for embedding encode/decode roundtrip."""

    def test_roundtrip_384_dimensions(self):
        """Test roundtrip with 384-dimension embeddings (MiniLM)."""
        original = np.random.randn(384).astype(np.float32)
        encoded = MUDConversationEntry.encode_embedding(original)

        entry = MUDConversationEntry(
            role="user",
            content="test",
            tokens=10,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
            embedding=encoded,
        )

        decoded = entry.get_embedding_vector()
        assert decoded is not None
        np.testing.assert_array_equal(decoded, original)

    def test_roundtrip_768_dimensions(self):
        """Test roundtrip with 768-dimension embeddings (BERT base)."""
        original = np.random.randn(768).astype(np.float32)
        encoded = MUDConversationEntry.encode_embedding(original)

        entry = MUDConversationEntry(
            role="assistant",
            content="test response",
            tokens=20,
            document_type="DOC_MUD_ACTION",
            conversation_id="test-conv",
            sequence_no=2,
            embedding=encoded,
        )

        decoded = entry.get_embedding_vector()
        assert decoded is not None
        np.testing.assert_array_equal(decoded, original)

    def test_roundtrip_1024_dimensions(self):
        """Test roundtrip with 1024-dimension embeddings (larger models)."""
        original = np.random.randn(1024).astype(np.float32)
        encoded = MUDConversationEntry.encode_embedding(original)

        entry = MUDConversationEntry(
            role="user",
            content="another test",
            tokens=15,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=3,
            embedding=encoded,
        )

        decoded = entry.get_embedding_vector()
        assert decoded is not None
        np.testing.assert_array_equal(decoded, original)

    def test_roundtrip_preserves_float32_precision(self):
        """Test that float32 precision is preserved exactly."""
        # Use specific values that could lose precision in other formats
        original = np.array([
            1.0000001,
            -0.9999999,
            3.1415927,  # pi in float32
            1e-7,
            1e7,
        ], dtype=np.float32)

        encoded = MUDConversationEntry.encode_embedding(original)
        entry = MUDConversationEntry(
            role="user",
            content="precision test",
            tokens=5,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
            embedding=encoded,
        )

        decoded = entry.get_embedding_vector()
        assert decoded is not None
        np.testing.assert_array_equal(decoded, original)


class TestEmbeddingNoneHandling:
    """Tests for None embedding handling."""

    def test_get_embedding_vector_returns_none_when_no_embedding(self):
        """Test that get_embedding_vector returns None when embedding is None."""
        entry = MUDConversationEntry(
            role="user",
            content="test",
            tokens=10,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
            embedding=None,
        )

        assert entry.get_embedding_vector() is None

    def test_get_embedding_vector_returns_none_when_embedding_empty_string(self):
        """Test that get_embedding_vector returns None for empty string."""
        entry = MUDConversationEntry(
            role="user",
            content="test",
            tokens=10,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
            embedding="",
        )

        assert entry.get_embedding_vector() is None

    def test_default_embedding_is_none(self):
        """Test that embedding field defaults to None."""
        entry = MUDConversationEntry(
            role="user",
            content="test",
            tokens=10,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
        )

        assert entry.embedding is None
        assert entry.get_embedding_vector() is None


class TestEmbeddingDeterminism:
    """Tests that encoding is deterministic."""

    def test_encoding_is_deterministic(self):
        """Test that same input produces same output."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        encoded1 = MUDConversationEntry.encode_embedding(original)
        encoded2 = MUDConversationEntry.encode_embedding(original)
        encoded3 = MUDConversationEntry.encode_embedding(original)

        assert encoded1 == encoded2
        assert encoded2 == encoded3

    def test_encoding_deterministic_with_copy(self):
        """Test that encoding a copy produces same result."""
        original = np.array([1.5, -2.5, 3.5], dtype=np.float32)
        copy = original.copy()

        encoded_original = MUDConversationEntry.encode_embedding(original)
        encoded_copy = MUDConversationEntry.encode_embedding(copy)

        assert encoded_original == encoded_copy


class TestEmbeddingTypeCoercion:
    """Tests for type coercion in encoding."""

    def test_float64_coerced_to_float32(self):
        """Test that float64 arrays are properly coerced to float32."""
        original_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        expected_f32 = original_f64.astype(np.float32)

        encoded = MUDConversationEntry.encode_embedding(original_f64)
        entry = MUDConversationEntry(
            role="user",
            content="test",
            tokens=5,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
            embedding=encoded,
        )

        decoded = entry.get_embedding_vector()
        assert decoded is not None
        assert decoded.dtype == np.float32
        np.testing.assert_array_equal(decoded, expected_f32)

    def test_int_array_coerced_to_float32(self):
        """Test that integer arrays are properly coerced to float32."""
        original_int = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        expected_f32 = original_int.astype(np.float32)

        encoded = MUDConversationEntry.encode_embedding(original_int)
        entry = MUDConversationEntry(
            role="user",
            content="test",
            tokens=5,
            document_type="DOC_MUD_WORLD",
            conversation_id="test-conv",
            sequence_no=1,
            embedding=encoded,
        )

        decoded = entry.get_embedding_vector()
        assert decoded is not None
        assert decoded.dtype == np.float32
        np.testing.assert_array_equal(decoded, expected_f32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
