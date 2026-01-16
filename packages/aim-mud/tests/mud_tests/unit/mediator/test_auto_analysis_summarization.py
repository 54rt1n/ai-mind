# packages/aim-mud/tests/unit/mediator/test_auto_analysis_summarization.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for auto-analysis context size checking and summarization.

Tests the context size checking logic that determines whether a conversation
needs summarization before analysis.
"""

import json
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from aim_mud_types.profile import AgentProfile
from aim_mud_types.helper import _utc_now


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration with auto-analysis enabled."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
        auto_analysis_enabled=True,
        auto_analysis_idle_seconds=60,
        auto_analysis_cooldown_seconds=10,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.expire = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)
    redis.incr = AsyncMock(side_effect=lambda key: 1)
    redis.set = AsyncMock(return_value=True)
    return redis


class TestContextSizeChecking:
    """Test context size checking before analysis."""

    @pytest.mark.asyncio
    async def test_should_summarize_large_conversation(self, mock_redis, mediator_config):
        """Test that large conversations trigger summarization."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock agent profile
        async def mock_get_agent_profile(agent_id):
            return AgentProfile(agent_id="andi", persona_id="andi")

        # Mock CVM with large conversation (50000 tokens)
        mock_cvm = MagicMock()
        mock_cvm.query = MagicMock(return_value=[
            {"content": "x" * 200000}  # Large document
        ])

        # Mock model with 32k context window
        mock_model = MagicMock()
        mock_model.max_tokens = 32768

        # Mock token counting to return high count
        def mock_count_tokens(text):
            return 50000  # Simulated: 50k tokens

        with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                   side_effect=mock_get_agent_profile):
            with patch('aim.conversation.model.ConversationModel', return_value=mock_cvm):
                with patch.dict('sys.modules', {'aim.llm.models': MagicMock(
                    LanguageModelV2=MagicMock(index_models=lambda config: {
                        mediator.chat_config.default_model: mock_model
                    })
                )}):
                    with patch('aim.utils.tokens.count_tokens', side_effect=mock_count_tokens):
                        # Conversation with no summary
                        doc_counts = {"mud-world": 5, "mud-agent": 5, "summary": 0}

                        should_summarize, reason = await mediator._should_summarize_before_analysis(
                            agent_id="andi",
                            conversation_id="conv_001",
                            doc_counts=doc_counts
                        )

                        assert should_summarize is True
                        assert reason == "over_threshold"

    @pytest.mark.asyncio
    async def test_should_not_summarize_small_conversation(self, mock_redis, mediator_config):
        """Test that small conversations go directly to analysis."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock agent profile
        async def mock_get_agent_profile(agent_id):
            return AgentProfile(agent_id="andi", persona_id="andi")

        # Mock CVM with small conversation
        mock_cvm = MagicMock()
        mock_cvm.query = MagicMock(return_value=[
            {"content": "x" * 1000} for _ in range(2)  # 2 docs × 1k chars = 2k chars
        ])

        # Mock model with 32k context window
        mock_model = MagicMock()
        mock_model.max_tokens = 32768

        with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                   side_effect=mock_get_agent_profile):
            with patch('aim.conversation.model.ConversationModel', return_value=mock_cvm):
                with patch('aim.llm.models.LanguageModelV2.index_models',
                           return_value={"default": mock_model}):
                    # Conversation with no summary
                    doc_counts = {"mud-world": 1, "mud-agent": 1, "summary": 0}

                    should_summarize, reason = await mediator._should_summarize_before_analysis(
                        agent_id="andi",
                        conversation_id="conv_001",
                        doc_counts=doc_counts
                    )

                    assert should_summarize is False
                    assert reason == "under_threshold"

    @pytest.mark.asyncio
    async def test_skip_check_if_has_summary(self, mock_redis, mediator_config):
        """Test that conversations with summary docs skip context check."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Conversation with summary - should not even check CVM
        doc_counts = {"mud-world": 5, "mud-agent": 5, "summary": 3, "analysis": 0}

        should_summarize, reason = await mediator._should_summarize_before_analysis(
            agent_id="andi",
            conversation_id="conv_001",
            doc_counts=doc_counts
        )

        assert should_summarize is False
        assert reason == "has_summary"

    @pytest.mark.asyncio
    async def test_handle_missing_profile(self, mock_redis, mediator_config):
        """Test handling of missing agent profile."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock missing profile
        async def mock_get_agent_profile(agent_id):
            return None

        with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                   side_effect=mock_get_agent_profile):
            doc_counts = {"mud-world": 5, "mud-agent": 5}

            should_summarize, reason = await mediator._should_summarize_before_analysis(
                agent_id="andi",
                conversation_id="conv_001",
                doc_counts=doc_counts
            )

            # Should fail open
            assert should_summarize is False
            assert reason == "no_profile"

    @pytest.mark.asyncio
    async def test_handle_cvm_error(self, mock_redis, mediator_config):
        """Test handling of CVM query errors."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Mock agent profile
        async def mock_get_agent_profile(agent_id):
            return AgentProfile(agent_id="andi", persona_id="andi")

        # Mock CVM that raises error
        def mock_cvm_init(*args, **kwargs):
            raise Exception("CVM connection failed")

        with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                   side_effect=mock_get_agent_profile):
            with patch('aim.conversation.model.ConversationModel', side_effect=mock_cvm_init):
                doc_counts = {"mud-world": 5, "mud-agent": 5}

                should_summarize, reason = await mediator._should_summarize_before_analysis(
                    agent_id="andi",
                    conversation_id="conv_001",
                    doc_counts=doc_counts
                )

                # Should fail open
                assert should_summarize is False
                assert reason == "check_failed"


class TestScanLogicWithContextCheck:
    """Test scan logic integration with context checking."""

    @pytest.mark.asyncio
    async def test_scan_finds_summarized_conversations(self, mock_redis, mediator_config):
        """Test that scan finds conversations with summary but no analysis."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Create conversation report with summarized but not analyzed
        sample_report = {
            "conv_001": {
                "mud-world": 5,
                "mud-agent": 3,
                "summary": 2,  # Has summary
                "analysis": 0,  # But no analysis
                "timestamp_max": "2026-01-01T10:00:00+00:00",
            },
            "conv_002": {
                "mud-world": 3,
                "mud-agent": 2,
                "summary": 0,  # No summary
                "analysis": 1,  # Already analyzed
                "timestamp_max": "2026-01-02T10:00:00+00:00",
            },
        }

        # Mock Redis and profile
        async def mock_get_conversation_report(agent_id):
            return sample_report

        async def mock_get_agent_profile(agent_id):
            return AgentProfile(agent_id="andi", persona_id="andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"completed_at": (_utc_now() - timedelta(seconds=400)).isoformat().encode(),
        })

        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        with patch('aim_mud_types.client.RedisMUDClient.get_conversation_report',
                   side_effect=mock_get_conversation_report):
            with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                       side_effect=mock_get_agent_profile):
                with patch.object(mediator, '_refresh_conversation_reports'):
                    with patch.object(mediator, '_handle_analysis_command',
                                      return_value=True) as mock_handle:
                        with patch.object(mediator, '_is_paused', return_value=False):
                            await mediator._scan_for_unanalyzed_conversations()

                            # Should find and trigger for conv_001 (has summary)
                            mock_handle.assert_called_once()
                            call_args = mock_handle.call_args
                            assert call_args[1]["conversation_id"] == "conv_001"

    @pytest.mark.asyncio
    async def test_scan_triggers_summarizer_for_large_context(self, mock_redis, mediator_config):
        """Test that scan triggers summarizer for large conversations."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Create conversation report with large unanalyzed conversation
        sample_report = {
            "conv_001": {
                "mud-world": 50,
                "mud-agent": 30,
                "summary": 0,  # No summary yet
                "analysis": 0,  # Not analyzed
                "timestamp_max": "2026-01-01T10:00:00+00:00",
            },
        }

        # Mock Redis and profile
        async def mock_get_conversation_report(agent_id):
            return sample_report

        async def mock_get_agent_profile(agent_id):
            return AgentProfile(agent_id="andi", persona_id="andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"completed_at": (_utc_now() - timedelta(seconds=400)).isoformat().encode(),
        })

        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        # Mock CVM with large conversation
        mock_cvm = MagicMock()
        mock_cvm.query = MagicMock(return_value=[
            {"content": "x" * 200000}
        ])

        # Mock model
        mock_model = MagicMock()
        mock_model.max_tokens = 32768

        # Mock token counting to return high count
        def mock_count_tokens(text):
            return 50000  # Simulated: 50k tokens

        with patch('aim_mud_types.client.RedisMUDClient.get_conversation_report',
                   side_effect=mock_get_conversation_report):
            with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                       side_effect=mock_get_agent_profile):
                with patch.object(mediator, '_refresh_conversation_reports'):
                    with patch('aim.conversation.model.ConversationModel', return_value=mock_cvm):
                        with patch.dict('sys.modules', {'aim.llm.models': MagicMock(
                            LanguageModelV2=MagicMock(index_models=lambda config: {
                                mediator.chat_config.default_model: mock_model
                            })
                        )}):
                            with patch('aim.utils.tokens.count_tokens', side_effect=mock_count_tokens):
                                with patch.object(mediator, '_handle_analysis_command',
                                                  return_value=True) as mock_handle:
                                    with patch.object(mediator, '_is_paused', return_value=False):
                                        await mediator._scan_for_unanalyzed_conversations()

                                        # Should trigger summarizer
                                        mock_handle.assert_called_once()
                                        call_args = mock_handle.call_args
                                        assert call_args[1]["scenario"] == "summarizer"

    @pytest.mark.asyncio
    async def test_scan_triggers_analysis_for_small_context(self, mock_redis, mediator_config):
        """Test that scan triggers analysis directly for small conversations."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")

        # Create conversation report with small unanalyzed conversation
        sample_report = {
            "conv_001": {
                "mud-world": 2,
                "mud-agent": 1,
                "summary": 0,  # No summary
                "analysis": 0,  # Not analyzed
                "timestamp_max": "2026-01-01T10:00:00+00:00",
            },
        }

        # Mock Redis and profile
        async def mock_get_conversation_report(agent_id):
            return sample_report

        async def mock_get_agent_profile(agent_id):
            return AgentProfile(agent_id="andi", persona_id="andi")

        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
            b"sequence_id": b"1",
            b"completed_at": (_utc_now() - timedelta(seconds=400)).isoformat().encode(),
        })

        mediator._last_auto_analysis_check = _utc_now() - timedelta(seconds=120)

        # Mock CVM with small conversation
        mock_cvm = MagicMock()
        mock_cvm.query = MagicMock(return_value=[
            {"content": "x" * 500},
            {"content": "x" * 500},
        ])

        # Mock model
        mock_model = MagicMock()
        mock_model.max_tokens = 32768

        with patch('aim_mud_types.client.RedisMUDClient.get_conversation_report',
                   side_effect=mock_get_conversation_report):
            with patch('aim_mud_types.client.RedisMUDClient.get_agent_profile',
                       side_effect=mock_get_agent_profile):
                with patch.object(mediator, '_refresh_conversation_reports'):
                    with patch('aim.conversation.model.ConversationModel', return_value=mock_cvm):
                        with patch('aim.llm.models.LanguageModelV2.index_models',
                                   return_value={"default": mock_model}):
                            with patch.object(mediator, '_handle_analysis_command',
                                              return_value=True) as mock_handle:
                                with patch.object(mediator, '_is_paused', return_value=False):
                                    await mediator._scan_for_unanalyzed_conversations()

                                    # Should trigger analysis
                                    mock_handle.assert_called_once()
                                    call_args = mock_handle.call_args
                                    assert call_args[1]["scenario"] == "analysis_dialogue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
