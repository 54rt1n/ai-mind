# packages/aim-mud/tests/mud_tests/unit/worker/test_conversation_report.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for conversation report caching in MUDAgentWorker."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone

from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from aim.config import ChatConfig
from aim_mud_types import RedisKeys


@pytest.fixture
def chat_config():
    """Create a test ChatConfig."""
    config = ChatConfig()
    config.llm_provider = "anthropic"
    config.model_name = "claude-opus-4-5-20251101"
    config.anthropic_api_key = "test-key"
    return config


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def mock_cvm():
    """Create a mock ConversationModel with sample report."""
    cvm = Mock()
    cvm.insert = Mock()
    # Sample conversation report data
    cvm.get_conversation_report = Mock(return_value={
        "conv1": {
            "conversation": 10,
            "analysis": 1,
            "timestamp_max": 1704412800
        },
        "conv2": {
            "conversation": 5,
            "timestamp_max": 1704412900
        }
    })
    return cvm


@pytest.mark.asyncio
async def test_update_conversation_report_success(mud_config, mock_redis, mock_cvm):
    """Test that _update_conversation_report() successfully caches report."""
    worker = MUDAgentWorker(mud_config, mock_redis)
    worker.cvm = mock_cvm

    await worker._update_conversation_report()

    # Verify report was fetched from CVM
    mock_cvm.get_conversation_report.assert_called_once()

    # Verify report was stored in Redis
    expected_key = RedisKeys.agent_conversation_report(mud_config.agent_id)
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == expected_key

    # Verify JSON serialization
    stored_data = json.loads(call_args[1])
    assert "conv1" in stored_data
    assert stored_data["conv1"]["conversation"] == 10
    assert "conv2" in stored_data


@pytest.mark.asyncio
async def test_update_conversation_report_empty(mud_config, mock_redis, mock_cvm):
    """Test _update_conversation_report() with empty report."""
    mock_cvm.get_conversation_report = Mock(return_value={})
    worker = MUDAgentWorker(mud_config, mock_redis)
    worker.cvm = mock_cvm

    await worker._update_conversation_report()

    # Should still call Redis set even with empty dict
    expected_key = RedisKeys.agent_conversation_report(mud_config.agent_id)
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == expected_key
    assert json.loads(call_args[1]) == {}


@pytest.mark.asyncio
async def test_update_conversation_report_error(mud_config, mock_redis, mock_cvm, caplog):
    """Test _update_conversation_report() handles errors gracefully."""
    mock_cvm.get_conversation_report = Mock(side_effect=Exception("Database error"))
    worker = MUDAgentWorker(mud_config, mock_redis)
    worker.cvm = mock_cvm

    # Should not raise exception
    await worker._update_conversation_report()

    # Should log error
    assert "Failed to update conversation report" in caplog.text
    assert "Database error" in caplog.text

    # Should not call Redis set
    mock_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_conversation_report_updates_called_correctly(mud_config, mock_redis, mock_cvm):
    """Test that _update_conversation_report can be called multiple times."""
    worker = MUDAgentWorker(mud_config, mock_redis)
    worker.cvm = mock_cvm

    # Call multiple times to simulate startup, flush, and dream
    await worker._update_conversation_report()
    await worker._update_conversation_report()
    await worker._update_conversation_report()

    # Verify report was fetched from CVM three times
    assert mock_cvm.get_conversation_report.call_count == 3

    # Verify report was stored in Redis three times
    expected_key = RedisKeys.agent_conversation_report(mud_config.agent_id)
    assert mock_redis.set.call_count == 3

    # Verify all calls used the same key
    for call in mock_redis.set.call_args_list:
        assert call[0][0] == expected_key
