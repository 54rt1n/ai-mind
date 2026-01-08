# packages/aim-mud/tests/mud_tests/unit/worker/test_conversation_report.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for conversation report caching in MUDAgentWorker."""

import pytest
import json
from unittest.mock import Mock
from datetime import datetime, timezone

from andimud_worker.worker import MUDAgentWorker
from aim_mud_types import RedisKeys


@pytest.mark.asyncio
async def test_update_conversation_report_success(test_worker, test_cvm, mock_redis, monkeypatch):
    """Test that _update_conversation_report() successfully caches report."""
    import pandas as pd

    # Use real worker and CVM, but monkeypatch get_conversation_report to return test data
    test_worker.cvm = test_cvm

    # Create sample report data
    report_data = {
        'conversation_id': ['conv1', 'conv2'],
        'conversation': [10, 5],
        'analysis': [1, None],
        'timestamp_max': [1704412800, 1704412900]
    }
    report_df = pd.DataFrame(report_data)

    # Monkeypatch the report method to return our test data
    monkeypatch.setattr(test_cvm, 'get_conversation_report', Mock(return_value=report_df))

    await test_worker._update_conversation_report()

    # Verify report was fetched from CVM
    test_cvm.get_conversation_report.assert_called_once()

    # Verify report was stored in Redis
    expected_key = RedisKeys.agent_conversation_report(test_worker.config.agent_id)
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == expected_key

    # Verify JSON serialization
    stored_data = json.loads(call_args[1])
    assert "conv1" in stored_data
    assert stored_data["conv1"]["conversation"] == 10
    assert "conv2" in stored_data


@pytest.mark.asyncio
async def test_update_conversation_report_empty(test_worker, test_cvm, mock_redis, monkeypatch):
    """Test _update_conversation_report() with empty report."""
    import pandas as pd

    test_worker.cvm = test_cvm

    empty_df = pd.DataFrame()
    monkeypatch.setattr(test_cvm, 'get_conversation_report', Mock(return_value=empty_df))

    await test_worker._update_conversation_report()

    expected_key = RedisKeys.agent_conversation_report(test_worker.config.agent_id)
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args[0]
    assert call_args[0] == expected_key
    assert json.loads(call_args[1]) == {}


@pytest.mark.asyncio
async def test_update_conversation_report_error(test_worker, test_cvm, mock_redis, monkeypatch, caplog):
    """Test _update_conversation_report() handles errors gracefully."""
    test_worker.cvm = test_cvm

    monkeypatch.setattr(test_cvm, 'get_conversation_report', Mock(side_effect=Exception("Database error")))

    # Should not raise exception
    await test_worker._update_conversation_report()

    # Should log error
    assert "Failed to update conversation report" in caplog.text
    assert "Database error" in caplog.text

    # Should not call Redis set
    mock_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_conversation_report_updates_called_correctly(test_worker, test_cvm, mock_redis, monkeypatch):
    """Test that _update_conversation_report can be called multiple times."""
    import pandas as pd

    test_worker.cvm = test_cvm

    report_data = {
        'conversation_id': ['conv1', 'conv2'],
        'conversation': [10, 5],
        'analysis': [1, None],
        'timestamp_max': [1704412800, 1704412900]
    }
    report_df = pd.DataFrame(report_data)
    monkeypatch.setattr(test_cvm, 'get_conversation_report', Mock(return_value=report_df))

    # Call multiple times to simulate startup, flush, and dream
    await test_worker._update_conversation_report()
    await test_worker._update_conversation_report()
    await test_worker._update_conversation_report()

    # Verify report was fetched from CVM three times
    assert test_cvm.get_conversation_report.call_count == 3

    # Verify report was stored in Redis three times
    expected_key = RedisKeys.agent_conversation_report(test_worker.config.agent_id)
    assert mock_redis.set.call_count == 3

    # Verify all calls used the same key
    for call in mock_redis.set.call_args_list:
        assert call[0][0] == expected_key
