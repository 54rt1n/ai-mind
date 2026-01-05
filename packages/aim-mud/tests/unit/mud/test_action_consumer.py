# tests/unit/mud/test_action_consumer.py
"""Tests for Evennia action consumer retry logic and stream trimming.

These tests verify the retry decorator and stream trimming functionality
added to the action_consumer to handle Redis connection failures gracefully.
"""

import pytest
import time
from unittest.mock import MagicMock, patch, call
import redis


# Import the retry decorator and ActionConsumer from the Evennia codebase
# Note: This assumes the andimud directory structure is accessible
import sys
from pathlib import Path

# Add andimud to path for imports
andimud_path = Path(__file__).parent.parent.parent.parent.parent / "andimud" / "andimud"
if andimud_path.exists():
    sys.path.insert(0, str(andimud_path.parent))

# Try to import action_consumer - skip all tests if not available or Django not configured
EVENNIA_AVAILABLE = False
retry_redis = None
ActionConsumer = None

try:
    from andimud.server.services.action_consumer import retry_redis, ActionConsumer
    EVENNIA_AVAILABLE = True
except Exception:
    # Skip all tests if Evennia is not available or Django is not configured
    pass

if not EVENNIA_AVAILABLE:
    pytestmark = pytest.mark.skip(reason="Evennia action_consumer not available or Django not configured")


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestRetryRedisDecorator:
    """Tests for the retry_redis decorator with exponential backoff."""

    def test_retry_decorator_succeeds_on_first_attempt(self):
        """Test that decorator returns immediately on success."""
        @retry_redis(max_retries=3, backoff_base=1.0)
        def mock_func():
            return "success"

        result = mock_func()
        assert result == "success"

    def test_retry_decorator_retries_on_redis_error(self):
        """Test that decorator retries on RedisError."""
        call_count = {"count": 0}

        @retry_redis(max_retries=3, backoff_base=0.1)
        def mock_func():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise redis.RedisError("Connection failed")
            return "success"

        with patch('time.sleep') as mock_sleep:
            result = mock_func()

        assert result == "success"
        assert call_count["count"] == 3
        # Should have slept twice (0.1s, then 0.2s)
        assert mock_sleep.call_count == 2

    def test_retry_decorator_raises_after_max_retries(self):
        """Test that decorator raises after exhausting retries."""
        @retry_redis(max_retries=3, backoff_base=0.1)
        def mock_func():
            raise redis.RedisError("Persistent failure")

        with patch('time.sleep'):
            with pytest.raises(redis.RedisError, match="Persistent failure"):
                mock_func()

    def test_retry_decorator_exponential_backoff(self):
        """Test that backoff doubles with each retry."""
        @retry_redis(max_retries=4, backoff_base=1.0)
        def mock_func():
            raise redis.RedisError("Always fails")

        with patch('time.sleep') as mock_sleep:
            with pytest.raises(redis.RedisError):
                mock_func()

        # Should have slept 3 times before final failure: 1s, 2s, 4s
        assert mock_sleep.call_count == 3
        calls = [call(1.0), call(2.0), call(4.0)]
        mock_sleep.assert_has_calls(calls)

    def test_retry_decorator_does_not_retry_non_redis_errors(self):
        """Test that decorator doesn't retry non-RedisError exceptions."""
        @retry_redis(max_retries=3, backoff_base=0.1)
        def mock_func():
            raise ValueError("Not a Redis error")

        with pytest.raises(ValueError, match="Not a Redis error"):
            mock_func()


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestActionConsumerRetryLogic:
    """Tests for ActionConsumer retry logic on poll_actions and mark_action_processed."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def consumer(self, mock_redis):
        """Create an ActionConsumer with mocked Redis."""
        with patch('andimud.server.services.action_consumer.redis.from_url', return_value=mock_redis):
            consumer = ActionConsumer(redis_url="redis://localhost:6379")
            consumer.redis = mock_redis
            return consumer

    def test_poll_actions_retries_on_redis_error(self, consumer, mock_redis):
        """Test that poll_actions retries on RedisError."""
        call_count = {"count": 0}

        def mock_xread(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise redis.RedisError("Connection lost")
            return []

        mock_redis.xread.side_effect = mock_xread

        with patch('time.sleep'):
            result = consumer.poll_actions()

        assert result == []
        assert call_count["count"] == 3

    def test_mark_action_processed_retries_on_redis_error(self, consumer, mock_redis):
        """Test that _mark_action_processed retries on RedisError."""
        call_count = {"count": 0}

        def mock_hset(*args, **kwargs):
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise redis.RedisError("Connection lost")
            return 1

        mock_redis.hset.side_effect = mock_hset

        with patch('time.sleep'):
            consumer._mark_action_processed("123-0", "andi", True)

        assert call_count["count"] == 3

    def test_poll_actions_raises_after_max_retries(self, consumer, mock_redis):
        """Test that poll_actions raises after exhausting retries."""
        mock_redis.xread.side_effect = redis.RedisError("Persistent failure")

        with patch('time.sleep'):
            with pytest.raises(redis.RedisError, match="Persistent failure"):
                consumer.poll_actions()

    def test_mark_action_processed_raises_after_max_retries(self, consumer, mock_redis):
        """Test that _mark_action_processed raises after exhausting retries."""
        mock_redis.hset.side_effect = redis.RedisError("Persistent failure")

        with patch('time.sleep'):
            with pytest.raises(redis.RedisError, match="Persistent failure"):
                consumer._mark_action_processed("123-0", "andi", True)


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestActionConsumerStreamTrimming:
    """Tests for ActionConsumer stream trimming logic."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def consumer(self, mock_redis):
        """Create an ActionConsumer with mocked Redis."""
        with patch('andimud.server.services.action_consumer.redis.from_url', return_value=mock_redis):
            consumer = ActionConsumer(redis_url="redis://localhost:6379")
            consumer.redis = mock_redis
            return consumer

    def test_trim_action_stream_keeps_last_1000(self, consumer, mock_redis):
        """Test that _trim_action_stream trims to maxlen=1000."""
        consumer._trim_action_stream()

        mock_redis.xtrim.assert_called_once()
        args, kwargs = mock_redis.xtrim.call_args
        assert kwargs['maxlen'] == 1000
        assert kwargs['approximate'] is True

    def test_trim_action_stream_handles_redis_error(self, consumer, mock_redis):
        """Test that _trim_action_stream handles RedisError gracefully."""
        mock_redis.xtrim.side_effect = redis.RedisError("Connection lost")

        # Should not raise - just log the error
        consumer._trim_action_stream()

    def test_run_trims_stream_every_100_iterations(self, consumer, mock_redis):
        """Test that run() calls _trim_action_stream every 100 iterations."""
        mock_redis.xread.return_value = []
        iteration_count = {"count": 0}

        def mock_run_once():
            iteration_count["count"] += 1
            if iteration_count["count"] >= 250:
                consumer.running = False
            return 0

        with patch.object(consumer, 'run_once', side_effect=mock_run_once):
            with patch.object(consumer, '_trim_action_stream') as mock_trim:
                with patch.object(consumer, '_cleanup_processed_hash'):
                    consumer.run()

        # Should have trimmed at iterations 100 and 200
        assert mock_trim.call_count == 2

    def test_run_cleans_hash_every_100_iterations(self, consumer, mock_redis):
        """Test that run() calls _cleanup_processed_hash every 100 iterations."""
        mock_redis.xread.return_value = []
        iteration_count = {"count": 0}

        def mock_run_once():
            iteration_count["count"] += 1
            if iteration_count["count"] >= 250:
                consumer.running = False
            return 0

        with patch.object(consumer, 'run_once', side_effect=mock_run_once):
            with patch.object(consumer, '_trim_action_stream'):
                with patch.object(consumer, '_cleanup_processed_hash') as mock_cleanup:
                    consumer.run()

        # Should have cleaned hash at iterations 100 and 200
        assert mock_cleanup.call_count == 2


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestActionConsumerIdempotency:
    """Tests for ActionConsumer idempotency via processed hash."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def consumer(self, mock_redis):
        """Create an ActionConsumer with mocked Redis."""
        with patch('andimud.server.services.action_consumer.redis.from_url', return_value=mock_redis):
            consumer = ActionConsumer(redis_url="redis://localhost:6379")
            consumer.redis = mock_redis
            return consumer

    def test_is_action_processed_checks_hash(self, consumer, mock_redis):
        """Test that _is_action_processed checks the hash."""
        mock_redis.hexists.return_value = True

        result = consumer._is_action_processed("123-0")

        assert result is True
        mock_redis.hexists.assert_called_once()

    def test_is_action_processed_handles_redis_error(self, consumer, mock_redis):
        """Test that _is_action_processed returns False on RedisError."""
        mock_redis.hexists.side_effect = redis.RedisError("Connection lost")

        result = consumer._is_action_processed("123-0")

        # Should return False (better to duplicate than to lose)
        assert result is False

    def test_mark_action_processed_writes_to_hash(self, consumer, mock_redis):
        """Test that _mark_action_processed writes to the hash."""
        with patch('time.sleep'):  # Prevent actual retry delays
            consumer._mark_action_processed("123-0", "andi", True)

        mock_redis.hset.assert_called_once()
        args, kwargs = mock_redis.hset.call_args
        # Should include timestamp, agent_id, and status
        assert "andi" in args[2]
        assert "success" in args[2]

    def test_run_once_skips_already_processed_actions(self, consumer, mock_redis):
        """Test that run_once skips actions already in the processed hash."""
        mock_redis.xread.return_value = [
            (b"mud:actions", [
                (b"123-0", {b"data": b'{"agent_id": "andi", "command": "say test"}'})
            ])
        ]
        mock_redis.hexists.return_value = True  # Action already processed

        with patch.object(consumer, 'execute_action') as mock_execute:
            result = consumer.run_once()

        # Should not execute action if already processed
        assert result == 0
        mock_execute.assert_not_called()


@pytest.mark.skipif(not EVENNIA_AVAILABLE, reason="Evennia not available")
class TestActionConsumerThreadSafety:
    """Tests for potential concurrency issues in ActionConsumer."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        return MagicMock(spec=redis.Redis)

    @pytest.fixture
    def consumer(self, mock_redis):
        """Create an ActionConsumer with mocked Redis."""
        with patch('andimud.server.services.action_consumer.redis.from_url', return_value=mock_redis):
            consumer = ActionConsumer(redis_url="redis://localhost:6379")
            consumer.redis = mock_redis
            return consumer

    def test_iteration_counter_is_local_variable(self, consumer, mock_redis):
        """Test that iteration counter is a local variable (not shared state)."""
        # The iteration counter in run() is a local variable, so it's thread-safe
        # This test verifies the implementation uses local state
        import inspect
        source = inspect.getsource(consumer.run)
        assert "iteration = 0" in source
        assert "iteration += 1" in source
        # Counter is local, not self.iteration, so no race condition

    def test_running_flag_is_safe_to_modify(self, consumer):
        """Test that the running flag can be safely modified."""
        # The running flag is simple boolean - safe for single writer (stop())
        consumer.running = True
        assert consumer.running is True

        consumer.stop()
        assert consumer.running is False

    def test_redis_client_is_created_once(self, consumer, mock_redis):
        """Test that Redis client is created once and reused."""
        # Calling get_redis multiple times should return same instance
        client1 = consumer.get_redis()
        client2 = consumer.get_redis()

        assert client1 is client2
        assert client1 is mock_redis
