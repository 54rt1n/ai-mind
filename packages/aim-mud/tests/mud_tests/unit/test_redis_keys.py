# packages/aim-mud/tests/mud_tests/unit/test_redis_keys.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for RedisKeys utility class."""

import pytest

from aim_mud_types import RedisKeys


class TestRedisKeysAgentThought:
    """Tests for RedisKeys.agent_thought() method."""

    def test_agent_thought_key_format(self):
        """Test agent_thought returns correct key format."""
        key = RedisKeys.agent_thought("andi")
        assert key == "agent:andi:thought"

    def test_agent_thought_different_agent_ids(self):
        """Test agent_thought with various agent IDs."""
        assert RedisKeys.agent_thought("test_agent") == "agent:test_agent:thought"
        assert RedisKeys.agent_thought("nova") == "agent:nova:thought"
        assert RedisKeys.agent_thought("tiberius") == "agent:tiberius:thought"

    def test_agent_thought_unique_per_agent(self):
        """Test that different agents get different keys."""
        key1 = RedisKeys.agent_thought("agent1")
        key2 = RedisKeys.agent_thought("agent2")
        assert key1 != key2
        assert key1 == "agent:agent1:thought"
        assert key2 == "agent:agent2:thought"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
