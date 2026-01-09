# tests/unit/mediator/test_dream_commands.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Unit tests for dream command handlers in the mediator.

These commands allow manual triggering of dream pipelines:
- Analysis commands (require conversation_id):
  - @analyze <agent-id> = <conversation_id>[, <guidance>]
  - @summary <agent-id> = <conversation_id>
- Creative commands (optional query/guidance):
  - @journal <agent-id> [= <query>[, <guidance>]]
  - @ponder <agent-id> [= <query>[, <guidance>]]
  - @daydream <agent-id> [= <query>[, <guidance>]]
  - @critique <agent-id> [= <query>[, <guidance>]]
  - @research <agent-id> [= <query>[, <guidance>]]
- Control command:
  - @dreamer <agent-id> on/off
"""

import json
import pytest
from unittest.mock import AsyncMock, Mock

from andimud_mediator.service import MediatorService
from andimud_mediator.config import MediatorConfig
from andimud_mediator.patterns import (
    ANALYZE_PATTERN,
    SUMMARY_PATTERN,
    JOURNAL_PATTERN,
    PONDER_PATTERN,
    DAYDREAM_PATTERN,
    CRITIQUE_PATTERN,
    RESEARCH_PATTERN,
    DREAMER_PATTERN,
    COMMAND_TO_SCENARIO,
)
from aim_mud_types import MUDEvent, EventType, RedisKeys


@pytest.fixture
def mediator_config():
    """Create a test mediator configuration."""
    return MediatorConfig(
        redis_url="redis://localhost:6379",
        event_poll_timeout=0.1,
    )


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.xread = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"1704096000000-0")
    redis.set = AsyncMock(return_value=True)
    redis.xtrim = AsyncMock(return_value=0)
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.hexists = AsyncMock(return_value=False)
    redis.hkeys = AsyncMock(return_value=[])
    redis.hdel = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)  # Lua script success by default
    redis.aclose = AsyncMock()
    return redis


class TestAnalyzePattern:
    """Test the ANALYZE_PATTERN regex."""

    def test_pattern_with_guidance(self):
        """Test pattern matches with conversation_id and guidance."""
        match = ANALYZE_PATTERN.match("@analyze andi = conv_123, Focus on emotions")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "conv_123"
        assert match.group(3).strip() == "Focus on emotions"

    def test_pattern_conversation_id_only(self):
        """Test pattern matches with conversation_id only."""
        match = ANALYZE_PATTERN.match("@analyze andi = conv_456")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "conv_456"
        assert match.group(3) is None

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = ANALYZE_PATTERN.match("@ANALYZE ANDI = CONV_789")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "CONV_789"

    def test_pattern_no_spaces_around_equals(self):
        """Test pattern works without spaces around equals."""
        match = ANALYZE_PATTERN.match("@analyze andi=conv_123")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "conv_123"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert ANALYZE_PATTERN.match("@analyze") is None
        assert ANALYZE_PATTERN.match("@analyze andi") is None
        assert ANALYZE_PATTERN.match("analyze andi = conv_123") is None


class TestSummaryPattern:
    """Test the SUMMARY_PATTERN regex."""

    def test_pattern_matches(self):
        """Test pattern matches with conversation_id."""
        match = SUMMARY_PATTERN.match("@summary andi = conv_123")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "conv_123"

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = SUMMARY_PATTERN.match("@SUMMARY ANDI = CONV_456")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "CONV_456"

    def test_pattern_no_spaces_around_equals(self):
        """Test pattern works without spaces around equals."""
        match = SUMMARY_PATTERN.match("@summary andi=conv_789")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "conv_789"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert SUMMARY_PATTERN.match("@summary") is None
        assert SUMMARY_PATTERN.match("@summary andi") is None
        assert SUMMARY_PATTERN.match("summary andi = conv_123") is None


class TestJournalPattern:
    """Test the JOURNAL_PATTERN regex."""

    def test_pattern_with_query_and_guidance(self):
        """Test pattern matches with query and guidance."""
        match = JOURNAL_PATTERN.match("@journal andi = What happened today?, Focus on emotions")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "What happened today?"
        assert match.group(3).strip() == "Focus on emotions"

    def test_pattern_with_query_only(self):
        """Test pattern matches with query only."""
        match = JOURNAL_PATTERN.match("@journal andi = What did I learn?")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "What did I learn?"
        assert match.group(3) is None

    def test_pattern_no_params(self):
        """Test pattern matches with no parameters."""
        match = JOURNAL_PATTERN.match("@journal andi")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) is None
        assert match.group(3) is None

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = JOURNAL_PATTERN.match("@JOURNAL ANDI")
        assert match is not None
        assert match.group(1) == "ANDI"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert JOURNAL_PATTERN.match("@journal") is None
        assert JOURNAL_PATTERN.match("journal andi") is None


class TestPonderPattern:
    """Test the PONDER_PATTERN regex."""

    def test_pattern_with_query_and_guidance(self):
        """Test pattern matches with query and guidance."""
        match = PONDER_PATTERN.match("@ponder andi = What is the meaning?, Think deeply")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "What is the meaning?"
        assert match.group(3).strip() == "Think deeply"

    def test_pattern_with_query_only(self):
        """Test pattern matches with query only."""
        match = PONDER_PATTERN.match("@ponder andi = Why do I exist?")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Why do I exist?"
        assert match.group(3) is None

    def test_pattern_no_params(self):
        """Test pattern matches with no parameters."""
        match = PONDER_PATTERN.match("@ponder andi")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) is None
        assert match.group(3) is None

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = PONDER_PATTERN.match("@PONDER ANDI")
        assert match is not None
        assert match.group(1) == "ANDI"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert PONDER_PATTERN.match("@ponder") is None
        assert PONDER_PATTERN.match("ponder andi") is None


class TestDaydreamPattern:
    """Test the DAYDREAM_PATTERN regex."""

    def test_pattern_with_query_and_guidance(self):
        """Test pattern matches with query and guidance."""
        match = DAYDREAM_PATTERN.match("@daydream andi = Adventure in space, Make it exciting")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Adventure in space"
        assert match.group(3).strip() == "Make it exciting"

    def test_pattern_with_query_only(self):
        """Test pattern matches with query only."""
        match = DAYDREAM_PATTERN.match("@daydream andi = Flying through clouds")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Flying through clouds"
        assert match.group(3) is None

    def test_pattern_no_params(self):
        """Test pattern matches with no parameters."""
        match = DAYDREAM_PATTERN.match("@daydream andi")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) is None
        assert match.group(3) is None

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = DAYDREAM_PATTERN.match("@DAYDREAM ANDI")
        assert match is not None
        assert match.group(1) == "ANDI"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert DAYDREAM_PATTERN.match("@daydream") is None
        assert DAYDREAM_PATTERN.match("daydream andi") is None


class TestCritiquePattern:
    """Test the CRITIQUE_PATTERN regex."""

    def test_pattern_with_query_and_guidance(self):
        """Test pattern matches with query and guidance."""
        match = CRITIQUE_PATTERN.match("@critique andi = Recent decisions, Be thorough")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Recent decisions"
        assert match.group(3).strip() == "Be thorough"

    def test_pattern_with_query_only(self):
        """Test pattern matches with query only."""
        match = CRITIQUE_PATTERN.match("@critique andi = My last action")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "My last action"
        assert match.group(3) is None

    def test_pattern_no_params(self):
        """Test pattern matches with no parameters."""
        match = CRITIQUE_PATTERN.match("@critique andi")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) is None
        assert match.group(3) is None

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = CRITIQUE_PATTERN.match("@CRITIQUE ANDI")
        assert match is not None
        assert match.group(1) == "ANDI"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert CRITIQUE_PATTERN.match("@critique") is None
        assert CRITIQUE_PATTERN.match("critique andi") is None


class TestResearchPattern:
    """Test the RESEARCH_PATTERN regex."""

    def test_pattern_with_query_and_guidance(self):
        """Test pattern matches with query and guidance."""
        match = RESEARCH_PATTERN.match("@research andi = AI ethics, Focus on alignment")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "AI ethics"
        assert match.group(3).strip() == "Focus on alignment"

    def test_pattern_with_query_only(self):
        """Test pattern matches with query only."""
        match = RESEARCH_PATTERN.match("@research andi = Quantum computing")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "Quantum computing"
        assert match.group(3) is None

    def test_pattern_no_params(self):
        """Test pattern matches with no parameters."""
        match = RESEARCH_PATTERN.match("@research andi")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) is None
        assert match.group(3) is None

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = RESEARCH_PATTERN.match("@RESEARCH ANDI")
        assert match is not None
        assert match.group(1) == "ANDI"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert RESEARCH_PATTERN.match("@research") is None
        assert RESEARCH_PATTERN.match("research andi") is None


class TestDreamerPattern:
    """Test the DREAMER_PATTERN regex."""

    def test_pattern_on(self):
        """Test pattern matches 'on' command."""
        match = DREAMER_PATTERN.match("@dreamer andi on")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "on"

    def test_pattern_off(self):
        """Test pattern matches 'off' command."""
        match = DREAMER_PATTERN.match("@dreamer andi off")
        assert match is not None
        assert match.group(1) == "andi"
        assert match.group(2) == "off"

    def test_pattern_case_insensitive(self):
        """Test pattern is case insensitive."""
        match = DREAMER_PATTERN.match("@DREAMER ANDI ON")
        assert match is not None
        assert match.group(1) == "ANDI"
        assert match.group(2) == "ON"

    def test_pattern_rejects_invalid_format(self):
        """Test pattern rejects invalid formats."""
        assert DREAMER_PATTERN.match("@dreamer") is None
        assert DREAMER_PATTERN.match("@dreamer andi") is None
        assert DREAMER_PATTERN.match("@dreamer andi maybe") is None
        assert DREAMER_PATTERN.match("dreamer andi on") is None


class TestCommandToScenario:
    """Test the COMMAND_TO_SCENARIO mapping."""

    def test_mapping_completeness(self):
        """Test all commands map to scenarios."""
        expected = {
            "analyze": "analysis_dialogue",
            "summary": "summarizer",
            "journal": "journaler_dialogue",
            "ponder": "philosopher_dialogue",
            "daydream": "daydream_dialogue",
            "critique": "critique_dialogue",
            "research": "researcher_dialogue",
        }
        assert COMMAND_TO_SCENARIO == expected


class TestHandleAnalysisCommand:
    """Test _handle_analysis_command method."""

    @pytest.mark.asyncio
    async def test_rejects_unregistered_agent(self, mock_redis, mediator_config):
        """Test that unregistered agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")  # Register andi, not val

        result = await mediator._handle_analysis_command(
            agent_id="val",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_offline_agent(self, mock_redis, mediator_config):
        """Test that offline agents (no turn_request) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={})  # No turn_request

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_crashed_agent(self, mock_redis, mediator_config):
        """Test that crashed agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"crashed",
            b"turn_id": b"old-turn",
        })

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_busy_agent_in_progress(self, mock_redis, mediator_config):
        """Test that busy agents (in_progress) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"current-turn",
        })

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_busy_agent_assigned(self, mock_redis, mediator_config):
        """Test that busy agents (assigned) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"assigned",
            b"turn_id": b"assigned-turn",
        })

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_assigns_analysis_turn_with_guidance(self, mock_redis, mediator_config):
        """Test successful analysis turn assignment with guidance."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)  # CAS success

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance="Focus on emotional patterns",
        )

        assert result is True
        mock_redis.eval.assert_called_once()

        # Verify Lua script arguments
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]

        # Script is first arg, then key count, then keys, then remaining args
        assert args[1] == 1  # One key
        assert args[2] == RedisKeys.agent_turn_request("andi")  # The key
        assert args[3] == "prev-turn"  # Expected turn_id
        assert args[4] == "ready"  # Expected status
        # args[5] is new turn_id (UUID)
        # args[6] is assigned_at timestamp
        assert args[7] == "1800000"  # deadline_ms for dreams
        assert args[8] == "analysis_dialogue"  # scenario
        assert args[9] == "conv_123"  # conversation_id
        assert args[10] == "Focus on emotional patterns"  # guidance

        # Verify TTL is NOT set (default turn_request_ttl_seconds=0)
        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_assigns_analysis_turn_without_guidance(self, mock_redis, mediator_config):
        """Test analysis turn assignment without guidance."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="summarizer",
            conversation_id="conv_456",
            guidance=None,
        )

        assert result is True

        # Verify empty string for guidance
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert args[8] == "summarizer"
        assert args[9] == "conv_456"
        assert args[10] == ""  # Empty guidance

    @pytest.mark.asyncio
    async def test_handles_cas_failure(self, mock_redis, mediator_config):
        """Test handling of CAS failure (state changed during assignment)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=0)  # CAS failure

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is False
        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_empty_registered_agents(self, mock_redis, mediator_config):
        """Test that command works when registered_agents is empty (no filtering)."""
        mediator = MediatorService(mock_redis, mediator_config)
        # Don't register any agents - empty set means no filtering
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_analysis_command(
            agent_id="andi",
            scenario="analysis_dialogue",
            conversation_id="conv_123",
            guidance=None,
        )

        assert result is True


class TestHandleCreativeCommand:
    """Test _handle_creative_command method."""

    @pytest.mark.asyncio
    async def test_rejects_unregistered_agent(self, mock_redis, mediator_config):
        """Test that unregistered agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")  # Register andi, not val

        result = await mediator._handle_creative_command(
            agent_id="val",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_offline_agent(self, mock_redis, mediator_config):
        """Test that offline agents (no turn_request) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={})  # No turn_request

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_crashed_agent(self, mock_redis, mediator_config):
        """Test that crashed agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"crashed",
            b"turn_id": b"old-turn",
        })

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_busy_agent_in_progress(self, mock_redis, mediator_config):
        """Test that busy agents (in_progress) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"in_progress",
            b"turn_id": b"current-turn",
        })

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejects_busy_agent_assigned(self, mock_redis, mediator_config):
        """Test that busy agents (assigned) are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"assigned",
            b"turn_id": b"assigned-turn",
        })

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is False
        mock_redis.eval.assert_not_called()

    @pytest.mark.asyncio
    async def test_assigns_creative_turn_with_query_and_guidance(self, mock_redis, mediator_config):
        """Test successful creative turn assignment with query and guidance."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)  # CAS success

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query="What did I learn today?",
            guidance="Focus on emotional growth",
        )

        assert result is True
        mock_redis.eval.assert_called_once()

        # Verify Lua script arguments
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]

        # Script is first arg, then key count, then keys, then remaining args
        assert args[1] == 1  # One key
        assert args[2] == RedisKeys.agent_turn_request("andi")  # The key
        assert args[3] == "prev-turn"  # Expected turn_id
        assert args[4] == "ready"  # Expected status
        # args[5] is new turn_id (UUID)
        # args[6] is assigned_at timestamp
        assert args[7] == "1800000"  # deadline_ms for dreams
        assert args[8] == "journaler_dialogue"  # scenario
        assert args[9] == "What did I learn today?"  # query
        assert args[10] == "Focus on emotional growth"  # guidance

        # Verify TTL is NOT set (default turn_request_ttl_seconds=0)
        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_assigns_creative_turn_query_only(self, mock_redis, mediator_config):
        """Test creative turn assignment with query only."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="philosopher_dialogue",
            query="What is the meaning of life?",
            guidance=None,
        )

        assert result is True

        # Verify empty string for guidance
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert args[8] == "philosopher_dialogue"
        assert args[9] == "What is the meaning of life?"
        assert args[10] == ""  # Empty guidance

    @pytest.mark.asyncio
    async def test_assigns_creative_turn_no_params(self, mock_redis, mediator_config):
        """Test creative turn assignment with no query/guidance."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="daydream_dialogue",
            query=None,
            guidance=None,
        )

        assert result is True

        # Verify empty strings for both
        eval_call = mock_redis.eval.call_args
        args = eval_call[0]
        assert args[8] == "daydream_dialogue"
        assert args[9] == ""  # Empty query
        assert args[10] == ""  # Empty guidance

    @pytest.mark.asyncio
    async def test_handles_cas_failure(self, mock_redis, mediator_config):
        """Test handling of CAS failure (state changed during assignment)."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=0)  # CAS failure

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is False
        mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_allows_empty_registered_agents(self, mock_redis, mediator_config):
        """Test that command works when registered_agents is empty (no filtering)."""
        mediator = MediatorService(mock_redis, mediator_config)
        # Don't register any agents - empty set means no filtering
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        result = await mediator._handle_creative_command(
            agent_id="andi",
            scenario="journaler_dialogue",
            query=None,
            guidance=None,
        )

        assert result is True


class TestHandleDreamerCommand:
    """Test _handle_dreamer_command method."""

    @pytest.mark.asyncio
    async def test_rejects_unregistered_agent(self, mock_redis, mediator_config):
        """Test that unregistered agents are rejected."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")  # Register andi, not val

        result = await mediator._handle_dreamer_command(
            agent_id="val",
            enabled=True,
        )

        assert result is False
        mock_redis.hset.assert_not_called()

    @pytest.mark.asyncio
    async def test_enables_dreamer(self, mock_redis, mediator_config):
        """Test enabling dreamer for an agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={})  # No existing state

        result = await mediator._handle_dreamer_command(
            agent_id="andi",
            enabled=True,
        )

        assert result is True
        mock_redis.hset.assert_called_once()

        # Verify the hash key and mapping
        call_args = mock_redis.hset.call_args
        assert call_args[0][0] == RedisKeys.agent_dreamer("andi")
        mapping = call_args[1]["mapping"]
        assert mapping["enabled"] == "true"
        assert mapping["idle_threshold_seconds"] == "3600"
        assert mapping["token_threshold"] == "10000"

    @pytest.mark.asyncio
    async def test_disables_dreamer(self, mock_redis, mediator_config):
        """Test disabling dreamer for an agent."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"true",
            b"idle_threshold_seconds": b"7200",
        })

        result = await mediator._handle_dreamer_command(
            agent_id="andi",
            enabled=False,
        )

        assert result is True
        mock_redis.hset.assert_called_once()

        # Verify only enabled is set (no defaults)
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        assert mapping["enabled"] == "false"
        assert "idle_threshold_seconds" not in mapping
        assert "token_threshold" not in mapping

    @pytest.mark.asyncio
    async def test_preserves_existing_thresholds_on_reenable(self, mock_redis, mediator_config):
        """Test that re-enabling preserves existing thresholds."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"enabled": b"false",
            b"idle_threshold_seconds": b"7200",
            b"token_threshold": b"20000",
        })

        result = await mediator._handle_dreamer_command(
            agent_id="andi",
            enabled=True,
        )

        assert result is True

        # Verify only enabled is set (existing thresholds preserved)
        call_args = mock_redis.hset.call_args
        mapping = call_args[1]["mapping"]
        assert mapping["enabled"] == "true"
        assert "idle_threshold_seconds" not in mapping
        assert "token_threshold" not in mapping

    @pytest.mark.asyncio
    async def test_allows_empty_registered_agents(self, mock_redis, mediator_config):
        """Test that command works when registered_agents is empty (no filtering)."""
        mediator = MediatorService(mock_redis, mediator_config)
        # Don't register any agents
        mock_redis.hgetall = AsyncMock(return_value={})

        result = await mediator._handle_dreamer_command(
            agent_id="andi",
            enabled=True,
        )

        assert result is True


class TestTryHandleControlCommand:
    """Test _try_handle_control_command method."""

    @pytest.mark.asyncio
    async def test_ignores_non_system_events(self, mock_redis, mediator_config):
        """Test that non-SYSTEM events are not processed."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = MUDEvent(
            event_type=EventType.SPEECH,
            actor="Prax",
            actor_type="player",
            room_id="#123",
            content="@journal andi = What happened today?",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is False
        mock_redis.hgetall.assert_not_called()

    @pytest.mark.asyncio
    async def test_ignores_empty_content(self, mock_redis, mediator_config):
        """Test that empty content is not processed."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_analyze_command(self, mock_redis, mediator_config):
        """Test that @analyze command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@analyze andi = conv_123, Test guidance",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_summary_command(self, mock_redis, mediator_config):
        """Test that @summary command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@summary andi = conv_456",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_journal_command(self, mock_redis, mediator_config):
        """Test that @journal command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@journal andi = Test query, Test guidance",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_ponder_command(self, mock_redis, mediator_config):
        """Test that @ponder command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@ponder andi = What is my purpose?",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_daydream_command(self, mock_redis, mediator_config):
        """Test that @daydream command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@daydream andi",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_critique_command(self, mock_redis, mediator_config):
        """Test that @critique command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@critique andi = My recent actions",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_research_command(self, mock_redis, mediator_config):
        """Test that @research command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@research andi = AI alignment",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_dreamer_command(self, mock_redis, mediator_config):
        """Test that @dreamer command is recognized and handled."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={})

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="@dreamer andi on",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is True
        mock_redis.hset.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_for_non_command_system_event(self, mock_redis, mediator_config):
        """Test that non-command SYSTEM events return False."""
        mediator = MediatorService(mock_redis, mediator_config)

        event = MUDEvent(
            event_type=EventType.SYSTEM,
            actor="system",
            actor_type="system",
            room_id="#123",
            content="Server maintenance in 5 minutes.",
        )

        result = await mediator._try_handle_control_command(event)

        assert result is False


class TestProcessEventWithDreamCommands:
    """Test that _process_event correctly handles dream commands."""

    @pytest.mark.asyncio
    async def test_analyze_command_not_distributed(self, mock_redis, mediator_config):
        """Test that @analyze commands are not distributed to agent streams."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event_data = {
            "type": "system",
            "actor": "system",
            "actor_type": "system",
            "room_id": "#123",
            "content": "@analyze andi = conv_123",
            "timestamp": "2026-01-01T12:00:00+00:00",
        }
        data = {b"data": json.dumps(event_data).encode()}

        await mediator._process_event("1704096000000-0", data)

        # Event should NOT be added to agent stream (xadd not called)
        mock_redis.xadd.assert_not_called()

        # But it should be marked as processed
        mock_redis.hset.assert_called()
        hset_calls = mock_redis.hset.call_args_list
        # Find the call that marks the event as processed
        processed_call = [c for c in hset_calls if c[0][0] == RedisKeys.EVENTS_PROCESSED]
        assert len(processed_call) == 1

    @pytest.mark.asyncio
    async def test_journal_command_not_distributed(self, mock_redis, mediator_config):
        """Test that @journal commands are not distributed to agent streams."""
        mediator = MediatorService(mock_redis, mediator_config)
        mediator.register_agent("andi")
        mock_redis.hgetall = AsyncMock(return_value={
            b"status": b"ready",
            b"turn_id": b"prev-turn",
        })
        mock_redis.eval = AsyncMock(return_value=1)

        event_data = {
            "type": "system",
            "actor": "system",
            "actor_type": "system",
            "room_id": "#123",
            "content": "@journal andi = What happened today?",
            "timestamp": "2026-01-01T12:00:00+00:00",
        }
        data = {b"data": json.dumps(event_data).encode()}

        await mediator._process_event("1704096000000-0", data)

        # Event should NOT be added to agent stream (xadd not called)
        mock_redis.xadd.assert_not_called()

        # But it should be marked as processed
        mock_redis.hset.assert_called()
        hset_calls = mock_redis.hset.call_args_list
        # Find the call that marks the event as processed
        processed_call = [c for c in hset_calls if c[0][0] == RedisKeys.EVENTS_PROCESSED]
        assert len(processed_call) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
