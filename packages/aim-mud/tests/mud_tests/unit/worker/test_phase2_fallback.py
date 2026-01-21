# packages/aim-mud/tests/mud_tests/unit/worker/test_phase2_fallback.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Integration tests for Phase 2 fallback model retry logic.

DEPRECATED: These tests were written for the old PhasedTurnProcessor architecture.

The architecture has been refactored:
- PhasedTurnProcessor -> DecisionProcessor + SpeakingProcessor
- Fallback retry logic is now in SpeakingProcessor
- The test structure needs to be updated to test SpeakingProcessor directly

These tests need to be rewritten to test SpeakingProcessor.
Skipping for now pending rewrite.

Original test coverage:
- Retry loop for emotional state header validation
- Fallback model activation after 3 chat model failures
- Format guidance escalation (gentle → strong → fallback)
- Loss for words emote when all attempts fail
- Abort detection during retry/fallback
- Early success scenarios (no fallback needed)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime, timezone

from andimud_worker.worker import MUDAgentWorker
from andimud_worker.config import MUDConfig
from andimud_worker.turns.processor.speaking import SpeakingProcessor
from aim_mud_types import MUDSession, MUDTurnRequest, MUDEvent, EventType, MUDAction
from aim.config import ChatConfig
from aim.llm.model_set import ModelSet
from aim.agents.persona import Persona


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mud_config():
    """Create a test MUD configuration."""
    return MUDConfig(
        agent_id="test_agent",
        persona_id="test_persona",
        redis_url="redis://localhost:6379",
    )


@pytest.fixture
def chat_config():
    """Create a test ChatConfig with fallback configured.

    WORKAROUND: Due to implementation deficiency in model_set.py line 116,
    fallback must be set on config, not persona.models["fallback"].
    See IMPLEMENTATION DEFICIENCY report for details.
    """
    config = ChatConfig(
        persona_path="config/persona",
        tools_path="config/tools",
        memory_path="memory",
        user_id="test_user",
        persona_id="test_persona",
        conversation_id="test_conversation",
    )
    config.default_model = "anthropic/claude-sonnet-4-5-20250929"
    config.thought_model = "anthropic/claude-opus-4-5-20251101"
    config.codex_model = "anthropic/claude-opus-4-5-20251101"
    config.fallback = "anthropic/claude-3.5-haiku"  # Fallback model
    config.llm_provider = "anthropic"
    config.anthropic_api_key = "test-key"
    return config


@pytest.fixture
def mock_persona():
    """Create a mock Persona with fallback model configured."""
    persona = MagicMock(spec=Persona)
    persona.persona_id = "test_persona"
    persona.models = {
        "chat": "anthropic/claude-sonnet-4-5-20250929",
        "fallback": "anthropic/claude-3.5-haiku",
    }
    persona.system_prompt.return_value = "You are a test persona."
    persona.xml_decorator = MagicMock(side_effect=lambda xml, **kwargs: xml)
    persona.thoughts = []
    persona.get_wakeup.return_value = ""
    return persona


@pytest.fixture
def chat_config_no_fallback():
    """Create a test ChatConfig without fallback configured.

    WORKAROUND: Due to implementation deficiency in model_set.py line 116,
    fallback must be set on config, not persona.models["fallback"].
    This fixture omits config.fallback to test the "no fallback" scenario.
    """
    config = ChatConfig(
        persona_path="config/persona",
        tools_path="config/tools",
        memory_path="memory",
        user_id="test_user",
        persona_id="test_persona",
        conversation_id="test_conversation",
    )
    config.default_model = "anthropic/claude-sonnet-4-5-20250929"
    config.thought_model = "anthropic/claude-opus-4-5-20251101"
    config.codex_model = "anthropic/claude-opus-4-5-20251101"
    # No config.fallback set - will default to default_model
    config.llm_provider = "anthropic"
    config.anthropic_api_key = "test-key"
    return config


@pytest.fixture
def mock_persona_no_fallback():
    """Create a mock Persona without distinct fallback model."""
    persona = MagicMock(spec=Persona)
    persona.persona_id = "test_persona"
    persona.models = {
        "chat": "anthropic/claude-sonnet-4-5-20250929",
        "fallback": "anthropic/claude-sonnet-4-5-20250929",  # Same as chat
    }
    persona.system_prompt.return_value = "You are a test persona."
    persona.xml_decorator = MagicMock(side_effect=lambda xml, **kwargs: xml)
    persona.thoughts = []
    persona.get_wakeup.return_value = ""
    return persona


@pytest.fixture
def model_set(chat_config, mock_persona):
    """Create a ModelSet with fallback configured."""
    return ModelSet.from_config(chat_config, persona=mock_persona)


@pytest.fixture
def model_set_no_fallback(chat_config_no_fallback, mock_persona_no_fallback):
    """Create a ModelSet without distinct fallback."""
    return ModelSet.from_config(chat_config_no_fallback, persona=mock_persona_no_fallback)


@pytest.fixture
def mock_redis():
    """Create a mock async Redis client."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hget = AsyncMock(return_value=None)
    redis.hset = AsyncMock(return_value=1)
    redis.get = AsyncMock(return_value=None)
    redis.eval = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    redis.lrange = AsyncMock(return_value=[])
    redis.xadd = AsyncMock(return_value=b"stream-id-123")
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def session():
    """Create a test MUDSession."""
    return MUDSession(
        agent_id="test_agent",
        persona_id="test_persona",
    )


@pytest.fixture
def turn_request():
    """Create a test MUDTurnRequest."""
    return MUDTurnRequest(
        turn_id="test_turn_123",
        agent_id="test_agent",
        reason="events",
        sequence_id=100,
    )


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    return [
        MUDEvent(
            event_id="event-1",
            event_type=EventType.SPEECH,
            actor="OtherAgent",
            actor_id="other_agent",
            room_id="room1",
            room_name="Test Room",
            content="Hello there!",
            timestamp=datetime.now(timezone.utc),
            metadata={"sequence_id": 100},
        ),
    ]


@pytest.fixture
def worker(mud_config, mock_redis, chat_config, mock_persona, model_set):
    """Create a MUDAgentWorker for testing."""
    worker = MUDAgentWorker(
        config=mud_config,
        redis_client=mock_redis,
        chat_config=chat_config,
    )
    worker.persona = mock_persona
    worker.model = MagicMock()
    worker.model.max_tokens = 128000
    worker.model_set = model_set
    worker.chat_config = chat_config
    worker.chat_config.max_tokens = 4096

    # Mock internal methods
    worker._emit_actions = AsyncMock()
    worker._is_fresh_session = AsyncMock(return_value=False)
    worker._response_strategy = MagicMock()
    worker._response_strategy.build_turns = AsyncMock(return_value=[
        {"role": "user", "content": "Test user message"}
    ])

    # Mock decision strategy (needed for aura tool check)
    worker._decision_strategy = MagicMock()
    worker._decision_strategy.is_aura_tool = MagicMock(return_value=False)

    return worker


# =============================================================================
# Test 1: Fallback succeeds after 3 chat failures
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestFallbackSucceedsAfterChatFailures:
    """Test fallback model succeeds after 3 chat model failures."""

    @pytest.mark.asyncio
    async def test_fallback_succeeds_on_fourth_attempt(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Fallback model succeeds after 3 chat failures.

        Validates:
        - 3 chat attempts with invalid responses (missing ESH)
        - 4th attempt uses fallback model
        - Fallback returns valid response with ESH
        - Valid response is processed normally
        - No loss for words emote
        """
        worker.session = session

        # Track call_llm invocations to verify model role and response
        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1

            # First 3 attempts (chat): invalid response (no ESH)
            if call_count <= 3:
                assert role == "chat", f"Attempt {call_count} should use chat role"
                return "This is a response without emotional state header."

            # 4th attempt (fallback): valid response with ESH
            if call_count == 4:
                assert role == "fallback", "4th attempt should use fallback role"
                return "[== test_persona's Emotional State: +Focused+ +Helpful+ ==]\nThis is a proper response."

            pytest.fail(f"Unexpected call_llm attempt: {call_count}")

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        # Mock decision phase to return "speak"
        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute phase 2 via processor
        processor = PhasedTurnProcessor(worker)
        processor.user_guidance = ""

        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify 4 LLM calls were made
        assert call_count == 4, "Should make exactly 4 LLM calls (3 chat + 1 fallback)"

        # Verify valid response was processed (speak action emitted)
        assert len(actions) == 1
        assert actions[0].tool == "speak"
        assert "proper response" in actions[0].args["text"]

        # Verify no loss for words emote
        assert not any(a.tool == "emote" and "loss for words" in a.args.get("action", "") for a in actions)


# =============================================================================
# Test 2: Loss for words after all 4 attempts fail
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestLossForWordsAfterAllFailures:
    """Test loss for words emote after all retry attempts fail."""

    @pytest.mark.asyncio
    async def test_loss_for_words_after_all_attempts_fail(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Loss for words emote after 3 chat + 1 fallback failures.

        Validates:
        - 4 LLM calls total (3 chat + 1 fallback)
        - All return invalid responses (missing ESH)
        - Loss for words emote is emitted
        - Error in thinking output
        - No speak action emitted
        """
        worker.session = session

        # Track call count
        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1
            # All attempts return invalid response
            return "This response is missing the emotional state header."

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        # Mock decision phase
        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify 4 LLM calls
        assert call_count == 4, "Should make 4 calls (3 chat + 1 fallback)"

        # Verify loss for words emote
        assert len(actions) == 1
        assert actions[0].tool == "emote"
        assert "loss for words" in actions[0].args["action"]

        # Verify error in thinking
        assert "[ERROR]" in thinking
        assert "Failed to generate valid response format" in thinking


# =============================================================================
# Test 3: Skipping fallback when not configured
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestSkipFallbackWhenNotConfigured:
    """Test fallback is skipped when not configured or same as chat."""

    @pytest.mark.asyncio
    async def test_skip_fallback_when_same_as_chat(
        self, mud_config, mock_redis, chat_config_no_fallback, mock_persona_no_fallback,
        model_set_no_fallback, session, turn_request, sample_events
    ):
        """Test: Only 3 attempts when fallback is same as chat model.

        Validates:
        - 3 LLM calls only (no fallback attempt)
        - Loss for words emote after 3 attempts
        """
        # Create worker with no distinct fallback
        worker = MUDAgentWorker(
            config=mud_config,
            redis_client=mock_redis,
            chat_config=chat_config_no_fallback,
        )
        worker.persona = mock_persona_no_fallback
        worker.model = MagicMock()
        worker.model.max_tokens = 128000
        worker.model_set = model_set_no_fallback
        worker.chat_config = chat_config_no_fallback
        worker.chat_config.max_tokens = 4096
        worker.session = session

        worker._emit_actions = AsyncMock()
        worker._is_fresh_session = AsyncMock(return_value=False)
        worker._response_strategy = MagicMock()
        worker._response_strategy.build_turns = AsyncMock(return_value=[
            {"role": "user", "content": "Test"}
        ])

        # Mock decision strategy (needed for aura tool check)
        worker._decision_strategy = MagicMock()
        worker._decision_strategy.is_aura_tool = MagicMock(return_value=False)

        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1
            return "Response without ESH"

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify only 3 calls (no fallback)
        assert call_count == 3, "Should only make 3 chat calls when fallback not configured"

        # Verify loss for words emote
        assert len(actions) == 1
        assert actions[0].tool == "emote"
        assert "loss for words" in actions[0].args["action"]


# =============================================================================
# Test 4: Abort during fallback attempt
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestAbortDuringFallback:
    """Test abort detection during fallback attempt."""

    @pytest.mark.asyncio
    async def test_abort_during_fallback_attempt(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Abort detected during fallback attempt.

        Validates:
        - 3 chat attempts complete
        - Abort detected before fallback LLM call
        - Abort is caught by exception handler
        - Error added to thinking
        - Loss for words emote (generic error handling)
        """
        from andimud_worker.exceptions import AbortRequestedException

        worker.session = session

        call_count = 0

        async def mock_check_abort():
            nonlocal call_count
            # Allow first 3 attempts, abort on 4th
            return call_count >= 3

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1
            return "Response without ESH"

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(side_effect=mock_check_abort)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify 3 calls were made before abort
        assert call_count == 3

        # Verify abort error is in thinking
        assert "[ERROR]" in thinking
        assert "Turn aborted" in thinking or "aborted" in thinking.lower()

        # Verify loss for words emote (generic error handling)
        assert len(actions) == 1
        assert actions[0].tool == "emote"
        assert "loss for words" in actions[0].args["action"]


# =============================================================================
# Test 5: First attempt succeeds (no fallback needed)
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestFirstAttemptSucceeds:
    """Test early success - no retry or fallback needed."""

    @pytest.mark.asyncio
    async def test_first_attempt_succeeds(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Valid response on first attempt.

        Validates:
        - Only 1 LLM call
        - No fallback attempted
        - Valid response processed normally
        """
        worker.session = session

        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1
            assert role == "chat", "First attempt should use chat role"
            return "[== test_persona's Emotional State: +Happy+ ==]\nGreat response!"

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify only 1 call
        assert call_count == 1, "Should only make 1 call when first attempt succeeds"

        # Verify speak action
        assert len(actions) == 1
        assert actions[0].tool == "speak"
        assert "Great response" in actions[0].args["text"]


# =============================================================================
# Test 6: Second attempt succeeds (no fallback needed)
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestSecondAttemptSucceeds:
    """Test success on second attempt with format guidance."""

    @pytest.mark.asyncio
    async def test_second_attempt_succeeds(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Valid response on second attempt after format guidance.

        Validates:
        - 2 LLM calls total
        - Format guidance added after first failure
        - No fallback attempted
        - Valid response processed
        """
        worker.session = session

        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return "Invalid response without ESH"

            if call_count == 2:
                # Check that format guidance was added
                assert any("Gentle reminder" in str(turn.get("content", "")) for turn in turns), \
                    "Format guidance should be added after first failure"
                return "[== test_persona's Emotional State: +Focused+ ==]\nNow with proper format!"

            pytest.fail(f"Unexpected call: {call_count}")

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify 2 calls
        assert call_count == 2, "Should make 2 calls (1 fail + 1 success)"

        # Verify speak action
        assert len(actions) == 1
        assert actions[0].tool == "speak"
        assert "proper format" in actions[0].args["text"]


# =============================================================================
# Test 7: Third attempt succeeds (no fallback needed)
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestThirdAttemptSucceeds:
    """Test success on third attempt with stronger guidance."""

    @pytest.mark.asyncio
    async def test_third_attempt_succeeds(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Valid response on third attempt with escalated guidance.

        Validates:
        - 3 LLM calls total
        - Progressive format guidance (gentle → gentle → strong)
        - No fallback attempted
        - Valid response processed
        """
        worker.session = session

        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1

            if call_count in [1, 2]:
                return "Invalid response"

            if call_count == 3:
                # Check for stronger guidance before 3rd attempt
                # Note: The stronger guidance is added AFTER the 2nd failure,
                # not before the 3rd attempt in current implementation
                return "[== test_persona's Emotional State: +Determined+ ==]\nFinally got it right!"

            pytest.fail(f"Unexpected call: {call_count}")

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify 3 calls
        assert call_count == 3, "Should make 3 calls (2 fail + 1 success)"

        # Verify speak action
        assert len(actions) == 1
        assert actions[0].tool == "speak"
        assert "got it right" in actions[0].args["text"]


# =============================================================================
# Test 8: Model role verification
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestModelRoleVerification:
    """Test that correct model roles are used for each attempt."""

    @pytest.mark.asyncio
    async def test_correct_model_roles_for_each_attempt(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Verify chat vs fallback role usage.

        Validates:
        - Attempts 1-3 use "chat" role
        - Attempt 4 uses "fallback" role
        """
        worker.session = session

        roles_used = []

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            roles_used.append(role)
            return "Invalid response"

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify roles
        assert len(roles_used) == 4, "Should have 4 attempts"
        assert roles_used[0] == "chat", "1st attempt should use chat"
        assert roles_used[1] == "chat", "2nd attempt should use chat"
        assert roles_used[2] == "chat", "3rd attempt should use chat"
        assert roles_used[3] == "fallback", "4th attempt should use fallback"


# =============================================================================
# Test 9: Heartbeat callback verification
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestHeartbeatCallback:
    """Test that heartbeat callback is passed to LLM calls."""

    @pytest.mark.asyncio
    async def test_heartbeat_callback_passed_to_call_llm(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Heartbeat callback is provided for all LLM calls.

        Validates:
        - heartbeat_callback parameter is not None
        - Callback is async callable
        """
        worker.session = session

        heartbeat_callbacks = []

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            heartbeat_callbacks.append(heartbeat_callback)
            return "[== test_persona's Emotional State: +Ready+ ==]\nResponse"

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        await processor._decide_action(turn_request, sample_events)

        # Verify heartbeat callback was provided
        assert len(heartbeat_callbacks) == 1
        assert heartbeat_callbacks[0] is not None
        assert callable(heartbeat_callbacks[0])


# =============================================================================
# Test 10: Think tag extraction during retry
# =============================================================================


@pytest.mark.skip(reason="Needs rewrite for SpeakingProcessor architecture")
class TestThinkTagExtraction:
    """Test that <think> tags are extracted during retry validation."""

    @pytest.mark.asyncio
    async def test_think_content_extracted_from_all_attempts(
        self, worker, session, turn_request, sample_events
    ):
        """Test: Think content is extracted and accumulated across retries.

        Validates:
        - Think tags from all attempts are extracted
        - Think content is accumulated in thinking output
        """
        worker.session = session

        call_count = 0

        async def mock_call_llm(turns, role=None, heartbeat_callback=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return "<think>Attempt 1 thinking</think>Response without ESH"

            if call_count == 2:
                return "<think>Attempt 2 thinking</think>[== test_persona's Emotional State: +Thinking+ ==]\nGood!"

            pytest.fail(f"Unexpected call: {call_count}")

        worker._call_llm = AsyncMock(side_effect=mock_call_llm)
        worker._check_abort_requested = AsyncMock(return_value=False)

        async def mock_decide_action(idle_mode, role, action_guidance, user_guidance):
            return "speak", {}, "", "", ""

        worker._decide_action = AsyncMock(side_effect=mock_decide_action)

        # Execute
        processor = PhasedTurnProcessor(worker)
        actions, thinking = await processor._decide_action(turn_request, sample_events)

        # Verify think content from both attempts is in thinking
        assert "Attempt 1 thinking" in thinking
        assert "Attempt 2 thinking" in thinking


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
