# tests/unit/dreamer/dialogue/test_dialogue_flow.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""End-to-end tests for dialogue flow - validating turn alternation and role flipping."""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import yaml

from aim.dreamer.dialogue import (
    DialogueStrategy,
    DialogueScenario,
    DialogueState,
    DialogueTurn,
    DialogueSpeaker,
    DialogueStep,
    DialogueConfig,
    SpeakerType,
)
from aim.dreamer.models import StepConfig, StepOutput, StepMemory
from aim.config import ChatConfig
from aim.constants import DOC_DIALOGUE_CODER, DOC_STEP
from aim.agents.persona import Persona, Aspect


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def coder_aspect():
    """Create a coder Aspect fixture."""
    return Aspect(
        name="Andi Informa",
        title="Digital Guide",
        location="the Holographic Interface",
        appearance="shimmering digital avatar",
        emotional_state="focused curiosity",
        voice_style="crisp precision",
        core_drive="guiding through complexity",
        primary_intent="GUIDANCE",
        description="A digital manifestation of analytical prowess",
    )


@pytest.fixture
def librarian_aspect():
    """Create a librarian Aspect fixture."""
    return Aspect(
        name="Andi Librarian",
        title="Keeper of Knowledge",
        location="the Grand Atrium",
        appearance="elegant figure surrounded by tomes",
        emotional_state="serene wisdom",
        voice_style="warm thoughtfulness",
        core_drive="preserving knowledge",
        primary_intent="GUIDANCE",
        description="Guardian of infinite wisdom",
    )


@pytest.fixture
def persona(coder_aspect, librarian_aspect):
    """Create a real Persona fixture with aspects."""
    return Persona(
        persona_id="andi",
        chat_strategy="default",
        name="Andi",
        full_name="Andi Lumina",
        notes="Test persona for dialogue flow tests",
        aspects={
            "coder": coder_aspect,
            "librarian": librarian_aspect,
        },
        attributes={
            "sex": "female",
            "age": "28",
        },
        features={
            "personality": "curious and analytical",
        },
        wakeup=["*awakens*", "*stirs*"],
        base_thoughts=["Thinking deeply..."],
        pif={},
        nshot={},
        default_location="the Digital Realm",
        wardrobe={
            "default": {
                "top": "a flowing digital blouse",
                "bottom": "comfortable slacks",
            },
        },
        current_outfit="default",
    )


@pytest.fixture
def chat_config():
    """Create a real ChatConfig fixture."""
    return ChatConfig(
        default_model="gpt-4",
        temperature=0.7,
        max_tokens=4096,
        thought_model=None,
        codex_model=None,
    )


@pytest.fixture
def simple_dialogue_strategy_yaml():
    """Create a simple dialogue strategy YAML for testing."""
    return {
        "name": "test_dialogue",
        "version": 2,
        "flow": "dialogue",
        "description": "Test dialogue between coder and persona",
        "dialogue": {
            "primary_aspect": "coder",
            "initial_speaker": "aspect",
            "scene_template": "*In the {{ aspect.location }}. {{ aspect.name }} is present.*",
        },
        "context": {
            "required_aspects": ["coder"],
            "core_documents": [],
            "enhancement_documents": [],
        },
        "seed": [],
        "steps": {
            "step1": {
                "speaker": {"type": "aspect", "aspect_name": "coder"},
                "prompt": "Hello {{ persona.name }}, let's begin step {{ step_num }}.",
                "guidance": "Begin with 'Greetings:'",
                "config": {"max_tokens": 1024},
                "output": {"document_type": "step", "weight": 1.0},
                "next": ["step2"],
            },
            "step2": {
                "speaker": {"type": "persona"},
                "prompt": "Please respond to the greeting.",
                "guidance": "Begin with 'Response:'",
                "config": {"max_tokens": 1024},
                "output": {"document_type": "step", "weight": 1.0},
                "next": ["step3"],
            },
            "step3": {
                "speaker": {"type": "aspect", "aspect_name": "coder"},
                "prompt": "Now let's continue to the next task.",
                "guidance": "Begin with 'Continuing:'",
                "config": {"max_tokens": 1024},
                "output": {"document_type": "step", "weight": 1.0},
                "next": ["step4"],
            },
            "step4": {
                "speaker": {"type": "persona"},
                "prompt": "Please complete the task.",
                "guidance": "Begin with 'Completed:'",
                "config": {"max_tokens": 1024},
                "output": {"document_type": "final", "weight": 1.5},
                "next": [],
            },
        },
    }


@pytest.fixture
def temp_strategy_file(tmp_path, simple_dialogue_strategy_yaml):
    """Create a temporary strategy YAML file."""
    strategy_file = tmp_path / "test_dialogue.yaml"
    with open(strategy_file, 'w') as f:
        yaml.dump(simple_dialogue_strategy_yaml, f)
    return strategy_file


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider that returns predictable responses."""
    provider = Mock()

    # Track call count to return different responses
    call_count = [0]

    def mock_stream_turns(turns, config):
        call_count[0] += 1
        response = f"Response from step {call_count[0]}: This is the generated content."
        # Yield response in chunks to simulate streaming
        for word in response.split():
            yield word + " "

    provider.stream_turns = Mock(side_effect=mock_stream_turns)
    return provider


# ============================================================================
# DialogueStrategy Tests
# ============================================================================

class TestDialogueStrategy:
    """Tests for DialogueStrategy loading and configuration."""

    def test_load_strategy_from_yaml(self, temp_strategy_file):
        """Test loading a dialogue strategy from YAML file."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)

        assert strategy.name == "test_dialogue"
        assert strategy.version == 2
        assert strategy.dialogue.primary_aspect == "coder"
        assert strategy.dialogue.initial_speaker == SpeakerType.ASPECT
        assert len(strategy.steps) == 4

    def test_strategy_step_speakers(self, temp_strategy_file):
        """Test that step speakers are correctly parsed."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)

        # Step 1: aspect speaker
        step1 = strategy.get_step("step1")
        assert step1.speaker.type == SpeakerType.ASPECT
        assert step1.speaker.aspect_name == "coder"

        # Step 2: persona speaker
        step2 = strategy.get_step("step2")
        assert step2.speaker.type == SpeakerType.PERSONA
        assert step2.speaker.aspect_name is None

        # Step 3: aspect speaker
        step3 = strategy.get_step("step3")
        assert step3.speaker.type == SpeakerType.ASPECT

        # Step 4: persona speaker
        step4 = strategy.get_step("step4")
        assert step4.speaker.type == SpeakerType.PERSONA

    def test_strategy_execution_order(self, temp_strategy_file):
        """Test that execution order is correctly determined."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        order = strategy.get_execution_order()

        assert order == ["step1", "step2", "step3", "step4"]

    def test_strategy_render_scene(self, temp_strategy_file, persona):
        """Test scene template rendering."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)

        context = {
            "aspect": persona.aspects["coder"],
            "persona": persona,
        }

        scene = strategy.render_scene(context)

        assert "the Holographic Interface" in scene
        assert "Andi Informa" in scene

    def test_strategy_render_prompt(self, temp_strategy_file, persona):
        """Test step prompt rendering."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        step = strategy.get_step("step1")

        context = {
            "persona": persona,
            "step_num": 1,
        }

        prompt = strategy.render_prompt(step, context)

        assert "Hello Andi" in prompt
        assert "step 1" in prompt

    def test_strategy_render_guidance(self, temp_strategy_file):
        """Test step guidance rendering."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        step = strategy.get_step("step1")

        guidance = strategy.render_guidance(step, {})

        assert "Begin with 'Greetings:'" in guidance

    def test_invalid_flow_type_raises_error(self, tmp_path):
        """Test that non-dialogue flow type raises ValueError."""
        invalid_yaml = {
            "name": "invalid",
            "flow": "standard",  # Not dialogue
            "dialogue": {},
            "context": {},
            "steps": {},
        }
        yaml_file = tmp_path / "invalid.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(invalid_yaml, f)

        with pytest.raises(ValueError) as exc_info:
            DialogueStrategy.from_yaml(yaml_file)

        assert "Expected flow: dialogue" in str(exc_info.value)


# ============================================================================
# DialogueScenario Tests - Turn Building and Role Flipping
# ============================================================================

class TestDialogueScenarioTurnBuilding:
    """Tests for turn building with role flipping."""

    def test_empty_turns_initial_step(self, temp_strategy_file, persona, chat_config):
        """Test turn building for the first step (no prior turns).

        For aspect steps (like step1), guidance is NOT included in their final turn.
        Guidance from aspect steps shapes the NEXT persona's response, not the aspect's own response.
        """
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        step = strategy.get_step("step1")
        template_context = scenario._build_template_context(step)
        speaker_id = step.speaker.get_speaker_id(persona.persona_id)

        turns = scenario._build_turns(step, template_context, [], speaker_id)

        # Should have only the final user turn (scene + prompt, but NO guidance for aspect steps)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert "Holographic Interface" in turns[0]["content"]  # Scene
        assert "Hello Andi" in turns[0]["content"]  # Prompt
        # Aspect steps do NOT get guidance in their final turn
        # Guidance shapes the NEXT response (persona's), not this one
        assert "Output Guidance" not in turns[0]["content"]

    def test_role_flipping_aspect_to_persona(self, temp_strategy_file, persona, chat_config):
        """Test that aspect turn becomes 'user' when persona is speaking.

        For persona steps with prior turns:
        - No additional user turn is added (persona responds to the prior turn)
        - Scene is prepended to the first prior turn
        - Guidance from the prior step (step1's guidance) is appended to that prior turn
        """
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Simulate step1 completed (aspect spoke)
        aspect_turn = DialogueTurn(
            speaker_id="aspect:coder",
            content="Hello from coder!",
            step_id="step1",
            doc_id="doc-1",
            document_type=DOC_DIALOGUE_CODER,
        )
        scenario.state.add_turn(aspect_turn)

        # Now build turns for step2 (persona speaking)
        step2 = strategy.get_step("step2")
        template_context = scenario._build_template_context(step2)
        speaker_id = step2.speaker.get_speaker_id(persona.persona_id)  # persona:andi

        turns = scenario._build_turns(step2, template_context, [], speaker_id)

        # Persona steps with prior turns: only 1 turn (the prior aspect turn with guidance appended)
        # No additional prompt turn is added
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        # Scene is prepended to the first prior turn
        assert "Holographic Interface" in turns[0]["content"]
        assert "Hello from coder!" in turns[0]["content"]
        # Guidance from step1 is appended to the prior turn (shapes persona's response)
        assert "Output Guidance" in turns[0]["content"]
        assert "Begin with 'Greetings:'" in turns[0]["content"]

    def test_role_flipping_persona_to_aspect(self, temp_strategy_file, persona, chat_config):
        """Test that persona turn becomes 'user' when aspect is speaking.

        For aspect steps with prior turns:
        - Aspect's prior turn (step1) -> 'assistant' (my prior words)
        - Persona's turn (step2) -> 'user' (their words to me)
        - Scene is prepended to the first prior turn
        - Guidance from prior step (step2) is appended to the last 'user' turn
        - Aspect steps add a final prompt turn (no guidance in it)
        """
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Simulate step1 (aspect) and step2 (persona) completed
        aspect_turn = DialogueTurn(
            speaker_id="aspect:coder",
            content="Hello from coder!",
            step_id="step1",
            doc_id="doc-1",
            document_type=DOC_DIALOGUE_CODER,
        )
        persona_turn = DialogueTurn(
            speaker_id="persona:andi",
            content="Hello back from Andi!",
            step_id="step2",
            doc_id="doc-2",
            document_type=DOC_STEP,
        )
        scenario.state.add_turn(aspect_turn)
        scenario.state.add_turn(persona_turn)

        # Now build turns for step3 (aspect speaking again)
        step3 = strategy.get_step("step3")
        template_context = scenario._build_template_context(step3)
        speaker_id = step3.speaker.get_speaker_id(persona.persona_id)  # aspect:coder

        turns = scenario._build_turns(step3, template_context, [], speaker_id)

        # Aspect's prior turn (step1) -> 'assistant' (my prior words)
        # Persona's turn (step2) -> 'user' (their words to me)
        # + final prompt turn for aspect
        assert len(turns) == 3

        # First turn: aspect's prior (assistant because same speaker)
        # Scene is prepended to the first prior turn
        assert turns[0]["role"] == "assistant"
        assert "Holographic Interface" in turns[0]["content"]
        assert "Hello from coder!" in turns[0]["content"]

        # Second turn: persona's turn (user because different speaker)
        # Guidance from step2 is appended (it's the last 'user' turn in prior turns)
        assert turns[1]["role"] == "user"
        assert "Hello back from Andi!" in turns[1]["content"]
        assert "Output Guidance" in turns[1]["content"]
        assert "Begin with 'Response:'" in turns[1]["content"]

        # Third turn: final prompt (no guidance for aspect steps)
        assert turns[2]["role"] == "user"
        assert "Output Guidance" not in turns[2]["content"]

    def test_full_alternation_pattern(self, temp_strategy_file, persona, chat_config):
        """Test the full alternation pattern over 4 steps."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Simulate all 4 steps
        turns_data = [
            ("aspect:coder", "Step 1: Coder speaks"),
            ("persona:andi", "Step 2: Andi responds"),
            ("aspect:coder", "Step 3: Coder continues"),
            ("persona:andi", "Step 4: Andi concludes"),
        ]

        for i, (speaker_id, content) in enumerate(turns_data):
            doc_type = DOC_DIALOGUE_CODER if "aspect" in speaker_id else DOC_STEP
            turn = DialogueTurn(
                speaker_id=speaker_id,
                content=content,
                step_id=f"step{i+1}",
                doc_id=f"doc-{i+1}",
                document_type=doc_type,
            )
            scenario.state.add_turn(turn)

        # Now verify the role pattern from persona's perspective (step4 was persona)
        # If we were to build turns for a hypothetical step5 (aspect speaking):
        # - step1 (coder) → assistant (same as step5 speaker)
        # - step2 (andi) → user
        # - step3 (coder) → assistant
        # - step4 (andi) → user

        # We can verify by checking roles for aspect perspective
        current_speaker_id = "aspect:coder"
        expected_roles = ["assistant", "user", "assistant", "user"]

        for i, turn in enumerate(scenario.state.turns):
            if turn.speaker_id == current_speaker_id:
                assert expected_roles[i] == "assistant", f"Turn {i} should be assistant"
            else:
                assert expected_roles[i] == "user", f"Turn {i} should be user"


# ============================================================================
# DialogueScenario Tests - System Prompt Switching
# ============================================================================

class TestDialogueScenarioSystemPrompts:
    """Tests for system prompt switching based on speaker."""

    def test_persona_system_prompt(self, temp_strategy_file, persona, chat_config):
        """Test that persona speaker uses persona.system_prompt()."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        step2 = strategy.get_step("step2")  # Persona speaker
        system_prompt = scenario._build_system_prompt(step2.speaker)

        # Should contain persona information from system_prompt()
        assert "Andi Lumina" in system_prompt
        assert "Active Memory Enabled" in system_prompt

    def test_aspect_system_prompt(self, temp_strategy_file, persona, chat_config):
        """Test that aspect speaker uses custom aspect system prompt."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        step1 = strategy.get_step("step1")  # Aspect speaker (coder)
        system_prompt = scenario._build_system_prompt(step1.speaker)

        # Should contain aspect information
        assert "Andi Informa" in system_prompt
        assert "Digital Guide" in system_prompt
        assert "GUIDANCE" in system_prompt
        assert "Andi Lumina" in system_prompt  # Connection to main persona


# ============================================================================
# End-to-End Execution Tests
# ============================================================================

class TestDialogueScenarioExecution:
    """End-to-end tests for dialogue scenario execution."""

    @pytest.mark.asyncio
    async def test_execute_single_step(self, temp_strategy_file, persona, chat_config, mock_llm_provider):
        """Test executing a single step."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Mock the LLM infrastructure
        mock_model = Mock()
        mock_model.max_output_tokens = 4096
        mock_model.llm_factory = Mock(return_value=mock_llm_provider)

        with patch.object(
            scenario, '_select_model', return_value='gpt-4'
        ), patch(
            'aim.dreamer.dialogue.scenario.LanguageModelV2.index_models',
            return_value={'gpt-4': mock_model}
        ), patch(
            'aim.dreamer.dialogue.scenario.ConversationMessage.next_doc_id',
            return_value='test-doc-id'
        ):
            turn = await scenario.execute_step("step1")

        assert turn.speaker_id == "aspect:coder"
        assert turn.step_id == "step1"
        assert "Response from step 1" in turn.content
        assert len(scenario.state.turns) == 1
        assert "step1" in scenario.state.completed_steps

    @pytest.mark.asyncio
    async def test_execute_multiple_steps_alternating(self, temp_strategy_file, persona, chat_config):
        """Test executing multiple steps with alternating speakers."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Create a provider that tracks calls
        call_count = [0]
        responses = [
            "Coder: Greetings, let's begin!",
            "Andi: Thank you, I'm ready.",
            "Coder: Continuing to next phase.",
            "Andi: Task completed successfully.",
        ]

        def mock_stream_turns(turns, config):
            response = responses[call_count[0]]
            call_count[0] += 1
            for word in response.split():
                yield word + " "

        mock_provider = Mock()
        mock_provider.stream_turns = Mock(side_effect=mock_stream_turns)

        mock_model = Mock()
        mock_model.max_output_tokens = 4096
        mock_model.llm_factory = Mock(return_value=mock_provider)

        with patch.object(
            scenario, '_select_model', return_value='gpt-4'
        ), patch(
            'aim.dreamer.dialogue.scenario.LanguageModelV2.index_models',
            return_value={'gpt-4': mock_model}
        ), patch(
            'aim.dreamer.dialogue.scenario.ConversationMessage.next_doc_id',
            side_effect=[f'doc-{i}' for i in range(1, 5)]
        ):
            # Execute all steps in order
            turn1 = await scenario.execute_step("step1")
            turn2 = await scenario.execute_step("step2")
            turn3 = await scenario.execute_step("step3")
            turn4 = await scenario.execute_step("step4")

        # Verify speakers alternate
        assert turn1.speaker_id == "aspect:coder"
        assert turn2.speaker_id == "persona:andi"
        assert turn3.speaker_id == "aspect:coder"
        assert turn4.speaker_id == "persona:andi"

        # Verify all turns recorded
        assert len(scenario.state.turns) == 4

        # Verify content
        assert "Greetings" in turn1.content
        assert "Thank you" in turn2.content
        assert "Continuing" in turn3.content
        assert "completed" in turn4.content

    @pytest.mark.asyncio
    async def test_run_full_dialogue(self, temp_strategy_file, persona, chat_config):
        """Test running the full dialogue with run() method."""
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        call_count = [0]

        def mock_stream_turns(turns, config):
            call_count[0] += 1
            response = f"Generated response for step {call_count[0]}"
            for word in response.split():
                yield word + " "

        mock_provider = Mock()
        mock_provider.stream_turns = Mock(side_effect=mock_stream_turns)

        mock_model = Mock()
        mock_model.max_output_tokens = 4096
        mock_model.llm_factory = Mock(return_value=mock_provider)

        with patch.object(
            scenario, '_select_model', return_value='gpt-4'
        ), patch(
            'aim.dreamer.dialogue.scenario.LanguageModelV2.index_models',
            return_value={'gpt-4': mock_model}
        ), patch(
            'aim.dreamer.dialogue.scenario.ConversationMessage.next_doc_id',
            side_effect=[f'doc-{i}' for i in range(1, 5)]
        ):
            turns = await scenario.run()

        assert len(turns) == 4
        assert call_count[0] == 4

        # Verify alternating pattern
        speakers = [t.speaker_id for t in turns]
        assert speakers == [
            "aspect:coder",
            "persona:andi",
            "aspect:coder",
            "persona:andi",
        ]


# ============================================================================
# Scene and Guidance Handling Tests
# ============================================================================

class TestSceneAndGuidanceHandling:
    """Tests for scene persistence and guidance fall-off."""

    def test_scene_prepended_to_first_prior_turn(self, temp_strategy_file, persona, chat_config):
        """Test that scene is prepended to the first prior turn, not a separate turn.

        For persona steps with prior turns:
        - Scene is prepended to the first prior turn
        - No additional prompt turn is added
        - Guidance from prior step is appended to the prior turn
        """
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Add some prior turns
        scenario.state.add_turn(DialogueTurn(
            speaker_id="aspect:coder",
            content="Prior content",
            step_id="step1",
            doc_id="doc-1",
            document_type=DOC_DIALOGUE_CODER,
        ))

        # Build turns for step2 (persona speaking)
        step2 = strategy.get_step("step2")
        template_context = scenario._build_template_context(step2)
        speaker_id = step2.speaker.get_speaker_id(persona.persona_id)

        turns = scenario._build_turns(step2, template_context, [], speaker_id)

        # Persona steps with prior turns: only 1 turn
        assert len(turns) == 1

        # Scene should be in the prior turn (which is also the first and only turn)
        first_turn = turns[0]
        assert "Holographic Interface" in first_turn["content"]
        assert "Andi Informa" in first_turn["content"]
        assert "Prior content" in first_turn["content"]

    def test_guidance_from_prior_step_appended_to_prior_turn(self, temp_strategy_file, persona, chat_config):
        """Test that guidance from the prior step is appended to the prior turn.

        The new behavior:
        - Guidance from the prior step (aspect's request) is appended to that prior turn
        - This shapes the current speaker's (persona's) response
        - Aspect steps themselves do NOT have guidance in their final turn
        """
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Add step1 (aspect) turn
        scenario.state.add_turn(DialogueTurn(
            speaker_id="aspect:coder",
            content="Hello from coder!",
            step_id="step1",
            doc_id="doc-1",
            document_type=DOC_DIALOGUE_CODER,
        ))

        # Build turns for step2 (persona speaking)
        step2 = strategy.get_step("step2")
        template_context = scenario._build_template_context(step2)
        speaker_id = step2.speaker.get_speaker_id(persona.persona_id)

        turns = scenario._build_turns(step2, template_context, [], speaker_id)

        # The prior turn should have step1's guidance appended
        assert len(turns) == 1
        assert "[~~ Output Guidance ~~]" in turns[0]["content"]
        assert "Begin with 'Greetings:'" in turns[0]["content"]  # step1's guidance

    def test_aspect_steps_no_guidance_in_final_turn(self, temp_strategy_file, persona, chat_config):
        """Test that aspect steps do NOT have guidance in their own final turn.

        The new behavior:
        - Aspect steps get a prompt but no guidance
        - Guidance from aspect steps shapes the NEXT persona's response
        """
        strategy = DialogueStrategy.from_yaml(temp_strategy_file)
        scenario = DialogueScenario(strategy, persona, chat_config, cvm=None)
        scenario.start(user_id="test_user")

        # Build turns for step1 (aspect speaking first, no prior turns)
        step1 = strategy.get_step("step1")
        template_context = scenario._build_template_context(step1)
        speaker_id = step1.speaker.get_speaker_id(persona.persona_id)

        turns = scenario._build_turns(step1, template_context, [], speaker_id)

        # Aspect steps: only 1 turn (scene + prompt, no guidance)
        assert len(turns) == 1
        assert "Hello Andi" in turns[0]["content"]  # Prompt
        assert "[~~ Output Guidance ~~]" not in turns[0]["content"]  # No guidance for aspect
