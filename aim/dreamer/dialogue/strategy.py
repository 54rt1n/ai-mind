# aim/dreamer/dialogue/strategy.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DialogueStrategy: Loads and represents dialogue flow configuration from YAML."""

from pathlib import Path
from typing import Optional
import yaml

from .models import (
    DialogueStep,
    DialogueSpeaker,
    DialogueConfig,
    DialogueState,
    ScenarioContext,
    SpeakerType,
)
from ..models import MemoryAction, StepConfig, StepOutput
from ..scenario import get_jinja_environment


class DialogueStrategy:
    """
    Loads and validates dialogue strategy from YAML.

    The Strategy knows WHAT to do (configuration),
    the Scenario knows HOW to do it (execution).

    Factory pattern: DialogueStrategy.from_yaml() builds instances from config.

    Attributes:
        name: Strategy name
        version: Version number
        description: Human-readable description
        dialogue: Dialogue-specific configuration
        context: Scenario context (aspects, documents)
        seed: Initial data loading actions
        steps: Dictionary of step definitions
    """

    def __init__(
        self,
        name: str,
        version: int,
        description: str,
        dialogue: DialogueConfig,
        context: ScenarioContext,
        seed: list[MemoryAction],
        steps: dict[str, DialogueStep],
    ):
        self.name = name
        self.version = version
        self.description = description
        self.dialogue = dialogue
        self.context = context
        self.seed = seed
        self.steps = steps
        self._jinja_env = get_jinja_environment()

    @classmethod
    def from_yaml(cls, path: Path) -> "DialogueStrategy":
        """
        Factory method: Build strategy instance from YAML config file.

        Args:
            path: Path to the YAML strategy file

        Returns:
            Configured DialogueStrategy instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If flow type is not 'dialogue'
            pydantic.ValidationError: If config structure is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate flow type
        flow = data.get('flow')
        if flow != 'dialogue':
            raise ValueError(f"Expected flow: dialogue, got: {flow}")

        # Build dialogue config
        dialogue_data = data.get('dialogue', {})
        dialogue = DialogueConfig(
            primary_aspect=dialogue_data.get('primary_aspect', 'coder'),
            initial_speaker=SpeakerType(dialogue_data.get('initial_speaker', 'aspect')),
            scene_template=dialogue_data.get('scene_template', ''),
        )

        # Build context
        context_data = data.get('context', {})
        context = ScenarioContext(
            required_aspects=context_data.get('required_aspects', []),
            core_documents=context_data.get('core_documents', []),
            enhancement_documents=context_data.get('enhancement_documents', []),
            location=context_data.get('location', ''),
            thoughts=context_data.get('thoughts', []),
        )

        # Build seed actions using unified MemoryAction
        seed = []
        for seed_data in data.get('seed', []):
            seed.append(MemoryAction(
                action=seed_data.get('action', 'load_conversation'),
                target=seed_data.get('target', 'current'),
                document_types=seed_data.get('document_types'),
                exclude_types=seed_data.get('exclude_types'),
                top_n=seed_data.get('top_n'),
                min_memories=seed_data.get('min_memories'),
                conversation_id=seed_data.get('conversation_id'),
                by=seed_data.get('by'),
                direction=seed_data.get('direction'),
                match=seed_data.get('match'),
            ))

        # Build step definitions
        steps = {}
        for step_id, step_data in data.get('steps', {}).items():
            # Parse speaker
            speaker_data = step_data.get('speaker', {})
            if isinstance(speaker_data, str):
                # Handle shorthand: "aspect" or "persona"
                speaker = DialogueSpeaker(type=SpeakerType(speaker_data))
            else:
                speaker = DialogueSpeaker(
                    type=SpeakerType(speaker_data.get('type', 'persona')),
                    aspect_name=speaker_data.get('aspect_name'),
                )

            # Parse config
            config_data = step_data.get('config', {})
            config = StepConfig(
                max_tokens=config_data.get('max_tokens', 1024),
                use_guidance=config_data.get('use_guidance', False),
                is_thought=config_data.get('is_thought', False),
                is_codex=config_data.get('is_codex', False),
                temperature=config_data.get('temperature'),
                model_override=config_data.get('model_override'),
            )

            # Parse output
            output_data = step_data.get('output', {})
            output = StepOutput(
                document_type=output_data.get('document_type', 'step'),
                weight=output_data.get('weight', 1.0),
                add_to_turns=output_data.get('add_to_turns', True),
            )

            # Parse context DSL using unified MemoryAction
            # (memory: config is deprecated - use context: with search_memories action)
            context_actions = None
            if 'context' in step_data:
                context_actions = [
                    MemoryAction(**action_data)
                    for action_data in step_data['context']
                ]

            steps[step_id] = DialogueStep(
                id=step_id,
                speaker=speaker,
                guidance=step_data.get('guidance', ''),
                config=config,
                output=output,
                context=context_actions,
                next=step_data.get('next', []),
            )

        return cls(
            name=data.get('name', 'unnamed'),
            version=data.get('version', 2),
            description=data.get('description', ''),
            dialogue=dialogue,
            context=context,
            seed=seed,
            steps=steps,
        )

    @classmethod
    def load(cls, name: str, scenarios_dir: Optional[Path] = None) -> "DialogueStrategy":
        """
        Load strategy by name from scenarios directory.

        Args:
            name: Strategy name (without .yaml extension)
            scenarios_dir: Directory containing strategy files (default: config/scenario/)

        Returns:
            Configured DialogueStrategy instance
        """
        if scenarios_dir is None:
            scenarios_dir = Path("config/scenario")
        return cls.from_yaml(scenarios_dir / f"{name}.yaml")

    def get_step(self, step_id: str) -> DialogueStep:
        """Get step by ID.

        Args:
            step_id: The step identifier

        Returns:
            The DialogueStep definition

        Raises:
            KeyError: If step not found
        """
        if step_id not in self.steps:
            raise KeyError(f"Step '{step_id}' not found in strategy '{self.name}'")
        return self.steps[step_id]

    def get_root_steps(self) -> list[str]:
        """Get entry point steps (steps with no dependencies).

        Returns:
            List of step IDs that have no incoming edges
        """
        # Find steps that are not referenced in any 'next' lists
        referenced = set()
        for step in self.steps.values():
            referenced.update(step.next)

        roots = [
            step_id for step_id in self.steps.keys()
            if step_id not in referenced
        ]

        # If no roots found (all steps are referenced), return first step
        if not roots and self.steps:
            roots = [next(iter(self.steps.keys()))]

        return roots

    def get_execution_order(self) -> list[str]:
        """Get steps in topological execution order.

        Returns:
            List of step IDs in order of execution
        """
        order = []
        visited = set()

        def visit(step_id: str):
            if step_id in visited:
                return
            visited.add(step_id)
            order.append(step_id)
            step = self.steps.get(step_id)
            if step:
                for next_id in step.next:
                    visit(next_id)

        for root in self.get_root_steps():
            visit(root)

        return order

    def render_scene(self, context: dict) -> str:
        """Render scene template with Jinja2 context.

        Args:
            context: Dictionary of template variables

        Returns:
            Rendered scene string
        """
        if not self.dialogue.scene_template:
            return ""
        template = self._jinja_env.from_string(self.dialogue.scene_template)
        return template.render(**context)

    def render_guidance(self, step: DialogueStep, context: dict) -> str:
        """Render step guidance with Jinja2 context.

        Args:
            step: The step definition
            context: Dictionary of template variables

        Returns:
            Rendered guidance string
        """
        if not step.guidance:
            return ""
        template = self._jinja_env.from_string(step.guidance)
        return template.render(**context)
