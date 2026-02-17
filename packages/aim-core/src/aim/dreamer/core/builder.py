# aim/dreamer/core/builder.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""ScenarioBuilder - loads YAML and builds ScenarioFramework objects."""

from pathlib import Path
from typing import Optional, Any
import logging
import yaml
from pydantic import TypeAdapter

from .models import (
    ScenarioTool,
    MemoryAction,
    StepConfig,
    StepOutput,
    NewStepDefinition,
    Condition,
    ContextOnlyStepDefinition,
    StandardStepDefinition,
    ToolCallingStepDefinition,
    RenderingStepDefinition,
    DialogueStepDefinition,
    DialogueSpeaker,
    SpeakerType,
)
from .framework import ScenarioFramework, DialogueConfig


logger = logging.getLogger(__name__)


class ScenarioBuilder:
    """Builder for creating ScenarioFramework from YAML files.

    Usage:
        builder = ScenarioBuilder()
        framework = builder.load("exploration_brainstorm")
        # or
        framework = builder.from_dict(yaml_data)
    """

    def __init__(self, scenarios_dir: Optional[Path] = None):
        """Initialize builder with scenario directory.

        Args:
            scenarios_dir: Directory containing scenario YAML files.
                           Defaults to config/scenario/
        """
        self.scenarios_dir = scenarios_dir or Path("config/scenario")
        self._step_adapter = TypeAdapter(NewStepDefinition)

    def load(self, name: str) -> ScenarioFramework:
        """Load scenario from YAML file.

        Args:
            name: Scenario name (without .yaml extension)

        Returns:
            Validated ScenarioFramework

        Raises:
            FileNotFoundError: If scenario file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValueError: If validation fails
        """
        scenario_path = self.scenarios_dir / f"{name}.yaml"

        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

        with open(scenario_path, 'r') as f:
            data = yaml.safe_load(f)

        return self.from_dict(data)

    def from_dict(self, data: dict) -> ScenarioFramework:
        """Build ScenarioFramework from dictionary (e.g., parsed YAML).

        Args:
            data: Scenario dictionary with name, steps, tools, etc.

        Returns:
            Validated ScenarioFramework

        Raises:
            ValueError: If validation fails
        """
        # Parse tools
        tools = self._parse_tools(data.get('tools', {}))

        # Parse seed memory actions (scenario-level initial context)
        seed = [
            MemoryAction(**action) if isinstance(action, dict) else action
            for action in data.get('seed', [])
        ]

        # Parse dialogue config if present (pass context for required_aspects)
        dialogue_config = self._parse_dialogue_config(
            data.get('dialogue'), data.get('context')
        )

        # Extract required_aspects from context section (for non-dialogue scenarios)
        required_aspects = data.get('context', {}).get('required_aspects', [])

        # Parse steps
        steps = self._parse_steps(data.get('steps', {}))

        # Determine first step if not explicit
        first_step = data.get('first_step')
        if not first_step and steps:
            # Default to first step in dict (or find root step)
            first_step = next(iter(steps.keys()))

        # Build framework
        framework = ScenarioFramework(
            name=data.get('name', ''),
            version=data.get('version', 1),
            description=data.get('description'),
            first_step=first_step or '',
            steps=steps,
            tools=tools,
            seed=seed,
            dialogue=dialogue_config,
            required_aspects=required_aspects,
        )

        # Validate at build time
        framework.validate()

        logger.info(
            f"Built ScenarioFramework '{framework.name}' with "
            f"{len(framework.steps)} steps, {len(framework.tools)} tools"
            f"{', dialogue mode' if dialogue_config else ''}"
        )

        return framework

    def _parse_dialogue_config(
        self, dialogue_data: Optional[dict], context_data: Optional[dict] = None
    ) -> Optional[DialogueConfig]:
        """Parse dialogue section into DialogueConfig.

        Args:
            dialogue_data: Dict with dialogue configuration or None
            context_data: Dict with context configuration (for required_aspects)

        Returns:
            DialogueConfig or None
        """
        if not dialogue_data:
            return None

        # Parse initial_speaker as SpeakerType enum
        initial_speaker_str = dialogue_data.get('initial_speaker', 'aspect')
        initial_speaker = SpeakerType(initial_speaker_str)

        # Get required_aspects from dialogue section first, fall back to context section
        required_aspects = dialogue_data.get('required_aspects', [])
        if not required_aspects and context_data:
            required_aspects = context_data.get('required_aspects', [])

        return DialogueConfig(
            primary_aspect=dialogue_data.get('primary_aspect', 'coder'),
            initial_speaker=initial_speaker,
            scene_template=dialogue_data.get('scene_template', ''),
            required_aspects=required_aspects,
        )

    def _parse_tools(self, tools_data: dict[str, Any]) -> dict[str, ScenarioTool]:
        """Parse tools section into ScenarioTool objects.

        Args:
            tools_data: Dict mapping tool_name -> tool definition

        Returns:
            Dict mapping tool_name -> ScenarioTool
        """
        tools = {}
        for tool_name, tool_def in tools_data.items():
            tools[tool_name] = ScenarioTool(
                name=tool_name,
                description=tool_def.get('description', ''),
                parameters=tool_def.get('parameters', {}),
                required=tool_def.get('required', []),
            )
        return tools

    def _parse_steps(self, steps_data: dict[str, Any]) -> dict[str, NewStepDefinition]:
        """Parse steps section into typed step definitions.

        Uses Pydantic's discriminated union to parse each step into
        the correct StepDefinition subclass based on 'type' field.

        Args:
            steps_data: Dict mapping step_id -> step definition

        Returns:
            Dict mapping step_id -> NewStepDefinition
        """
        steps = {}
        for step_id, step_def in steps_data.items():
            step = self._parse_step(step_id, step_def)
            steps[step_id] = step
        return steps

    def _parse_step(self, step_id: str, step_data: dict[str, Any]) -> NewStepDefinition:
        """Parse a single step definition.

        Injects step_id into the data and uses discriminated union parsing.

        Args:
            step_id: The step's ID (from dict key)
            step_data: Step definition data

        Returns:
            Appropriate StepDefinition subclass instance
        """
        # Inject id from dict key
        step_data = {**step_data, 'id': step_id}

        # Infer type if not specified
        if 'type' not in step_data:
            step_data['type'] = self._infer_step_type(step_data)

        # Parse context actions if present
        if 'context' in step_data and step_data['context']:
            step_data['context'] = [
                MemoryAction(**action) if isinstance(action, dict) else action
                for action in step_data['context']
            ]

        # Parse config if present (as dict)
        if 'config' in step_data and isinstance(step_data['config'], dict):
            step_data['config'] = StepConfig(**step_data['config'])

        # Parse output if present (as dict)
        if 'output' in step_data and isinstance(step_data['output'], dict):
            step_data['output'] = StepOutput(**step_data['output'])

        # Parse next_conditions if present
        if 'next_conditions' in step_data:
            step_data['next_conditions'] = [
                Condition(**cond) if isinstance(cond, dict) else cond
                for cond in step_data['next_conditions']
            ]

        # Parse speaker field for dialogue steps
        if 'speaker' in step_data and isinstance(step_data['speaker'], dict):
            speaker_data = step_data['speaker']
            step_data['speaker'] = DialogueSpeaker(
                type=SpeakerType(speaker_data.get('type', 'aspect')),
                aspect_name=speaker_data.get('aspect_name'),
            )

        # Use TypeAdapter for discriminated union parsing
        return self._step_adapter.validate_python(step_data)

    def _infer_step_type(self, step_data: dict[str, Any]) -> str:
        """Infer step type from fields present.

        Args:
            step_data: Step definition data

        Returns:
            Inferred type string
        """
        # dialogue: has speaker field
        if 'speaker' in step_data:
            return 'dialogue'

        # tool_calling: has tools and next_conditions
        if 'tools' in step_data and 'next_conditions' in step_data:
            return 'tool_calling'

        # rendering: has template
        if 'template' in step_data:
            return 'rendering'

        # context_only: has context but no prompt and no output
        if 'context' in step_data and 'prompt' not in step_data and 'output' not in step_data:
            return 'context_only'

        # Default to standard
        return 'standard'


def load_scenario_framework(
    name: str,
    scenarios_dir: Optional[Path] = None,
) -> ScenarioFramework:
    """Convenience function to load a scenario framework.

    Args:
        name: Scenario name (without .yaml extension)
        scenarios_dir: Directory containing scenario files

    Returns:
        Validated ScenarioFramework
    """
    builder = ScenarioBuilder(scenarios_dir)
    return builder.load(name)
