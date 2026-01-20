# aim/dreamer/core/framework.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""ScenarioFramework - immutable scenario definition for strategy-based execution."""

from typing import Optional, Any
from pydantic import BaseModel, Field

from .models import ScenarioTool, NewStepDefinition, SpeakerType


class DialogueConfig(BaseModel):
    """Dialogue-specific configuration at the scenario level.

    Defines the primary aspect guiding the dialogue, who speaks first,
    and optional scene template that can be shared across steps.
    """
    primary_aspect: str
    """Name of the primary aspect guiding the dialogue (e.g., 'coder')."""

    initial_speaker: SpeakerType = SpeakerType.ASPECT
    """Who speaks first in the dialogue."""

    scene_template: str = ""
    """Default Jinja2 template for scene descriptions."""

    required_aspects: list[str] = Field(default_factory=list)
    """Aspects required for this dialogue scenario."""


class ScenarioFramework(BaseModel):
    """Immutable scenario definition. Serialized to Redis alongside ScenarioState.

    Built once from YAML by ScenarioBuilder, validated, then persisted.
    Never re-read from YAML after dream starts.

    Attributes:
        name: Scenario identifier (e.g., "exploration_brainstorm")
        version: Schema version for compatibility checking
        description: Human-readable description
        first_step: Entry point step ID
        steps: Step definitions keyed by step_id
        tools: Tool definitions keyed by tool name
    """

    # Identity
    name: str
    version: int = 1
    description: Optional[str] = None

    # Entry point
    first_step: str

    # Steps (step_id -> definition)
    steps: dict[str, NewStepDefinition]

    # Tools (tool_name -> definition)
    tools: dict[str, ScenarioTool] = Field(default_factory=dict)

    # Dialogue configuration (only for dialogue scenarios)
    dialogue: Optional[DialogueConfig] = None

    def get_tools(self, tool_names: list[str]) -> list:
        """Get Tool objects for the given names, ready for ToolUser.

        Args:
            tool_names: Names of tools to load (from step definition)

        Returns:
            List of Tool objects

        Raises:
            ValueError: If tool name not found
        """
        from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters

        tools = []
        for name in tool_names:
            if name not in self.tools:
                raise ValueError(f"Tool '{name}' not defined in scenario '{self.name}'")

            tool_def = self.tools[name]
            tool = Tool(
                type="scenario",
                function=ToolFunction(
                    name=name,
                    description=tool_def.description,
                    parameters=ToolFunctionParameters(
                        type="object",
                        properties=tool_def.parameters,
                        required=tool_def.required,
                    )
                )
            )
            tools.append(tool)
        return tools

    def get_root_steps(self) -> list[str]:
        """Get step IDs with no dependencies."""
        return [
            step_id for step_id, step_def in self.steps.items()
            if not step_def.depends_on
        ]

    def topological_order(self) -> list[str]:
        """Return step IDs in dependency order (Kahn's algorithm)."""
        # Build in-degree map
        in_degree = {
            step_id: len(step_def.depends_on)
            for step_id, step_def in self.steps.items()
        }

        # Start with root steps (no dependencies)
        queue = [s for s, d in in_degree.items() if d == 0]
        result = []

        while queue:
            step_id = queue.pop(0)
            result.append(step_id)

            # Decrease in-degree for steps that depend on this one
            for other_id, other_def in self.steps.items():
                if step_id in other_def.depends_on:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(result) != len(self.steps):
            raise ValueError("Cycle detected in step dependencies")

        return result

    def validate_goto_targets(self) -> None:
        """Validate all goto targets exist. Called at load time.

        Raises:
            ValueError: If any step references an invalid target
        """
        valid_targets = set(self.steps.keys()) | {"end", "abort"}

        for step_id, step_def in self.steps.items():
            # Check next field (non-tool-calling steps)
            if hasattr(step_def, 'next'):
                for target in step_def.next:
                    if target not in valid_targets:
                        raise ValueError(
                            f"Step '{step_id}' has invalid next target '{target}'"
                        )

            # Check next_conditions (tool_calling steps)
            if hasattr(step_def, 'next_conditions'):
                for cond in step_def.next_conditions:
                    target = cond.goto or cond.default
                    if target and target not in valid_targets:
                        raise ValueError(
                            f"Step '{step_id}' has invalid condition target '{target}'"
                        )

    def validate_tools(self) -> None:
        """Validate all tool references exist.

        Raises:
            ValueError: If any step references an undefined tool
        """
        for step_id, step_def in self.steps.items():
            if hasattr(step_def, 'tools'):
                for tool_name in step_def.tools:
                    if tool_name not in self.tools:
                        raise ValueError(
                            f"Step '{step_id}' references undefined tool '{tool_name}'"
                        )

    def validate(self) -> None:
        """Run all validations.

        Raises:
            ValueError: If any validation fails
        """
        self.validate_goto_targets()
        self.validate_tools()
