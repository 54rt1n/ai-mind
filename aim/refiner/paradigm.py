# aim/refiner/paradigm.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Paradigm domain object.

A Paradigm is an exploration strategy that encapsulates:
- Document gathering configuration
- Scenario routing logic
- Prompt building for both selection and validation phases
- Tool definitions for LLM interaction

Config (YAML) → Paradigm (strategy) → Engine (context)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from aim.tool.dto import Tool, ToolFunction, ToolFunctionParameters
from aim.tool.formatting import ToolUser
from aim.utils.xml import XmlFormatter
from aim.agents.aspects import (
    get_aspect,
    create_default_aspect,
    build_librarian_scene,
    build_dreamer_scene,
    build_philosopher_scene,
    build_writer_scene,
    build_psychologist_scene,
)

if TYPE_CHECKING:
    from aim.agents.persona import Persona

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "paradigm"

# Map aspect names to scene builders
SCENE_BUILDERS = {
    "librarian": build_librarian_scene,
    "dreamer": build_dreamer_scene,
    "philosopher": build_philosopher_scene,
    "writer": build_writer_scene,
    "psychologist": build_psychologist_scene,
}


@dataclass
class Paradigm:
    """
    Domain object representing an exploration paradigm.

    Loaded from YAML config. Encapsulates all paradigm-specific behavior.
    """

    name: str
    aspect: str
    doc_types: list[str]
    approach_doc_types: dict[str, list[str]]
    scenario: Optional[str]
    scenarios_by_approach: dict[str, str]
    prompts: dict
    tools_data: dict

    # Cache for Tool objects
    _tools_cache: dict = None

    @classmethod
    def load(cls, name: str) -> "Paradigm":
        """
        Load paradigm from YAML config.

        Raises ValueError if config is missing or invalid.
        """
        path = CONFIG_DIR / f"{name}.yaml"
        if not path.exists():
            raise ValueError(f"No config found for paradigm '{name}' at {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required = ["aspect", "doc_types", "prompts", "tools"]
        for field in required:
            if field not in data:
                raise ValueError(f"Paradigm '{name}' missing required field: {field}")

        # Validate prompts structure
        prompts = data["prompts"]
        for phase in ["selection", "validation"]:
            if phase not in prompts:
                raise ValueError(f"Paradigm '{name}' missing prompts.{phase}")
            for field in ["scene", "think", "instructions"]:
                if field not in prompts[phase]:
                    raise ValueError(f"Paradigm '{name}' missing prompts.{phase}.{field}")

        # Handle approach_doc_types - can be dict or list
        approach_doc_types = data.get("approach_doc_types", {})
        if isinstance(approach_doc_types, list):
            approach_doc_types = {"default": approach_doc_types}

        return cls(
            name=name,
            aspect=data["aspect"],
            doc_types=data["doc_types"],
            approach_doc_types=approach_doc_types,
            scenario=data.get("scenario"),
            scenarios_by_approach=data.get("scenarios_by_approach", {}),
            prompts=prompts,
            tools_data=data["tools"],
            _tools_cache={},
        )

    @classmethod
    def available(cls, exclude: Optional[list[str]] = None) -> list[str]:
        """List available paradigm names from config directory."""
        if not CONFIG_DIR.exists():
            return []

        names = [p.stem for p in CONFIG_DIR.glob("*.yaml")]
        if exclude:
            names = [n for n in names if n not in exclude]
        return sorted(names)

    def get_scenario(self, approach: Optional[str] = None) -> str:
        """
        Get scenario for the given approach.

        Priority:
        1. scenarios_by_approach[approach] if exists
        2. self.scenario if set
        3. approach itself as scenario name
        """
        if approach and approach in self.scenarios_by_approach:
            return self.scenarios_by_approach[approach]
        if self.scenario:
            return self.scenario
        return approach or self.name

    def get_approach_doc_types(self, approach: str) -> list[str]:
        """Get document types for targeted gathering."""
        if approach in self.approach_doc_types:
            return self.approach_doc_types[approach]
        if "default" in self.approach_doc_types:
            return self.approach_doc_types["default"]
        return self.doc_types

    def get_tool(self, name: str) -> Tool:
        """Get a tool definition by name."""
        if name in self._tools_cache:
            return self._tools_cache[name]

        if name not in self.tools_data:
            raise ValueError(f"Paradigm '{self.name}' has no tool '{name}'")

        tool = self._parse_tool(name, self.tools_data[name])
        self._tools_cache[name] = tool
        return tool

    def get_select_tool(self) -> Tool:
        """Get the select_topic tool."""
        return self.get_tool("select_topic")

    def get_validate_tool(self) -> Tool:
        """Get the validate_exploration tool."""
        return self.get_tool("validate_exploration")

    def build_selection_prompt(
        self,
        documents: list[dict],
        persona: "Persona",
        seed_query: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Build Phase 1 (selection) prompt.

        Returns:
            Tuple of (system_message, user_message)
        """
        # Build system prompt with persona and tool
        xml = XmlFormatter()
        xml = persona.xml_decorator(xml, disable_pif=True, disable_guidance=True)

        tool = self.get_select_tool()
        tool_user = ToolUser([tool])
        xml = tool_user.xml_decorator(xml)

        system_prompt = xml.render()

        # Get aspect for scene building
        aspect_obj = get_aspect(persona, self.aspect)
        if not aspect_obj:
            aspect_obj = create_default_aspect(self.aspect)

        # Build scene from aspect
        scene_builder = SCENE_BUILDERS.get(self.aspect)
        if scene_builder:
            scene = scene_builder(persona, aspect_obj)
        else:
            scene = f"*You find yourself in {aspect_obj.name}'s presence*"

        # Format documents
        docs_formatted = self._format_documents(documents)

        # Get prompt content from config
        prompt_config = self.prompts["selection"]

        # Build format kwargs for template substitution
        format_kwargs = {
            "persona_name": persona.name,
            "aspect_name": aspect_obj.name,
            "subj": persona.pronouns.get("subj", "they"),
            "obj": persona.pronouns.get("obj", "them"),
            "poss": persona.pronouns.get("poss", "their"),
            "Subj": persona.pronouns.get("subj", "they").capitalize(),
            "Poss": persona.pronouns.get("poss", "their").capitalize(),
            "voice_style": aspect_obj.voice_style or "a thoughtful cadence",
        }

        scene_text = prompt_config["scene"].format(**format_kwargs)
        think_text = prompt_config["think"]
        instructions_text = prompt_config["instructions"].format(**format_kwargs)

        # Add seed query hint if provided
        seed_hint = ""
        if seed_query:
            seed_hint = f"\n\n*A whisper echoes: perhaps consider '{seed_query}'...*\n"

        user_prompt = f"""{scene}

<context_documents>
{docs_formatted}
</context_documents>

{scene_text}
{seed_hint}
<think>
{think_text}
</think>

<instructions>
{instructions_text}
</instructions>"""

        return system_prompt, user_prompt

    def build_validation_prompt(
        self,
        documents: list[dict],
        persona: "Persona",
        topic: str,
        approach: str,
    ) -> tuple[str, str]:
        """
        Build Phase 2 (validation) prompt.

        Returns:
            Tuple of (system_message, user_message)
        """
        # Build system prompt with persona and tool
        xml = XmlFormatter()
        xml = persona.xml_decorator(xml, disable_pif=True, disable_guidance=True)

        tool = self.get_validate_tool()
        tool_user = ToolUser([tool])
        xml = tool_user.xml_decorator(xml)

        system_prompt = xml.render()

        # Get aspect for scene building
        aspect_obj = get_aspect(persona, self.aspect)
        if not aspect_obj:
            aspect_obj = create_default_aspect(self.aspect)

        # Build scene from aspect
        scene_builder = SCENE_BUILDERS.get(self.aspect)
        if scene_builder:
            scene = scene_builder(persona, aspect_obj)
        else:
            scene = f"*You find yourself in {aspect_obj.name}'s presence*"

        # Format documents
        docs_formatted = self._format_documents(documents)

        # Get prompt content from config
        prompt_config = self.prompts["validation"]

        # Build format kwargs for template substitution
        format_kwargs = {
            "persona_name": persona.name,
            "aspect_name": aspect_obj.name,
            "subj": persona.pronouns.get("subj", "they"),
            "obj": persona.pronouns.get("obj", "them"),
            "poss": persona.pronouns.get("poss", "their"),
            "Subj": persona.pronouns.get("subj", "they").capitalize(),
            "Poss": persona.pronouns.get("poss", "their").capitalize(),
            "voice_style": aspect_obj.voice_style or "a thoughtful cadence",
            "topic": topic,
            "approach": approach,
        }

        scene_text = prompt_config["scene"].format(**format_kwargs)
        think_text = prompt_config["think"]
        instructions_text = prompt_config["instructions"].format(**format_kwargs)

        user_prompt = f"""{scene}

<topic>{topic}</topic>
<approach>{approach}</approach>

<focused_context>
{docs_formatted}
</focused_context>

{scene_text}

<think>
{think_text}
</think>

<instructions>
{instructions_text}
</instructions>"""

        return system_prompt, user_prompt

    def _format_documents(self, documents: list[dict]) -> str:
        """Format documents for prompt inclusion."""
        if not documents:
            return "(No documents available.)"

        formatted = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            doc_type = doc.get("document_type", "unknown")
            date = doc.get("timestamp", "")
            formatted.append(f"[{i+1}] ({doc_type}) {date}\n{content}")

        return "\n\n".join(formatted)

    def _parse_tool(self, name: str, tool_data: dict) -> Tool:
        """Parse tool definition from config."""
        params = tool_data.get("parameters", {})

        properties = {}
        for param_name, param_def in params.items():
            prop = {"type": param_def.get("type", "string")}
            if "description" in param_def:
                prop["description"] = param_def["description"]
            if "enum" in param_def:
                prop["enum"] = param_def["enum"]
            properties[param_name] = prop

        examples = []
        for ex in tool_data.get("examples", []):
            examples.append({name: ex})

        return Tool(
            type="refiner",
            function=ToolFunction(
                name=name,
                description=tool_data.get("description", ""),
                parameters=ToolFunctionParameters(
                    type="object",
                    properties=properties,
                    required=tool_data.get("required", []),
                    examples=examples,
                ),
            ),
        )
