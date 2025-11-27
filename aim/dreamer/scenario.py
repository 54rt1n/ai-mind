# aim/dreamer/scenario.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""YAML scenario loading, validation, and Jinja2 template rendering."""

from pathlib import Path
from typing import Optional
import yaml
from jinja2 import Environment, BaseLoader, StrictUndefined

from .models import Scenario, PipelineState


def load_scenario(name: str, scenarios_dir: Optional[Path] = None) -> Scenario:
    """
    Load and validate a scenario from YAML file.

    Args:
        name: Name of the scenario (without .yaml extension)
        scenarios_dir: Directory containing scenario files (defaults to config/scenario/)

    Returns:
        Validated Scenario model

    Raises:
        FileNotFoundError: If scenario file doesn't exist
        yaml.YAMLError: If YAML is invalid
        pydantic.ValidationError: If scenario structure is invalid
    """
    if scenarios_dir is None:
        scenarios_dir = Path("config/scenario")

    scenario_path = scenarios_dir / f"{name}.yaml"

    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {scenario_path}")

    with open(scenario_path, 'r') as f:
        data = yaml.safe_load(f)

    # Validate against Pydantic model
    scenario = Scenario(**data)

    return scenario


def get_jinja_environment() -> Environment:
    """
    Create configured Jinja2 environment for prompt rendering.

    Returns:
        Configured Jinja2 Environment with:
        - StrictUndefined for missing variable errors
        - BaseLoader for string templates
        - No autoescape (we're generating prompts, not HTML)
    """
    env = Environment(
        loader=BaseLoader(),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return env


def render_template(template: str, context: dict) -> str:
    """
    Render a Jinja2 template string with the given context.

    Args:
        template: Jinja2 template string
        context: Dictionary of variables to inject

    Returns:
        Rendered template string

    Raises:
        jinja2.UndefinedError: If template references undefined variables
        jinja2.TemplateSyntaxError: If template syntax is invalid
    """
    env = get_jinja_environment()
    jinja_template = env.from_string(template)
    return jinja_template.render(**context)


def build_template_context(
    state: PipelineState,
    scenario: Scenario,
    persona: 'Persona',  # Type hint as string to avoid circular import
) -> dict:
    """
    Build Jinja2 context from Persona and pipeline state.

    Creates a context dictionary with:
    - persona: Full Persona object
    - pronouns: Persona pronouns dictionary
    - step_num: Current step counter
    - guidance: Optional guidance text
    - query_text: Optional query text
    - conversation_id: Current conversation ID
    - {aspect_name}: Each required aspect from the persona

    Args:
        state: Current pipeline state
        scenario: Scenario definition with required_aspects
        persona: Persona object with aspects

    Returns:
        Dictionary suitable for Jinja2 template rendering
    """
    ctx = {
        # Persona
        'persona': persona,
        'pronouns': persona.pronouns,

        # State
        'step_num': state.step_counter,
        'guidance': state.guidance,
        'query_text': state.query_text,

        # Convenience
        'conversation_id': state.conversation_id,
    }

    # Add required aspects as top-level variables
    for aspect_name in scenario.context.required_aspects:
        if aspect_name in persona.aspects:
            ctx[aspect_name] = persona.aspects[aspect_name]

    return ctx
