#!/usr/bin/env python3
"""
Validate dialogue scenario configuration files.

Usage:
    python scripts/validate_dialogue.py config/scenario/analysis_dialogue.yaml
    python scripts/validate_dialogue.py config/scenario/daydream_dialogue.yaml
    python scripts/validate_dialogue.py config/scenario/*.yaml  # validate all
"""

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, TemplateSyntaxError, meta

# Valid memory DSL action types
VALID_MEMORY_ACTIONS = {
    "load_conversation",
    "get_memory",
    "search_memories",
    "sort",
    "filter",
    "truncate",
    "drop",
    "flush",
    "clear",
}


class ValidationResult:
    """Collects validation errors and warnings."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def print_report(self) -> None:
        """Print validation results."""
        status = "PASS" if self.is_valid else "FAIL"
        print(f"\n{'='*60}")
        print(f"File: {self.filepath}")
        print(f"Status: {status}")
        print(f"{'='*60}")

        if self.errors:
            print(f"\nErrors ({len(self.errors)}):")
            for err in self.errors:
                print(f"  - {err}")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warn in self.warnings:
                print(f"  - {warn}")

        if not self.errors and not self.warnings:
            print("\n  All checks passed.")


def validate_structure(data: dict[str, Any], result: ValidationResult) -> bool:
    """Validate top-level structure of dialogue scenario."""
    required_fields = ["name", "version", "flow", "dialogue", "context", "steps"]

    for field in required_fields:
        if field not in data:
            result.error(f"Missing required top-level field: '{field}'")

    if data.get("flow") != "dialogue":
        result.error(f"Field 'flow' must be 'dialogue', got: '{data.get('flow')}'")

    # Validate dialogue block
    dialogue = data.get("dialogue", {})
    if not isinstance(dialogue, dict):
        result.error("'dialogue' must be a mapping")
        return False

    dialogue_fields = ["primary_aspect", "initial_speaker", "scene_template"]
    for field in dialogue_fields:
        if field not in dialogue:
            result.error(f"Missing required dialogue field: '{field}'")

    if dialogue.get("initial_speaker") not in ("aspect", "persona"):
        result.error(
            f"'initial_speaker' must be 'aspect' or 'persona', "
            f"got: '{dialogue.get('initial_speaker')}'"
        )

    # Validate context block
    context = data.get("context", {})
    if not isinstance(context, dict):
        result.error("'context' must be a mapping")
        return False

    if "required_aspects" not in context:
        result.error("Missing 'required_aspects' in context")

    return result.is_valid


def validate_steps(data: dict[str, Any], result: ValidationResult) -> None:
    """Validate step definitions."""
    steps = data.get("steps", {})
    if not isinstance(steps, dict):
        result.error("'steps' must be a mapping")
        return

    if not steps:
        result.error("No steps defined")
        return

    required_aspects = set(data.get("context", {}).get("required_aspects", []))
    step_ids = set(steps.keys())
    terminal_steps = []
    prev_speaker_type = None

    for step_id, step in steps.items():
        if not isinstance(step, dict):
            result.error(f"Step '{step_id}' must be a mapping")
            continue

        # Validate speaker
        speaker = step.get("speaker")
        if not speaker:
            result.error(f"Step '{step_id}': missing 'speaker' field")
            continue

        if not isinstance(speaker, dict):
            result.error(f"Step '{step_id}': 'speaker' must be a mapping")
            continue

        speaker_type = speaker.get("type")
        if speaker_type not in ("aspect", "persona"):
            result.error(
                f"Step '{step_id}': speaker type must be 'aspect' or 'persona', "
                f"got: '{speaker_type}'"
            )

        if speaker_type == "aspect":
            aspect_name = speaker.get("aspect_name")
            if not aspect_name:
                result.error(
                    f"Step '{step_id}': aspect speaker missing 'aspect_name'"
                )
            elif aspect_name not in required_aspects:
                result.error(
                    f"Step '{step_id}': aspect '{aspect_name}' not in required_aspects: "
                    f"{sorted(required_aspects)}"
                )

        # Check for speaker alternation (warning only)
        if prev_speaker_type is not None:
            if prev_speaker_type == speaker_type:
                # Get previous step's next to see if this is actually a successor
                # This is a simplified check - the actual flow might differ
                result.warn(
                    f"Step '{step_id}': consecutive {speaker_type} speakers "
                    f"(may be intentional if not direct successors)"
                )
        prev_speaker_type = speaker_type

        # Validate next references
        next_steps = step.get("next", [])
        if not isinstance(next_steps, list):
            result.error(f"Step '{step_id}': 'next' must be a list")
            continue

        if not next_steps:
            terminal_steps.append(step_id)

        for next_id in next_steps:
            if next_id not in step_ids:
                result.error(
                    f"Step '{step_id}': references non-existent step '{next_id}'"
                )

        # Validate output
        output = step.get("output", {})
        if not output.get("document_type"):
            result.error(f"Step '{step_id}': missing output.document_type")

        weight = output.get("weight")
        if weight is not None:
            if not isinstance(weight, (int, float)):
                result.error(
                    f"Step '{step_id}': output.weight must be a number, "
                    f"got: {type(weight).__name__}"
                )
            elif weight < 0:
                result.warn(f"Step '{step_id}': negative weight ({weight})")

        # Validate config
        config = step.get("config", {})
        if not config.get("max_tokens"):
            result.warn(f"Step '{step_id}': no max_tokens specified in config")

    # Check for terminal steps
    if not terminal_steps:
        result.error("No terminal steps (steps with next: [])")

    # Validate DAG (check for cycles)
    validate_dag(steps, result)


def validate_dag(steps: dict[str, Any], result: ValidationResult) -> None:
    """Validate that steps form a valid DAG (no cycles)."""
    # Build adjacency list
    graph: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {step_id: 0 for step_id in steps}

    for step_id, step in steps.items():
        next_steps = step.get("next", [])
        if isinstance(next_steps, list):
            graph[step_id] = next_steps
            for next_id in next_steps:
                if next_id in in_degree:
                    in_degree[next_id] += 1
        else:
            graph[step_id] = []

    # Kahn's algorithm for topological sort
    queue = deque([node for node, degree in in_degree.items() if degree == 0])
    visited_count = 0

    while queue:
        node = queue.popleft()
        visited_count += 1

        for neighbor in graph.get(node, []):
            if neighbor in in_degree:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

    if visited_count != len(steps):
        result.error(
            f"Step DAG contains a cycle (visited {visited_count} of {len(steps)} steps)"
        )

    # Check for unreachable steps (no incoming edges and not a start step)
    start_candidates = [step_id for step_id, step in steps.items()
                        if not any(step_id in s.get("next", []) for s in steps.values())]

    if len(start_candidates) > 1:
        result.warn(
            f"Multiple potential start steps (no incoming edges): {start_candidates}"
        )


def validate_templates(data: dict[str, Any], result: ValidationResult) -> None:
    """Validate Jinja2 templates in prompts and guidance."""
    env = Environment()

    # Known template variables
    known_vars = {
        "persona", "pronouns", "step_num", "guidance", "query_text",
        "conversation_id"
    }

    # Add aspect names as known variables
    required_aspects = data.get("context", {}).get("required_aspects", [])
    known_vars.update(required_aspects)

    steps = data.get("steps", {})
    for step_id, step in steps.items():
        if not isinstance(step, dict):
            continue

        # Check for deprecated prompt field
        if "prompt" in step:
            result.warn(
                f"Step '{step_id}': deprecated 'prompt' field. "
                f"Use 'guidance' instead (prompt+guidance consolidated into guidance)"
            )

        # Check guidance template
        guidance = step.get("guidance", "")
        if guidance:
            try:
                ast = env.parse(guidance)
                undefined = meta.find_undeclared_variables(ast)
                unknown = undefined - known_vars
                if unknown:
                    result.warn(
                        f"Step '{step_id}' guidance: unknown template variables: {unknown}"
                    )
            except TemplateSyntaxError as e:
                result.error(f"Step '{step_id}' guidance: invalid Jinja2 syntax: {e}")

    # Check scene_template
    scene_template = data.get("dialogue", {}).get("scene_template", "")
    if scene_template:
        try:
            ast = env.parse(scene_template)
            undefined = meta.find_undeclared_variables(ast)
            unknown = undefined - known_vars
            if unknown:
                result.warn(
                    f"scene_template: unknown template variables: {unknown}"
                )
        except TemplateSyntaxError as e:
            result.error(f"scene_template: invalid Jinja2 syntax: {e}")


def validate_memory_dsl(data: dict[str, Any], result: ValidationResult) -> None:
    """Validate memory DSL usage in steps.

    Checks:
    - Warns on deprecated `memory:` config (should use context DSL)
    - Validates context actions have valid action types
    - Validates search_memories has required params
    - Validates flush placement for flush_before patterns
    """
    steps = data.get("steps", {})

    for step_id, step in steps.items():
        if not isinstance(step, dict):
            continue

        # Check for deprecated memory: config
        memory_config = step.get("memory")
        if memory_config and isinstance(memory_config, dict):
            top_n = memory_config.get("top_n", 0)
            if top_n > 0:
                result.error(
                    f"Step '{step_id}': deprecated 'memory:' config with top_n={top_n}. "
                    f"Migrate to context DSL: add 'search_memories' action to context"
                )
            flush_before = memory_config.get("flush_before", False)
            if flush_before:
                result.error(
                    f"Step '{step_id}': deprecated 'memory.flush_before'. "
                    f"Migrate to context DSL: add 'flush' action before 'search_memories'"
                )

        # Validate context DSL actions
        context = step.get("context")
        if context and isinstance(context, list):
            has_flush = False
            has_search_after_flush = False

            for i, action in enumerate(context):
                if not isinstance(action, dict):
                    result.error(
                        f"Step '{step_id}': context[{i}] must be a mapping, "
                        f"got {type(action).__name__}"
                    )
                    continue

                action_type = action.get("action")
                if not action_type:
                    result.error(
                        f"Step '{step_id}': context[{i}] missing 'action' field"
                    )
                    continue

                if action_type not in VALID_MEMORY_ACTIONS:
                    result.error(
                        f"Step '{step_id}': context[{i}] has invalid action '{action_type}'. "
                        f"Valid actions: {sorted(VALID_MEMORY_ACTIONS)}"
                    )
                    continue

                # Track flush/search ordering
                if action_type in ("flush", "clear"):
                    has_flush = True

                if action_type == "search_memories":
                    if has_flush:
                        has_search_after_flush = True

                    # Validate search_memories has top_n
                    top_n = action.get("top_n")
                    if not top_n or top_n <= 0:
                        result.warn(
                            f"Step '{step_id}': search_memories without top_n "
                            f"(will not retrieve any memories)"
                        )

                # Validate sort action
                if action_type == "sort":
                    sort_by = action.get("by")
                    if sort_by not in ("timestamp", "relevance"):
                        result.warn(
                            f"Step '{step_id}': sort action with by='{sort_by}'. "
                            f"Expected 'timestamp' or 'relevance'"
                        )


def validate_file(filepath: Path) -> ValidationResult:
    """Validate a single dialogue scenario file."""
    result = ValidationResult(str(filepath))

    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.error(f"YAML parse error: {e}")
        return result
    except FileNotFoundError:
        result.error(f"File not found: {filepath}")
        return result

    if not isinstance(data, dict):
        result.error("File must contain a YAML mapping at top level")
        return result

    # Always validate memory DSL (applies to all scenarios)
    validate_memory_dsl(data, result)

    # Skip additional dialogue validations for non-dialogue scenarios
    if data.get("flow") != "dialogue":
        result.warn("Not a dialogue scenario (flow != 'dialogue'), skipping dialogue-specific checks")
        return result

    # Run dialogue-specific validations
    if validate_structure(data, result):
        validate_steps(data, result)
        validate_templates(data, result)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate dialogue scenario configuration files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Path(s) to scenario YAML files"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only print failures"
    )
    args = parser.parse_args()

    all_valid = True
    results = []

    for filepath in args.files:
        if filepath.is_dir():
            # If directory, find all yaml files
            yaml_files = list(filepath.glob("*.yaml")) + list(filepath.glob("*.yml"))
            for f in yaml_files:
                results.append(validate_file(f))
        else:
            results.append(validate_file(filepath))

    for result in results:
        if not args.quiet or not result.is_valid:
            result.print_report()
        if not result.is_valid:
            all_valid = False

    # Summary
    print(f"\n{'='*60}")
    total = len(results)
    passed = sum(1 for r in results if r.is_valid)
    print(f"Summary: {passed}/{total} files passed validation")
    print(f"{'='*60}")

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
