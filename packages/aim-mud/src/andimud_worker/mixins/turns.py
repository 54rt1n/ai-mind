# aim/app/mud/worker/turns.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Turn processing for the MUD worker.

Handles full turn processing pipeline including decision and response phases.
Extracted from worker.py lines 779-1869
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from aim_mud_types import MUDAction, MUDTurnRequest
from aim.utils.think import extract_think_tags
from ..adapter import build_current_context
from aim_mud_types import MUDEvent, MUDTurn
from ..turns.validation import (
    resolve_target_name,
    resolve_move_location,
    is_superuser_persona,
    get_room_objects,
    get_inventory_items,
    get_valid_give_targets,
)
from ..turns.response import (
    normalize_response,
    has_emotional_state_header,
    parse_agent_action_response,
    sanitize_response,
)
from ..turns.decision import (
    validate_move,
    validate_take,
    validate_drop,
    validate_give,
)
from ..turns.processor import PhasedTurnProcessor, AgentTurnProcessor
from ..exceptions import AbortRequestedException

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class TurnsMixin:
    """Mixin for turn processing methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    def _init_agent_action_spec(self: "MUDAgentWorker") -> None:
        """Load @agent action specification from YAML.

        Originally from worker.py lines 833-854
        """
        tool_file = Path(self.config.agent_tool_file)
        if not tool_file.is_absolute():
            if "/" not in str(tool_file):
                tool_file = Path(self.chat_config.tools_path) / tool_file
            elif not tool_file.exists():
                candidate = Path(self.chat_config.tools_path) / tool_file.name
                if candidate.exists():
                    tool_file = candidate

        if not tool_file.exists():
            raise ValueError(f"Agent tool file not found: {tool_file}")

        try:
            with open(tool_file, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except Exception as e:
            raise ValueError(f"Failed to load agent tool file {tool_file}: {e}") from e

        self._agent_action_spec = data
        logger.info("Loaded @agent action spec from %s", tool_file)

    def _agent_action_list(self: "MUDAgentWorker") -> list[dict]:
        """Return the list of actions from the @agent spec.

        Originally from worker.py lines 856-862
        """
        if not self._agent_action_spec:
            return []
        actions = self._agent_action_spec.get("actions", [])
        return actions if isinstance(actions, list) else []

    def _resolve_target_name(self: "MUDAgentWorker", target_id: str) -> str:
        """Backward compatibility wrapper for resolve_target_name function."""
        return resolve_target_name(self.session, target_id)

    def _resolve_move_location(self: "MUDAgentWorker", location: Optional[str]) -> Optional[str]:
        """Backward compatibility wrapper for resolve_move_location function."""
        return resolve_move_location(self.session, location)

    def _is_superuser_persona(self: "MUDAgentWorker") -> bool:
        """Backward compatibility wrapper for is_superuser_persona function."""
        return is_superuser_persona(self.persona)

    def _get_room_objects(self: "MUDAgentWorker") -> list[str]:
        """Backward compatibility wrapper for get_room_objects function."""
        return get_room_objects(self.session)

    def _get_inventory_items(self: "MUDAgentWorker") -> list[str]:
        """Backward compatibility wrapper for get_inventory_items function."""
        return get_inventory_items(self.session)

    def _get_valid_give_targets(self: "MUDAgentWorker") -> list[str]:
        """Backward compatibility wrapper for get_valid_give_targets function."""
        return get_valid_give_targets(self.session)

    def _build_agent_guidance(self: "MUDAgentWorker", user_guidance: str) -> str:
        """Build guidance for @agent action selection.

        Originally from worker.py lines 863-898
        """
        spec = self._agent_action_spec or {}
        instructions = spec.get("instructions", "")
        actions = self._agent_action_list()

        lines = []
        if instructions:
            lines.append(instructions)
        lines.append("Include any other text inside of the JSON response instead.")
        lines.append("You are in your memory palace. Respond as yourself.")
        lines.append(
            "For describe, write paragraph-long descriptions infused with your personality."
        )
        if user_guidance:
            lines.append(f"User guidance: {user_guidance}")

        if actions:
            action_names = ", ".join([a.get("name", "") for a in actions if a.get("name")])
            lines.append(f"Allowed actions: {action_names}")
            lines.append('Output format: {"action": "<name>", ...}')
            for action in actions:
                name = action.get("name")
                desc = action.get("description", "")
                examples = action.get("examples") or action.get("parameters", {}).get("examples")
                if name:
                    if desc:
                        lines.append(f"- {name}: {desc}")
                    if examples:
                        first = examples[0]
                        if isinstance(first, dict):
                            lines.append(f"Example: {json.dumps(first)}")

        if self._decision_strategy and self.session:
            lines.extend(self._decision_strategy._build_agent_action_hints(self.session))
        return "\n".join([line for line in lines if line])

    async def _decide_action(
        self: "MUDAgentWorker",
        idle_mode: bool,
        role: str = "tool",
        action_guidance: str = "",
        user_guidance: str = "",
    ) -> tuple[Optional[str], dict, str, str, str]:
        """Phase 1 decision: choose speak or move via tool call.

        Originally from worker.py lines 943-1114

        Args:
            idle_mode: Whether this is an idle/spontaneous action
            role: Model role to use (defaults to "tool" for fast decisions)
            action_guidance: Optional guidance from prior action results to include
                at the start of the user turn.
            user_guidance: Optional guidance from user (@choose) to include.

        Returns:
            Tuple of (tool_name, args, raw_response, thinking, cleaned_text)
        """
        if not self._decision_strategy.tool_user:
            raise ValueError("Decision tools not initialized")

        turns = await self._decision_strategy.build_turns(
            persona=self.persona,
            session=self.session,
            idle_mode=idle_mode,
            action_guidance=action_guidance,
            user_guidance=user_guidance,
        )
        last_response = ""
        last_cleaned = ""
        last_thinking = ""

        for attempt in range(self.config.decision_max_retries):
            response = await self._call_llm(turns, role=role)
            last_response = response
            logger.debug("Phase1 LLM response: %s...", response[:500])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Phase1 LLM response (full):\n%s", response)
            cleaned, think_content = extract_think_tags(response)
            cleaned = sanitize_response(cleaned)
            last_cleaned = cleaned.strip()
            last_thinking = think_content or ""

            result = self._decision_strategy.tool_user.process_response(response)
            if result.is_valid and result.function_name:
                tool_name = result.function_name
                args = result.arguments or {}

                if tool_name == "move":
                    result = validate_move(self.session, args)
                    if result.is_valid:
                        return "move", result.args, last_response, last_thinking, last_cleaned
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": result.guidance})
                    logger.warning(
                        "Invalid move; retrying with guidance (attempt %d/%d)",
                        attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "speak":
                    # Args are used to enhance memory query, not as action parameters
                    return "speak", args, last_response, last_thinking, last_cleaned

                if tool_name == "wait":
                    # Explicit choice to do nothing this turn
                    return "wait", {}, last_response, last_thinking, last_cleaned

                if tool_name == "take":
                    result = validate_take(self.session, args)
                    if result.is_valid:
                        return "take", result.args, last_response, last_thinking, last_cleaned
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": result.guidance})
                    logger.warning(
                        "Invalid take; retrying with guidance (attempt %d/%d)",
                        attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "drop":
                    result = validate_drop(self.session, args)
                    if result.is_valid:
                        return "drop", result.args, last_response, last_thinking, last_cleaned
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": result.guidance})
                    logger.warning(
                        "Invalid drop; retrying with guidance (attempt %d/%d)",
                        attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "give":
                    result = validate_give(self.session, args)
                    if result.is_valid:
                        return "give", result.args, last_response, last_thinking, last_cleaned
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": result.guidance})
                    logger.warning(
                        "Invalid give; retrying with guidance (attempt %d/%d)",
                        attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                # Unexpected tool - give guidance and retry
                error_guidance = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools are: speak, wait, move, take, drop, give. "
                    f"Please try again with a valid tool."
                )
                turns.append({"role": "assistant", "content": response})
                turns.append({"role": "user", "content": error_guidance})
                logger.warning(
                    "Unexpected tool '%s'; retrying with guidance (attempt %d/%d)",
                    tool_name, attempt + 1, self.config.decision_max_retries,
                )
                continue

            # Invalid tool call format - give guidance and retry
            error_guidance = (
                f"Invalid response format: {result.error}. "
                f"Please respond with exactly one JSON tool call, e.g. {{\"speak\": {{}}}} or {{\"move\": {{\"location\": \"north\"}}}}."
            )
            turns.append({"role": "assistant", "content": response})
            turns.append({"role": "user", "content": error_guidance})
            logger.warning(
                "Invalid tool call (attempt %d/%d): %s",
                attempt + 1,
                self.config.decision_max_retries,
                result.error,
            )

        # Fallback: return "confused" if tool call keeps failing after all retries
        logger.warning("All %d decision attempts failed; returning confused", self.config.decision_max_retries)
        return "confused", {}, last_response, last_thinking, last_cleaned


    @staticmethod
    def _normalize_response(response: str) -> str:
        """Backward compatibility wrapper for normalize_response function."""
        return normalize_response(response)

    @staticmethod
    def _has_emotional_state_header(response: str) -> bool:
        """Backward compatibility wrapper for has_emotional_state_header function."""
        return has_emotional_state_header(response)

    @staticmethod
    def _parse_agent_action_response(response: str) -> tuple[Optional[str], dict, str]:
        """Backward compatibility wrapper for parse_agent_action_response function."""
        return parse_agent_action_response(response)

    async def _is_fresh_session(self: "MUDAgentWorker") -> bool:
        """Check if this is a fresh session (no conversation history).

        Originally from worker.py lines 1294-1299
        """
        if not self.conversation_manager:
            return True
        total = await self.conversation_manager.get_total_tokens()
        return total == 0

    async def process_turn(self: "MUDAgentWorker", turn_request: MUDTurnRequest, events: list[MUDEvent], user_guidance: str = "") -> None:
        """Process a batch of events into a single agent turn.

        Two-phase strategy:
        Phase 1: Decision with TOOL role (fast, cheap)
        Phase 2: Response with CHAT role (only if Phase 1 decided to speak)

        Args:
            turn_request: MUDTurnRequest with sequence_id and attempt_count.
            events: List of MUDEvent objects to process.
            user_guidance: Optional guidance for the turn (used by @choose).
        """
        logger.info(f"Processing turn with {len(events)} events")
        processor = PhasedTurnProcessor(self)
        processor.user_guidance = user_guidance
        await processor.execute(turn_request, events)

    async def process_agent_turn(self: "MUDAgentWorker", turn_request: MUDTurnRequest, events: list[MUDEvent], user_guidance: str) -> None:
        """Process a guided @agent turn using mud_agent.yaml action schema.

        Single-phase strategy: direct action with full guidance,
        skip decision phase entirely.

        Args:
            turn_request: MUDTurnRequest with sequence_id and attempt_count.
            events: List of events to process
            user_guidance: Optional guidance for the agent
        """
        logger.info(f"Processing @agent turn with {len(events)} events")
        processor = AgentTurnProcessor(self)
        processor.user_guidance = user_guidance
        await processor.execute(turn_request, events)
