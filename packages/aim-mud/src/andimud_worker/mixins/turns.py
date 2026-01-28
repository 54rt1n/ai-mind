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

import numpy as np
import yaml

from aim_mud_types import MUDAction, MUDTurnRequest
from aim_mud_types import MUDEvent, MUDTurn
from aim_mud_types.models.decision import DecisionType, DecisionResult
from aim_mud_types.helper import _utc_now
from aim_mud_types.redis_keys import RedisKeys
from aim.utils.think import extract_think_tags
from ..adapter import build_current_context
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
    validate_emote,
    validate_ring,
)
from ..turns.processor import AgentTurnProcessor, ThinkingTurnProcessor
from ..exceptions import AbortRequestedException, ContextOverflowError
from aim.utils.tokens import count_tokens

if TYPE_CHECKING:
    from ..worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class TurnsMixin:
    """Mixin for turn processing methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    # Immediate requery constants for overflow handling
    IMMEDIATE_REQUERY_MAX_RETRIES = 2
    OVERFLOW_ERROR_TEMPLATE = """[OVERFLOW ERROR] Context budget exceeded by {overflow} tokens.
Total: {input_tokens} input + {output_tokens} output = {total} tokens
Model limit: {model_limit} tokens

The focused code is too large. Try:
1. Narrower line range (focus on specific methods, not entire files)
2. Fewer files (focus on 1-2 files instead of many)
3. Smaller height/depth for call graph traversal

Current focus will be cleared. Please refocus with a smaller scope."""

    def _measure_total_tokens(self: "MUDAgentWorker", turns: list[dict], system_message: str) -> int:
        """Measure total tokens for turns + system message.

        Args:
            turns: List of chat turns with role and content.
            system_message: The system message that will be prepended.

        Returns:
            Total token count for the entire context.
        """
        total = count_tokens(system_message)
        for turn in turns:
            content = turn.get("content", "")
            if content:
                total += count_tokens(content)
        return total

    def _get_model_context_limit(self: "MUDAgentWorker") -> int:
        """Get current model's context limit.

        Returns:
            Maximum context tokens for the current model.
        """
        if self.model and hasattr(self.model, "max_tokens"):
            return self.model.max_tokens
        # Fallback to common default
        return 32768

    def _format_overflow_error(
        self: "MUDAgentWorker",
        total_tokens: int,
        model_limit: int,
        output_tokens: int = 4096,
    ) -> str:
        """Format overflow error message.

        Args:
            total_tokens: Total input tokens used.
            model_limit: Maximum context tokens for the model.
            output_tokens: Reserved tokens for output (default 4096).

        Returns:
            Formatted error message with actionable guidance.
        """
        input_tokens = total_tokens
        overflow = (input_tokens + output_tokens) - model_limit
        return self.OVERFLOW_ERROR_TEMPLATE.format(
            overflow=overflow,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total=input_tokens + output_tokens,
            model_limit=model_limit,
        )

    def _is_code_strategy(self: "MUDAgentWorker") -> bool:
        """Check if using CODE_RAG strategy (has clear_focus method).

        Returns:
            True if the decision strategy has clear_focus method.
        """
        return hasattr(self._decision_strategy, "clear_focus")

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
        actions = self._agent_action_spec.get("functions", [])
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

    def _compute_drain_signature(self: "MUDAgentWorker", events: list[MUDEvent]) -> Optional[tuple[str, ...]]:
        """Compute a stable signature for a drained event batch."""
        if not events:
            return None
        signature: list[str] = []
        for event in events:
            if event.event_id:
                signature.append(str(event.event_id))
            else:
                sequence_id = event.metadata.get("sequence_id", "")
                signature.append(f"seq:{sequence_id}")
        return tuple(signature)

    def _refresh_emote_tools(self: "MUDAgentWorker", events: list[MUDEvent]) -> None:
        """Enable/disable emote tool based on current drain and prior emote usage."""
        if not self._decision_strategy:
            return
        signature = self._compute_drain_signature(events)
        if signature is None:
            self._last_drain_signature = None
            self._emote_used_in_drain = False
        elif signature != self._last_drain_signature:
            self._last_drain_signature = signature
            self._emote_used_in_drain = False
        self._decision_strategy.set_emote_allowed(not self._emote_used_in_drain)

    def _mark_emote_used_in_drain(self: "MUDAgentWorker") -> None:
        """Mark that an emote occurred for the current drain."""
        self._emote_used_in_drain = True
        if self._decision_strategy:
            self._decision_strategy.set_emote_allowed(False)

    def _get_available_tool_names(self: "MUDAgentWorker") -> list[str]:
        """Return current decision tool names."""
        if self._decision_strategy:
            names = self._decision_strategy.get_available_tool_names()
            if names:
                return names
        return ["speak", "wait", "move", "take", "drop", "give", "emote"]

    def _format_available_tools(self: "MUDAgentWorker") -> str:
        """Return a human-readable list of available tools."""
        return ", ".join(self._get_available_tool_names())

    def _build_agent_guidance(self: "MUDAgentWorker", user_guidance: str, required_tool: str = "") -> str:
        """Build guidance for @agent action selection.

        Originally from worker.py lines 863-898

        Args:
            user_guidance: Optional user-provided guidance string
            required_tool: If set, only show this tool in the guidance
        """
        spec = self._agent_action_spec or {}
        instructions = spec.get("instructions", "")
        actions = self._agent_action_list()

        # Filter actions to required tool if specified
        if required_tool:
            required_lower = required_tool.lower()
            actions = [a for a in actions if a.get("name", "").lower() == required_lower]

        lines = []
        if instructions:
            lines.append(instructions)
        lines.append("Include any other text inside of the JSON response instead.")
        lines.append("You are in your memory palace. Respond as yourself.")

        # Only show relevant description guidance
        action_names_lower = {a.get("name", "").lower() for a in actions}
        if "desc_room" in action_names_lower:
            lines.append(
                "For desc_room, write paragraph-long room descriptions infused with your personality."
            )
        if "desc_object" in action_names_lower:
            lines.append(
                "For desc_object, use the object ID from hints and write paragraph-long descriptions."
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

    async def _emit_decision_action(
        self: "MUDAgentWorker",
        decision: DecisionResult,
        events: list[MUDEvent],
    ) -> list[MUDAction]:
        """Emit actions for non-speak decisions.

        Handles all decision types except SPEAK:
        - MOVE: Emit move action
        - TAKE: Emit get action
        - DROP: Emit drop action
        - GIVE: Emit give action
        - EMOTE: Emit emote action
        - WAIT: Emit subtle emote with mood
        - PLAN_UPDATE: Emit progress emote
        - CONFUSED: Emit confused emote
        - AURA_TOOL: Emit generic MUDAction for room aura tools

        Creates turn record and updates session state.

        Args:
            decision: DecisionResult from DecisionProcessor
            events: Events that triggered this decision

        Returns:
            List of actions emitted
        """
        actions_taken: list[MUDAction] = []

        if decision.decision_type == DecisionType.MOVE:
            action = MUDAction(tool="move", args=decision.args)
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        elif decision.decision_type == DecisionType.TAKE:
            obj = decision.args.get("object")
            if obj:
                action = MUDAction(tool="get", args={"object": obj})
                actions_taken.append(action)
                await self._emit_actions(actions_taken)
            else:
                logger.warning("TAKE decision missing object; no action emitted")

        elif decision.decision_type == DecisionType.DROP:
            obj = decision.args.get("object")
            if obj:
                action = MUDAction(tool="drop", args={"object": obj})
                actions_taken.append(action)
                await self._emit_actions(actions_taken)
            else:
                logger.warning("DROP decision missing object; no action emitted")

        elif decision.decision_type == DecisionType.GIVE:
            obj = decision.args.get("object")
            target = decision.args.get("target")
            if obj and target:
                action = MUDAction(tool="give", args={"object": obj, "target": target})
                actions_taken.append(action)
                await self._emit_actions(actions_taken)
            else:
                logger.warning("GIVE decision missing object or target; no action emitted")

        elif decision.decision_type == DecisionType.EMOTE:
            action_text = (decision.args.get("action") or "").strip()
            if action_text:
                action = MUDAction(tool="emote", args={"action": action_text})
                actions_taken.append(action)
                await self._emit_actions(actions_taken)
            else:
                logger.warning("EMOTE decision missing action text; no action emitted")

        elif decision.decision_type == DecisionType.NON_PUBLISHED:
            action_text = (decision.args.get("action") or "").strip()
            if action_text:
                action = MUDAction(tool="emote", args={
                    "action": action_text},
                    metadata={MUDAction.META_NON_PUBLISHED: True},
                )
                actions_taken.append(action)
                await self._emit_actions(actions_taken)
            else:
                logger.warning("EMOTE decision missing action text; no action emitted")

        elif decision.decision_type == DecisionType.NON_REACTIVE:
            action_text = (decision.args.get("action") or "").strip()
            if action_text:
                action = MUDAction(tool="emote", args={
                    "action": action_text},
                    metadata={MUDAction.META_NON_REACTIVE: True},
                )
                actions_taken.append(action)
                await self._emit_actions(actions_taken)
            else:
                logger.warning("EMOTE decision missing action text; no action emitted")

        elif decision.decision_type == DecisionType.WAIT:
            logger.info("Decision to wait; emitting subtle emote")
            # Emit a subtle emote based on mood (if provided)
            mood = decision.args.get("mood", "").strip()
            if mood:
                emote_text = f"waits {mood}."
                logger.info(f"Wait emote with mood: {emote_text}")
                # Custom mood: non_reactive (published but doesn't trigger reactions)
                metadata = {MUDAction.META_NON_REACTIVE: True}
            else:
                emote_text = "waits quietly."
                logger.info("Wait emote with default mood")
                # Default: non_published (not routed to event streams at all)
                metadata = {MUDAction.META_NON_PUBLISHED: True}
            action = MUDAction(
                tool="emote",
                args={"action": emote_text},
                metadata=metadata,
            )
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        elif decision.decision_type == DecisionType.PLAN_UPDATE:
            # Plan task status was updated - emit emote about progress
            plan_status = decision.args.get("plan_status", "unknown")
            next_task = decision.args.get("next_task")

            if plan_status == "completed":
                emote_text = "completed the plan successfully."
            elif next_task:
                emote_text = f"completed a task and moved on to: {next_task}"
            else:
                emote_text = "updated the plan status."

            logger.info(f"Plan update: status={plan_status}, next_task={next_task}")
            action = MUDAction(tool="emote", args={
                "action": emote_text},
                metadata={MUDAction.META_NON_PUBLISHED: True},
            )
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        elif decision.decision_type == DecisionType.CONFUSED:
            # Decision processor failed to parse valid response
            logger.info("Decision returned CONFUSED; emitting confused emote")
            emote_text = "looks confused."
            action = MUDAction(tool="emote", args={
                "action": emote_text},
                metadata={MUDAction.META_NON_PUBLISHED: True},
            )
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        elif decision.decision_type == DecisionType.CLOSE_BOOK:
            # close_book is handled internally (workspace cleared in _decide_action)
            # - no MUDAction needed, no visible action in the world
            logger.info("close_book: workspace cleared, no MUDAction emitted")

        elif decision.decision_type == DecisionType.FOCUS:
            # focus is handled internally (focus set on strategy in _decide_action)
            # Emit a non-published action to report the focus result
            focus_files = decision.args.get("focused_files", [])
            symbols_count = decision.args.get("symbols_in_focus", 0)
            line_range = decision.args.get("line_range", "all")
            focus_report = f"Focused on {', '.join(focus_files)} (lines {line_range}, {symbols_count} symbols)"
            action = MUDAction(
                tool="focus",
                args=decision.args,
                metadata={MUDAction.META_NON_PUBLISHED: True},
            )
            actions_taken.append(action)
            await self._emit_actions(actions_taken)
            logger.info("focus: %s", focus_report)

            # Persist focus to agent profile for @status display
            # Get focus from strategy and save to Redis
            if self._decision_strategy and hasattr(self._decision_strategy, "focus") and self._decision_strategy.focus:
                focus = self._decision_strategy.focus
                focus_data = {
                    "files": focus.files,  # list[dict] with path, start, end
                    "height": focus.height,
                    "depth": focus.depth,
                }
                from aim_mud_types.client import RedisMUDClient
                client = RedisMUDClient(self.redis)
                await client.update_agent_profile_fields(
                    self.config.agent_id,
                    focus=json.dumps(focus_data),
                )
                logger.debug("Focus persisted to agent profile")

        elif decision.decision_type == DecisionType.AURA_TOOL:
            tool_name = decision.aura_tool_name or "unknown_aura_tool"

            # Special-case: sleep and wake have dedicated commands with side effects
            # (emote generation, conversation ID rotation, CVM flush)
            if tool_name in ("sleep", "wake"):
                command = self.command_registry.get_command(tool_name)
                if command:
                    logger.info("Aura tool '%s' routing to dedicated command", tool_name)
                    # Invoke command with agent=True to trigger conversation management
                    await command.execute(
                        self,
                        turn_id=f"agent_{tool_name}",
                        reason=tool_name,
                        sequence_id=0,  # Not validated by command
                        metadata={"agent": True},
                        events=events,
                    )
                    # Command emitted emote action, now emit sleep/wake action for Evennia
                    action = MUDAction(tool=tool_name, args=decision.args)
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Command '%s' not found in registry", tool_name)
            else:
                # Generic aura tool handling - emit MUDAction for Evennia to execute
                action = MUDAction(tool=tool_name, args=decision.args)
                actions_taken.append(action)
                logger.info("Aura tool '%s' emitting action with args: %s", tool_name, decision.args)
                await self._emit_actions(actions_taken)

        # Create and store turn record
        turn = MUDTurn(
            timestamp=_utc_now(),
            events_received=events,
            room_context=self.session.current_room,
            entities_context=self.session.entities_present,
            thinking=decision.thinking,
            actions_taken=actions_taken,
        )
        self.session.add_turn(turn)

        logger.info(
            f"Emitted {len(actions_taken)} action(s) for {decision.decision_type.name}. "
            f"Session now has {len(self.session.recent_turns)} turns"
        )

        return actions_taken

    async def _decide_action(
        self: "MUDAgentWorker",
        idle_mode: bool,
        role: str = "tool",
        action_guidance: str = "",
        user_guidance: str = "",
    ) -> tuple[Optional[str], dict, str, str, str]:
        """Phase 1 decision: choose speak or move via tool call.

        Originally from worker.py lines 943-1114

        For CODE_RAG strategies, includes immediate requery on overflow:
        - Detects context overflow BEFORE sending to LLM
        - Clears focus and injects error message
        - Retries up to IMMEDIATE_REQUERY_MAX_RETRIES times

        Args:
            idle_mode: Whether this is an idle/spontaneous action
            role: Model role to use (defaults to "tool" for fast decisions)
            action_guidance: Optional guidance from prior action results to include
                at the start of the user turn.
            user_guidance: Optional guidance from user (@choose) to include.

        Returns:
            Tuple of (tool_name, args, raw_response, thinking, cleaned_text)

        Raises:
            ContextOverflowError: If context exceeds limit after max retries.
        """
        if not self._decision_strategy.tool_user:
            raise ValueError("Decision tools not initialized")

        # Immediate requery loop for overflow handling (CODE_RAG only)
        overflow_retry_count = 0
        overflow_error_message: Optional[str] = None

        while overflow_retry_count <= self.IMMEDIATE_REQUERY_MAX_RETRIES:
            # Build turns (includes consciousness with focused code)
            turns = await self._decision_strategy.build_turns(
                persona=self.persona,
                session=self.session,
                idle_mode=idle_mode,
                action_guidance=action_guidance,
                user_guidance=user_guidance,
                max_context_tokens=self.model.max_tokens,       # Model context window
                max_output_tokens=self.chat_config.max_tokens,  # Max output tokens
            )

            # If there was an overflow error from previous iteration, inject it
            if overflow_error_message:
                turns.append({"role": "user", "content": overflow_error_message})

            # Check for overflow BEFORE sending to LLM (CODE_RAG strategies only)
            if self._is_code_strategy():
                system_message = self._decision_strategy.get_system_message(self.persona)
                total_tokens = self._measure_total_tokens(turns, system_message)
                model_limit = self._get_model_context_limit()
                output_reservation = self.chat_config.max_tokens or 4096

                if total_tokens + output_reservation > model_limit:
                    # Overflow detected
                    if overflow_retry_count >= self.IMMEDIATE_REQUERY_MAX_RETRIES:
                        # Max retries exceeded - fail with actionable error
                        raise ContextOverflowError(
                            f"Focus too large after {self.IMMEDIATE_REQUERY_MAX_RETRIES} retries. "
                            f"Total: {total_tokens} tokens, limit: {model_limit}"
                        )

                    # Clear focus and prepare error message for next iteration
                    self._decision_strategy.clear_focus()
                    logger.warning(
                        f"Immediate requery: overflow detected ({total_tokens} tokens), "
                        f"clearing focus (attempt {overflow_retry_count + 1}/{self.IMMEDIATE_REQUERY_MAX_RETRIES})"
                    )
                    overflow_error_message = self._format_overflow_error(
                        total_tokens, model_limit, output_reservation
                    )
                    overflow_retry_count += 1
                    continue

            # No overflow (or not a code strategy) - proceed with LLM call
            break

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

                if tool_name == "emote":
                    result = validate_emote(args)
                    if result.is_valid:
                        return "emote", result.args, last_response, last_thinking, last_cleaned
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": result.guidance})
                    logger.warning(
                        "Invalid emote; retrying with guidance (attempt %d/%d)",
                        attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "plan_update":
                    # Plan update requires async execution via PlanExecutionTool
                    plan_tool_impl = self._decision_strategy.get_plan_tool_impl()
                    if plan_tool_impl is None:
                        available_tools = self._format_available_tools()
                        error_guidance = (
                            "plan_update tool not available. "
                            f"This tool requires an active plan. Use {available_tools}."
                        )
                        turns.append({"role": "assistant", "content": response})
                        turns.append({"role": "user", "content": error_guidance})
                        logger.warning(
                            "plan_update called but no plan tool impl (attempt %d/%d)",
                            attempt + 1, self.config.decision_max_retries,
                        )
                        continue

                    # Execute async tool
                    tool_result = await plan_tool_impl.execute_async(tool_name, args)
                    if "error" in tool_result:
                        error_guidance = f"plan_update failed: {tool_result['error']}"
                        turns.append({"role": "assistant", "content": response})
                        turns.append({"role": "user", "content": error_guidance})
                        logger.warning(
                            "plan_update error (attempt %d/%d): %s",
                            attempt + 1, self.config.decision_max_retries, tool_result["error"],
                        )
                        continue

                    # Return plan_update result - processor will handle appropriately
                    return "plan_update", tool_result, last_response, last_thinking, last_cleaned

                # close_book: clear workspace internally, no MUDAction
                if tool_name == "close_book":
                    workspace_key = RedisKeys.agent_workspace(self.config.agent_id)
                    await self.redis.delete(workspace_key)
                    if self._chat_manager:
                        self._chat_manager.current_workspace = None
                    if self._decision_strategy:
                        self._decision_strategy.set_workspace_active(False)
                    logger.info("close_book: cleared workspace for agent %s", self.config.agent_id)
                    return "close_book", {"success": True}, last_response, last_thinking, last_cleaned

                # focus: CODE_RAG tool to set code focus on strategy
                if tool_name == "focus":
                    focus_tool = getattr(self._decision_strategy, "get_focus_tool", lambda: None)()
                    if focus_tool is None:
                        error_guidance = (
                            "focus tool not available. "
                            "This tool requires a code agent with CODE_RAG strategy."
                        )
                        turns.append({"role": "assistant", "content": response})
                        turns.append({"role": "user", "content": error_guidance})
                        logger.warning(
                            "focus called but no focus tool impl (attempt %d/%d)",
                            attempt + 1, self.config.decision_max_retries,
                        )
                        continue

                    # Execute focus tool (sets focus on decision strategy)
                    # Pass entities for name resolution (e.g., "model.py" -> "/repo/src/model.py")
                    entities = self.session.entities_present if self.session else []
                    tool_result = focus_tool.focus(
                        files=args.get("files", []),
                        start_line=args.get("start_line"),
                        end_line=args.get("end_line"),
                        height=args.get("height", 1),
                        depth=args.get("depth", 1),
                        entities=entities,
                    )

                    # Also set focus on response strategy so Phase 2 has access
                    # Use resolved files from tool_result to ensure consistency
                    if hasattr(self, "_response_strategy") and self._response_strategy:
                        from aim_code.strategy import FocusRequest
                        focus_request = FocusRequest(
                            files=tool_result.get("focused_files", args.get("files", [])),
                            start_line=args.get("start_line"),
                            end_line=args.get("end_line"),
                            height=args.get("height", 1),
                            depth=args.get("depth", 1),
                        )
                        self._response_strategy.set_focus(focus_request)

                    logger.info("focus: set focus on %s", args.get("files", []))
                    return "focus", tool_result, last_response, last_thinking, last_cleaned

                # Generic aura tool handling - all aura tools emit MUDActions
                if self._decision_strategy.is_aura_tool(tool_name):
                    # For ring specifically, validate targets
                    if tool_name == "ring":
                        result = validate_ring(self.session, args)
                        if not result.is_valid:
                            turns.append({"role": "assistant", "content": response})
                            turns.append({"role": "user", "content": result.guidance})
                            logger.warning(
                                "Invalid ring; retrying with guidance (attempt %d/%d)",
                                attempt + 1, self.config.decision_max_retries,
                            )
                            continue
                        args = result.args
                    # All aura tools pass through - processor will emit MUDAction
                    return tool_name, args, last_response, last_thinking, last_cleaned

                # Unexpected tool - give guidance and retry
                error_guidance = (
                    f"Unknown tool '{tool_name}'. "
                    f"Available tools are: {self._format_available_tools()}. "
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

    def get_current_turn_embedding(self: "MUDAgentWorker") -> Optional[np.ndarray]:
        """Get the embedding from the last user entry in current turn.

        Used by Phase 2 processors to pass pre-computed embeddings to CVM
        queries, enabling same-embedding reranking for document retrieval
        and FAISS similarity scoring.

        The embedding comes from the mediator, which computes it during
        event compilation. Using the same embedding for both indexing
        and querying ensures consistent similarity scores.

        Returns:
            numpy array of the last user entry's embedding, or None if:
            - No entries in current turn
            - No user entries found
            - Entry has no embedding
        """
        entries = getattr(self, "_current_turn_entries", None)
        if not entries:
            return None

        # Find the last user entry (matches query text selection logic)
        for entry in reversed(entries):
            if entry.role == "user":
                return entry.get_embedding_vector()

        return None

    async def _setup_turn_context(
        self: "MUDAgentWorker",
        events: list[MUDEvent]
    ) -> None:
        """Setup turn context - called by commands before processors.

        This is the business logic that was previously in BaseTurnProcessor.setup_turn().
        Commands call this once at the start, then all processors use the prepared state.

        Performs the following setup steps:
        1. Update decision tool availability based on drained events
        2. Refresh world state snapshot from agent + room profiles
        3. Log event details for debugging
        4. Update session context from events
        5. Push user turn to conversation history

        Args:
            worker: Worker instance with session, conversation_manager, etc.
            events: Events to process

        Example:
            ```python
            # In a command's execute() method:
            await setup_turn_context(worker, events)
            processor = PhasedTurnProcessor(worker)
            await processor.execute(turn_request, events)
            ```
        """
        # Load thought content from Redis and set on strategies
        await self._load_thought_content()

        # Load workspace state from Redis and set on chat manager
        await self._load_workspace_state()

        # Load focus state from Redis and set on strategies (for code agents)
        await self._load_focus_state()

        # Update decision tool availability based on drained events
        self._refresh_emote_tools(events)

        # Refresh world state snapshot from agent + room profiles
        room_id, character_id = await self._load_agent_world_state()
        if not room_id and self.session.current_room and self.session.current_room.room_id:
            room_id = self.session.current_room.room_id
        if not room_id and events:
            room_id = events[-1].room_id
        await self._load_room_profile(room_id, character_id)

        # Log event details for debugging
        for event in events:
            logger.info(
                f"  Event: {event.event_type.value} | "
                f"Actor: {event.actor} | "
                f"Room: {event.room_name or event.room_id} | "
                f"Content: {event.content[:100] if event.content else '(none)'}..."
            )

        # Push events to conversation history
        await self._push_events_to_conversation(events)

    async def take_turn(
        self: "MUDAgentWorker",
        turn_id: str,
        events: list[MUDEvent],
        turn_request: MUDTurnRequest,
        user_guidance: str = "",
    ) -> DecisionResult | None:
        """Take a turn."""
        self._last_turn_error = None
        self._last_turn_error_type = None
        self._current_turn_entries = []  # Reset entries for this turn
        try:
            # Setup turn context ONCE
            await self._setup_turn_context(events)

            from ..turns.processor.decision import DecisionProcessor
            from ..turns.processor.speaking import SpeakingProcessor
            from ..turns.processor.thinking import ThinkingTurnProcessor

            # Run DecisionProcessor for Phase 1
            decision_processor = DecisionProcessor(self)
            decision_processor.user_guidance = user_guidance
            await decision_processor.execute(turn_request, events)

            # Route based on decision type
            decision = self._last_decision

            if decision.decision_type == DecisionType.SPEAK:
                # Get new conversation entries that arrived during Phase 1
                # (entries are already in conversation list from mediator)
                # settle=True waits for cascading events to stop arriving
                new_entries = await self.get_new_conversation_entries(settle=True)
                new_entries = self.collapse_consecutive_entries(new_entries)
                # Store entries on worker for processors to access embeddings
                self._current_turn_entries = new_entries
                if new_entries:
                    logger.info(f"[{turn_id}] Captured {len(new_entries)} new entries (settled) for Phase 2 (SPEAK)")

                speaking_processor = SpeakingProcessor(self)
                speaking_processor.user_guidance = user_guidance
                await speaking_processor.execute(turn_request, events)
            elif decision.decision_type == DecisionType.THINK:
                # Get new conversation entries that arrived during Phase 1
                # settle=True waits for cascading events to stop arriving
                new_entries = await self.get_new_conversation_entries(settle=True)
                new_entries = self.collapse_consecutive_entries(new_entries)
                # Store entries on worker for processors to access embeddings
                self._current_turn_entries = new_entries
                if new_entries:
                    logger.info(f"[{turn_id}] Captured {len(new_entries)} new entries (settled) for Phase 2 (THINK)")

                thinking_processor = ThinkingTurnProcessor(self)
                thinking_processor.user_guidance = user_guidance
                await thinking_processor.execute(turn_request, events)
            else:
                # Direct action (move, take, drop, give, emote, wait, etc.)
                await self._emit_decision_action(decision, events)

            # Clear decision
            self._last_decision = None

            return decision
        except Exception as e:
            self._last_turn_error = str(e)
            self._last_turn_error_type = type(e).__name__
            logger.error(f"[{turn_id}] Error taking turn: {e}")
            return None
