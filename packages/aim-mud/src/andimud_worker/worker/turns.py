# aim/app/mud/worker/turns.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Turn processing for the MUD worker.

Handles full turn processing pipeline including decision and response phases.
Extracted from worker.py lines 779-1869
"""

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from aim_mud_types import MUDAction
from aim.dreamer.executor import extract_think_tags
from aim.tool.loader import ToolLoader
from aim.tool.formatting import ToolUser
from ..adapter import build_current_context
from ..session import MUDEvent, MUDTurn
from ..utils import sanitize_response
from .utils import _utc_now


if TYPE_CHECKING:
    from .main import MUDAgentWorker, AbortRequestedException
else:
    # Avoid circular import - import at runtime
    AbortRequestedException = None


logger = logging.getLogger(__name__)


class TurnsMixin:
    """Mixin for turn processing methods.

    These methods are mixed into MUDAgentWorker in main.py.
    """

    def _init_decision_tools(self: "MUDAgentWorker") -> None:
        """Initialize phase 1 decision tools (speak/move).

        Originally from worker.py lines 806-831
        """
        if self.chat_config is None:
            raise ValueError("ChatConfig must be initialized before loading tools")

        tool_file = Path(self.config.decision_tool_file)
        if not tool_file.is_absolute():
            # Allow passing just a filename; resolve relative to tools_path
            if "/" not in str(tool_file):
                tool_file = Path(self.chat_config.tools_path) / tool_file
            elif not tool_file.exists():
                candidate = Path(self.chat_config.tools_path) / tool_file.name
                if candidate.exists():
                    tool_file = candidate

        loader = ToolLoader(self.chat_config.tools_path)
        tools = loader.load_tool_file(str(tool_file))
        if not tools:
            raise ValueError(f"No tools loaded from {tool_file}")

        self._decision_tool_user = ToolUser(tools)
        logger.info(
            "Loaded %d phase 1 decision tools from %s",
            len(tools),
            tool_file,
        )

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

    def _resolve_target_name(self: "MUDAgentWorker", target_id: str) -> str:
        """Resolve a target id to a display name for emotes.

        Originally from worker.py lines 900-921
        """
        if not target_id or not self.session:
            return target_id or "object"

        world_state = self.session.world_state
        if world_state and world_state.room_state:
            room = world_state.room_state
            if room.room_id == target_id:
                return room.name or "room"

        # Check entities present
        if world_state:
            for entity in world_state.entities_present:
                if entity.entity_id == target_id:
                    return entity.name or target_id

            for item in world_state.inventory:
                if getattr(item, "item_id", None) == target_id:
                    return item.name or target_id

        return target_id

    def _resolve_move_location(self: "MUDAgentWorker", location: Optional[str]) -> Optional[str]:
        """Validate and normalize a move location against current exits.

        Originally from worker.py lines 923-942
        """
        if not location:
            return None

        room = self.session.current_room if self.session else None
        exits = room.exits if room else None
        if not exits:
            return location

        if location in exits:
            return location

        lowered = location.lower()
        for exit_name in exits.keys():
            if exit_name.lower() == lowered:
                return exit_name

        return None

    def _is_superuser_persona(self: "MUDAgentWorker") -> bool:
        """Return True if persona should have builder tools.

        Originally from worker.py lines 1116-1134
        """
        if not self.persona:
            return False

        attrs = self.persona.attributes or {}
        role = str(attrs.get("mud_role", "")).lower()
        perms = attrs.get("mud_permissions")

        if role in ("superuser", "builder"):
            return True

        if isinstance(perms, list):
            return any(str(p).lower() in ("superuser", "builder") for p in perms)

        if isinstance(perms, str):
            return perms.lower() in ("superuser", "builder")

        return False

    def _get_room_objects(self: "MUDAgentWorker") -> list[str]:
        """Get names of objects available to take in the current room.

        Originally from worker.py lines 1136-1157
        """
        objects: list[str] = []
        world_state = self.session.world_state if self.session else None
        if world_state:
            for entity in world_state.entities_present:
                if entity.is_self:
                    continue
                # Objects are entities that aren't players/AIs/NPCs
                if entity.entity_type not in ("player", "ai", "npc"):
                    if entity.name:
                        objects.append(entity.name)
        else:
            # Fall back to session entities
            if self.session:
                for entity in self.session.entities_present:
                    if entity.is_self:
                        continue
                    if entity.entity_type not in ("player", "ai", "npc"):
                        if entity.name:
                            objects.append(entity.name)
        return objects

    def _get_inventory_items(self: "MUDAgentWorker") -> list[str]:
        """Get names of items in the agent's inventory.

        Originally from worker.py lines 1159-1167
        """
        items: list[str] = []
        world_state = self.session.world_state if self.session else None
        if world_state:
            for item in world_state.inventory:
                if item.name:
                    items.append(item.name)
        return items

    def _get_valid_give_targets(self: "MUDAgentWorker") -> list[str]:
        """Get names of valid targets for giving items (players, AIs, NPCs, objects).

        Originally from worker.py lines 1169-1189
        """
        targets: list[str] = []
        world_state = self.session.world_state if self.session else None
        if world_state:
            for entity in world_state.entities_present:
                if entity.is_self:
                    continue
                if entity.entity_type in ("player", "ai", "npc", "object"):
                    if entity.name:
                        targets.append(entity.name)
        else:
            # Fall back to session entities
            if self.session:
                for entity in self.session.entities_present:
                    if entity.is_self:
                        continue
                    if entity.entity_type in ("player", "ai", "npc", "object"):
                        if entity.name:
                            targets.append(entity.name)
        return targets

    async def _decide_action(self: "MUDAgentWorker", idle_mode: bool) -> tuple[Optional[str], dict, str, str, str]:
        """Phase 1 decision: choose speak or move via tool call.

        Originally from worker.py lines 943-1114

        Returns:
            Tuple of (tool_name, args, raw_response, thinking, cleaned_text)
        """
        if not self._decision_tool_user:
            raise ValueError("Decision tools not initialized")

        turns = await self._decision_strategy.build_turns(
            persona=self.persona,
            session=self.session,
            idle_mode=idle_mode,
        )
        last_response = ""
        last_cleaned = ""
        last_thinking = ""

        for attempt in range(self.config.decision_max_retries):
            response = await self._call_llm(turns)
            last_response = response
            logger.debug("Phase1 LLM response: %s...", response[:500])
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Phase1 LLM response (full):\n%s", response)
            cleaned, think_content = extract_think_tags(response)
            cleaned = sanitize_response(cleaned)
            last_cleaned = cleaned.strip()
            last_thinking = think_content or ""

            result = self._decision_tool_user.process_response(response)
            if result.is_valid and result.function_name:
                tool_name = result.function_name
                args = result.arguments or {}

                if tool_name == "move":
                    location = args.get("location") or args.get("direction")
                    resolved = self._resolve_move_location(location)
                    if not resolved:
                        # Get valid exits for guidance
                        valid_exits = []
                        if self.session.current_room and self.session.current_room.exits:
                            valid_exits = list(self.session.current_room.exits.keys())
                        exits_str = ", ".join(valid_exits) if valid_exits else "none available"
                        error_guidance = (
                            f"Invalid move location '{location}'. "
                            f"Valid exits are: {exits_str}. "
                            f"Please try again with a valid exit, or use {{\"speak\": {{}}}} to respond instead."
                        )
                        turns.append({"role": "assistant", "content": response})
                        turns.append({"role": "user", "content": error_guidance})
                        logger.warning(
                            "Invalid move location '%s'; retrying with guidance (attempt %d/%d)",
                            location, attempt + 1, self.config.decision_max_retries,
                        )
                        continue
                    return "move", {"location": resolved}, last_response, last_thinking, last_cleaned

                if tool_name == "speak":
                    # Args are used to enhance memory query, not as action parameters
                    return "speak", args, last_response, last_thinking, last_cleaned

                if tool_name == "wait":
                    # Explicit choice to do nothing this turn
                    return "wait", {}, last_response, last_thinking, last_cleaned

                if tool_name == "take":
                    obj = args.get("object")
                    # Get available items in the room
                    room_objects = self._get_room_objects()
                    if obj and obj.lower() in [o.lower() for o in room_objects]:
                        return "take", args, last_response, last_thinking, last_cleaned
                    # Invalid - give guidance
                    objects_str = ", ".join(room_objects) if room_objects else "nothing here to take"
                    error_guidance = (
                        f"Cannot take '{obj}'. "
                        f"Available items: {objects_str}. "
                        f"Please try again with a valid item, or use {{\"speak\": {{}}}} to respond instead."
                    )
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": error_guidance})
                    logger.warning(
                        "Invalid take object '%s'; retrying with guidance (attempt %d/%d)",
                        obj, attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "drop":
                    obj = args.get("object")
                    # Get inventory items
                    inventory = self._get_inventory_items()
                    if obj and obj.lower() in [i.lower() for i in inventory]:
                        return "drop", args, last_response, last_thinking, last_cleaned
                    # Invalid - give guidance
                    inventory_str = ", ".join(inventory) if inventory else "nothing in inventory"
                    error_guidance = (
                        f"Cannot drop '{obj}'. "
                        f"Your inventory: {inventory_str}. "
                        f"Please try again with an item you're carrying, or use {{\"speak\": {{}}}} to respond instead."
                    )
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": error_guidance})
                    logger.warning(
                        "Invalid drop object '%s'; retrying with guidance (attempt %d/%d)",
                        obj, attempt + 1, self.config.decision_max_retries,
                    )
                    continue

                if tool_name == "give":
                    obj = args.get("object")
                    target = args.get("target")
                    # Get inventory and valid targets
                    inventory = self._get_inventory_items()
                    valid_targets = self._get_valid_give_targets()

                    obj_valid = obj and obj.lower() in [i.lower() for i in inventory]
                    target_valid = target and target.lower() in [t.lower() for t in valid_targets]

                    if obj_valid and target_valid:
                        return "give", args, last_response, last_thinking, last_cleaned

                    # Build specific guidance
                    errors = []
                    if not obj_valid:
                        inventory_str = ", ".join(inventory) if inventory else "nothing"
                        errors.append(f"Cannot give '{obj}'. Your inventory: {inventory_str}.")
                    if not target_valid:
                        targets_str = ", ".join(valid_targets) if valid_targets else "no one here"
                        errors.append(f"Cannot give to '{target}'. People present: {targets_str}.")

                    error_guidance = (
                        " ".join(errors) + " "
                        f"Please try again with valid item and target, or use {{\"speak\": {{}}}} to respond instead."
                    )
                    turns.append({"role": "assistant", "content": response})
                    turns.append({"role": "user", "content": error_guidance})
                    logger.warning(
                        "Invalid give (object='%s', target='%s'); retrying with guidance (attempt %d/%d)",
                        obj, target, attempt + 1, self.config.decision_max_retries,
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
        """Normalize a free-text response for emission.

        Originally from worker.py lines 1270-1292
        """
        if not response:
            return ""

        stripped = response.strip()
        if not stripped:
            return ""

        lines = [line.rstrip() for line in stripped.splitlines()]
        normalized: list[str] = []
        blank = False
        for line in lines:
            if not line.strip():
                if not blank:
                    normalized.append("")
                    blank = True
                continue
            normalized.append(line)
            blank = False

        return "\n".join(normalized).strip()

    async def _is_fresh_session(self: "MUDAgentWorker") -> bool:
        """Check if this is a fresh session (no conversation history).

        Originally from worker.py lines 1294-1299
        """
        if not self.conversation_manager:
            return True
        total = await self.conversation_manager.get_total_tokens()
        return total == 0

    @staticmethod
    def _has_emotional_state_header(response: str) -> bool:
        """Check if response starts with emotional state header after think block.

        Originally from worker.py lines 1301-1307
        """
        # Remove think block first
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        # Check if it starts with [== ... Emotional State ... ==]
        return bool(re.match(r'\[==.*Emotional State.*==\]', cleaned, re.IGNORECASE))

    @staticmethod
    def _extract_speak_text_from_tool_call(response: str) -> Optional[str]:
        """Extract speak text if the response is a tool-call-like JSON blob.

        Originally from worker.py lines 1309-1344
        """
        if not response:
            return None

        stripped = response.strip()
        if not stripped:
            return None

        parsed = None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            try:
                parsed = ToolUser([])._extract_tool_call(stripped)
            except Exception:
                parsed = None

        if not isinstance(parsed, dict):
            return None

        if "speak" not in parsed:
            return None

        payload = parsed.get("speak")
        if isinstance(payload, str):
            return payload

        if isinstance(payload, dict):
            for key in ("text", "say", "message", "content"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value

        return None

    def _parse_agent_action_response(self: "MUDAgentWorker", response: str) -> tuple[Optional[str], dict, str]:
        """Parse @agent JSON response into (action, args, error).

        Originally from worker.py lines 1346-1400
        """
        cleaned, _think = extract_think_tags(response)
        cleaned = sanitize_response(cleaned)
        text = cleaned.strip()
        parsed = None
        json_text = None

        try:
            parsed = json.loads(text)
            json_text = text
        except json.JSONDecodeError:
            # Try to extract a JSON object from mixed text
            json_candidates = []
            brace_depth = 0
            start_idx = None
            for i, char in enumerate(text):
                if char == "{":
                    if brace_depth == 0:
                        start_idx = i
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                    if brace_depth == 0 and start_idx is not None:
                        json_candidates.append(text[start_idx : i + 1])
                        start_idx = None
            for candidate in reversed(json_candidates):
                try:
                    parsed = json.loads(candidate)
                    json_text = candidate.strip()
                    break
                except json.JSONDecodeError:
                    continue

        if not isinstance(parsed, dict):
            # Include truncated response in error for debugging
            preview = text[:200] + "..." if len(text) > 200 else text
            return None, {}, f"Could not parse JSON: {preview}"

        # Preferred format: {"action": "<name>", ...}
        if "action" in parsed:
            action = parsed.get("action")
            if not isinstance(action, str):
                return None, {}, "Action must be a string"
            args = {k: v for k, v in parsed.items() if k != "action"}
            return action.lower(), args, ""

        # Alternate tool-call format: {"describe": {...}}
        if len(parsed) == 1:
            action = next(iter(parsed))
            args = parsed.get(action)
            if isinstance(action, str) and isinstance(args, dict):
                return action.lower(), args, ""

        return None, {}, "Missing action field"

    async def process_turn(self: "MUDAgentWorker", events: list[MUDEvent]) -> None:
        """Process a batch of events into a single agent turn.

        Originally from worker.py lines 1402-1643

        Implements the full turn processing pipeline:
        1. Update session context from events
        2. Build chat turns from session state
        3. Call LLM to generate response
        4. Normalize free-text response
        5. Emit a single `speak` action (or noop)
        6. Create turn record and add to session history

        Args:
            events: List of MUDEvent objects to process.
        """
        logger.info(f"Processing turn with {len(events)} events")

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

        # Step 1: Update session context from events
        self.session.pending_events = events
        if events:
            latest = events[-1]
            self.session.last_event_time = latest.timestamp

        # Push user turn to conversation list
        if self.conversation_manager and events:
            await self.conversation_manager.push_user_turn(
                events=events,
                world_state=self.session.world_state,
                room_id=self.session.current_room.room_id if self.session.current_room else None,
                room_name=self.session.current_room.name if self.session.current_room else None,
            )

        # Step 2: Phase 1 decision (speak/move)
        idle_mode = len(events) == 0
        thinking_parts: list[str] = []
        actions_taken: list[MUDAction] = []
        raw_responses: list[str] = []

        try:
            # Check for abort before decision LLM call
            if await self._check_abort_requested():
                from .main import AbortRequestedException
                raise AbortRequestedException("Turn aborted before decision")

            decision_tool, decision_args, decision_raw, decision_thinking, decision_cleaned = (
                await self._decide_action(idle_mode=idle_mode)
            )
            # Phase 1 thinking captured for debugging, but not the raw JSON response
            if decision_thinking:
                thinking_parts.append(decision_thinking)

            if decision_tool == "move":
                action = MUDAction(tool="move", args=decision_args)
                actions_taken.append(action)
                await self._emit_actions(actions_taken)

            elif decision_tool == "take":
                obj = decision_args.get("object")
                if obj:
                    action = MUDAction(tool="get", args={"object": obj})
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 take missing object; no action emitted")

            elif decision_tool == "drop":
                obj = decision_args.get("object")
                if obj:
                    action = MUDAction(tool="drop", args={"object": obj})
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 drop missing object; no action emitted")

            elif decision_tool == "give":
                obj = decision_args.get("object")
                target = decision_args.get("target")
                if obj and target:
                    action = MUDAction(tool="give", args={"object": obj, "target": target})
                    actions_taken.append(action)
                    await self._emit_actions(actions_taken)
                else:
                    logger.warning("Phase1 give missing object or target; no action emitted")

            elif decision_tool == "wait":
                # Explicit choice to do nothing - skip Phase 2
                logger.info("Phase 1 decided to wait; no action this turn")

            elif decision_tool == "speak":
                # Phase 2: full response turn with memory via response strategy
                coming_online = await self._is_fresh_session()

                # Extract memory query from speak args (enhances CVM search)
                memory_query = decision_args.get("query") or decision_args.get("focus") or ""
                if memory_query:
                    logger.info(f"Phase 2 memory query: {memory_query[:100]}...")

                # Build user input with current context (events/guidance)
                # Events already pushed to conversation history, so exclude here
                user_input = build_current_context(
                    self.session,
                    idle_mode=idle_mode,
                    guidance=None,
                    coming_online=coming_online,
                    include_events=False,
                )

                # Use response strategy for full context (consciousness + memory)
                chat_turns = await self._response_strategy.build_turns(
                    persona=self.persona,
                    user_input=user_input,
                    session=self.session,
                    coming_online=coming_online,
                    max_context_tokens=self.model.max_tokens,
                    max_output_tokens=self.chat_config.max_tokens,
                    memory_query=memory_query,
                )

                # Retry loop for emotional state header validation
                max_format_retries = 3
                cleaned_response = ""
                for format_attempt in range(max_format_retries):
                    # Check for abort before response LLM call
                    if await self._check_abort_requested():
                        from .main import AbortRequestedException
                        raise AbortRequestedException("Turn aborted before response")

                    response = await self._call_llm(chat_turns)
                    raw_responses.append(response)
                    logger.debug(f"LLM response: {response[:500]}...")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("LLM response (full):\n%s", response)

                    cleaned_response, think_content = extract_think_tags(response)
                    cleaned_response = sanitize_response(cleaned_response)
                    cleaned_response = cleaned_response.strip()
                    if think_content:
                        thinking_parts.append(think_content)

                    # Validate emotional state header
                    if self._has_emotional_state_header(cleaned_response):
                        break  # Valid format, continue

                    # Missing header - retry with stronger guidance
                    logger.warning(
                        f"Response missing Emotional State header (attempt {format_attempt + 1}/{max_format_retries})"
                    )
                    if format_attempt < max_format_retries - 1:
                        persona_name = self.session.persona_id if self.session else "Agent"
                        format_guidance = (
                            f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                            f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotions+> ==] then continue with prose.]"
                        )
                        # Append guidance to the last user turn
                        if chat_turns and chat_turns[-1]["role"] == "user":
                            chat_turns[-1]["content"] += format_guidance
                        else:
                            chat_turns.append({"role": "user", "content": format_guidance})

                extracted_text = self._extract_speak_text_from_tool_call(cleaned_response)
                if extracted_text is not None:
                    logger.debug(
                        "Phase2 response looked like a tool call; extracted speak text (%d chars)",
                        len(extracted_text),
                    )
                normalized = self._normalize_response(
                    extracted_text if extracted_text is not None else cleaned_response
                )

                if normalized:
                    action = MUDAction(tool="speak", args={"text": normalized})
                    actions_taken.append(action)
                    logger.info("Prepared speak action (%d chars)", len(normalized))
                    await self._emit_actions(actions_taken)
                else:
                    logger.info("No response content to emit")

            elif decision_tool == "confused":
                # Phase 1 failed to parse a valid decision - emit confused emote
                logger.info("Phase 1 returned confused; emitting confused emote")
                action = MUDAction(tool="emote", args={"action": "looks confused."})
                actions_taken.append(action)
                await self._emit_actions(actions_taken)

            else:
                # Unknown decision tool - log warning and skip
                logger.warning(
                    "Unknown phase 1 decision tool '%s'; skipping turn",
                    decision_tool,
                )

        except Exception as e:
            logger.error(f"Error during LLM inference: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] LLM inference failed: {e}")
            raw_responses.append(f"[ERROR] LLM inference failed: {e}")
            # Emit a graceful emote when LLM fails
            action = MUDAction(tool="emote", args={"action": "was at a loss for words."})
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()

        # Push assistant turn to conversation list - ONLY for speak actions
        # Non-speak actions (move/take/drop/give) are mechanical tool calls,
        # not narrative content, so we don't save them to conversation history.
        if self.conversation_manager:
            for action in actions_taken:
                if action.tool == "speak":
                    speak_text = action.args.get("text", "")
                    if speak_text:
                        await self.conversation_manager.push_assistant_turn(
                            content=speak_text,
                            think=thinking if thinking else None,
                            actions=actions_taken,
                        )
                    break

        # Step 7: Create turn record
        turn = MUDTurn(
            timestamp=_utc_now(),
            events_received=events,
            room_context=self.session.current_room,
            entities_context=self.session.entities_present,
            thinking=thinking,
            actions_taken=actions_taken,
        )

        # Add turn to session history
        self.session.add_turn(turn)
        self.session.clear_pending_events()

        logger.info(
            f"Turn processed. Actions: {len(actions_taken)}. "
            f"Session now has {len(self.session.recent_turns)} turns"
        )

    async def process_agent_turn(self: "MUDAgentWorker", events: list[MUDEvent], user_guidance: str) -> None:
        """Process a guided @agent turn using mud_agent.yaml action schema.

        Originally from worker.py lines 1644-1869
        """
        logger.info(f"Processing @agent turn with {len(events)} events")

        # Refresh world state snapshot from agent + room profiles
        room_id, character_id = await self._load_agent_world_state()
        if not room_id and self.session.current_room and self.session.current_room.room_id:
            room_id = self.session.current_room.room_id
        if not room_id and events:
            room_id = events[-1].room_id
        await self._load_room_profile(room_id, character_id)

        for event in events:
            logger.info(
                f"  Event: {event.event_type.value} | "
                f"Actor: {event.actor} | "
                f"Room: {event.room_name or event.room_id} | "
                f"Content: {event.content[:100] if event.content else '(none)'}..."
            )

        self.session.pending_events = events
        if events:
            latest = events[-1]
            self.session.last_event_time = latest.timestamp

        # Push user turn to conversation list
        if self.conversation_manager and events:
            await self.conversation_manager.push_user_turn(
                events=events,
                world_state=self.session.world_state,
                room_id=self.session.current_room.room_id if self.session.current_room else None,
                room_name=self.session.current_room.name if self.session.current_room else None,
            )

        guidance = self._build_agent_guidance(user_guidance)
        coming_online = await self._is_fresh_session()

        # Build user input with current context and agent guidance
        # Events already pushed to conversation history, so exclude here
        user_input = build_current_context(
            self.session,
            idle_mode=False,
            guidance=guidance,
            coming_online=coming_online,
            include_events=False,
        )

        # Use response strategy for full context (consciousness + memory)
        chat_turns = await self._response_strategy.build_turns(
            persona=self.persona,
            user_input=user_input,
            session=self.session,
            coming_online=coming_online,
            max_context_tokens=self.model.max_tokens,
            max_output_tokens=self.chat_config.max_tokens,
        )

        actions_taken: list[MUDAction] = []
        raw_responses: list[str] = []
        thinking_parts: list[str] = []

        allowed = {a.get("name", "").lower() for a in self._agent_action_list() if a.get("name")}

        try:
            for attempt in range(self.config.decision_max_retries):
                # Check for abort before @agent LLM call
                if await self._check_abort_requested():
                    from .main import AbortRequestedException
                    raise AbortRequestedException("Turn aborted before @agent action")

                response = await self._call_llm(chat_turns)
                raw_responses.append(response)
                cleaned, think_content = extract_think_tags(response)
                cleaned = sanitize_response(cleaned)
                cleaned = cleaned.strip()
                if think_content:
                    thinking_parts.append(think_content)

                action, args, error = self._parse_agent_action_response(cleaned)
                if not action:
                    logger.warning(
                        "Invalid @agent response (attempt %d/%d): %s",
                        attempt + 1,
                        self.config.decision_max_retries,
                        error,
                    )
                    continue

                if allowed and action not in allowed:
                    logger.warning(
                        "Invalid @agent action '%s' (attempt %d/%d)",
                        action,
                        attempt + 1,
                        self.config.decision_max_retries,
                    )
                    continue

                # Valid action -> emit
                if action == "speak":
                    text = args.get("text", "")
                    # Validate emotional state header for speak actions
                    if not self._has_emotional_state_header(text):
                        logger.warning(
                            "Agent speak missing Emotional State header (attempt %d/%d)",
                            attempt + 1,
                            self.config.decision_max_retries,
                        )
                        if attempt < self.config.decision_max_retries - 1:
                            persona_name = self.session.persona_id if self.session else "Agent"
                            format_guidance = (
                                f"\n\n[Gentle reminder from your link: Please begin with your emotional state, "
                                f"e.g. [== {persona_name}'s Emotional State: <list of your +Emotion+> ==] then continue with prose.]"
                            )
                            if chat_turns and chat_turns[-1]["role"] == "user":
                                chat_turns[-1]["content"] += format_guidance
                            else:
                                chat_turns.append({"role": "user", "content": format_guidance})
                            continue
                    normalized = self._normalize_response(text)
                    if normalized:
                        action_obj = MUDAction(tool="speak", args={"text": normalized})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.info("Agent speak had no text; no action emitted")
                    break

                if action == "move":
                    location = args.get("location") or args.get("direction")
                    resolved = self._resolve_move_location(location)
                    if resolved:
                        action_obj = MUDAction(tool="move", args={"location": resolved})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent move missing/invalid location")
                    break

                if action == "take":
                    obj = args.get("object")
                    if obj:
                        action_obj = MUDAction(tool="get", args={"object": obj})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent take missing object")
                    break

                if action == "drop":
                    obj = args.get("object")
                    if obj:
                        action_obj = MUDAction(tool="drop", args={"object": obj})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent drop missing object")
                    break

                if action == "give":
                    obj = args.get("object")
                    target = args.get("target")
                    if obj and target:
                        action_obj = MUDAction(tool="give", args={"object": obj, "target": target})
                        actions_taken.append(action_obj)
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent give missing object or target")
                    break

                if action == "describe":
                    target = args.get("target")
                    description = args.get("description")
                    if target and description:
                        action_obj = MUDAction(
                            tool="describe",
                            args={"target": target, "description": description},
                        )
                        actions_taken.append(action_obj)
                        object_name = self._resolve_target_name(target)
                        emote_text = f"adjusted the {object_name}."
                        actions_taken.append(
                            MUDAction(tool="emote", args={"action": emote_text})
                        )
                        await self._emit_actions(actions_taken)
                    else:
                        logger.warning("Agent describe missing target or description")
                    break

        except Exception as e:
            logger.error(f"Error during @agent LLM inference: {e}", exc_info=True)
            thinking_parts.append(f"[ERROR] LLM inference failed: {e}")
            raw_responses.append(f"[ERROR] LLM inference failed: {e}")
            # Emit a graceful emote when LLM fails
            action = MUDAction(tool="emote", args={"action": "was at a loss for words."})
            actions_taken.append(action)
            await self._emit_actions(actions_taken)

        thinking = "\n\n".join(thinking_parts).strip()

        # Push assistant turn to conversation list - ONLY for speak actions
        # Non-speak actions (move/take/drop/give/describe) are mechanical tool calls,
        # not narrative content, so we don't save them to conversation history.
        if self.conversation_manager:
            for action in actions_taken:
                if action.tool == "speak":
                    speak_text = action.args.get("text", "")
                    if speak_text:
                        await self.conversation_manager.push_assistant_turn(
                            content=speak_text,
                            think=thinking if thinking else None,
                            actions=actions_taken,
                        )
                    break

        turn = MUDTurn(
            timestamp=_utc_now(),
            events_received=events,
            room_context=self.session.current_room,
            entities_context=self.session.entities_present,
            thinking=thinking,
            actions_taken=actions_taken,
        )

        self.session.add_turn(turn)
        self.session.clear_pending_events()

        logger.info(
            f"@agent turn processed. Actions: {len(actions_taken)}. "
            f"Session now has {len(self.session.recent_turns)} turns"
        )
