# andimud_mediator/mixins/agents.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Agents mixin for the mediator service."""

import logging
from datetime import datetime, timezone
from typing import Optional

from aim.conversation.model import ConversationModel
from aim_mud_types import MUDTurnRequest, TurnRequestStatus, TurnReason
from aim_mud_types.helper import _utc_now

logger = logging.getLogger(__name__)


class AgentsMixin:
    """Agents mixin for the mediator service."""

    async def _check_agent_states(self) -> None:
        """Check agent states when no events arrive (XREAD timeout).

        Checks for:
        1. Failed turns past retry time -> trigger retry
        2. Stale heartbeats (>5min) -> mark as crashed
        """
        for agent_id in self.registered_agents:
            try:
                turn_request = await self._get_turn_request(agent_id)
                if not turn_request:
                    continue

                status = turn_request.status

                # Case 1: Failed/retry turn ready to retry
                if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
                    if turn_request.next_attempt_at:
                        next_attempt_at = datetime.fromisoformat(turn_request.next_attempt_at)
                        if datetime.now(timezone.utc) >= next_attempt_at:
                            # Check if ANY agent is busy before assigning retry
                            # This enforces one-at-a-time concurrency
                            if await self._any_agent_processing():
                                logger.debug(f"Skipping retry for {agent_id} - another agent is processing")
                                continue

                            logger.info(f"Retrying turn for {agent_id} (was {status.value})")
                            await self._maybe_assign_turn(agent_id, reason=TurnReason.RETRY)

                # Case 2: Stale heartbeat (worker crashed)
                elif status == TurnRequestStatus.IN_PROGRESS:
                    if turn_request.heartbeat_at:
                        stale_seconds = (datetime.now(timezone.utc) - turn_request.heartbeat_at).total_seconds()
                        if stale_seconds > 300:  # 5 minutes
                            stale_duration = f"{int(stale_seconds // 60)}m{int(stale_seconds % 60)}s"

                            from aim_mud_types.turn_request_helpers import (
                                transition_turn_request_and_update_async,
                            )
                            updated = await transition_turn_request_and_update_async(
                                self.redis,
                                agent_id,
                                turn_request,
                                expected_turn_id=turn_request.turn_id,
                                status=TurnRequestStatus.CRASHED,
                                status_reason=f"Heartbeat stale for {stale_duration}",
                                set_completed=True,
                                update_heartbeat=False,
                            )

                            if updated:
                                logger.error(
                                    f"Worker {agent_id} crashed (no heartbeat for {stale_seconds:.0f}s)"
                                )
                            else:
                                logger.debug(
                                    f"Agent '{agent_id}' turn_request missing or changed, skipping crash detection"
                                )
            except Exception as e:
                logger.error(f"Error checking state for {agent_id}: {e}", exc_info=True)
                continue

        # Check for auto-analysis trigger
        await self._check_auto_analysis_trigger()

    async def _any_agent_processing(self) -> bool:
        """Check if any agent is currently processing a turn.

        Used to enforce one-at-a-time concurrency across all assignment paths.

        Returns:
            True if any agent is in ASSIGNED, IN_PROGRESS, EXECUTING, or EXECUTE status
        """
        for agent_id in self.registered_agents:
            turn_request = await self._get_turn_request(agent_id)
            if not turn_request:
                continue

            if turn_request.status in (
                TurnRequestStatus.ASSIGNED,
                TurnRequestStatus.IN_PROGRESS,
                TurnRequestStatus.EXECUTING,
                TurnRequestStatus.EXECUTE,
                TurnRequestStatus.ABORT_REQUESTED,
            ):
                return True

        return False

    async def _maybe_assign_turn(self, agent_id: str, reason: "str | TurnReason" = TurnReason.EVENTS) -> bool:
        """Assign turn if agent is available (including ready to retry).

        Checks agent availability:
        - Worker offline: no turn_request hash exists
        - Worker paused: mud:agent:{id}:paused key is set to "1"
        - Worker crashed: turn_request status is "crashed"
        - Worker busy: turn_request status is "assigned", "in_progress", or "abort_requested"
        - Failed turn in backoff: status is "fail" and current time < next_attempt_at

        Returns:
            True if turn was assigned, False if agent is busy/offline/crashed/paused.
        """
        current = await self._get_turn_request(agent_id)

        # No turn_request = worker offline
        if not current:
            logger.debug(f"Agent {agent_id} offline (no turn_request)")
            return False

        # Check if agent is paused via Redis key
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        is_paused = await client.is_agent_paused(agent_id)
        if is_paused:
            logger.debug(f"Agent {agent_id} paused, not assigning turn")
            return False

        status = current.status

        # Block crashed workers
        if status == TurnRequestStatus.CRASHED:
            logger.debug(f"Agent {agent_id} crashed, not assigning")
            return False

        # Block if actively processing or being aborted
        if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                      TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING):
            return False

        # Check if retry/fail turn in backoff period
        if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
            if current.next_attempt_at:
                next_attempt_at = datetime.fromisoformat(current.next_attempt_at)
                if datetime.now(timezone.utc) < next_attempt_at:
                    # Still in backoff period - not available
                    return False
            # Else: no next_attempt_at (max attempts) or past retry time, fall through to assign

        # Available: status is "ready" or "fail" past retry time
        # Convert reason to TurnReason enum
        turn_reason_enum = reason if isinstance(reason, TurnReason) else TurnReason(reason)

        from aim_mud_types.turn_request_helpers import assign_turn_request_async

        # Determine initial status based on reason
        # Immediate commands (FLUSH, CLEAR, NEW) get EXECUTE status for priority handling
        if turn_reason_enum.is_immediate_command():
            initial_status = TurnRequestStatus.EXECUTE
        else:
            initial_status = TurnRequestStatus.ASSIGNED

        # Preserve attempt_count and metadata if retrying a failed turn
        attempt_count = 0
        metadata = None
        if status in (TurnRequestStatus.RETRY, TurnRequestStatus.FAIL):
            attempt_count = current.attempt_count
            metadata = current.metadata  # Preserve metadata on retry

            # Validate: DREAM turns require metadata with scenario
            if current.reason == TurnReason.DREAM and not metadata:
                logger.warning(
                    f"DREAM turn for {agent_id} has no metadata, marking as FAIL"
                )
                from aim_mud_types.turn_request_helpers import (
                    transition_turn_request_and_update_async,
                )
                await transition_turn_request_and_update_async(
                    self.redis,
                    agent_id,
                    current,
                    expected_turn_id=current.turn_id,
                    status=TurnRequestStatus.FAIL,
                    message="Dream turn missing metadata",
                    next_attempt_at=None,
                    set_completed=True,
                    update_heartbeat=False,
                )
                return False
            if current.reason == TurnReason.DREAM and metadata and not metadata.get("scenario"):
                logger.warning(
                    f"DREAM turn for {agent_id} metadata missing scenario, marking as FAIL"
                )
                from aim_mud_types.turn_request_helpers import (
                    transition_turn_request_and_update_async,
                )
                await transition_turn_request_and_update_async(
                    self.redis,
                    agent_id,
                    current,
                    expected_turn_id=current.turn_id,
                    status=TurnRequestStatus.FAIL,
                    message="Dream turn missing scenario in metadata",
                    next_attempt_at=None,
                    set_completed=True,
                    update_heartbeat=False,
                )
                return False

        success, turn_request, result = await assign_turn_request_async(
            self.redis,
            agent_id,
            turn_reason_enum,
            attempt_count=attempt_count,
            status=initial_status,
            expected_turn_id=current.turn_id,
            skip_availability_check=True,
            **(metadata or {}),
        )

        if success and turn_request:
            logger.info(
                f"Assigned turn to {agent_id} "
                f"(sequence_id={turn_request.sequence_id}, "
                f"status={turn_request.status.value}, "
                f"reason={turn_request.reason.value}, "
                f"attempt={turn_request.attempt_count})"
            )
            return True
        else:
            # CAS failed - state changed between check and assign
            logger.debug(f"Assign failed for {agent_id}: {result}")
            return False

    async def _get_turn_request(self, agent_id: str) -> Optional[MUDTurnRequest]:
        """Fetch the current turn request hash for an agent."""
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)
        return await client.get_turn_request(agent_id)

    async def _agents_from_room_profile(self, room_id: str) -> list[str]:
        """Lookup agent_ids present in a room profile."""
        if not room_id:
            return []
        try:
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            room_profile = await client.get_room_profile(room_id)
            if not room_profile:
                return []

            agent_ids: set[str] = set()
            for entity in room_profile.entities:
                if entity.entity_type != "ai":
                    continue
                if entity.agent_id:
                    agent_ids.add(str(entity.agent_id))
            return list(agent_ids)
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return []

    async def _agent_id_from_actor(self, room_id: str, actor_id: str) -> Optional[str]:
        """Lookup agent_id for an actor by their entity_id (dbref).

        Args:
            room_id: The room where the event occurred.
            actor_id: The actor's entity_id (e.g., "#3").

        Returns:
            The agent_id if the actor is an AI agent, None otherwise.
        """
        if not room_id or not actor_id:
            return None
        try:
            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            room_profile = await client.get_room_profile(room_id)
            if not room_profile:
                return None

            for entity in room_profile.entities:
                if entity.entity_id == actor_id:
                    return entity.agent_id
            return None
        except Exception as e:
            logger.error(f"Failed to read room profile for {room_id}: {e}")
            return None

    def register_agent(self, agent_id: str, initial_room: str = "") -> None:
        """Register an agent with the mediator.

        Args:
            agent_id: Unique identifier for the agent.
            initial_room: Optional initial room ID for the agent.
        """
        self.registered_agents.add(agent_id)
        logger.info(
            f"Registered agent {agent_id}"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the mediator.

        Args:
            agent_id: Agent ID to unregister.
        """
        self.registered_agents.discard(agent_id)
        logger.info(f"Unregistered agent {agent_id}")

    async def _check_auto_analysis_trigger(self) -> None:
        """Check if we should trigger auto-analysis.

        Uses completed_at field to track idle duration persistently.
        No in-memory state needed - survives mediator restarts.
        """
        if not self.config.auto_analysis_enabled:
            return

        if await self._is_paused():
            return

        now = _utc_now()

        # Check cooldown
        elapsed = (now - self._last_auto_analysis_check).total_seconds()
        if elapsed < self.config.auto_analysis_cooldown_seconds:
            return

        # Check if ALL non-sleeping agents are idle
        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        # Track whether all agents are sleeping - important for handling
        # cases where streams don't exist (system reset) but agents should
        # still receive analysis tasks overnight
        all_agents_sleeping = True

        for agent_id in self.registered_agents:
            # Skip sleeping agents
            is_sleeping = await client.get_agent_is_sleeping(agent_id)
            if is_sleeping:
                continue

            # If we get here, at least one agent is NOT sleeping
            all_agents_sleeping = False

            turn_request = await self._get_turn_request(agent_id)
            if not turn_request:
                # Agent offline - skip it (offline agents don't affect idle state)
                continue

            status = turn_request.status

            # Active states block idle detection
            if status in (TurnRequestStatus.ASSIGNED, TurnRequestStatus.IN_PROGRESS,
                          TurnRequestStatus.ABORT_REQUESTED, TurnRequestStatus.EXECUTING,
                          TurnRequestStatus.EXECUTE):
                return  # System not idle

            # RETRY in backoff blocks idle detection
            if status == TurnRequestStatus.RETRY:
                if turn_request.next_attempt_at:
                    try:
                        next_attempt = datetime.fromisoformat(turn_request.next_attempt_at)
                        if datetime.now(timezone.utc) < next_attempt:
                            return  # Still in backoff - not idle
                    except (ValueError, TypeError):
                        pass  # Invalid timestamp, treat as idle

            # FAIL in backoff blocks idle detection (backward compatibility)
            if status == TurnRequestStatus.FAIL:
                if turn_request.next_attempt_at:
                    try:
                        next_attempt = datetime.fromisoformat(turn_request.next_attempt_at)
                        if datetime.now(timezone.utc) < next_attempt:
                            return  # Still in backoff - not idle
                    except (ValueError, TypeError):
                        pass

        # Query streams for last activity
        try:
            from aim_mud_types import RedisKeys
            import redis.exceptions

            # Get stream info to extract last message timestamps
            events_info = await self.redis.xinfo_stream(RedisKeys.MUD_EVENTS)
            actions_info = await self.redis.xinfo_stream(RedisKeys.MUD_ACTIONS)

            # Extract last-generated-id from stream info
            last_event_id = events_info.get("last-generated-id") or events_info.get(b"last-generated-id")
            last_action_id = actions_info.get("last-generated-id") or actions_info.get(b"last-generated-id")

            # Decode if bytes
            if isinstance(last_event_id, bytes):
                last_event_id = last_event_id.decode("utf-8")
            if isinstance(last_action_id, bytes):
                last_action_id = last_action_id.decode("utf-8")

            # Parse timestamps from stream IDs (format: "timestamp_ms-sequence")
            timestamps = []

            if last_event_id and last_event_id not in ("0-0", "0"):
                event_timestamp_ms = int(last_event_id.split("-")[0])
                timestamps.append(datetime.fromtimestamp(event_timestamp_ms / 1000, tz=timezone.utc))

            if last_action_id and last_action_id not in ("0-0", "0"):
                action_timestamp_ms = int(last_action_id.split("-")[0])
                timestamps.append(datetime.fromtimestamp(action_timestamp_ms / 1000, tz=timezone.utc))

            if not timestamps:
                # No events or actions yet - streams exist but are empty
                if all_agents_sleeping:
                    # All agents sleeping with no stream activity = treat as infinitely idle
                    # This allows overnight analysis to proceed after system reset
                    logger.info(
                        "Auto-analysis: no stream activity, but all agents sleeping - "
                        "treating as idle"
                    )
                    idle_duration = float('inf')
                else:
                    # Non-sleeping agents should create activity eventually
                    logger.debug("Auto-analysis: no events or actions in streams yet")
                    return
            else:
                # Calculate idle time from most recent activity
                # (Handles case where one stream is empty, e.g., user event but no agent action yet)
                last_activity = max(timestamps)
                idle_duration = (now - last_activity).total_seconds()

        except redis.exceptions.ResponseError as e:
            # Stream doesn't exist - system just started or streams not initialized
            if all_agents_sleeping:
                # All agents sleeping with no streams = treat as infinitely idle
                # This allows overnight analysis to proceed after system reset
                logger.info(
                    f"Auto-analysis: streams not initialized, but all agents sleeping - "
                    f"treating as idle (error: {e})"
                )
                idle_duration = float('inf')
            else:
                # Non-sleeping agents should create streams with activity
                logger.debug(f"Auto-analysis: streams not yet initialized (first startup): {e}")
                return
        except Exception as e:
            logger.error(f"Auto-analysis: error checking stream timestamps: {e}", exc_info=True)
            return

        if idle_duration < self.config.auto_analysis_idle_seconds:
            logger.debug(
                f"Auto-analysis: system idle for {idle_duration:.0f}s "
                f"(threshold: {self.config.auto_analysis_idle_seconds}s)"
            )
            return

        # Threshold reached!
        logger.info(f"Auto-analysis: triggered after {idle_duration:.0f}s idle")
        self._last_auto_analysis_check = now

        await self._scan_for_unanalyzed_conversations()

    async def _refresh_conversation_reports(self) -> None:
        """Refresh conversation reports for all registered agents.

        Generates fresh conversation reports and stores them in Redis
        for use by auto-analysis scanning. This ensures we're scanning
        with up-to-date conversation data.
        """
        if not self.registered_agents:
            return

        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        refreshed_count = 0
        for agent_id in self.registered_agents:
            try:
                # Get agent profile to lookup persona_id
                profile = await client.get_agent_profile(agent_id)
                if not profile or not profile.persona_id:
                    logger.debug(f"Auto-analysis: no persona_id for {agent_id}, skipping report refresh")
                    continue

                # Create ConversationModel for this agent
                # Using defaults similar to ChatConfig defaults
                memory_path = f"memory/{profile.persona_id}"
                cvm = ConversationModel(
                    memory_path=memory_path,
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    user_timezone=None,
                    embedding_device=None,
                )

                # Generate fresh conversation report
                report_df = cvm.get_conversation_report()

                if not report_df.empty:
                    report_dict = report_df.set_index('conversation_id').T.to_dict()
                else:
                    report_dict = {}

                await client.set_conversation_report(agent_id, report_dict)
                refreshed_count += 1

            except Exception as e:
                logger.error(
                    f"Auto-analysis: failed to refresh conversation report for {agent_id}: {e}",
                    exc_info=True
                )
                continue

        logger.debug(f"Auto-analysis: refreshed conversation reports for {refreshed_count} agent(s)")

    async def _should_summarize_before_analysis(
        self,
        agent_id: str,
        conversation_id: str,
        doc_counts: dict
    ) -> tuple[bool, str]:
        """Check if conversation needs summarization before analysis.

        Args:
            agent_id: Target agent ID
            conversation_id: Conversation to check
            doc_counts: Document counts from conversation report

        Returns:
            (should_summarize, reason)
            - (False, "has_summary"): Already summarized, can analyze
            - (False, "under_threshold"): Context small enough, can analyze
            - (True, "over_threshold"): Context too large, summarize first
        """
        from aim.utils.tokens import count_tokens
        from aim.llm.models import LanguageModelV2
        from aim_mud_types.client import RedisMUDClient

        # If conversation has summary documents, always analyze
        # (summary compressed the context already)
        has_summary = doc_counts.get("summary", 0) > 0
        if has_summary:
            return False, "has_summary"

        # Get agent profile to find persona_id
        client = RedisMUDClient(self.redis)
        profile = await client.get_agent_profile(agent_id)
        if not profile or not profile.persona_id:
            logger.warning(
                f"Auto-analysis: no persona_id for {agent_id}, "
                f"skipping context check"
            )
            return False, "no_profile"

        try:
            # Load CVM for this agent
            from aim.conversation.model import ConversationModel

            memory_path = f"memory/{profile.persona_id}"
            cvm = ConversationModel(
                memory_path=memory_path,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                user_timezone=None,
                embedding_device=None,
            )

            # Load conversation documents
            query_result = cvm.query(
                conversation_id=conversation_id,
                document_types=['conversation', 'summary', 'mud-world', 'mud-agent'],
                limit=1000  # Safety limit
            )

            # Count tokens in all documents
            total_tokens = 0
            for doc in query_result:
                content = doc.get('content', '')
                total_tokens += count_tokens(content)

            logger.debug(
                f"Auto-analysis: conversation {conversation_id} has "
                f"{len(query_result)} docs, {total_tokens} tokens"
            )

            # Get model context window and calculate threshold
            # Using config default model (usually set in MediatorConfig)
            models = LanguageModelV2.index_models(self.chat_config)
            default_model_name = getattr(self.chat_config, 'default_model', 'qwen2.5:7b')
            model = models.get(default_model_name)

            if not model:
                logger.warning(
                    f"Auto-analysis: model {default_model_name} not found, "
                    f"using default context window 32768"
                )
                model_context_window = 32768
            else:
                model_context_window = model.max_tokens

            # Threshold: 80% of (context_window - safety_margin)
            # TODO: Make ratio configurable via MediatorConfig (default 0.8)
            threshold_ratio = 0.8
            safety_margin = 1024 + 4096  # System prompt + typical analysis output
            effective_window = model_context_window - safety_margin
            threshold = int(effective_window * threshold_ratio)

            should_summarize = total_tokens > threshold

            if should_summarize:
                logger.info(
                    f"Auto-analysis: conversation {conversation_id} has {total_tokens} tokens > "
                    f"{threshold} threshold ({threshold_ratio:.1%} of {model_context_window}) - "
                    f"will summarize first"
                )
                return True, "over_threshold"
            else:
                logger.debug(
                    f"Auto-analysis: conversation {conversation_id} has {total_tokens} tokens <= "
                    f"{threshold} threshold - can analyze directly"
                )
                return False, "under_threshold"

        except Exception as e:
            logger.error(
                f"Auto-analysis: error checking context size for {conversation_id}: {e}",
                exc_info=True
            )
            # Fail open - allow analysis without check
            return False, "check_failed"

    async def _scan_for_unanalyzed_conversations(self) -> None:
        """Scan agents for conversations needing analysis.

        Uses round-robin to assign analysis to ONE agent per trigger.
        Tries agents in order starting from self._turn_index until one
        successfully receives an assignment.

        Uses existing turn assignment infrastructure via _handle_analysis_command().
        """
        # Refresh conversation reports before scanning
        await self._refresh_conversation_reports()

        from aim_mud_types.client import RedisMUDClient
        client = RedisMUDClient(self.redis)

        # Convert set to list for indexing
        agents_list = list(self.registered_agents)
        if not agents_list:
            logger.debug("Auto-analysis: no registered agents")
            return

        n = len(agents_list)
        assigned_agent = None

        # Try each agent in round-robin order
        for i in range(n):
            agent_id = agents_list[(self._turn_index + i) % n]

            try:
                # Load cached conversation report from Redis
                report = await client.get_conversation_report(agent_id)
                if not report:
                    logger.debug(f"Auto-analysis: no conversation report for {agent_id}")
                    continue

                if not isinstance(report, dict):
                    logger.warning(
                        f"Auto-analysis: invalid report format for {agent_id} "
                        f"(expected dict, got {type(report).__name__})"
                    )
                    continue

                # Find unanalyzed conversations
                # Document types come from conversation report structure:
                # - "mud-world": documents from world events (user turns)
                # - "mud-agent": documents from agent actions (assistant turns)
                # - "analysis": analysis documents created by analysis_dialogue scenario
                # These column names come from ConversationModel.get_conversation_report()
                unanalyzed = []
                for conversation_id, doc_counts in report.items():
                    if not isinstance(doc_counts, dict):
                        logger.warning(
                            f"Auto-analysis: invalid doc_counts for {conversation_id} in {agent_id}"
                        )
                        continue

                    has_conversation_docs = doc_counts.get("conversation", 0) > 0
                    has_mud_docs = (
                        doc_counts.get("mud-world", 0) > 0
                        or doc_counts.get("mud-agent", 0) > 0
                    )
                    has_summary = doc_counts.get("summary", 0) > 0
                    has_analysis = doc_counts.get("analysis", 0) > 0

                    # Include conversations that:
                    # 1. Have conversation/MUD docs AND no analysis
                    # 2. Have summary docs AND no analysis (summarized but not yet analyzed)
                    needs_processing = (
                        (has_conversation_docs or has_mud_docs or has_summary)
                        and not has_analysis
                    )

                    if needs_processing:
                        timestamp = doc_counts.get("timestamp_max", "")
                        unanalyzed.append((conversation_id, timestamp))

                if not unanalyzed:
                    logger.debug(f"Auto-analysis: no unanalyzed conversations for {agent_id}")
                    continue

                # Sort by timestamp (oldest first)
                unanalyzed.sort(key=lambda x: x[1])
                conversation_id, timestamp = unanalyzed[0]

                logger.info(
                    f"Auto-analysis: found {len(unanalyzed)} unanalyzed conversation(s) "
                    f"for {agent_id}, checking oldest: {conversation_id}"
                )

                # Check if we need to summarize first
                # Get doc_counts for context size check
                doc_counts = report.get(conversation_id, {})
                should_summarize, reason = await self._should_summarize_before_analysis(
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    doc_counts=doc_counts
                )

                # Determine scenario based on context check
                if should_summarize:
                    scenario = "summarizer"
                    logger.info(
                        f"Auto-analysis: context too large ({reason}), "
                        f"assigning summarizer for {conversation_id}"
                    )
                else:
                    scenario = "analysis_dialogue"
                    logger.info(
                        f"Auto-analysis: context ok ({reason}), "
                        f"assigning analysis for {conversation_id}"
                    )

                # TODO - Phase 4: Implement self-turns where agents can initiate
                # their own processing without explicit conversation targets. This
                # will enable agents to autonomously explore topics, reflect on
                # recent experiences, or pursue creative initiatives during idle time.
                # Note: Self-turns should respect sleeping state (no self-turns for
                # sleeping agents), but analysis tasks are allowed for sleeping agents.

                # Assign turn using existing infrastructure
                success = await self._handle_analysis_command(
                    agent_id=agent_id,
                    scenario=scenario,
                    conversation_id=conversation_id,
                    guidance=None,
                )

                if success:
                    assigned_agent = agent_id
                    self._turn_index = (self._turn_index + i + 1) % n
                    logger.info(
                        f"Auto-analysis: assigned analysis to {agent_id} "
                        f"(round-robin index updated to {self._turn_index})"
                    )
                    break  # Stop after first successful assignment
                else:
                    logger.debug(
                        f"Auto-analysis: {agent_id} unavailable, trying next agent"
                    )

            except Exception as e:
                logger.error(
                    f"Auto-analysis: error scanning {agent_id}: {e}",
                    exc_info=True
                )
                continue

        if not assigned_agent:
            logger.debug("Auto-analysis: no agents had unanalyzed conversations or were available")
