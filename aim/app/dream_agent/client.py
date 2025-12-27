# aim/app/dream_agent/client.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""DreamerClient - Client for interacting with Dreamer pipelines.

Supports two modes:
1. Direct mode: Connect directly to Redis (for local development)
2. HTTP mode: Connect via REST API (for remote server)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal, AsyncIterator, Callable
import httpx
import redis.asyncio as redis

from ...config import ChatConfig
from ...dreamer.api import (
    start_pipeline,
    get_status,
    cancel_pipeline,
    resume_pipeline,
    list_pipelines,
    restart_from_step,
    get_restart_info,
    PipelineStatus,
)
from ...dreamer.state import StateStore
from ...dreamer.scheduler import Scheduler
from ...dreamer.scenario import load_scenario


@dataclass
class PipelineResult:
    """Result of a pipeline operation."""
    success: bool
    pipeline_id: Optional[str] = None
    message: Optional[str] = None
    status: Optional[PipelineStatus] = None
    error: Optional[str] = None


class DreamerClient:
    """Client for managing Dreamer pipelines.

    Supports both direct Redis connection and HTTP API modes.

    Usage (direct mode):
        async with DreamerClient.direct(config) as client:
            result = await client.start("analyst", "conv-123", "claude-3-5-sonnet")
            status = await client.watch(result.pipeline_id)

    Usage (HTTP mode):
        async with DreamerClient.http("http://localhost:8000", "api-key") as client:
            result = await client.start("analyst", "conv-123", "claude-3-5-sonnet")
            status = await client.watch(result.pipeline_id)
    """

    def __init__(
        self,
        mode: Literal["direct", "http"],
        config: Optional[ChatConfig] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize DreamerClient.

        Args:
            mode: Connection mode ("direct" or "http")
            config: ChatConfig for direct mode
            base_url: Base URL for HTTP mode (e.g., "http://localhost:8000")
            api_key: API key for HTTP mode authentication
        """
        self.mode = mode
        self.config = config
        self.base_url = base_url.rstrip("/") if base_url else None
        self.api_key = api_key

        # Direct mode resources
        self._redis_client: Optional[redis.Redis] = None
        self._state_store: Optional[StateStore] = None
        self._scheduler: Optional[Scheduler] = None

        # HTTP mode resources
        self._http_client: Optional[httpx.AsyncClient] = None

    @classmethod
    def direct(cls, config: ChatConfig) -> "DreamerClient":
        """Create a client in direct Redis mode.

        Args:
            config: ChatConfig with Redis connection settings

        Returns:
            DreamerClient configured for direct mode
        """
        return cls(mode="direct", config=config)

    @classmethod
    def http(cls, base_url: str, api_key: str) -> "DreamerClient":
        """Create a client in HTTP API mode.

        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8000")
            api_key: API key for authentication

        Returns:
            DreamerClient configured for HTTP mode
        """
        return cls(mode="http", base_url=base_url, api_key=api_key)

    async def __aenter__(self) -> "DreamerClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Establish connections based on mode."""
        if self.mode == "direct":
            if self.config is None:
                raise ValueError("ChatConfig required for direct mode")

            self._redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=False,
            )
            self._state_store = StateStore(self._redis_client, key_prefix="dreamer")
            self._scheduler = Scheduler(self._redis_client, self._state_store)

        elif self.mode == "http":
            if not self.base_url or not self.api_key:
                raise ValueError("base_url and api_key required for HTTP mode")

            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )

    async def close(self) -> None:
        """Close connections."""
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            self._state_store = None
            self._scheduler = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def start(
        self,
        scenario_name: str,
        conversation_id: str,
        model_name: str,
        query_text: Optional[str] = None,
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        guidance: Optional[str] = None,
        context_documents: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Start a new pipeline.

        Args:
            scenario_name: Name of the scenario (e.g., "analyst", "journaler")
            conversation_id: Conversation ID to process
            model_name: Model to use (e.g., "claude-3-5-sonnet")
            query_text: Optional query text for journaler/philosopher
            persona_id: Optional persona ID
            user_id: Optional user ID (defaults to persona_id if not set)
            guidance: Optional guidance text for the pipeline
            context_documents: Optional list of context documents to pass to the pipeline

        Returns:
            PipelineResult with pipeline_id on success
        """
        if self.mode == "direct":
            return await self._start_direct(
                scenario_name, conversation_id, model_name, query_text, persona_id, user_id, guidance, context_documents
            )
        else:
            return await self._start_http(
                scenario_name, conversation_id, model_name, query_text, persona_id, user_id, guidance, context_documents
            )

    async def _start_direct(
        self,
        scenario_name: str,
        conversation_id: str,
        model_name: str,
        query_text: Optional[str],
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        guidance: Optional[str] = None,
        context_documents: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Start pipeline via direct Redis connection."""
        try:
            pipeline_id = await start_pipeline(
                scenario_name=scenario_name,
                conversation_id=conversation_id,
                config=self.config,
                model_name=model_name,
                state_store=self._state_store,
                scheduler=self._scheduler,
                query_text=query_text,
                persona_id=persona_id,
                user_id=user_id,
                guidance=guidance,
                context_documents=context_documents,
            )
            return PipelineResult(
                success=True,
                pipeline_id=pipeline_id,
                message=f"Pipeline {pipeline_id} started successfully",
            )
        except FileNotFoundError as e:
            return PipelineResult(success=False, error=f"Scenario not found: {e}")
        except ValueError as e:
            return PipelineResult(success=False, error=f"Validation error: {e}")
        except Exception as e:
            return PipelineResult(success=False, error=f"Failed to start pipeline: {e}")

    async def _start_http(
        self,
        scenario_name: str,
        conversation_id: str,
        model_name: str,
        query_text: Optional[str],
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        guidance: Optional[str] = None,
        context_documents: Optional[list[dict]] = None,
    ) -> PipelineResult:
        """Start pipeline via HTTP API."""
        try:
            response = await self._http_client.post(
                "/api/dreamer/pipeline",
                json={
                    "scenario_name": scenario_name,
                    "conversation_id": conversation_id,
                    "model_name": model_name,
                    "query_text": query_text,
                    "persona_id": persona_id,
                    "user_id": user_id,
                    "guidance": guidance,
                    "context_documents": context_documents,
                },
            )
            response.raise_for_status()
            data = response.json()
            return PipelineResult(
                success=True,
                pipeline_id=data["pipeline_id"],
                message=data.get("message"),
            )
        except httpx.HTTPStatusError as e:
            return PipelineResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def get_status(self, pipeline_id: str) -> PipelineResult:
        """Get the current status of a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            PipelineResult with status on success
        """
        if self.mode == "direct":
            return await self._get_status_direct(pipeline_id)
        else:
            return await self._get_status_http(pipeline_id)

    async def _get_status_direct(self, pipeline_id: str) -> PipelineResult:
        """Get status via direct Redis connection."""
        try:
            state = await self._state_store.load_state(pipeline_id)
            if state is None:
                return PipelineResult(
                    success=False, error=f"Pipeline {pipeline_id} not found"
                )

            scenario = load_scenario(state.scenario_name)
            status = await get_status(pipeline_id, self._state_store, scenario)

            if status is None:
                return PipelineResult(
                    success=False, error=f"Pipeline {pipeline_id} not found"
                )

            return PipelineResult(success=True, pipeline_id=pipeline_id, status=status)
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def _get_status_http(self, pipeline_id: str) -> PipelineResult:
        """Get status via HTTP API."""
        try:
            response = await self._http_client.get(
                f"/api/dreamer/pipeline/{pipeline_id}"
            )
            response.raise_for_status()
            data = response.json()

            status = PipelineStatus(
                pipeline_id=data["pipeline_id"],
                scenario_name=data["scenario_name"],
                status=data["status"],
                current_step=data.get("current_step"),
                completed_steps=data.get("completed_steps", []),
                failed_steps=data.get("failed_steps", []),
                progress_percent=data.get("progress_percent", 0.0),
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
            )

            return PipelineResult(success=True, pipeline_id=pipeline_id, status=status)
        except httpx.HTTPStatusError as e:
            return PipelineResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def cancel(self, pipeline_id: str) -> PipelineResult:
        """Cancel a running pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            PipelineResult indicating success or failure
        """
        if self.mode == "direct":
            return await self._cancel_direct(pipeline_id)
        else:
            return await self._cancel_http(pipeline_id)

    async def _cancel_direct(self, pipeline_id: str) -> PipelineResult:
        """Cancel via direct Redis connection."""
        try:
            success = await cancel_pipeline(pipeline_id, self._state_store)
            if success:
                return PipelineResult(
                    success=True,
                    pipeline_id=pipeline_id,
                    message=f"Pipeline {pipeline_id} cancelled",
                )
            else:
                return PipelineResult(
                    success=False, error=f"Pipeline {pipeline_id} not found"
                )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def _cancel_http(self, pipeline_id: str) -> PipelineResult:
        """Cancel via HTTP API."""
        try:
            response = await self._http_client.post(
                f"/api/dreamer/pipeline/{pipeline_id}/cancel"
            )
            response.raise_for_status()
            data = response.json()
            return PipelineResult(
                success=True,
                pipeline_id=pipeline_id,
                message=data.get("message"),
            )
        except httpx.HTTPStatusError as e:
            return PipelineResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def resume(self, pipeline_id: str) -> PipelineResult:
        """Resume a failed or cancelled pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            PipelineResult indicating success or failure
        """
        if self.mode == "direct":
            return await self._resume_direct(pipeline_id)
        else:
            return await self._resume_http(pipeline_id)

    async def _resume_direct(self, pipeline_id: str) -> PipelineResult:
        """Resume via direct Redis connection."""
        try:
            success = await resume_pipeline(
                pipeline_id, self._state_store, self._scheduler
            )
            if success:
                return PipelineResult(
                    success=True,
                    pipeline_id=pipeline_id,
                    message=f"Pipeline {pipeline_id} resumed",
                )
            else:
                return PipelineResult(
                    success=False, error=f"Pipeline {pipeline_id} not found"
                )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def _resume_http(self, pipeline_id: str) -> PipelineResult:
        """Resume via HTTP API."""
        try:
            response = await self._http_client.post(
                f"/api/dreamer/pipeline/{pipeline_id}/resume"
            )
            response.raise_for_status()
            data = response.json()
            return PipelineResult(
                success=True,
                pipeline_id=pipeline_id,
                message=data.get("message"),
            )
        except httpx.HTTPStatusError as e:
            return PipelineResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def list(
        self,
        status_filter: Optional[str] = None,
        limit: int = 100,
    ) -> list[PipelineStatus]:
        """List pipelines.

        Args:
            status_filter: Optional status filter (running, complete, failed, pending)
            limit: Maximum number of pipelines to return

        Returns:
            List of PipelineStatus objects
        """
        if self.mode == "direct":
            return await self._list_direct(status_filter, limit)
        else:
            return await self._list_http(status_filter, limit)

    async def _list_direct(
        self, status_filter: Optional[str], limit: int
    ) -> list[PipelineStatus]:
        """List via direct Redis connection."""
        try:
            return await list_pipelines(
                self._state_store, status=status_filter, limit=limit
            )
        except Exception:
            return []

    async def _list_http(
        self, status_filter: Optional[str], limit: int
    ) -> list[PipelineStatus]:
        """List via HTTP API."""
        try:
            params = {"limit": limit}
            if status_filter:
                params["status"] = status_filter

            response = await self._http_client.get(
                "/api/dreamer/pipelines", params=params
            )
            response.raise_for_status()
            data = response.json()

            return [
                PipelineStatus(
                    pipeline_id=p["pipeline_id"],
                    scenario_name=p["scenario_name"],
                    status=p["status"],
                    current_step=p.get("current_step"),
                    completed_steps=p.get("completed_steps", []),
                    failed_steps=p.get("failed_steps", []),
                    progress_percent=p.get("progress_percent", 0.0),
                    created_at=datetime.fromisoformat(p["created_at"]),
                    updated_at=datetime.fromisoformat(p["updated_at"]),
                )
                for p in data.get("pipelines", [])
            ]
        except Exception:
            return []

    async def watch(
        self,
        pipeline_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[PipelineStatus]:
        """Watch a pipeline until completion, yielding status updates.

        Args:
            pipeline_id: Pipeline identifier
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds

        Yields:
            PipelineStatus on each update

        Raises:
            TimeoutError: If timeout is reached before completion
        """
        start_time = time.time()
        last_progress = -1.0

        while True:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Pipeline {pipeline_id} did not complete in {timeout}s")

            result = await self.get_status(pipeline_id)

            if not result.success:
                raise RuntimeError(result.error)

            status = result.status

            # Yield if progress changed
            if status.progress_percent != last_progress:
                last_progress = status.progress_percent
                yield status

            # Check for terminal states
            if status.status in ("complete", "failed"):
                break

            await asyncio.sleep(poll_interval)

    async def run_and_wait(
        self,
        scenario_name: str,
        conversation_id: str,
        model_name: str,
        query_text: Optional[str] = None,
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        guidance: Optional[str] = None,
        context_documents: Optional[list[dict]] = None,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable] = None,
    ) -> PipelineResult:
        """Start a pipeline and wait for completion.

        Convenience method that combines start() and watch().

        Args:
            scenario_name: Name of the scenario
            conversation_id: Conversation ID to process
            model_name: Model to use
            query_text: Optional query text
            persona_id: Optional persona ID
            user_id: Optional user ID (defaults to persona_id if not set)
            guidance: Optional guidance text for the pipeline
            context_documents: Optional list of context documents
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds
            on_progress: Optional callback(status) for progress updates

        Returns:
            PipelineResult with final status
        """
        # Start the pipeline
        result = await self.start(scenario_name, conversation_id, model_name, query_text, persona_id, user_id, guidance, context_documents)

        if not result.success:
            return result

        pipeline_id = result.pipeline_id

        # Watch for completion
        try:
            async for status in self.watch(pipeline_id, poll_interval, timeout):
                if on_progress:
                    on_progress(status)

            # Get final status
            final_result = await self.get_status(pipeline_id)
            return final_result

        except TimeoutError as e:
            return PipelineResult(success=False, pipeline_id=pipeline_id, error=str(e))
        except Exception as e:
            return PipelineResult(success=False, pipeline_id=pipeline_id, error=str(e))

    async def restart(
        self,
        conversation_id: str,
        branch: int,
        step_id: str,
        model_name: str,
        scenario_name: Optional[str] = None,
        query_text: Optional[str] = None,
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        guidance: Optional[str] = None,
        mood: Optional[str] = None,
        include_all_history: bool = False,
        same_branch: bool = False,
    ) -> PipelineResult:
        """Restart a scenario pipeline from a specific step.

        This method allows replaying a scenario from a particular step by:
        1. Loading existing conversation data for the specified branch
        2. Inferring the scenario type (if not provided) from document types
        3. Creating a new pipeline that starts from the target step

        Args:
            conversation_id: ID of the conversation to restart from
            branch: Branch number to restart from
            step_id: Step ID to restart from (this step will be re-executed)
            model_name: Model to use for execution
            scenario_name: Optional scenario name (inferred from docs if not provided)
            query_text: Optional query text for journaler/philosopher
            persona_id: Optional persona ID
            user_id: Optional user ID
            guidance: Optional guidance text
            mood: Optional persona mood
            include_all_history: Load entire conversation (all branches) into context
            same_branch: Continue on the same branch instead of creating a new one

        Returns:
            PipelineResult with pipeline_id on success
        """
        if self.mode == "direct":
            return await self._restart_direct(
                conversation_id, branch, step_id, model_name,
                scenario_name, query_text, persona_id, user_id, guidance, mood,
                include_all_history, same_branch
            )
        else:
            return await self._restart_http(
                conversation_id, branch, step_id, model_name,
                scenario_name, query_text, persona_id, user_id, guidance, mood,
                include_all_history, same_branch
            )

    async def _restart_direct(
        self,
        conversation_id: str,
        branch: int,
        step_id: str,
        model_name: str,
        scenario_name: Optional[str],
        query_text: Optional[str],
        persona_id: Optional[str],
        user_id: Optional[str],
        guidance: Optional[str],
        mood: Optional[str],
        include_all_history: bool = False,
        same_branch: bool = False,
    ) -> PipelineResult:
        """Restart via direct Redis connection."""
        try:
            pipeline_id = await restart_from_step(
                conversation_id=conversation_id,
                branch=branch,
                step_id=step_id,
                config=self.config,
                model_name=model_name,
                state_store=self._state_store,
                scheduler=self._scheduler,
                scenario_name=scenario_name,
                persona_id=persona_id,
                user_id=user_id,
                query_text=query_text,
                guidance=guidance,
                mood=mood,
                include_all_history=include_all_history,
                same_branch=same_branch,
            )
            return PipelineResult(
                success=True,
                pipeline_id=pipeline_id,
                message=f"Pipeline {pipeline_id} started from step '{step_id}'",
            )
        except FileNotFoundError as e:
            return PipelineResult(success=False, error=f"Scenario not found: {e}")
        except ValueError as e:
            return PipelineResult(success=False, error=f"Validation error: {e}")
        except Exception as e:
            return PipelineResult(success=False, error=f"Failed to restart pipeline: {e}")

    async def _restart_http(
        self,
        conversation_id: str,
        branch: int,
        step_id: str,
        model_name: str,
        scenario_name: Optional[str],
        query_text: Optional[str],
        persona_id: Optional[str],
        user_id: Optional[str],
        guidance: Optional[str],
        mood: Optional[str],
        include_all_history: bool = False,
        same_branch: bool = False,
    ) -> PipelineResult:
        """Restart via HTTP API."""
        try:
            response = await self._http_client.post(
                "/api/dreamer/pipeline/restart",
                json={
                    "conversation_id": conversation_id,
                    "branch": branch,
                    "step_id": step_id,
                    "model_name": model_name,
                    "scenario_name": scenario_name,
                    "query_text": query_text,
                    "persona_id": persona_id,
                    "user_id": user_id,
                    "guidance": guidance,
                    "mood": mood,
                    "include_all_history": include_all_history,
                    "same_branch": same_branch,
                },
            )
            response.raise_for_status()
            data = response.json()
            return PipelineResult(
                success=True,
                pipeline_id=data["pipeline_id"],
                message=data.get("message"),
            )
        except httpx.HTTPStatusError as e:
            return PipelineResult(
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text}",
            )
        except Exception as e:
            return PipelineResult(success=False, error=str(e))

    async def inspect(
        self,
        conversation_id: str,
        branch: int,
        scenario_name: Optional[str] = None,
    ) -> dict:
        """Inspect a conversation branch to see restart options.

        Returns information about what scenario was run, which steps
        completed, and what restart points are available.

        Args:
            conversation_id: ID of the conversation
            branch: Branch number to inspect
            scenario_name: Optional scenario name (if not provided, will try to infer)

        Returns:
            Dict containing:
            - scenario_name: Inferred scenario name (or None)
            - doc_types: Set of document types found
            - step_outputs: Dict mapping step_id -> doc_id
            - available_restart_points: List of step_ids
        """
        if self.mode == "direct":
            return await self._inspect_direct(conversation_id, branch, scenario_name)
        else:
            return await self._inspect_http(conversation_id, branch, scenario_name)

    async def _inspect_direct(
        self, conversation_id: str, branch: int, scenario_name: Optional[str] = None
    ) -> dict:
        """Inspect via direct connection."""
        try:
            return await get_restart_info(
                conversation_id, branch, self.config, scenario_name
            )
        except Exception as e:
            return {"error": str(e)}

    async def _inspect_http(
        self, conversation_id: str, branch: int, scenario_name: Optional[str] = None
    ) -> dict:
        """Inspect via HTTP API."""
        try:
            params = {}
            if scenario_name:
                params["scenario"] = scenario_name
            response = await self._http_client.get(
                f"/api/dreamer/inspect/{conversation_id}/{branch}",
                params=params,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"error": str(e)}

    async def restart_and_wait(
        self,
        conversation_id: str,
        branch: int,
        step_id: str,
        model_name: str,
        scenario_name: Optional[str] = None,
        query_text: Optional[str] = None,
        persona_id: Optional[str] = None,
        user_id: Optional[str] = None,
        guidance: Optional[str] = None,
        mood: Optional[str] = None,
        include_all_history: bool = False,
        same_branch: bool = False,
        poll_interval: float = 2.0,
        timeout: Optional[float] = None,
        on_progress: Optional[Callable] = None,
    ) -> PipelineResult:
        """Restart a pipeline from a step and wait for completion.

        Convenience method that combines restart() and watch().

        Args:
            conversation_id: ID of the conversation to restart from
            branch: Branch number to restart from
            step_id: Step ID to restart from
            model_name: Model to use
            scenario_name: Optional scenario name (inferred if not provided)
            query_text: Optional query text
            persona_id: Optional persona ID
            user_id: Optional user ID
            guidance: Optional guidance text
            mood: Optional persona mood
            include_all_history: Load entire conversation (all branches) into context
            same_branch: Continue on the same branch instead of creating a new one
            poll_interval: Seconds between status checks
            timeout: Optional timeout in seconds
            on_progress: Optional callback(status) for progress updates

        Returns:
            PipelineResult with final status
        """
        # Restart the pipeline
        result = await self.restart(
            conversation_id, branch, step_id, model_name,
            scenario_name, query_text, persona_id, user_id, guidance, mood,
            include_all_history, same_branch
        )

        if not result.success:
            return result

        pipeline_id = result.pipeline_id

        # Watch for completion
        try:
            async for status in self.watch(pipeline_id, poll_interval, timeout):
                if on_progress:
                    on_progress(status)

            # Get final status
            final_result = await self.get_status(pipeline_id)
            return final_result

        except TimeoutError as e:
            return PipelineResult(success=False, pipeline_id=pipeline_id, error=str(e))
        except Exception as e:
            return PipelineResult(success=False, pipeline_id=pipeline_id, error=str(e))
