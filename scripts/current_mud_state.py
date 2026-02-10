#!/usr/bin/env python3
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Generate a quick MUD state report for idle-turn debugging."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional
import shlex
import subprocess

import redis


def _ensure_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    packages_root = repo_root / "packages" / "aim-mud" / "src"
    for path in (repo_root, packages_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_import_paths()

from aim_mud_types.client import SyncRedisMUDClient  # noqa: E402
from aim_mud_types.redis_keys import RedisKeys  # noqa: E402
from aim_mud_types.models.coordination import DreamingState, MUDTurnRequest  # noqa: E402


def _decode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    text = _decode(value)
    if text is None:
        return False
    return text.strip().lower() in ("1", "true", "yes", "on")


def _format_dt(dt: Optional[datetime]) -> str:
    if dt is None:
        return "n/a"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _age_seconds(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max(0.0, (now - dt).total_seconds())


def _format_age(dt: Optional[datetime]) -> str:
    age = _age_seconds(dt)
    if age is None:
        return "n/a"
    return f"{age:.1f}s"


def _stream_info(redis_client: redis.Redis, key: str) -> dict[str, Any]:
    try:
        info = redis_client.xinfo_stream(key)
    except Exception as e:
        return {"error": str(e)}

    last_id = info.get("last-generated-id") or info.get(b"last-generated-id")
    last_id = _decode(last_id) or "0-0"
    length = info.get("length") or info.get(b"length") or 0

    idle_seconds = None
    try:
        ts_ms = int(last_id.split("-")[0])
        now_ms = int(time.time() * 1000)
        idle_seconds = max(0.0, (now_ms - ts_ms) / 1000.0)
    except Exception:
        idle_seconds = None

    return {
        "last_id": last_id,
        "length": int(length),
        "idle_seconds": idle_seconds,
    }


def _get_processed_events_summary(client: SyncRedisMUDClient) -> dict[str, Any]:
    try:
        ids = client.get_mud_event_processed_ids()
    except Exception as e:
        return {"error": str(e)}

    if not ids:
        return {"count": 0}

    ids.sort()
    min_id = ids[0]
    max_id = ids[-1]

    # Extract timestamp from processed hash value if present
    value = client.redis.hget(RedisKeys.EVENTS_PROCESSED, max_id)
    value = _decode(value) or ""
    timestamp = None
    agents = None
    if value:
        parts = value.split("|", 1)
        if parts:
            timestamp = parts[0]
        if len(parts) > 1:
            agents = parts[1] or None

    return {
        "count": len(ids),
        "min_id": min_id,
        "max_id": max_id,
        "last_processed_at": timestamp,
        "last_processed_agents": agents,
    }


def _load_agent_rooms(redis_client: redis.Redis) -> dict[str, str]:
    raw = redis_client.hgetall(RedisKeys.AGENT_ROOMS)
    if not raw:
        return {}
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        k = _decode(k)
        v = _decode(v)
        if k is not None and v is not None:
            decoded[k] = v
    return decoded


def _get_process_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return {
            "lines": [],
            "mediator_agents": [],
            "worker_agents": [],
        }

    lines: list[str] = []
    mediator_agents: list[str] = []
    worker_agents: list[str] = []

    for line in result.stdout.splitlines():
        lower = line.lower()
        if not (
            "andimud_mediator" in lower
            or "aim.app.mud.mediator" in lower
            or "mud.mediator" in lower
            or "andimud_worker" in lower
            or "aim.app.mud.worker" in lower
            or "mud.worker" in lower
        ):
            continue

        line = line.strip()
        if not line:
            continue
        lines.append(line)

        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        cmd = parts[1]
        try:
            args = shlex.split(cmd)
        except ValueError:
            continue

        if "andimud_mediator" in lower or "aim.app.mud.mediator" in lower or "mud.mediator" in lower:
            if "--agents" in args:
                idx = args.index("--agents")
                for token in args[idx + 1 :]:
                    if token.startswith("--"):
                        break
                    mediator_agents.append(token)

        if "andimud_worker" in lower or "aim.app.mud.worker" in lower or "mud.worker" in lower:
            if "--agent-id" in args:
                idx = args.index("--agent-id")
                if idx + 1 < len(args):
                    worker_agents.append(args[idx + 1])

    return {
        "lines": lines,
        "mediator_agents": mediator_agents,
        "worker_agents": worker_agents,
    }


def _load_dreaming_state(redis_client: redis.Redis, agent_id: str) -> Optional[DreamingState]:
    key = RedisKeys.agent_dreaming_state(agent_id)
    raw = redis_client.hgetall(key)
    if not raw:
        return None
    decoded: dict[str, str] = {}
    for k, v in raw.items():
        k = _decode(k)
        v = _decode(v)
        if k is not None and v is not None:
            decoded[k] = v
    try:
        return DreamingState.model_validate(decoded)
    except Exception:
        return None


def _format_turn_request(turn_request: Optional[MUDTurnRequest]) -> list[str]:
    if turn_request is None:
        return ["turn_request: missing"]

    lines = [
        "turn_request:",
        f"status={turn_request.status.value} reason={turn_request.reason.value} sequence_id={turn_request.sequence_id}",
        f"turn_id={turn_request.turn_id} attempt={turn_request.attempt_count}",
        f"assigned_at={_format_dt(turn_request.assigned_at)} (age={_format_age(turn_request.assigned_at)})",
        f"heartbeat_at={_format_dt(turn_request.heartbeat_at)} (age={_format_age(turn_request.heartbeat_at)})",
    ]
    if turn_request.completed_at:
        lines.append(f"completed_at={_format_dt(turn_request.completed_at)} (age={_format_age(turn_request.completed_at)})")
    if turn_request.next_attempt_at:
        lines.append(f"next_attempt_at={_format_dt(turn_request.next_attempt_at)} (age={_format_age(turn_request.next_attempt_at)})")
    if turn_request.deadline_ms:
        lines.append(f"deadline_ms={turn_request.deadline_ms}")
    if turn_request.status_reason:
        lines.append(f"status_reason={turn_request.status_reason}")
    if turn_request.message:
        lines.append(f"message={turn_request.message}")
    if turn_request.metadata:
        lines.append(f"metadata={turn_request.metadata}")
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report current MUD mediator/worker state from Redis",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379"),
        help="Redis connection URL (default: env REDIS_URL or redis://localhost:6379)",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=None,
        help="Optional list of agent IDs to inspect (default: all with turn_request)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    redis_client = redis.from_url(args.redis_url, decode_responses=False)
    client = SyncRedisMUDClient(redis_client)

    now = datetime.now(timezone.utc)
    print(f"MUD State Report (UTC {now.strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"Redis URL: {args.redis_url}")

    mediator_paused = client.is_mediator_paused()
    print(f"Mediator paused: {mediator_paused}")

    events_info = _stream_info(redis_client, RedisKeys.MUD_EVENTS)
    if "error" in events_info:
        print(f"MUD events stream: error={events_info['error']}")
    else:
        idle = events_info.get("idle_seconds")
        idle_str = f"{idle:.1f}s" if idle is not None else "n/a"
        print(
            "MUD events stream: "
            f"length={events_info['length']} last_id={events_info['last_id']} idle_for={idle_str}"
        )

    last_player_activity_raw = redis_client.get(RedisKeys.LAST_PLAYER_ACTIVITY)
    last_player_activity = None
    if last_player_activity_raw is not None:
        try:
            last_ts_ms = int(_decode(last_player_activity_raw) or "0")
            if last_ts_ms:
                last_player_activity = datetime.fromtimestamp(last_ts_ms / 1000.0, tz=timezone.utc)
        except Exception:
            last_player_activity = None
    if last_player_activity:
        print(
            "Last player activity: "
            f"{_format_dt(last_player_activity)} (idle_for={_format_age(last_player_activity)})"
        )
    else:
        print("Last player activity: n/a")

    processed_summary = _get_processed_events_summary(client)
    if "error" in processed_summary:
        print(f"Processed events: error={processed_summary['error']}")
    else:
        count = processed_summary.get("count", 0)
        if count == 0:
            print("Processed events: count=0")
        else:
            print(
                "Processed events: "
                f"count={count} min_id={processed_summary.get('min_id')} "
                f"max_id={processed_summary.get('max_id')} "
                f"last_processed_at={processed_summary.get('last_processed_at') or 'n/a'} "
                f"last_processed_agents={processed_summary.get('last_processed_agents') or 'n/a'}"
            )

    agent_rooms = _load_agent_rooms(redis_client)
    if agent_rooms:
        print(f"Mediator agent rooms: {agent_rooms}")
    else:
        print("Mediator agent rooms: n/a")

    process_snapshot = _get_process_snapshot()
    process_lines = process_snapshot["lines"]
    mediator_agents = process_snapshot["mediator_agents"]
    worker_agents = process_snapshot["worker_agents"]

    if process_lines:
        print("Processes:")
        for line in process_lines:
            print(f"  {line}")
    else:
        print("Processes: n/a (no matching mediator/worker processes found)")

    if mediator_agents:
        print(f"Mediator registered agents (from process args): {sorted(set(mediator_agents))}")
    else:
        print("Mediator registered agents (from process args): n/a")

    if worker_agents:
        print(f"Worker agents (from process args): {sorted(set(worker_agents))}")
    else:
        print("Worker agents (from process args): n/a")

    if args.agents is None:
        all_turns = client.get_all_turn_requests()
        agents = sorted({agent_id for agent_id, _ in all_turns})
        turn_request_map = {agent_id: tr for agent_id, tr in all_turns}
    else:
        agents = sorted(args.agents)
        turn_request_map = {agent_id: client.get_turn_request(agent_id) for agent_id in agents}

    print(f"Agents: {len(agents)}")

    turn_request_present = {agent_id for agent_id, tr in turn_request_map.items() if tr is not None}
    if mediator_agents:
        mediator_set = set(mediator_agents)
        missing_turn_requests = sorted(mediator_set - turn_request_present)
        extra_turn_requests = sorted(turn_request_present - mediator_set)
        if missing_turn_requests:
            print(f"Missing turn_request for mediator agent(s): {missing_turn_requests}")
        if extra_turn_requests:
            print(f"Turn_request present for unregistered agent(s): {extra_turn_requests}")

    if mediator_agents and worker_agents:
        mediator_set = set(mediator_agents)
        worker_set = set(worker_agents)
        missing_workers = sorted(mediator_set - worker_set)
        extra_workers = sorted(worker_set - mediator_set)
        if missing_workers:
            print(f"Missing worker process for mediator agent(s): {missing_workers}")
        if extra_workers:
            print(f"Worker running for unregistered agent(s): {extra_workers}")

    all_ready = True
    any_processing = False
    for agent_id in agents:
        turn_request = turn_request_map.get(agent_id)
        if not turn_request or turn_request.status.value != "ready":
            all_ready = False
        if turn_request and turn_request.status.value in (
            "assigned",
            "in_progress",
            "executing",
            "execute",
            "abort_requested",
        ):
            any_processing = True

        paused = client.is_agent_paused(agent_id)
        sleeping = client.get_agent_is_sleeping(agent_id)
        idle_active = _truthy(redis_client.get(RedisKeys.agent_idle_active(agent_id)))
        events_stream = _stream_info(redis_client, RedisKeys.agent_events(agent_id))

        dreamer_state = client.get_dreamer_state(agent_id)
        dreaming_state = _load_dreaming_state(redis_client, agent_id)

        print(f"Agent: {agent_id}")
        print(f"  sleeping={sleeping} paused={paused} idle_active={idle_active}")
        if "error" in events_stream:
            print(f"  agent_events: error={events_stream['error']}")
        else:
            idle = events_stream.get("idle_seconds")
            idle_str = f"{idle:.1f}s" if idle is not None else "n/a"
            print(
                "  agent_events: "
                f"length={events_stream['length']} "
                f"last_id={events_stream['last_id']} "
                f"idle_for={idle_str}"
            )

        if dreamer_state:
            print(
                "  dreamer:"
                f" enabled={dreamer_state.enabled}"
                f" idle_threshold_seconds={dreamer_state.idle_threshold_seconds}"
                f" token_threshold={dreamer_state.token_threshold}"
                f" last_dream_at={dreamer_state.last_dream_at or 'n/a'}"
                f" last_dream_scenario={dreamer_state.last_dream_scenario or 'n/a'}"
                f" pending_pipeline_id={dreamer_state.pending_pipeline_id or 'n/a'}"
            )
        else:
            print("  dreamer: n/a")

        if dreaming_state:
            print(
                "  dreaming_state:"
                f" status={dreaming_state.status.value}"
                f" scenario={dreaming_state.scenario_name}"
                f" pipeline_id={dreaming_state.pipeline_id}"
                f" step_index={dreaming_state.step_index}"
                f" conversation_id={dreaming_state.conversation_id or 'n/a'}"
            )
        else:
            print("  dreaming_state: n/a")

        turn_lines = _format_turn_request(turn_request)
        if turn_lines:
            print("  " + turn_lines[0])
            for line in turn_lines[1:]:
                print("    " + line)

    print("Idle gating summary:")
    print(f"  all_ready={all_ready}")
    print(f"  any_processing={any_processing}")
    print(f"  mediator_paused={mediator_paused}")
    if "error" in events_info:
        print("  events_idle_for=n/a")
    else:
        idle = events_info.get("idle_seconds")
        idle_str = f"{idle:.1f}s" if idle is not None else "n/a"
        print(f"  events_idle_for={idle_str}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
