# Inline Pipeline Scheduler (LEGACY)

> **⚠️ LEGACY CODE**: This module has been moved to `aim_legacy.dreamer.inline`.
> For new code, use the strategy-based system in `aim.dreamer.core.strategy` with
> `ScenarioBuilder`, `ScenarioExecutor`, and `ScenarioState`.

The `aim_legacy.dreamer.inline` module provides synchronous, non-distributed pipeline execution for AI-Mind scenarios.

## Overview

Unlike the distributed `aim_legacy.dreamer.server` module (which uses Redis queues, state stores, and workers), the inline scheduler executes pipelines **synchronously in-process** with state held entirely in memory.

## When to Use

**Use the inline scheduler for:**
- CLI tools that need immediate results
- Testing and debugging scenarios
- Single-user applications
- Local development and exploration
- Jupyter notebooks and REPL workflows

**Use the distributed server for:**
- Production deployments
- Multi-worker processing
- High-concurrency scenarios
- Fault tolerance and retry logic
- Monitoring and observability

## Features

- **Synchronous execution**: Returns when pipeline completes
- **No infrastructure dependencies**: No Redis, no queues, no state stores
- **Simple error propagation**: Exceptions bubble up directly
- **Heartbeat support**: Optional callback for progress monitoring
- **Full scenario support**: Both standard and dialogue flows
- **Memory DSL**: Full support for seed actions and context building

## Usage

### Basic Example

```python
import asyncio
from aim.config import ChatConfig
from aim.agents.roster import Roster
from aim.conversation.model import ConversationModel
from aim_legacy.dreamer.inline import execute_pipeline_inline

async def run_philosophy():
    # Setup
    config = ChatConfig.from_env()
    roster = Roster()
    cvm = ConversationModel(roster)

    # Execute pipeline inline
    pipeline_id = await execute_pipeline_inline(
        scenario_name="philosopher",
        config=config,
        cvm=cvm,
        roster=roster,
        persona_id="andi",
        query_text="What is the nature of consciousness?",
    )

    print(f"Pipeline complete: {pipeline_id}")

asyncio.run(run_philosophy())
```

### With Heartbeat Monitoring

```python
def heartbeat(pipeline_id, step_id):
    print(f"[{pipeline_id[:8]}...] Executing step: {step_id}")

pipeline_id = await execute_pipeline_inline(
    scenario_name="journaler",
    config=config,
    cvm=cvm,
    roster=roster,
    persona_id="andi",
    heartbeat_callback=heartbeat,
)
```

### Conversation Analysis

```python
# For scenarios that analyze existing conversations
pipeline_id = await execute_pipeline_inline(
    scenario_name="analyst",
    config=config,
    cvm=cvm,
    roster=roster,
    persona_id="andi",
    conversation_id="conv_abc123",  # Required for analyst/summarizer
    guidance="Focus on emotional themes",
)
```

### Dialogue Scenarios

```python
# Dialogue flows work the same way
pipeline_id = await execute_pipeline_inline(
    scenario_name="analysis_dialogue",
    config=config,
    cvm=cvm,
    roster=roster,
    persona_id="andi",
    conversation_id="conv_xyz789",
)
```

## API Reference

### `execute_pipeline_inline()`

```python
async def execute_pipeline_inline(
    scenario_name: str,
    config: ChatConfig,
    cvm: ConversationModel,
    roster: Roster,
    persona_id: str,
    conversation_id: Optional[str] = None,
    query_text: Optional[str] = None,
    guidance: Optional[str] = None,
    heartbeat_callback: Optional[Callable[[str, str], None]] = None,
    scenarios_dir: Optional[Path] = None,
    user_id: str = "user",
    model: Optional[str] = None,
) -> str
```

**Parameters:**

- `scenario_name`: Name of scenario YAML file (without .yaml extension)
- `config`: ChatConfig with model settings and credentials
- `cvm`: ConversationModel for memory queries and result storage
- `roster`: Roster containing personas
- `persona_id`: ID of persona to execute as
- `conversation_id`: Optional conversation ID for context (required for analyst/summarizer)
- `query_text`: Optional query text for memory searches
- `guidance`: Optional user guidance for the scenario
- `heartbeat_callback`: Optional callback(pipeline_id, step_id) called before each step
- `scenarios_dir`: Optional directory containing scenario files (defaults to config/scenario/)
- `user_id`: User identifier (defaults to "user")
- `model`: Optional model override (defaults to config.default_model)

**Returns:**

- `pipeline_id`: Unique identifier for this execution

**Raises:**

- `FileNotFoundError`: If scenario file doesn't exist
- `ValueError`: If persona not found, scenario invalid, or required conversation missing
- `Exception`: Any error during step execution

## Execution Flow

### Standard Scenarios

1. Load scenario YAML
2. Validate persona and conversation requirements
3. Compute DAG dependencies and topological order
4. Create pipeline state
5. Execute seed actions (if present)
6. For each step in topological order:
   - Call heartbeat callback (if provided)
   - Execute step with LLM
   - Save result to CVM
   - Append to accumulated context
7. Return pipeline_id

### Dialogue Scenarios

1. Load scenario YAML
2. Load DialogueStrategy
3. Create DialogueScenario executor
4. Initialize dialogue state
5. For each step in execution order:
   - Call heartbeat callback (if provided)
   - Execute dialogue turn with role flipping
   - Save to CVM
6. Return pipeline_id

## Architecture

The inline scheduler is built on the same core components as the distributed system:

- **Core Executor** (`aim.dreamer.core.executor`): Step execution with LLM calls
- **Memory DSL** (`aim.dreamer.core.memory_dsl`): Context building and retrieval
- **Scenario Loader** (`aim.dreamer.core.scenario`): YAML validation and template rendering
- **Dialogue Support** (`aim_legacy.dreamer.core.dialogue`): Turn-based dialogue execution (legacy)

The key difference is the **absence of Redis infrastructure**:

| Component | Distributed | Inline |
|-----------|-------------|--------|
| State storage | Redis | Memory |
| Queue system | BullMQ/Redis | None |
| Worker pool | Multiple workers | Single process |
| Retry handling | Queue redelivery | Direct exception |
| Monitoring | Redis metrics | Heartbeat callback |

## Error Handling

Errors propagate directly as exceptions:

```python
try:
    pipeline_id = await execute_pipeline_inline(
        scenario_name="philosopher",
        config=config,
        cvm=cvm,
        roster=roster,
        persona_id="andi",
    )
except FileNotFoundError as e:
    print(f"Scenario not found: {e}")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except Exception as e:
    print(f"Execution failed: {e}")
```

No retry logic is built in - if you need retries, implement them in your calling code.

## Comparison with Distributed Execution

```python
# Inline (synchronous)
from aim_legacy.dreamer.inline import execute_pipeline_inline

pipeline_id = await execute_pipeline_inline(
    scenario_name="journaler",
    config=config,
    cvm=cvm,
    roster=roster,
    persona_id="andi",
)
# Returns when complete

# Distributed (asynchronous)
from aim_legacy.dreamer.server import start_pipeline, get_status

pipeline_id = start_pipeline(
    scenario_name="journaler",
    config=config,
    roster=roster,
    persona_id="andi",
)
# Returns immediately, pipeline runs in background

# Poll for completion
while True:
    status = get_status(pipeline_id)
    if status.is_complete:
        break
    await asyncio.sleep(1)
```

## Performance Considerations

- **Blocking**: The calling thread is blocked until execution completes
- **No parallelism**: Steps execute sequentially (even when DAG allows parallelism)
- **Memory usage**: All state and results held in memory
- **No persistence**: Pipeline state is lost if process crashes

For long-running or critical pipelines, use the distributed system instead.

## Testing

The inline scheduler is easier to test than the distributed system:

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_my_scenario(mock_config, mock_cvm, mock_roster):
    with patch('aim_legacy.dreamer.inline.scheduler.execute_step', new_callable=AsyncMock) as mock_step:
        mock_step.return_value = (mock_result, [], False)

        pipeline_id = await execute_pipeline_inline(
            scenario_name="test",
            config=mock_config,
            cvm=mock_cvm,
            roster=mock_roster,
            persona_id="andi",
        )

        assert mock_step.call_count == len(expected_steps)
```

## See Also

- [Scenario DSL Documentation](../../docs/scenarios-and-paradigms.md)
- [Memory DSL Reference](../core/memory_dsl.py)
- [Distributed Scheduler](../server/scheduler.py)
