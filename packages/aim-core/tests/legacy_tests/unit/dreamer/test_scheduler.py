# tests/unit/dreamer/test_scheduler.py
"""Unit tests for the Scheduler class."""

import pytest
import pytest_asyncio
import fakeredis.aioredis
from datetime import datetime, timezone
import asyncio

from aim_legacy.dreamer.server.scheduler import Scheduler
from aim_legacy.dreamer.server.state import StateStore
from aim.dreamer.core.models import (
    StepJob,
    StepStatus,
    Scenario,
    StepDefinition,
    StepOutput,
    ScenarioContext,
)


@pytest_asyncio.fixture
async def redis_client():
    """Create a fake Redis client for testing."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def state_store(redis_client):
    """Create a StateStore instance for testing."""
    return StateStore(redis_client, key_prefix="dreamer")


@pytest_asyncio.fixture
async def scheduler(redis_client, state_store):
    """Create a Scheduler instance for testing."""
    return Scheduler(redis_client, state_store)


@pytest.fixture
def simple_scenario():
    """Create a simple test scenario."""
    return Scenario(
        name="test",
        description="Test scenario",
        context=ScenarioContext(required_aspects=["coder"]),
        steps={
            "step1": StepDefinition(
                id="step1",
                prompt="Test prompt 1",
                output=StepOutput(document_type="test"),
                next=["step2"],
            ),
            "step2": StepDefinition(
                id="step2",
                prompt="Test prompt 2",
                output=StepOutput(document_type="test"),
                depends_on=["step1"],
            ),
        },
    )


@pytest.mark.asyncio
async def test_enqueue_step_basic(scheduler):
    """Test enqueue_step adds job to queue."""
    await scheduler.enqueue_step(
        pipeline_id="pipeline1",
        step_id="step1",
        priority=0,
    )

    # Pop the job to verify it was enqueued
    job = await scheduler.pop_step_job(timeout=1)

    assert job is not None
    assert job.pipeline_id == "pipeline1"
    assert job.step_id == "step1"
    assert job.priority == 0
    assert job.attempt == 1
    assert job.max_attempts == 3


@pytest.mark.asyncio
async def test_enqueue_step_with_delay(scheduler, redis_client):
    """Test enqueue_step with delay adds to delayed set."""
    await scheduler.enqueue_step(
        pipeline_id="pipeline1",
        step_id="step1",
        priority=0,
        delay=60,  # 60 seconds
    )

    # Check delayed queue has the job
    delayed_count = await redis_client.zcard(scheduler.delayed_key)
    assert delayed_count == 1

    # Main queue should be empty
    main_count = await redis_client.llen(scheduler.queue_key)
    assert main_count == 0


@pytest.mark.asyncio
async def test_pop_step_job_fifo_order(scheduler):
    """Test pop_step_job returns jobs in FIFO order."""
    # Enqueue multiple jobs
    await scheduler.enqueue_step("pipeline1", "step1")
    await scheduler.enqueue_step("pipeline1", "step2")
    await scheduler.enqueue_step("pipeline1", "step3")

    # Pop in order (FIFO)
    job1 = await scheduler.pop_step_job(timeout=1)
    job2 = await scheduler.pop_step_job(timeout=1)
    job3 = await scheduler.pop_step_job(timeout=1)

    assert job1.step_id == "step1"
    assert job2.step_id == "step2"
    assert job3.step_id == "step3"


@pytest.mark.asyncio
async def test_pop_step_job_empty_queue(scheduler):
    """Test pop_step_job returns None when queue is empty with timeout."""
    # Pop from empty queue with short timeout
    job = await scheduler.pop_step_job(timeout=1)
    assert job is None


@pytest.mark.asyncio
async def test_requeue_step_no_delay(scheduler):
    """Test requeue_step with no delay puts job back in queue."""
    # Create and enqueue a job
    await scheduler.enqueue_step("pipeline1", "step1")
    job = await scheduler.pop_step_job(timeout=1)

    # Requeue without delay
    await scheduler.requeue_step(job, delay=0)

    # Should be able to pop it again
    requeued_job = await scheduler.pop_step_job(timeout=1)
    assert requeued_job is not None
    assert requeued_job.pipeline_id == job.pipeline_id
    assert requeued_job.step_id == job.step_id


@pytest.mark.asyncio
async def test_requeue_step_with_delay(scheduler, redis_client):
    """Test requeue_step with delay adds to delayed set."""
    # Create and enqueue a job
    await scheduler.enqueue_step("pipeline1", "step1")
    job = await scheduler.pop_step_job(timeout=1)

    # Requeue with delay
    await scheduler.requeue_step(job, delay=30)

    # Should be in delayed queue
    delayed_count = await redis_client.zcard(scheduler.delayed_key)
    assert delayed_count == 1

    # Main queue should be empty
    main_count = await redis_client.llen(scheduler.queue_key)
    assert main_count == 0


@pytest.mark.asyncio
async def test_process_delayed_jobs_no_due_jobs(scheduler):
    """Test process_delayed_jobs with no due jobs."""
    # Add a job with future delay
    await scheduler.enqueue_step("pipeline1", "step1", delay=3600)  # 1 hour

    # Process delayed jobs
    moved_count = await scheduler.process_delayed_jobs()

    # No jobs should be moved
    assert moved_count == 0


@pytest.mark.asyncio
async def test_process_delayed_jobs_with_due_jobs(scheduler, redis_client):
    """Test process_delayed_jobs moves due jobs to main queue."""
    # Manually add jobs to delayed queue with past timestamps
    # (simulating jobs that were delayed but are now due)
    from datetime import datetime, timezone
    from aim.dreamer.core.models import StepJob

    # Create jobs with current time
    job1 = StepJob(
        pipeline_id="pipeline1",
        step_id="step1",
        enqueued_at=datetime.now(timezone.utc),
    )
    job2 = StepJob(
        pipeline_id="pipeline1",
        step_id="step2",
        enqueued_at=datetime.now(timezone.utc),
    )

    # Add directly to delayed queue with past scores (already due)
    past_score = datetime.now(timezone.utc).timestamp() - 10  # 10 seconds ago
    await redis_client.zadd(scheduler.delayed_key, {
        job1.model_dump_json(): past_score,
        job2.model_dump_json(): past_score,
    })

    # Process delayed jobs
    moved_count = await scheduler.process_delayed_jobs()

    # Both jobs should be moved
    assert moved_count == 2

    # Delayed queue should be empty
    delayed_count = await redis_client.zcard(scheduler.delayed_key)
    assert delayed_count == 0

    # Main queue should have 2 jobs
    main_count = await redis_client.llen(scheduler.queue_key)
    assert main_count == 2


@pytest.mark.asyncio
async def test_all_deps_complete_no_dependencies(scheduler, simple_scenario):
    """Test all_deps_complete with no dependencies."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # step1 has no dependencies
    step1 = simple_scenario.steps["step1"]
    result = await scheduler.all_deps_complete(pipeline_id, step1)

    assert result is True


@pytest.mark.asyncio
async def test_all_deps_complete_satisfied(scheduler, simple_scenario):
    """Test all_deps_complete with satisfied dependencies."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # Mark step1 as complete
    await scheduler.mark_complete(pipeline_id, "step1")

    # step2 depends on step1
    step2 = simple_scenario.steps["step2"]
    result = await scheduler.all_deps_complete(pipeline_id, step2)

    assert result is True


@pytest.mark.asyncio
async def test_all_deps_complete_unsatisfied(scheduler, simple_scenario):
    """Test all_deps_complete with unsatisfied dependencies."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # step1 is still pending
    step2 = simple_scenario.steps["step2"]
    result = await scheduler.all_deps_complete(pipeline_id, step2)

    assert result is False


@pytest.mark.asyncio
async def test_mark_complete(scheduler, simple_scenario):
    """Test mark_complete updates step status."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # Initially pending
    status = await scheduler.state_store.get_step_status(pipeline_id, "step1")
    assert status == StepStatus.PENDING

    # Mark complete
    await scheduler.mark_complete(pipeline_id, "step1")

    # Should be complete now
    status = await scheduler.state_store.get_step_status(pipeline_id, "step1")
    assert status == StepStatus.COMPLETE


@pytest.mark.asyncio
async def test_mark_failed(scheduler, simple_scenario):
    """Test mark_failed updates step status."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # Mark failed
    await scheduler.mark_failed(pipeline_id, "step1", "Test error")

    # Should be failed now
    status = await scheduler.state_store.get_step_status(pipeline_id, "step1")
    assert status == StepStatus.FAILED


@pytest.mark.asyncio
async def test_check_pipeline_complete_incomplete(scheduler, simple_scenario):
    """Test check_pipeline_complete with incomplete pipeline."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # Mark only step1 as complete
    await scheduler.mark_complete(pipeline_id, "step1")

    # Pipeline should not be complete
    result = await scheduler.check_pipeline_complete(pipeline_id, simple_scenario)
    assert result is False


@pytest.mark.asyncio
async def test_check_pipeline_complete_all_done(scheduler, simple_scenario):
    """Test check_pipeline_complete with all steps complete."""
    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, simple_scenario)

    # Mark all steps as complete
    await scheduler.mark_complete(pipeline_id, "step1")
    await scheduler.mark_complete(pipeline_id, "step2")

    # Pipeline should be complete
    result = await scheduler.check_pipeline_complete(pipeline_id, simple_scenario)
    assert result is True


@pytest.mark.asyncio
async def test_job_serialization_roundtrip(scheduler):
    """Test that StepJob can be serialized and deserialized correctly."""
    # Enqueue a job
    await scheduler.enqueue_step("pipeline1", "step1", priority=5)

    # Pop and verify all fields
    job = await scheduler.pop_step_job(timeout=1)

    assert job.pipeline_id == "pipeline1"
    assert job.step_id == "step1"
    assert job.priority == 5
    assert job.attempt == 1
    assert job.max_attempts == 3
    assert isinstance(job.enqueued_at, datetime)


@pytest.mark.asyncio
async def test_complex_dag_dependencies(scheduler):
    """Test dependency checking with complex DAG."""
    # Create a diamond DAG:
    #     A
    #    / \
    #   B   C
    #    \ /
    #     D
    scenario = Scenario(
        name="complex",
        description="Complex DAG",
        context=ScenarioContext(required_aspects=["coder"]),
        steps={
            "A": StepDefinition(
                id="A",
                prompt="Step A",
                output=StepOutput(document_type="test"),
                next=["B", "C"],
            ),
            "B": StepDefinition(
                id="B",
                prompt="Step B",
                output=StepOutput(document_type="test"),
                depends_on=["A"],
                next=["D"],
            ),
            "C": StepDefinition(
                id="C",
                prompt="Step C",
                output=StepOutput(document_type="test"),
                depends_on=["A"],
                next=["D"],
            ),
            "D": StepDefinition(
                id="D",
                prompt="Step D",
                output=StepOutput(document_type="test"),
                depends_on=["B", "C"],
            ),
        },
    )

    # Initialize pipeline
    pipeline_id = "pipeline1"
    await scheduler.state_store.init_dag(pipeline_id, scenario)

    # D should not be ready (B and C not complete)
    assert not await scheduler.all_deps_complete(pipeline_id, scenario.steps["D"])

    # Complete A
    await scheduler.mark_complete(pipeline_id, "A")

    # B and C should be ready
    assert await scheduler.all_deps_complete(pipeline_id, scenario.steps["B"])
    assert await scheduler.all_deps_complete(pipeline_id, scenario.steps["C"])

    # D still not ready
    assert not await scheduler.all_deps_complete(pipeline_id, scenario.steps["D"])

    # Complete B
    await scheduler.mark_complete(pipeline_id, "B")

    # D still not ready (C not complete)
    assert not await scheduler.all_deps_complete(pipeline_id, scenario.steps["D"])

    # Complete C
    await scheduler.mark_complete(pipeline_id, "C")

    # Now D should be ready
    assert await scheduler.all_deps_complete(pipeline_id, scenario.steps["D"])


@pytest.mark.asyncio
async def test_increment_attempt(scheduler):
    """Test that job attempt counter is incremented correctly."""
    # Create a job
    await scheduler.enqueue_step("pipeline1", "step1")
    job = await scheduler.pop_step_job(timeout=1)

    assert job.attempt == 1

    # Increment attempt
    job2 = job.increment_attempt()
    assert job2.attempt == 2
    assert job.attempt == 1  # Original should be unchanged

    # Requeue with incremented attempt
    await scheduler.requeue_step(job2, delay=0)
    job3 = await scheduler.pop_step_job(timeout=1)

    assert job3.attempt == 2
