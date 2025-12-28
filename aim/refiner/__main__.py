#!/usr/bin/env python3
# aim/refiner/__main__.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Refiner CLI - Test the ExplorationEngine with command-line arguments.

Usage:
    python -m aim.refiner [OPTIONS]

Examples:
    # Test with brainstorm paradigm, dry run
    python -m aim.refiner --paradigm brainstorm --dry-run

    # Random paradigm with verbose output (shows LLM responses)
    python -m aim.refiner -p random -v

    # Skip idle check for testing
    python -m aim.refiner --skip-idle-check --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import sys
from typing import Optional

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.app.dream_agent.client import DreamerClient
from aim.agents.persona import Persona
from aim.refiner.context import ContextGatherer
from aim.refiner.prompts import build_topic_selection_prompt, build_validation_prompt
from aim.refiner.tools import get_select_topic_tool, get_validate_tool
from aim.tool.formatting import ToolUser
from aim.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


async def run_exploration(args: argparse.Namespace) -> int:
    """Run the exploration engine step by step with verbose output."""
    verbose = args.verbose

    # Load config
    try:
        if args.env_file:
            print(f"Loading config from: {args.env_file}")
        else:
            print("Loading config from .env in current directory...")
        config = ChatConfig.from_env(args.env_file)
        if verbose:
            print(f"  PERSONA_ID: {config.persona_id}")
            print(f"  DEFAULT_MODEL: {config.default_model}")
            print(f"  PERSONA_PATH: {config.persona_path}")
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        return 1

    # Create CVM
    try:
        print("Initializing ConversationModel...")
        cvm = ConversationModel.from_config(config)
    except Exception as e:
        print(f"Error creating ConversationModel: {e}", file=sys.stderr)
        return 1

    # Load persona
    try:
        persona = Persona.from_config(config)
        print(f"Loaded persona: {persona.name}")
    except Exception as e:
        print(f"Error loading persona: {e}", file=sys.stderr)
        return 1

    # Get LLM provider with adequate max_tokens for thinking models
    try:
        from aim.llm.models import LanguageModelV2
        from dataclasses import replace

        model_name = args.model or config.default_model
        if not model_name:
            print("No model specified and DEFAULT_MODEL not set in env", file=sys.stderr)
            return 1

        # Override max_tokens for thinking models (need ~4096 for <think> + JSON)
        llm_config = replace(config, max_tokens=4096)

        models = LanguageModelV2.index_models(config)
        model = models.get(model_name)
        if not model:
            print(f"Model {model_name} not available", file=sys.stderr)
            return 1
        provider = model.llm_factory(llm_config)
        print(f"Using model: {model_name} (max_tokens: 4096)")
    except Exception as e:
        print(f"Error getting LLM provider: {e}", file=sys.stderr)
        return 1

    # Select paradigm
    if args.paradigm == "random":
        paradigm = random.choice(["brainstorm", "daydream", "knowledge"])
    else:
        paradigm = args.paradigm
    print(f"Selected paradigm: {paradigm}")

    # Check idle status (unless skipped)
    if not args.skip_idle_check:
        from aim.utils.redis_cache import RedisCache
        cache = RedisCache(config)
        last_activity = cache.get_api_last_activity()
        if last_activity is not None:
            import time
            elapsed = time.time() - last_activity
            if elapsed < args.idle_threshold:
                print(f"API is not idle (last activity {elapsed:.0f}s ago). Use --skip-idle-check to bypass.")
                return 0
        print("API is idle, proceeding with exploration...")
    else:
        print("Skipping idle check (--skip-idle-check)")

    # Create context gatherer
    context_gatherer = ContextGatherer(cvm=cvm, token_counter=count_tokens)

    # ================================================================
    # STEP 1: BROAD CONTEXT GATHERING + TOPIC SELECTION
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 1: BROAD CONTEXT GATHERING + TOPIC SELECTION")
    print("=" * 60)

    print(f"\nGathering broad context for {paradigm}...")
    broad_context = await context_gatherer.broad_gather(paradigm, token_budget=16000)

    if broad_context.empty:
        print("No documents found for exploration.")
        return 0

    broad_docs = broad_context.to_records()
    print(f"Found {len(broad_docs)} documents ({broad_context.tokens_used} tokens)")

    if verbose:
        print("\n--- BROAD CONTEXT DOCUMENTS ---")
        for i, doc in enumerate(broad_docs[:5]):  # Show first 5
            content = doc.get("content", "")[:200]
            doc_type = doc.get("document_type", "unknown")
            print(f"[{i+1}] ({doc_type}) {content}...")
        if len(broad_docs) > 5:
            print(f"... and {len(broad_docs) - 5} more")
        print("--- END DOCUMENTS ---\n")

    # Build topic selection prompt
    system_msg, user_msg = build_topic_selection_prompt(paradigm, broad_docs, persona)

    if verbose:
        print("\n--- TOPIC SELECTION PROMPT ---")
        print("SYSTEM:", system_msg[:500], "..." if len(system_msg) > 500 else "")
        print("\nUSER:", user_msg[:1000], "..." if len(user_msg) > 1000 else "")
        print("--- END PROMPT ---\n")

    # Call LLM for topic selection
    print("Calling LLM for topic selection...")
    turns = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    chunks = []
    for chunk in provider.stream_turns(turns, llm_config):
        if chunk:
            chunks.append(chunk)
    response1 = "".join(chunks)

    if verbose:
        print("\n--- LLM RESPONSE (TOPIC SELECTION) ---")
        print(response1)
        print("--- END RESPONSE ---\n")

    # Parse tool call
    select_tool = get_select_topic_tool()
    tool_user = ToolUser([select_tool])
    result = tool_user.process_response(response1)

    if not result.is_valid:
        print(f"Invalid tool call: {result.error}")
        return 1

    topic_result = result.arguments
    topic = topic_result.get("topic", "")
    approach = topic_result.get("approach", paradigm)
    reasoning = topic_result.get("reasoning", "")

    print(f"\nTopic selected: {topic}")
    print(f"Approach: {approach}")
    print(f"Reasoning: {reasoning}")

    # ================================================================
    # STEP 2: TARGETED RETRIEVAL + VALIDATION
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 2: TARGETED RETRIEVAL + VALIDATION")
    print("=" * 60)

    print(f"\nGathering targeted context for '{topic}'...")
    targeted_context = await context_gatherer.targeted_gather(
        topic=topic, approach=approach, token_budget=16000
    )

    if targeted_context.empty:
        print(f"No targeted documents found for topic '{topic}'")
        return 0

    targeted_docs = targeted_context.to_records()
    print(f"Found {len(targeted_docs)} targeted documents ({targeted_context.tokens_used} tokens)")

    if verbose:
        print("\n--- TARGETED CONTEXT DOCUMENTS ---")
        for i, doc in enumerate(targeted_docs[:5]):
            content = doc.get("content", "")[:200]
            doc_type = doc.get("document_type", "unknown")
            print(f"[{i+1}] ({doc_type}) {content}...")
        if len(targeted_docs) > 5:
            print(f"... and {len(targeted_docs) - 5} more")
        print("--- END DOCUMENTS ---\n")

    # Build validation prompt
    system_msg, user_msg = build_validation_prompt(
        paradigm=paradigm,
        topic=topic,
        approach=approach,
        reasoning=reasoning,
        documents=targeted_docs,
        persona=persona,
    )

    if verbose:
        print("\n--- VALIDATION PROMPT ---")
        print("SYSTEM:", system_msg[:500], "..." if len(system_msg) > 500 else "")
        print("\nUSER:", user_msg[:1000], "..." if len(user_msg) > 1000 else "")
        print("--- END PROMPT ---\n")

    # Call LLM for validation
    print("Calling LLM for validation...")
    turns = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    chunks = []
    for chunk in provider.stream_turns(turns, llm_config):
        if chunk:
            chunks.append(chunk)
    response2 = "".join(chunks)

    if verbose:
        print("\n--- LLM RESPONSE (VALIDATION) ---")
        print(response2)
        print("--- END RESPONSE ---\n")

    # Parse validation
    validate_tool = get_validate_tool()
    tool_user = ToolUser([validate_tool])
    result = tool_user.process_response(response2)

    if not result.is_valid:
        print(f"Invalid validation: {result.error}")
        return 1

    validation = result.arguments
    accepted = validation.get("accept", False)
    val_reasoning = validation.get("reasoning", "")
    query_text = validation.get("query_text", topic)
    guidance = validation.get("guidance")

    print(f"\nValidation result: {'ACCEPTED' if accepted else 'REJECTED'}")
    print(f"Reasoning: {val_reasoning}")
    if accepted:
        print(f"Query: {query_text}")
        if guidance:
            print(f"Guidance: {guidance}")

    if not accepted:
        print("\nExploration rejected by persona.")
        return 0

    # ================================================================
    # STEP 3: SCENARIO LAUNCH
    # ================================================================
    print("\n" + "=" * 60)
    print("STEP 3: SCENARIO LAUNCH")
    print("=" * 60)

    # Map paradigms to scenarios
    if paradigm == "daydream":
        scenario = "daydream"
    elif paradigm == "knowledge":
        scenario = "researcher"
    else:
        scenario = approach

    if args.dry_run:
        print("\n[DRY RUN] Would trigger pipeline:")
        print(f"  Scenario: {scenario}")
        print(f"  Query: {query_text}")
        print(f"  Guidance: {guidance}")
        print(f"  Model: {model_name}")
        print(f"  Persona: {config.persona_id}")
        print(f"  Context docs: {len(targeted_docs)}")
    else:
        print("Connecting to Redis...")
        async with DreamerClient.direct(config) as client:
            import uuid
            import time as time_module
            # Generate conversation ID matching webui format: {scenario}_{timestamp}_{random9}
            timestamp = int(time_module.time() * 1000)
            random_suffix = uuid.uuid4().hex[:9]
            conversation_id = f"{scenario}_{timestamp}_{random_suffix}"

            print(f"\nTriggering pipeline...")
            print(f"  Scenario: {scenario}")
            print(f"  Conversation: {conversation_id}")

            result = await client.start(
                scenario_name=scenario,
                conversation_id=conversation_id,
                model_name=model_name,
                query_text=query_text,
                persona_id=config.persona_id,
                guidance=guidance,
                context_documents=targeted_docs,
            )

            if result.success:
                print(f"\nPipeline started: {result.pipeline_id}")
            else:
                print(f"\nFailed to start pipeline: {result.error}")
                return 1

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Refiner CLI - Test the ExplorationEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with brainstorm paradigm, dry run
  python -m aim.refiner --paradigm brainstorm --dry-run

  # Random paradigm with verbose output (shows LLM responses)
  python -m aim.refiner -p random -v --dry-run

  # Skip idle check for testing
  python -m aim.refiner --skip-idle-check --dry-run

  # Use specific model
  python -m aim.refiner -p knowledge --model claude-3-5-sonnet --dry-run
        """
    )

    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file",
    )
    parser.add_argument(
        "-p", "--paradigm",
        type=str,
        default="random",
        choices=["brainstorm", "daydream", "knowledge", "critique", "random"],
        help="Paradigm to use (default: random)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually trigger pipelines, just show what would happen",
    )
    parser.add_argument(
        "--idle-threshold",
        type=int,
        default=300,
        help="Seconds for API idle check (default: 300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use for LLM decisions",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (shows LLM prompts and responses)",
    )
    parser.add_argument(
        "--skip-idle-check",
        action="store_true",
        help="Skip the API idle check (for testing)",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    return asyncio.run(run_exploration(args))


if __name__ == "__main__":
    sys.exit(main())
