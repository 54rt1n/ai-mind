# repo_watcher/main.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
CLI entry point for repo-watcher indexing service.

Usage:
    python -m repo_watcher --config config/sources/ai-mind.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

from aim.config import ChatConfig
from .config import RepoConfig
from .watcher import RepoWatcher

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for repo-watcher CLI.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Index source code for CODE_RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Index a repository
    python -m repo_watcher --config config/sources/ai-mind.yaml

    # With verbose logging
    python -m repo_watcher --config config/sources/ai-mind.yaml --verbose
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to repository configuration YAML file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path: Path = args.config
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        return 1

    try:
        config = RepoConfig(**config_dict)
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        return 1

    # Load .env settings
    chat_config = ChatConfig.from_env()
    memory_path = os.path.join(chat_config.memory_path, config.agent_id)

    print(f"Indexing {config.repo_id} for agent {config.agent_id}...")
    print(f"  Memory path: {memory_path}")
    print(f"  Embedding model: {chat_config.embedding_model}")
    print(f"  Device: {chat_config.embedding_device or 'cpu'}")
    print(f"  Sources: {len(config.sources)}")

    # Run indexing
    try:
        watcher = RepoWatcher(config)
        watcher.run()
    except Exception as e:
        logger.exception(f"Indexing failed: {e}")
        return 1

    print(f"Done. Index saved to {memory_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
