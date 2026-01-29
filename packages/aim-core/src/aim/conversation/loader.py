# aim/conversation/loader.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import json
import logging
from pathlib import Path
from tqdm import tqdm

from .message import ConversationMessage

logger = logging.getLogger(__name__)


class ConversationLoader:
    """Handles loading and saving conversations from JSONL files"""
    
    def __init__(self, conversations_dir: str = "memory/conversations"):
        self.conversations_dir = Path(conversations_dir)
        if not self.conversations_dir.exists():
            self.conversations_dir.mkdir(parents=True)

    def load_all(self, use_tqdm: bool = True) -> list[ConversationMessage]:
        """Load all conversations from JSONL files"""
        messages = []

        # Get list of files first to show total in progress bar
        jsonl_files = list(self.conversations_dir.glob("*.jsonl"))

        # Wrap with tqdm if requested
        file_iterator = tqdm(
            jsonl_files,
            desc="Loading conversation files",
            unit="file",
            position=0,
            leave=True
        ) if use_tqdm else jsonl_files

        for jsonl_file in file_iterator:
            try:
                file_messages = self.load_file(jsonl_file, use_tqdm=use_tqdm)
                messages.extend(file_messages)

                # Update description to show running total
                if use_tqdm:
                    file_iterator.set_postfix({"total_messages": len(messages)})
            except Exception as e:
                logger.error(f"Error loading {jsonl_file}: {e}")
                raise

        logger.info(f"Loaded {len(messages)} messages from {self.conversations_dir}")
        return messages

    def load_file(self, conversation_path: Path, use_tqdm: bool = False) -> list[ConversationMessage]:
        """Load a single conversation file"""

        if not conversation_path.exists():
            raise FileNotFoundError(f"Conversation {conversation_path.name} not found")

        messages = []

        # For progress bar, we need to count lines first (only if using tqdm for large files)
        if use_tqdm:
            with open(conversation_path, 'r') as f:
                total_lines = sum(1 for _ in f)

        with open(conversation_path, 'r') as f:
            # Wrap line iterator with tqdm if requested
            line_iterator = enumerate(f, 1)
            if use_tqdm and total_lines > 100:  # Only show progress for files with 100+ messages
                line_iterator = tqdm(
                    line_iterator,
                    total=total_lines,
                    desc=f"  {conversation_path.name}",
                    unit="msg",
                    position=1,
                    leave=False
                )

            for line_num, line in line_iterator:
                try:
                    entry = json.loads(line)
                    message = ConversationMessage.from_dict(entry)
                    messages.append(message)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {conversation_path}:{line_num}: {e}")
                    continue
                except KeyError as e:
                    logger.error(f"Missing required field in {conversation_path}:{line_num}: {e}")
                    raise

        return messages

    def load_conversation(self, conversation_id: str) -> list[ConversationMessage]:
        """
        Loads a conversation from the collection.
        """
        conversation_path = self.conversations_dir / f"{conversation_id}.jsonl"
        return self.load_file(conversation_path)

    def load_or_new(self, conversation_id: str) -> list[ConversationMessage]:
        """
        Loads a conversation from the collection. If the conversation does not exist,
        a new conversation is created.
        """
        conversation_path = self.conversations_dir / f"{conversation_id}.jsonl"
        if not conversation_path.exists():
            return []
        return self.load_file(conversation_path)

    def save_conversation(self, conversation_id: str, messages: list[ConversationMessage]) -> None:
        """Save messages to a conversation file"""
        file_path = self.conversations_dir / f"{conversation_id}.jsonl"
        with open(file_path, 'w') as f:
            for message in messages:
                json.dump(message.to_dict(), f)
                f.write('\n')
