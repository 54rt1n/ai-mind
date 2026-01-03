# aim/conversation/loader.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

import json
import logging
from pathlib import Path

from .message import ConversationMessage

logger = logging.getLogger(__name__)


class ConversationLoader:
    """Handles loading and saving conversations from JSONL files"""
    
    def __init__(self, conversations_dir: str = "memory/conversations"):
        self.conversations_dir = Path(conversations_dir)
        if not self.conversations_dir.exists():
            self.conversations_dir.mkdir(parents=True)

    def load_all(self) -> list[ConversationMessage]:
        """Load all conversations from JSONL files"""
        messages = []
        
        for jsonl_file in self.conversations_dir.glob("*.jsonl"):
            try:
                messages.extend(self.load_file(jsonl_file))
            except Exception as e:
                logger.error(f"Error loading {jsonl_file}: {e}")
                raise
                
        logger.info(f"Loaded {len(messages)} messages from {self.conversations_dir}")
        return messages

    def load_file(self, conversation_path: Path) -> list[ConversationMessage]:
        """Load a single conversation file"""

        if not conversation_path.exists():
            raise FileNotFoundError(f"Conversation {conversation_path.name} not found")

        messages = []
        with open(conversation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
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
