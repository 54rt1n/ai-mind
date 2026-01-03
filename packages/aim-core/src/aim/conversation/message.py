# aim/conversation/message.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from dataclasses import dataclass
import logging
import re
import time
from typing import Optional
from wonderwords import RandomWord

from ..constants import LISTENER_ALL, ROLE_USER, ROLE_ASSISTANT

logger = logging.getLogger(__name__)


VISIBLE_COLUMNS = ['doc_id', 'document_type', 'user_id', 'persona_id', 'conversation_id', 'role', 'content', 'think', 'branch', 'sequence_no',
                   'emotion_a', 'emotion_b', 'emotion_c', 'emotion_d', 'scenario_name', 'step_name']
CHUNK_COLUMNS = ["parent_doc_id", "chunk_level", "chunk_index", "chunk_start", "chunk_end", "chunk_count"]
QUERY_COLUMNS = VISIBLE_COLUMNS + ["weight", "sentiment_v", "sentiment_a", "sentiment_d", "importance", "timestamp"] + CHUNK_COLUMNS


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation"""

    # Our primary key
    doc_id: str
    document_type: str

    # Our natural key
    user_id: str
    persona_id: str
    conversation_id: str
    branch: int
    sequence_no: int

    # Conversation metadata
    speaker_id: str  # The name/id of the listener, or all, self, etc.
    listener_id: str  # The name/id of the listener, or all, self, etc.

    # The conversation turn
    role: str
    content: str
    timestamp: int
    think: str | None = None

    # Our turn emotion values
    emotion_a: str | None = None
    emotion_b: str | None = None
    emotion_c: str | None = None
    emotion_d: str | None = None

    # We are going to grade the conversation turn
    sentiment_v: float = 0 # Our angle V
    sentiment_a: float = 0 # Our angle A
    sentiment_d: float = 0 # Our angle D
    importance: float = 0 # Our magnitude
    observer: str = "" # The metric we are using to grade the conversation turn

    # We also need to be able to up/downweight the conversation turn
    weight: float = 1.0

    reference_id: str = ""  # If we are referencing another document or conversation
    inference_model: str = "unknown" # The model that generated the conversation turn
    metadata: str = ""  # catchall, internal use
    status: int = 0

    # Dreamer pipeline fields
    scenario_name: str | None = None  # The scenario that produced this document
    step_name: str | None = None  # The step within the scenario that produced this document

    def has_es_header(self) -> bool:
        """
        Returns True if the conversation turn has an ES header
        """
        # To do this, we search the first 500 characters for Emotional State:
        matches = re.findall(r"Emotional State:", self.content[:500])
        return len(matches) > 0
    
    def expanded_content(self) -> str:
        # We are going to use emotion_a, b, c, d to prepend an emotional state  to the content

        emotion_stack = [
            self.emotion_a,
            self.emotion_b,
            self.emotion_c,
            self.emotion_d,
        ].filter(lambda x: x is not None)

        if len(emotion_stack) == 0:
            return self.content

        result = ""
        if len(emotion_stack) == 1:
            result = f"*{emotion_stack[0]}*"
        elif len(emotion_stack) == 2:
            result = f"*{emotion_stack[0]}* with a sense of *{emotion_stack[1]}*"
        elif len(emotion_stack) == 3:
            result = f"*{emotion_stack[0]}* with a sense of *{emotion_stack[1]}* and *{emotion_stack[2]}*"
        elif len(emotion_stack) == 4:
            result = f"*{emotion_stack[0]}* with a sense of *{emotion_stack[1]}*, *{emotion_stack[2]}*, and *{emotion_stack[3]}*"
        
        name = self.speaker_id

        header = "[== {name}Emotional State: {result} ==]".format(name=name, result=result)
        return f"{header}\n\n{self.content}"

    @classmethod
    def next_doc_id(self) -> str:
        """
        Returns a valid unique doc ID. TODO ensure uniqueness
        """

        # A converation id is three random words separated by dashes
        random_words = RandomWord()
        doc_id = "-".join(random_words.word() for _ in range(3))

        return doc_id

    @classmethod
    def create(cls, conversation_id: str, sequence_no: int, role: str, content: str, speaker_id: str,
               document_type: str = "conversation",
               doc_id: Optional[str] = None, reference_id: Optional[str] = None,
               user_id: str = 'user', persona_id: str = 'assistant', listener_id: str = LISTENER_ALL,
               branch: int = 0, importance: float = 0.0, observer: str = "none", inference_model: Optional[str] = None,
               weight: float = 1.0, timestamp: Optional[int] = None, metadata: str = "", status: int = 0,
               emotion_a: Optional[str] = None, emotion_b: Optional[str] = None, emotion_c: Optional[str] = None, emotion_d: Optional[str] = None,
               sentiment_v: float = 0.0, sentiment_a: float = 0.0, sentiment_d: float = 0.0,
               think: Optional[str] = None,
               scenario_name: Optional[str] = None, step_name: Optional[str] = None,
               ) -> 'ConversationMessage':
        """
        Creates a new conversation message.
        """

        if doc_id is None:
            doc_id = cls.next_doc_id()

        if reference_id is None:
            reference_id = conversation_id

        if inference_model is None:
            inference_model = "default"

        if timestamp is None:
            timestamp = int(time.time())

        # Add the document and its embedding to the collection
        data = {
            "doc_id": doc_id,
            "document_type": document_type,
            "user_id": user_id,
            "persona_id": persona_id,
            "conversation_id": conversation_id,
            "branch": branch,
            "sequence_no": sequence_no,
            "speaker_id": speaker_id,
            "listener_id": listener_id,
            "reference_id": reference_id,
            "role": role,
            "content": content,
            "emotion_a": emotion_a,
            "emotion_b": emotion_b,
            "emotion_c": emotion_c,
            "emotion_d": emotion_d,
            "sentiment_v": sentiment_v,
            "sentiment_a": sentiment_a,
            "sentiment_d": sentiment_d,
            "importance": importance,
            "observer": observer,
            "weight": weight,
            "timestamp": timestamp,
            "inference_model": inference_model,
            "metadata": metadata,
            "status": status,
            "think": think,
            "scenario_name": scenario_name,
            "step_name": step_name,
        }

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        # Log a warning if timestamp is missing and being defaulted
        if "timestamp" not in data:
            logger.warning(f"Timestamp missing in message data (doc_id: {data.get('doc_id', 'Unknown')}). Defaulting to 0.")
        
        return cls(
            doc_id=data["doc_id"],
            document_type=data["document_type"],
            user_id=data["user_id"],
            persona_id=data["persona_id"],
            conversation_id=data["conversation_id"],
            branch=data["branch"],
            sequence_no=data["sequence_no"],
            speaker_id=data.get("speaker_id"),
            listener_id=data.get("listener_id"),
            reference_id=data.get("reference_id"),
            role=data["role"],
            content=data["content"],
            think=data.get("think"),
            emotion_a=data.get("emotion_a"),
            emotion_b=data.get("emotion_b"),
            emotion_c=data.get("emotion_c"),
            emotion_d=data.get("emotion_d"),
            sentiment_v=data.get("sentiment_v", 0.0),
            sentiment_a=data.get("sentiment_a", 0.0),
            sentiment_d=data.get("sentiment_d", 0.0),
            importance=data.get("importance", 0.0),
            observer=data.get("observer"),
            weight=data.get("weight", 1.0),
            timestamp=data.get("timestamp", 0),
            inference_model=data.get("inference_model"),
            metadata=data.get("metadata"),
            status=data.get("status", 0),
            scenario_name=data.get("scenario_name"),
            step_name=data.get("step_name"),
        )

    def to_dict(self) -> dict:
        result = {
            "doc_id": self.doc_id,
            "document_type": self.document_type,
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "conversation_id": self.conversation_id,
            "branch": self.branch,
            "sequence_no": self.sequence_no,
            "role": self.role,
            "content": self.content,
            "weight": self.weight,
            "timestamp": self.timestamp,
            "status": self.status,
        }

        # Add optional fields if they're not None
        optional_fields = [
            "think",
            "emotion_a",
            "emotion_b",
            "emotion_c",
            "emotion_d",
            "sentiment_v",
            "sentiment_a",
            "sentiment_d",
            "importance",
            "reference_id",
            "observer",
            "inference_model",
            "metadata",
            "speaker_id",
            "listener_id",
            "scenario_name",
            "step_name",
        ]
        
        default_fields = {
            "think": None,
            "emotion_a": None,
            "emotion_b": None,
            "emotion_c": None,
            "emotion_d": None,
            "sentiment_v": 0.0,
            "sentiment_a": 0.0,
            "sentiment_d": 0.0,
            "importance": 0.0,
            "reference_id": None,
            "observer": "none",
            "inference_model": "default",
            "metadata": "",
            "status": 0,
            "speaker_id": result['user_id'] if result['role'] == ROLE_USER else result['persona_id'],
            "listener_id": result['user_id'] if result['role'] == ROLE_ASSISTANT else result['persona_id'],
        }
        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                result[field] = value
            elif field in default_fields:
                result[field] = default_fields[field]

        return result
