# aim-mud-types/conversation.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

import base64
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field, field_validator, field_serializer

from ..helper import _utc_now, _datetime_to_unix, _unix_to_datetime

if TYPE_CHECKING:
    import numpy as np

class MUDConversationEntry(BaseModel):
    """Single entry in the Redis conversation list.

    Each entry represents either a user turn (world events compiled into
    a single document) or an assistant turn (agent response with actions).

    Attributes:
        role: Either "user" for world events or "assistant" for agent response.
        content: Formatted content ready for LLM consumption.
        timestamp: When the entry was created.
        tokens: Pre-counted tokens for budget management.
        saved: True after @write flushes to CVM (or skip_save is honored).
        doc_id: Set after CVM insert; used for deduplication.
        skip_save: If True, this entry is never persisted to CVM.
        document_type: DOC_MUD_WORLD, DOC_MUD_ACTION, or DOC_MUD_AGENT.
        conversation_id: Groups related turns together.
        sequence_no: Order within the conversation.
        metadata: Rich metadata (room info, actions, event details).
        think: Assistant's <think> content if present.
        speaker_id: "world" for user turns or persona_id for assistant.
    """

    # Core fields
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=_utc_now)

    # Token management
    tokens: int

    # Persistence tracking
    saved: bool = False
    doc_id: Optional[str] = None
    skip_save: bool = False

    # Document metadata (matches CVM schema)
    document_type: str
    conversation_id: str
    sequence_no: int

    # Rich metadata
    metadata: dict = Field(default_factory=dict)

    # Optional fields
    think: Optional[str] = None
    speaker_id: Optional[str] = None
    last_event_id: Optional[str] = None

    # Pre-computed embedding (base64-encoded float32 array)
    # Set by mediator during event compilation
    embedding: Optional[str] = None

    # Validators to parse Unix timestamps from Redis
    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_required_datetime(cls, v):
        return _unix_to_datetime(v) or _utc_now()

    # Serializers to output Unix timestamps
    @field_serializer("timestamp")
    def serialize_required_datetime(self, dt: datetime) -> int:
        return _datetime_to_unix(dt)

    def get_embedding_vector(self) -> Optional["np.ndarray"]:
        """Decode embedding from base64 to numpy array.

        Returns:
            numpy array of float32 values, or None if no embedding.
        """
        if not self.embedding:
            return None
        import numpy as np
        return np.frombuffer(base64.b64decode(self.embedding), dtype=np.float32)

    @staticmethod
    def encode_embedding(vector: "np.ndarray") -> str:
        """Encode numpy array to base64 string.

        Args:
            vector: numpy array to encode (will be cast to float32)

        Returns:
            Base64-encoded string representation.
        """
        import numpy as np
        return base64.b64encode(vector.astype(np.float32).tobytes()).decode('ascii')
