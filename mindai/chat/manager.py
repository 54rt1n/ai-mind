# mindai/chat/manager.py
# MindAI © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from typing import List, Dict, Optional

from ..constants import ROLE_USER, ROLE_ASSISTANT
from ..io.documents import Library
from ..llm.llm import ChatConfig
from ..conversation.model import ConversationModel
from ..agents import Roster


class ChatManager:
    def __init__(self, cvm: ConversationModel, config: ChatConfig, roster: Roster):
        self.cvm = cvm
        self.config = config
        self.roster = roster
        self.library = Library(documents_dir=config.documents_dir)
        self.current_document : Optional[str] = None
        self.current_workspace : Optional[str] = None

        self.history : List[Dict[str, str]] = []

    def render_conversation(self, messages: list[dict[str, str]]) -> list[str]:
        results = [
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        ]
        return results

    def add_history(self, role: str, content: str, author: Optional[str] = None) -> None:
        self.history.append({"role": role, "content": content, "author": author or self.config.user_id})

    def clear_history(self) -> None:
        self.history.clear()

    def new_conversation(self) -> None:
        self.clear_history()
        self.config.conversation_id = self.cvm.next_conversation_id(self.config.user_id)

    def insert_turn(self, user_id: str, persona_id: str, sequence_no: int, branch: int, user_turn: str, assistant_turn: str, usertime: int, assttime: int, author: Optional[str] = None, inference_model: Optional[str]= "unknown") -> None:
        base = {
            "conversation_id": self.config.conversation_id,
            "user_id": user_id,
            "persona_id": persona_id,
            "branch": branch,
            "inference_model": inference_model,
            "author": author or user_id,
        }
        user_entry = {
            "sequence_no": sequence_no,
            "role": ROLE_USER, 
            "content": user_turn,
            "timestamp": usertime,
        }
        assistant_entry = {
            "sequence_no": sequence_no + 1,
            "role": ROLE_ASSISTANT,
            "content": assistant_turn,
            "timestamp": assttime,
        }
        self.cvm.insert(**base, **user_entry)
        self.cvm.insert(**base, **assistant_entry)

    def search_memory(self, query: str) -> list[tuple[str, str]]:
        documents = self.cvm.query(query_texts=[query])
        if documents.empty:
            return []
        else:
            return [
                (f"Document {row['conversation_id']} (distance: {row['score']:.2f})", row['content'])
                for _, row in documents.reset_index()[::-1].iterrows()
            ]

    def __repr__(self):
        return f"ChatManager(history={len(self.history)} documents={self.cvm.to_pandas().count()} config={self.config})"

    @classmethod
    def from_config(cls, config: ChatConfig) -> "ChatManager":
        cvm = ConversationModel.from_config(config)
        roster = Roster.from_config(config)
        return cls(cvm=cvm, config=config, roster=roster)
