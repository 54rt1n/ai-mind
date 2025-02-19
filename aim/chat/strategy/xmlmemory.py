# aim/chat/strategy/memory.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
import copy
from datetime import datetime, timedelta
import logging
import pandas as pd
import random
from typing import Optional

from ..manager import ChatManager
from ...constants import TOKEN_CHARS
from ...utils.xml import XmlFormatter
from .base import ChatTurnStrategy
from ...utils.keywords import extract_semantic_keywords
from ...agents.persona import Persona
logger = logging.getLogger(__name__)


class XMLMemoryTurnStrategy(ChatTurnStrategy):
    def __init__(self, chat : ChatManager):
        self.chat = chat
        self.pinned : list[str] = []
        # TODO We need to calculate the actual tokens. This guesstimating is not working well.
        self.max_character_length = int((16384 - 4096) * (TOKEN_CHARS - 2.00))
        self.hud_name = "HUD Display Output"

    def user_turn_for(self, persona: Persona, user_input: str, history: list[dict[str, str]] = []) -> dict[str, str]:
        return {"role": "user", "content": user_input}

    def extract_memory_metadata(self, row: pd.Series, top_n_keywords: int = 5) -> tuple[list[str], list[str]]:
        """
        The memory metadata consists of two parts: The emotions (from 'emotion_a'..._d) and the semantic keywords (from matching all **keyword**s in the 'content').
        """
        # Extract emotions
        emotions = [row['emotion_a'], row['emotion_b'], row['emotion_c'], row['emotion_d']]
        # Extract semantic keywords
        keywords = extract_semantic_keywords(row['content'])
        keywords = [e[0] for e in sorted(keywords.items(), key=lambda x: x[1], reverse=True)][:top_n_keywords]
        return emotions, keywords

    def get_conscious_memory(self, persona: Persona, query: Optional[str] = None, user_queries: list[str] = [], assistant_queries: list[str] = [], content_len: int = 0) -> str:
        """
        Retrieves the conscious memory content to be included in the chat response.
        
        The conscious memory content includes the persona's thoughts, as well as relevant memories from the conversation history. It also includes any relevant documents that have been revealed to the user.
        
        Args:
            query (Optional[str]): The current user query, used to filter the retrieved memories.
            user_queries (List[str]): The history of user queries, used to retrieve relevant memories.
            assistant_queries (List[str]): The history of assistant queries, used to retrieve relevant memories.
        
        Returns:
            str: The conscious memory content, formatted as a string to be included in the chat response.
        """

        formatter = XmlFormatter()
        # First, lets add up the length of all the user queries and assistant queries.
        total_len = content_len
        my_emotions = defaultdict(int)
        my_keywords = defaultdict(int)

        logger.info(f"Initial Conscious Memory Length: {total_len}")
        document_content = []

        formatter.add_element("PraxOS", content="--== PraxOS Conscious Memory **Online** ==--", nowrap=True)
        
        # Document handling
        if self.chat.current_document is not None:
            logger.info(f"Current Document: {self.chat.current_document}")
            document_contents = self.chat.library.read_document(self.chat.current_document)
            doc_size = len(document_contents.split())
            formatter.add_element("document", content=document_contents,
                metadata=dict(
                    name=self.chat.current_document,
                    length=doc_size
                )
            )
        else:
            logger.info("No current document")

        # Workspace handling
        if self.chat.current_workspace is not None:
            workspace_contents = self.chat.current_workspace
            ws_size = len(workspace_contents.split())
            logger.debug(f"Workspace: {ws_size} words")
            formatter.add_element("workspace", content=workspace_contents,
                metadata=dict(
                    length=ws_size
                )
            )
        else:
            logger.info("No current workspace")

        my_emotions = defaultdict(int)
        my_keywords = defaultdict(int)
        def parse_row(row: pd.Series):
            emotions, keywords = self.extract_memory_metadata(row)
            for e in emotions:
                my_emotions[e] += 1
            for k in keywords:
                my_keywords[k] += 1
            
        motd = self.chat.cvm.get_motd(3)
        if not motd.empty:
            for _, row in motd.iterrows():
                # Check the date of the MOTD, if it's older than 3 days, skip it
                motd_date = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
                if motd_date < datetime.now() - timedelta(days=3):
                    logger.info(f"MOTD is older than 3 days, skipping: {row['date']}")
                    continue
                row_entry = f"xoxo MOTD: {row['date']}: {row['content']} oxox"
                formatter.add_element(self.hud_name, "Active Memory", "MOTD", content=row_entry)
                logger.debug(f"CMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")
                parse_row(row)

        logger.info(f"Total Conscious Memory Length: {total_len}")

        conscious = self.chat.cvm.get_conscious(persona.persona_id, top_n=self.chat.config.recall_size)
        for thought in persona.thoughts:
            formatter.add_element(self.hud_name, "thought", content=thought, nowrap=True)
        
        seen_docs = set()
        if not conscious.empty:
            for _, row in conscious.iterrows():
                if row['doc_id'] in seen_docs:
                    continue
                seen_docs.add(row['doc_id'])
                row_entry = row['content']
                formatter.add_element(self.hud_name, "Active Memory", "Journal",
                                      date=row['date'], type=row['document_type'],
                                      content=row_entry)
                parse_row(row)
                logger.debug(f"CMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")

        logger.info(f"Total Conscious Memory Length: {formatter.current_length}")
        if len(self.pinned) > 0:
            p_results = self.chat.cvm.get_documents(message_ids=self.pinned)
            for _, row in p_results.reset_index().iterrows():
                row_entry = row['content']
                formatter.add_element(self.hud_name, "Active Memory", "memory",
                                      date=row['date'], type=row['document_type'],
                                      content=row_entry)
                parse_row(row)
                logger.debug(f"PMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")

        logger.info(f"Total Conscious Memory Length: {formatter.current_length}")

        if query is not None:
            top_n = self.chat.config.memory_window - len(conscious)
            a_top = top_n // 2
            u_top = top_n - a_top
            available_len = self.max_character_length - formatter.current_length
            a_max = available_len // 2
            u_max = available_len - a_max

            a_results = self.chat.cvm.query(assistant_queries, filter_doc_ids=seen_docs, top_n=a_top, filter_metadocs=True, length_boost_factor=0, max_length=a_max)
            for _, row in a_results.reset_index().iterrows():
                row_entry = row['content']
                formatter.add_element(self.hud_name, "Active Memory", "memory",
                                      date=row['date'], type=row['document_type'],
                                      content=row_entry)
                parse_row(row)
                logger.debug(f"AMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")
        
            logger.info(f"Total Conscious Memory Length: {formatter.current_length}")

            u_results = self.chat.cvm.query(user_queries, filter_doc_ids=seen_docs, top_n=u_top, filter_metadocs=True, length_boost_factor=0.05, max_length=u_max)
            for _, row in u_results.reset_index().iterrows():
                row_entry = row['content']
                formatter.add_element(self.hud_name, "Active Memory", "memory",
                                      date=row['date'], type=row['document_type'],
                                      content=row_entry)
                parse_row(row)
                logger.debug(f"UMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")

        logger.info(f"Total Conscious Memory Length: {formatter.current_length}")

        if len(my_emotions) > 0:
            formatter.add_element(self.hud_name, "emotions", content=", ".join(e for e in my_emotions.keys() if e is not None))
        if len(my_keywords) > 0:
            formatter.add_element(self.hud_name, "keywords", content=", ".join(k for k in my_keywords.keys() if k is not None))

        logger.debug(f"Conscious Memory: Total Length: {formatter.current_length}/{self.max_character_length}")

        return "\n".join(document_content) + formatter.render()
        
    def chat_turns_for(self, persona: Persona, user_input: str, history: list[dict[str, str]] = [], content_len: Optional[int] = None) -> list[dict[str, str]]:
        """
        Generate a chat session, augmenting the response with information from the database.

        This is what will be passed to the chat complletion API.

        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history.
            
        Returns:
            List[Dict[str, str]]: The chat turns, in the alternating format [{"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_turn}].
        """
        
        # Make a deep copy of the history
        history = copy.deepcopy(history)

        history_len = sum(len(h['content']) for h in history)
        thought_len = len(self.thought_content or "")

        content_len_pct = (content_len or 0) / self.max_character_length
        history_len_pct = history_len / self.max_character_length
        logger.info(f"Generating chat turns. Thought Length: {thought_len} Current History Length: {len(history)} System: {content_len_pct:.2f} History: {history_len_pct:.2f}")

        fold_consciousness = 4
        history_cutoff_threshold = 0.5

        # if our history is over 50%, we need to remove some of the older user/assistant turns
        if history_len_pct > history_cutoff_threshold:
            # Calculate overage
            overage = int((history_len_pct - history_cutoff_threshold) * self.max_character_length)
            logger.info(f"History is over {history_cutoff_threshold:.2f}, removing older turns.")
            removed = 0
            while overage > 0:
                # Randomly select a turn to remove, weighting by the position in the history
                weights = range(len(history) - 2, 2, -1)
                total = sum(weights)
                weights = [w / total for w in weights]
                choices = len(history) - 4
                if choices <= 0:
                    break
                remove_turn_index = random.choices(range(choices), weights=weights)[0]
                # if this is an assistant turn, go back one
                if history[remove_turn_index]['role'] == 'assistant':
                    remove_turn_index -= 1
                
                # remove the turn
                overage -= len(history[remove_turn_index]['content'])
                del history[remove_turn_index]
                overage -= len(history[remove_turn_index]['content'])
                del history[remove_turn_index]
                removed += 2
                
            history = history[-(len(history) // 2):]
            history_len = sum(len(h['content']) for h in history)
            logger.info(f"History overage removed: {removed}")

        assistant_turn_history = [r['content'] for r in history if r['role'] == 'assistant'][::-1]
        user_turn_history = [r['content'] for r in history if r['role'] == 'user'][::-1]
        user_turn_history.append(user_input)
        consciousness = self.get_conscious_memory(
                persona=persona,
                query=user_input,
                user_queries=user_turn_history,
                assistant_queries=assistant_turn_history,
                content_len=(content_len or 0)+history_len+thought_len
                )
        
        consciousness_turn = {"role": "user", "content": consciousness}
        
        wakeup = persona.get_wakeup()
        wakeup_turn = {"role": "assistant", "content": wakeup}

        if len(history) > 0:
            if history[0]['role'] == 'assistant':
                turns = [consciousness_turn, *history]
            else:
                turns = [consciousness_turn, wakeup_turn, *history]
            
        else:
            turns = [consciousness_turn, wakeup_turn]

        turns.append({"role": "user", "content": user_input + "\n\n"})

        if self.thought_content:
            # Go back 3 turns and insert the thought content
            # step through, making sure we find a user turn
            for i in range(len(turns)-2, -1, -1):
                if turns[i]['role'] == 'user':
                    last_user_content = turns[i]['content']
                    last_user_content += f"\n\n{self.thought_content}"
                    turns[i]['content'] = last_user_content
                    logger.info(f"Thought inserted at {i}")
                    break
        
        return turns
