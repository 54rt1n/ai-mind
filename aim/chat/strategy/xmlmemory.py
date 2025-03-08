# aim/chat/strategy/memory.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
import copy
from datetime import datetime, timedelta
import logging
import pandas as pd
import random
from typing import Optional, List, Dict, Any, Tuple

from ..manager import ChatManager
from ...constants import TOKEN_CHARS
from ...utils.xml import XmlFormatter
from .base import ChatTurnStrategy
from ...utils.keywords import extract_semantic_keywords
from ...agents.persona import Persona
from aim.nlp.summarize import TextSummarizer, get_default_summarizer
from aim.utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)


class XMLMemoryTurnStrategy(ChatTurnStrategy):
    def __init__(self, chat : ChatManager):
        super().__init__(chat)
        # TODO We need to calculate the actual tokens. This guesstimating is not working well.
        self.max_character_length = int((16384 - 4096 - 1024) * (TOKEN_CHARS - 2.00))
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

        formatter.add_element("PraxOS", content="--== PraxOS Conscious Memory **Online** ==--", nowrap=True, priority=3)
        
        # Document handling
        if self.chat.current_document is not None:
            logger.info(f"Current Document: {self.chat.current_document}")
            document_contents = self.chat.library.read_document(self.chat.current_document)
            doc_size = len(document_contents.split())
            formatter.add_element("document", content=document_contents,
                metadata=dict(
                    name=self.chat.current_document,
                    length=doc_size
                ), priority=2
            )
        else:
            logger.info("No current document")

        # Workspace handling
        if self.chat.current_workspace is not None:
            ws_size = len(self.chat.current_workspace.split())
            content_len += ws_size
            logger.debug(f"Workspace: {ws_size} words")
        else:
            ws_size = 0
            logger.info("No current workspace")

        content_len += ws_size

        if self.scratch_pad:
            scratch_pad_size = len(self.scratch_pad.split())
            content_len += scratch_pad_size
            logger.debug(f"Scratch Pad: {scratch_pad_size} words")
        else:
            scratch_pad_size = 0
            logger.info("No scratch pad")

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
            logger.debug(f"MOTD: Found {len(motd)}")
            for _, row in motd.iterrows():
                # Check the date of the MOTD, if it's older than 3 days, skip it
                motd_date = datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')
                if motd_date < datetime.now() - timedelta(days=3):
                    logger.info(f"MOTD is older than 3 days, skipping: {row['date']}")
                    continue
                row_entry = f"xoxo MOTD: {row['date']}: {row['content']} oxox"
                formatter.add_element(self.hud_name, "Active Memory", "MOTD", content=row_entry, priority=2)
                logger.debug(f"XMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")
                parse_row(row)

        logger.info(f"Total Conscious Memory Length: {total_len}")

        conscious = self.chat.cvm.get_conscious(persona.persona_id, top_n=self.chat.config.recall_size)
        for thought in persona.thoughts:
            formatter.add_element(self.hud_name, "thought", content=thought, nowrap=True, priority=2)
        
        seen_docs = set()
        if not conscious.empty:
            for _, row in conscious.iterrows():
                if row['doc_id'] in seen_docs:
                    continue
                row_entry = row['content']
                formatter.add_element(self.hud_name, "Active Memory", "Journal",
                                      date=row['date'], type=row['document_type'],
                                      content=row_entry, priority=2)
                parse_row(row)
                seen_docs.add(row['doc_id'])
                logger.debug(f"CMemory: {len(row_entry)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")

        logger.info(f"Total Conscious Memory Length: {formatter.current_length}")

        if len(my_emotions) > 0:
            formatter.add_element(self.hud_name, "emotions", content=", ".join(e for e in my_emotions.keys() if e is not None), priority=1)
        if len(my_keywords) > 0:
            formatter.add_element(self.hud_name, "keywords", content=", ".join(k for k in my_keywords.keys() if k is not None), priority=1)

        if self.chat.current_workspace is not None:
            formatter.add_element(self.hud_name, "workspace", content="The user is sharing a workspace with you.", priority=3)
            formatter.add_element(self.hud_name, "workspace", content=self.chat.current_workspace,
                metadata=dict(
                    length=ws_size
                ), priority=3
            )

        if self.scratch_pad:
            formatter.add_element(self.hud_name, "scratchpad", content="You are sharding a scrachpad with yourself.", priority=3)
            formatter.add_element(self.hud_name, "scratchpad", content=self.scratch_pad,
                metadata=dict(
                    length=scratch_pad_size
                ), priority=3
            )

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
        logger.info(f"Generating chat turns.  System: {content_len} Thought : {thought_len} Current History: {history_len} ({len(history)}) | System: {content_len_pct:.2f} History: {history_len_pct:.2f}")

        fold_consciousness = 4
        history_cutoff_threshold = 0.5

        # if our history is over 50%, we need to reduce its size
        if history_len_pct > history_cutoff_threshold:
            # Calculate overage
            overage = int((history_len_pct - history_cutoff_threshold) * self.max_character_length)
            logger.info(f"History is over {history_cutoff_threshold:.2f}, applying compression strategy")
            
            # Choose history management strategy
            strategy = getattr(self.chat.config, 'history_management_strategy', 'sparsify')
            
            if strategy == "random_removal":
                logger.info(f"Using random removal strategy")
                history, removed = self._apply_random_removal_strategy(history, overage)
                logger.info(f"History overage removed: {removed} turns")
            elif strategy == "ai_summarize":
                logger.info(f"Using AI summarization strategy")
                history = self._apply_ai_summarization_strategy(history, overage, persona)
            else:  # Default to basic sparsification
                logger.info(f"Using basic sparsification strategy")
                history = self._apply_sparsification_strategy(history, overage)
            
            history_len = sum(len(h['content']) for h in history)
            logger.info(f"After history management: Length {history_len} chars ({len(history)} turns)")

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

        logger.info(f"Consciousness Length: {len(consciousness)}")
        
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

    def _apply_sparsification_strategy(self, history: list[dict[str, str]], overage: int) -> list[dict[str, str]]:
        """
        The new strategy that sparsifies messages rather than removing them completely.
        
        Args:
            history (list[dict[str, str]]): The chat history
            overage (int): The amount of characters we need to reduce
            
        Returns:
            list[dict[str, str]]: The updated history with sparsified messages
        """
        history_copy = copy.deepcopy(history)
        
        # Get or initialize text summarizer
        summarizer_method = getattr(self.chat.config, 'summarizer_method', 'auto')
        summarizer_model = getattr(self.chat.config, 'summarizer_model', None)
        use_gpu = getattr(self.chat.config, 'summarizer_use_gpu', False)
        
        # Current total length and target length
        current_length = sum(len(h['content']) for h in history_copy)
        target_length = current_length - overage
        
        # Use the conversation sparsification method from our new module
        summarizer = get_default_summarizer(model_name=summarizer_model, use_gpu=use_gpu)
        
        # Exclude the most recent 4 turns (2 exchanges) from sparsification
        preserve_recent = min(4, len(history_copy))
        
        # Sparsify the conversation to fit within the target length
        return summarizer.sparsify_conversation(
            messages=history_copy,
            max_total_length=target_length,
            preserve_recent=preserve_recent
        )

    def sparsify_message(self, message: str, target_length: int) -> str:
        """
        Intelligently reduce the size of a message while preserving semantic meaning.
        
        Uses the TextSummarizer from the nlp.summarize module.
        
        Args:
            message (str): The original message to sparsify
            target_length (int): The target length to reduce the message to
            
        Returns:
            str: The sparsified message
        """
        # Get or initialize text summarizer
        summarizer_method = getattr(self.chat.config, 'summarizer_method', 'auto')
        summarizer_model = getattr(self.chat.config, 'summarizer_model', None)
        use_gpu = getattr(self.chat.config, 'summarizer_use_gpu', False)
        
        summarizer = get_default_summarizer(model_name=summarizer_model, use_gpu=use_gpu)
        return summarizer.summarize(message, target_length, method=summarizer_method)

    def _apply_random_removal_strategy(self, history: list[dict[str, str]], overage: int) -> tuple[list[dict[str, str]], int]:
        """
        The original strategy that randomly removes conversation turns.
        
        Args:
            history (list[dict[str, str]]): The chat history
            overage (int): The amount of characters we need to reduce
            
        Returns:
            tuple[list[dict[str, str]], int]: The updated history and number of turns removed
        """
        history_copy = copy.deepcopy(history)
        removed = 0
        
        while overage > 0:
            # Randomly select a turn to remove, weighting by the position in the history
            weights = range(len(history_copy) - 2, 2, -1)
            total = sum(weights)
            if total == 0:
                break
                
            weights = [w / total for w in weights]
            choices = len(history_copy) - 4
            if choices <= 0:
                break
                
            remove_turn_index = random.choices(range(choices), weights=weights)[0]
            # if this is an assistant turn, go back one
            if history_copy[remove_turn_index]['role'] == 'assistant':
                remove_turn_index -= 1
            
            # remove the turn
            overage -= len(history_copy[remove_turn_index]['content'])
            del history_copy[remove_turn_index]
            overage -= len(history_copy[remove_turn_index]['content'])
            del history_copy[remove_turn_index]
            removed += 2
        
        # Keep only the most recent half of messages
        history_copy = history_copy[-(len(history_copy) // 2):]
        
        return history_copy, removed

    def _apply_ai_summarization_strategy(self, history: list[dict[str, str]], overage: int, persona: Persona) -> list[dict[str, str]]:
        """
        Apply AI-based summarization to compress messages in conversation history.
        
        Args:
            history (list[dict[str, str]]): The chat history
            overage (int): The amount of characters we need to reduce
            persona (Persona): The persona to use for customizing summaries
            
        Returns:
            list[dict[str, str]]: The updated history with AI-summarized messages
        """
        from aim.nlp.summarize import get_default_summarizer
        
        history_copy = copy.deepcopy(history)
        
        # Get summarization settings from config
        summarizer_model = getattr(self.chat.config, 'summarizer_model', "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        use_gpu = getattr(self.chat.config, 'summarizer_use_gpu', False)
        num_generations = getattr(self.chat.config, 'summarizer_num_generations', 2)
        num_beams = getattr(self.chat.config, 'summarizer_num_beams', 3)
        temperature = getattr(self.chat.config, 'summarizer_temperature', 0.7)
        
        # Current total length and target length
        current_length = sum(len(h['content']) for h in history_copy)
        target_length = current_length - overage
        
        # Initialize the summarizer
        try:
            summarizer = get_default_summarizer(model_name=summarizer_model, use_gpu=use_gpu)
            if summarizer._summarize_func is None:
                logger.warning("AI summarizer not available, falling back to sparsification")
                return self._apply_sparsification_strategy(history, overage)
        except Exception as e:
            logger.warning(f"Failed to initialize AI summarizer: {e}")
            return self._apply_sparsification_strategy(history, overage)
            
        # Initialize Redis cache
        cache = RedisCache(self.chat.config)
        
        # Exclude recent turns from summarization (preserve last 2 exchanges = 4 messages)
        preserve_recent = min(4, len(history_copy))
        older_messages = history_copy[:-preserve_recent] if preserve_recent < len(history_copy) else []
        recent_messages = history_copy[-preserve_recent:] if preserve_recent < len(history_copy) else history_copy
        
        # Nothing to summarize?
        if not older_messages:
            logger.warning("No older messages to summarize, falling back to removal")
            return self._apply_random_removal_strategy(history, overage)[0]
        
        # Calculate how much we need to reduce
        reduction_needed = overage
        
        # Sort older messages by length (longest first)
        older_messages.sort(key=lambda m: len(m.get('content', '')), reverse=True)
        
        # Track how much we've reduced
        reduced = 0
        
        # First pass: Summarize longest messages
        min_length_to_summarize = 350  # Only summarize messages longer than this
        summarized_messages = []
        
        for message in older_messages:
            content = message.get('content', '')
            
            # Skip short messages
            if len(content) <= min_length_to_summarize:
                summarized_messages.append(message)
                continue
                
            # Calculate target length based on role and position
            # Keep a higher percentage of user messages as they're often shorter and important for context
            if message['role'] == 'user':
                compression_ratio = 0.7  # Keep 70% of user messages
            else:  # assistant messages
                compression_ratio = 0.5  # Keep 50% of assistant messages
                
            # Set target length for this message
            target_msg_length = max(min_length_to_summarize // 2, int(len(content) * compression_ratio))
            
            # Don't bother with small reductions
            if len(content) - target_msg_length < 100:
                summarized_messages.append(message)
                continue
            
            # Create parameters dictionary for caching
            params = {
                'target_length': target_msg_length,
                'method': 'model',
                'num_generations': num_generations,
                'num_beams': num_beams,
                'temperature': temperature
            }
            
            # Define summarization function for cache miss
            def summarize_content(text, **kwargs):
                try:
                    return summarizer.summarize(text, **kwargs)
                except Exception as e:
                    logger.warning(f"Summarization error: {e}")
                    return text  # Return original on error
            
            try:
                # Try to get from cache or generate new summary
                summary = cache.get_or_cache(
                    content=content,
                    generator_func=summarize_content,
                    parameters=params
                )
                
                # Update message content
                new_message = message.copy()
                new_message['content'] = summary
                
                # Calculate reduction
                reduction = len(content) - len(summary)
                reduced += reduction
                
                summarized_messages.append(new_message)
                logger.info(f"AI-summarized a {message['role']} message from {len(content)} to {len(summary)} chars (cached={summary == cache.get(cache._hash_content(content, params))})")
                
                # If we've reduced enough, stop summarizing
                if reduced >= reduction_needed:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to summarize message with AI: {e}")
                summarized_messages.append(message)
        
        # Combine recent and summarized messages in correct order
        final_messages = summarized_messages + recent_messages
        
        # If we still need to reduce more (AI summaries weren't short enough),
        # fall back to removing oldest turns
        if reduced < reduction_needed:
            overage_remaining = reduction_needed - reduced
            logger.info(f"AI summarization wasn't enough, need to remove {overage_remaining} more chars")
            
            # Remove from oldest messages first, skipping the recent preserved ones
            while reduced < reduction_needed and len(final_messages) > preserve_recent:
                # Remove the oldest message
                removed_message = final_messages.pop(0)
                reduced += len(removed_message.get('content', ''))
                logger.info(f"Removed a {removed_message['role']} message of {len(removed_message.get('content', ''))} chars")
        
        return final_messages
