# aim/chat/strategy/xmlmemory.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
import copy
from datetime import datetime, timedelta
import logging
import pandas as pd
import random
import tiktoken
from typing import Optional, List, Dict, Any, Tuple

from ..manager import ChatManager
from ..util import insert_at_fold
from ...utils.xml import XmlFormatter
from .base import ChatTurnStrategy
from ...utils.keywords import extract_semantic_keywords
from ...agents.persona import Persona
from aim.nlp.summarize import TextSummarizer, get_default_summarizer
from aim.utils.redis_cache import RedisCache

logger = logging.getLogger(__name__)

MAX_CONTEXT=32768

class XMLMemoryTurnStrategy(ChatTurnStrategy):
    _encoder: tiktoken.Encoding = None

    @classmethod
    def get_encoder(cls) -> tiktoken.Encoding:
        if cls._encoder is None:
            cls._encoder = tiktoken.get_encoding("cl100k_base")
        return cls._encoder

    def count_tokens(self, text: str) -> int:
        return len(self.get_encoder().encode(text))

    def __init__(self, chat : ChatManager):
        super().__init__(chat)
        # Calculate max context tokens (reserve 2048 for output, 1024 for safety)
        self.max_context_tokens = MAX_CONTEXT - 2048 - 1024
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

    def get_conscious_memory(self, persona: Persona, query: Optional[str] = None, user_queries: list[str] = [], assistant_queries: list[str] = [], content_len: int = 0, thought_stream: list[str] = []) -> str:
        """
        Retrieves the conscious memory content to be included in the chat response.

        The conscious memory content includes the persona's thoughts, as well as relevant memories from the conversation history. It also includes any relevant documents that have been revealed to the user.

        Args:
            query (Optional[str]): The current user query, used to filter the retrieved memories.
            user_queries (List[str]): The history of user queries, used to retrieve relevant memories.
            assistant_queries (List[str]): The history of assistant queries, used to retrieve relevant memories.
            thought_stream (List[str]): Prior reasoning from assistant turns to include in header.

        Returns:
            str: The conscious memory content, formatted as a string to be included in the chat response.
        """

        formatter = XmlFormatter()
        total_len = content_len
        aggregated_emotions = defaultdict(int)
        aggregated_keywords = defaultdict(int)
        seen_docs = set() # Keep track of doc_ids we've already included

        #logger.info(f"Initial Conscious Memory Length: {total_len}")

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
            #logger.info("No current document")
            pass

        # Workspace handling (size calculation, actual element added later)
        if self.chat.current_workspace is not None:
            ws_size = len(self.chat.current_workspace.split())
            #logger.debug(f"Workspace: {ws_size} words")
        else:
            ws_size = 0
            #logger.info("No current workspace")
            pass

        # Scratchpad handling (size calculation, actual element added later)
        if self.scratch_pad:
            scratch_pad_size = len(self.scratch_pad.split())
            logger.debug(f"Scratch Pad: {scratch_pad_size} words")
        else:
            scratch_pad_size = 0
            #logger.info("No scratch pad")
        
        # --- Start Processing Specific Memory Types ---
        
        # 1. MOTD
        motd = self.chat.cvm.get_motd(1)
        if not motd.empty:
            #logger.debug(f"MOTD: Found {len(motd)}")
            for _, row in motd.iterrows():
                # Check the date of the MOTD, if it's older than 3 days, skip it
                motd_date = row['date']
                row_entry_content = f"xoxo MOTD: {motd_date}: {row['content']} oxox"
                formatter.add_element(self.hud_name, "Active Memory", "MOTD", content=row_entry_content, priority=2, noindent=True)
                emotions, keywords = self.extract_memory_metadata(row)
                for e in emotions: aggregated_emotions[e] += 1 if e else 0 # Check for None
                for k in keywords: aggregated_keywords[k] += 1 if k else 0 # Check for None
                seen_docs.add(row['doc_id'])
                logger.debug(f"MOTD: {len(row_entry_content)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")

        # 2. Pinned Messages
        if self.pinned:
            logger.info(f"Processing {len(self.pinned)} pinned messages.")
            all_pinned_doc_ids = list(self.pinned)
            if len(all_pinned_doc_ids) > 0:
                pinned_docs_df = self.chat.cvm.get_documents(all_pinned_doc_ids)
                if not pinned_docs_df.empty:
                    for _, row in pinned_docs_df.iterrows():
                         if row['doc_id'] in self.pinned:
                             row_entry_content = row.get('content', "Error: Pinned content not found.")
                             formatter.add_element(self.hud_name, "Active Memory", "Pinned",
                                                   date=row.get('date'), type=row.get('document_type'),
                                                   content=row_entry_content, noindent=True, priority=2)
                             emotions, keywords = self.extract_memory_metadata(row)
                             for e in emotions: aggregated_emotions[e] += 1 if e else 0 # Check for None
                             for k in keywords: aggregated_keywords[k] += 1 if k else 0 # Check for None
                             seen_docs.add(row['doc_id'])
                             logger.debug(f"PinnedMsg: {len(row_entry_content)} {row.get('conversation_id')}/{row.get('document_type')}/{row.get('date')}/{row.get('doc_id')}")
                else:
                    logger.warning(f"Could not retrieve any pinned documents for ids: {all_pinned_doc_ids}")
            else:
                logger.info("No document IDs in self.pinned to process.")


        # 3. Journal/Conscious Entries
        conscious = self.chat.cvm.get_conscious(persona.persona_id, top_n=self.chat.config.recall_size)
        if not conscious.empty:
             for _, row in conscious.iterrows():
                 if row['doc_id'] in seen_docs: continue
                 row_entry_content = row['content']
                 formatter.add_element(self.hud_name, "Active Memory", "Journal",
                                      date=row['date'], type=row['document_type'],
                                      noindent=True, content=row_entry_content, priority=2)
                 emotions, keywords = self.extract_memory_metadata(row)
                 for e in emotions: aggregated_emotions[e] += 1 if e else 0 # Check for None
                 for k in keywords: aggregated_keywords[k] += 1 if k else 0 # Check for None
                 seen_docs.add(row['doc_id'])
                 logger.debug(f"CMemory: {len(row_entry_content)} {row['conversation_id']}/{row['document_type']}/{row['date']}/{row['doc_id']}")

        # --- Add back the dynamic memory search based on queries ---
        #logger.info(f"Memory before dynamic query: {formatter.current_length} chars used.")
        thought_estimate_tokens = sum(self.count_tokens(t) for t in persona.thoughts) + 100
        current_tokens = self.count_tokens(formatter.render())
        # Estimate tokens for workspace/scratchpad (will be added later)
        ws_tokens_estimate = self.count_tokens(self.chat.current_workspace or "")
        scratch_tokens_estimate = self.count_tokens(self.scratch_pad or "")
        # Estimate tokens for thought_stream (if enabled and present)
        thought_stream_estimate = sum(self.count_tokens(t) for t in thought_stream) if thought_stream else 0
        available_tokens_for_dynamic_queries = (
            self.max_context_tokens
            - current_tokens
            - thought_estimate_tokens
            - ws_tokens_estimate
            - scratch_tokens_estimate
            - thought_stream_estimate
            - content_len  # External tokens: history, wakeup, user_input, etc.
        )
        logger.debug(f"Token budget: max={self.max_context_tokens}, current={current_tokens}, thoughts={thought_estimate_tokens}, ws={ws_tokens_estimate}, scratch={scratch_tokens_estimate}, stream={thought_stream_estimate}, external={content_len}, available={available_tokens_for_dynamic_queries}")

        if available_tokens_for_dynamic_queries > 50: # Min threshold for any dynamic querying (roughly 200 chars)
            query_sources_data = []
            workspace_content_for_query = self.chat.current_workspace if self.chat.current_workspace and self.chat.current_workspace.strip() else None

            if workspace_content_for_query:
                query_sources_data.append({
                    "name": "Workspace", 
                    "queries": [workspace_content_for_query], 
                    "length_boost": 0.05, 
                    "log_prefix": "WSMemory",
                    "chunk_size": 512,
                    "memory_type_tag": "memory_ws" # XML tag for this source
                })
            
            if assistant_queries:
                query_sources_data.append({
                    "name": "AssistantHistory", 
                    "queries": assistant_queries, 
                    "length_boost": 0, 
                    "log_prefix": "AMemory",
                    "chunk_size": 512,
                    "memory_type_tag": "memory_asst" # XML tag for this source
                })

            if user_queries:
                query_sources_data.append({
                    "name": "UserHistory", 
                    "queries": user_queries, 
                    "length_boost": 0.05, 
                    "log_prefix": "UMemory",
                    "chunk_size": 512,
                    "memory_type_tag": "memory_user" # XML tag for this source
                })
            
            # Add thought_content as a query source if it exists
            if self.thought_content and self.thought_content.strip():
                query_sources_data.append({
                    "name": "PersonaThoughts",
                    "queries": [self.thought_content.strip()],
                    "length_boost": 0.0,  # Thoughts are often concise, less need for length boosting
                    "log_prefix": "ThoughtMemory",
                    "chunk_size": 384,
                    "memory_type_tag": "memory_thought" # XML tag for this source
                })

            all_dynamic_results_dfs = []

            if query_sources_data:
                # Fetch results from all sources
                generous_top_n_per_source = self.chat.config.memory_window * 2 # Get more candidates
                if generous_top_n_per_source == 0 and self.chat.config.memory_window > 0 : generous_top_n_per_source = 2 # ensure at least a few if window is 1
                if self.chat.config.memory_window == 0 : generous_top_n_per_source = 0 # respect if window is 0


                for source_data in query_sources_data:
                    if not source_data["queries"]:
                        logger.debug(f"Skipping dynamic memory query for {source_data['name']} due to no queries.")
                        continue
                    if generous_top_n_per_source == 0:
                        logger.debug(f"Skipping dynamic memory query for {source_data['name']} as memory_window is 0.")
                        continue

                    #logger.debug(f"Querying dynamic memory ({source_data['name']}): queries_count={len(source_data['queries'])}, top_n_fetch={generous_top_n_per_source}, chunk_size=100")

                    results_df = self.chat.cvm.query(
                        source_data["queries"],
                        filter_doc_ids=seen_docs, # Avoid re-fetching docs already in XML (MOTD, Pinned, Journal)
                        top_n=generous_top_n_per_source,
                        filter_metadocs=True,
                        length_boost_factor=source_data["length_boost"],
                        max_length=None, # No max_length filtering at this stage per source
                        chunk_size=source_data["chunk_size"] # Use configured chunk_size for cvm.query's internal recursion
                    )

                    if not results_df.empty:
                        results_df['source_tag'] = source_data["memory_type_tag"] # Tag results with their source
                        results_df['log_prefix'] = source_data["log_prefix"] # For logging later
                        all_dynamic_results_dfs.append(results_df)
                        #logger.info(f"Fetched {len(results_df)} results from {source_data['name']}.")
                    else:
                        logger.info(f"No results from {source_data['name']}.")

            if all_dynamic_results_dfs:
                aggregated_results_df = pd.concat(all_dynamic_results_dfs, ignore_index=True)

                if not aggregated_results_df.empty:
                    #logger.info(f"Total dynamic results before deduplication: {len(aggregated_results_df)}")
                    # Sort by score (desc) then drop duplicates by doc_id, keeping the highest score entry
                    aggregated_results_df = aggregated_results_df.sort_values('score', ascending=False).drop_duplicates(subset=['doc_id'], keep='first')
                    #logger.info(f"Total dynamic results after deduplication: {len(aggregated_results_df)}")

                    dynamic_entries_added_count = 0
                    dynamic_tokens_added_to_formatter = 0 # Tracks formatter token increase from dynamic items
                    max_dynamic_entries_to_add = self.chat.config.memory_window

                    if max_dynamic_entries_to_add == 0:
                        logger.info("Max dynamic entries (memory_window) is 0, skipping addition of dynamic results.")
                    else:
                        logger.info(f"Processing {len(aggregated_results_df)} unique dynamic results. Budget: {available_tokens_for_dynamic_queries} tokens for new content, max_entries: {max_dynamic_entries_to_add}.")

                        for _, row in aggregated_results_df.iterrows():
                            if row['doc_id'] in seen_docs: # Already added by MOTD, Pinned, Journal or an earlier dynamic item
                                continue

                            if dynamic_entries_added_count >= max_dynamic_entries_to_add:
                                logger.debug(f"Reached max dynamic entries limit ({max_dynamic_entries_to_add}).")
                                break

                            row_entry_content = row['content']

                            # Estimate formatter increase for this item
                            temp_formatter = XmlFormatter()
                            temp_formatter.add_element(self.hud_name, "Active Memory", row['source_tag'],
                                                       date=row['date'], type=row['document_type'],
                                                       content=row_entry_content, priority=1, noindent=True)
                            estimated_formatter_token_increase = self.count_tokens(temp_formatter.render())

                            if (dynamic_tokens_added_to_formatter + estimated_formatter_token_increase) > available_tokens_for_dynamic_queries:
                                #logger.debug(f"Dynamic entry '{row['doc_id']}' from {row['log_prefix']} (est. formatter increase {estimated_formatter_token_increase} tokens, content len {len(row_entry_content)}) would exceed dynamic content budget ({available_tokens_for_dynamic_queries - dynamic_tokens_added_to_formatter} remaining). Trying next item.")
                                continue

                            # Add to the main formatter
                            formatter.add_element(self.hud_name, "Active Memory", row['source_tag'],
                                                  date=row['date'], type=row['document_type'],
                                                  content=row_entry_content, priority=1, noindent=True)

                            # Update budget and counts
                            # It's hard to get exact increase from formatter.add_element without re-rendering.
                            # Using the estimate and relying on the overall check.
                            dynamic_tokens_added_to_formatter += estimated_formatter_token_increase

                            emotions, keywords = self.extract_memory_metadata(row)
                            for e in emotions: aggregated_emotions[e] += 1 if e else 0
                            for k in keywords: aggregated_keywords[k] += 1 if k else 0
                            seen_docs.add(row['doc_id']) # Mark as added to XML
                            dynamic_entries_added_count += 1

                            current_formatter_tokens = self.count_tokens(formatter.render())
                            logger.debug(f"Added dynamic result ({row['log_prefix']}/{row['doc_id']}): content {len(row_entry_content)} chars, est. formatter increase {estimated_formatter_token_increase} tokens. Total dynamic entries: {dynamic_entries_added_count}/{max_dynamic_entries_to_add}. Dynamic budget used: {dynamic_tokens_added_to_formatter}/{available_tokens_for_dynamic_queries}. Overall formatter tokens: {current_formatter_tokens}")

                            # Final check against overall max length
                            if current_formatter_tokens + thought_estimate_tokens >= self.max_context_tokens:
                                logger.warning(f"Overall max context tokens ({self.max_context_tokens}) likely reached or exceeded after adding dynamic entry. Formatter: {current_formatter_tokens}, Thought_est: {thought_estimate_tokens}. Stopping dynamic additions.")
                                break
                else:
                    logger.info("No dynamic results to process after aggregation (aggregated_results_df is empty).")
            else:
                logger.info("No dynamic query sources had results, or no query sources were active.")
        else:
             logger.info(f"Skipping all dynamic memory queries due to insufficient initial space for dynamic content ({available_tokens_for_dynamic_queries} tokens available, need > 50).")

        # --- End Dynamic Memory Search ---


        # Add Persona Thoughts (as before)
        for thought in persona.thoughts:
            formatter.add_element(self.hud_name, "thought", content=thought, nowrap=True, priority=2)

        # Add Aggregated Emotions/Keywords (as before, but now includes dynamic results)
        if len(aggregated_emotions) > 0:
            formatter.add_element(self.hud_name, "emotions", content=", ".join(e for e in aggregated_emotions.keys() if e is not None), priority=1, nowrap=True) # Filter None keys
        if len(aggregated_keywords) > 0:
            formatter.add_element(self.hud_name, "keywords", content=", ".join(k for k in aggregated_keywords.keys() if k is not None), priority=1, nowrap=True) # Filter None keys

        # Add Workspace/Scratchpad (as before)
        if self.chat.current_workspace is not None:
            full_workspace_content = f"*The user is sharing a workspace with you.*\n{self.chat.current_workspace}"
            formatter.add_element(self.hud_name, "workspace",
                                content=full_workspace_content,
                                metadata=dict(
                                    length=ws_size # Use pre-calculated ws_size
                                ), 
                                priority=3,
                                noindent=True
                            )

        if self.scratch_pad:
             full_scratchpad_content = f"*You are sharing a scratchpad with yourself.*\n{self.scratch_pad}"
             formatter.add_element(self.hud_name, "scratchpad",
                                content=full_scratchpad_content,
                                metadata=dict(
                                    length=scratch_pad_size # Use pre-calculated scratch_pad_size
                                ),
                                priority=3,
                                noindent=True
                            )

        # Add thought_stream after memories if enabled
        include_thought_stream = getattr(self.chat.config, 'include_thought_stream', False)
        if include_thought_stream and thought_stream:
            for i, thought in enumerate(thought_stream):
                formatter.add_element(self.hud_name, "Thought Stream", f"Turn {i+1}",
                                      content=thought, priority=2, noindent=True)
            logger.info(f"Added thought_stream with {len(thought_stream)} prior thoughts to header")

        final_output = formatter.render()
        final_tokens = self.count_tokens(final_output)
        logger.debug(f"Final Conscious Memory: Total Tokens: {final_tokens}/{self.max_context_tokens}")

        return final_output
        
    def chat_turns_for(self, persona: Persona, user_input: str, history: list[dict[str, str]] = [], content_len: Optional[int] = None) -> list[dict[str, str]]:
        """
        Generate a chat session, augmenting the response with information from the database.

        This is what will be passed to the chat complletion API.

        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history (may include 'think' field).

        Returns:
            List[Dict[str, str]]: The chat turns, in the alternating format [{"role": "user", "content": user_input}, {"role": "assistant", "content": assistant_turn}].
        """

        # Make a deep copy of the history
        history = copy.deepcopy(history)

        # Build thought_stream from prior assistant think content
        thought_stream = [
            h['think'] for h in history
            if h.get('role') == 'assistant' and h.get('think')
        ]

        # Strip think from history - LLM only gets role/content
        for h in history:
            h.pop('think', None)

        history_tokens = sum(self.count_tokens(h['content']) for h in history)
        thought_tokens = self.count_tokens(self.thought_content or "")
        content_tokens = content_len or 0  # Assume content_len is now in tokens

        content_tokens_pct = content_tokens / self.max_context_tokens
        history_tokens_pct = history_tokens / self.max_context_tokens
        logger.info(f"Generating chat turns.  System: {content_tokens} tokens Thought : {thought_tokens} tokens Current History: {history_tokens} tokens ({len(history)} turns) | System: {content_tokens_pct:.2f} History: {history_tokens_pct:.2f}")

        fold_consciousness = 4
        history_cutoff_threshold = 0.5

        # if our history is over 50%, we need to reduce its size
        if history_tokens_pct > history_cutoff_threshold:
            # Calculate overage in tokens
            overage_tokens = int((history_tokens_pct - history_cutoff_threshold) * self.max_context_tokens)
            logger.info(f"History is over {history_cutoff_threshold:.2f}, applying compression strategy")
            
            # Choose history management strategy
            strategy = getattr(self.chat.config, 'history_management_strategy', 'sparsify')

            if strategy == "random_removal":
                logger.info(f"Using random removal strategy")
                history, removed = self._apply_random_removal_strategy(history, overage_tokens)
                logger.info(f"History overage removed: {removed} turns")
            elif strategy == "ai_summarize":
                logger.info(f"Using AI summarization strategy")
                history = self._apply_ai_summarization_strategy(history, overage_tokens, persona)
            else:  # Default to basic sparsification
                logger.info(f"Using basic sparsification strategy")
                history = self._apply_sparsification_strategy(history, overage_tokens)

            history_tokens = sum(self.count_tokens(h['content']) for h in history)
            logger.info(f"After history management: {history_tokens} tokens ({len(history)} turns)")

        assistant_turn_history = [r['content'] for r in history if r['role'] == 'assistant'][::-1]
        user_turn_history = [r['content'] for r in history if r['role'] == 'user'][::-1]
        user_turn_history.append(user_input)

        # Calculate tokens for items added after consciousness (wakeup, user_input)
        wakeup_tokens = self.count_tokens(persona.get_wakeup())
        user_input_tokens = self.count_tokens(user_input)

        # Total external tokens = history + thought + wakeup + user_input + content
        external_tokens = content_tokens + history_tokens + thought_tokens + wakeup_tokens + user_input_tokens

        consciousness = self.get_conscious_memory(
                persona=persona,
                query=user_input,
                user_queries=user_turn_history,
                assistant_queries=assistant_turn_history,
                content_len=external_tokens,
                thought_stream=thought_stream
                )

        consciousness_tokens = self.count_tokens(consciousness)
        logger.info(f"Consciousness Tokens: {consciousness_tokens}")
        
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
            # Insert current thought content above the fold
            turns = insert_at_fold(turns, f"{self.thought_content}\n\n", fold_depth=4)
            logger.info(f"Inserted thought_content above fold")

        return turns

    def _apply_sparsification_strategy(self, history: list[dict[str, str]], overage: int) -> list[dict[str, str]]:
        """
        The new strategy that sparsifies messages rather than removing them completely.

        Args:
            history (list[dict[str, str]]): The chat history
            overage (int): The amount of tokens we need to reduce

        Returns:
            list[dict[str, str]]: The updated history with sparsified messages
        """
        history_copy = copy.deepcopy(history)

        # Get or initialize text summarizer
        summarizer_method = getattr(self.chat.config, 'summarizer_method', 'auto')
        summarizer_model = getattr(self.chat.config, 'summarizer_model', None)
        use_gpu = getattr(self.chat.config, 'summarizer_use_gpu', False)

        # Current total length and target length (in tokens)
        current_tokens = sum(self.count_tokens(h['content']) for h in history_copy)
        target_tokens = current_tokens - overage

        # Use the conversation sparsification method from our new module
        summarizer = get_default_summarizer(model_name=summarizer_model, use_gpu=use_gpu)

        # Exclude the most recent 4 turns (2 exchanges) from sparsification
        preserve_recent = min(4, len(history_copy))

        # Sparsify the conversation to fit within the target length
        # Note: summarizer still uses character-based length, so we estimate
        # Approximate character length from tokens (roughly 4 chars per token)
        target_length_chars = target_tokens * 4
        return summarizer.sparsify_conversation(
            messages=history_copy,
            max_total_length=target_length_chars,
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
            overage (int): The amount of tokens we need to reduce

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

            # remove the turn (use token counting)
            overage -= self.count_tokens(history_copy[remove_turn_index]['content'])
            del history_copy[remove_turn_index]
            overage -= self.count_tokens(history_copy[remove_turn_index]['content'])
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
            overage (int): The amount of tokens we need to reduce
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

        # Current total length and target length (in tokens)
        current_tokens = sum(self.count_tokens(h['content']) for h in history_copy)
        target_tokens = current_tokens - overage
        
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

        # Calculate how much we need to reduce (in tokens)
        reduction_needed = overage
        
        # Sort older messages by token length (longest first)
        older_messages.sort(key=lambda m: self.count_tokens(m.get('content', '')), reverse=True)

        # Track how much we've reduced (in tokens)
        reduced = 0

        # First pass: Summarize longest messages
        min_tokens_to_summarize = 100  # Only summarize messages longer than this (roughly 350 chars)
        summarized_messages = []

        for message in older_messages:
            content = message.get('content', '')
            content_tokens = self.count_tokens(content)

            # Skip short messages
            if content_tokens <= min_tokens_to_summarize:
                summarized_messages.append(message)
                continue

            # Calculate target length based on role and position
            # Keep a higher percentage of user messages as they're often shorter and important for context
            if message['role'] == 'user':
                compression_ratio = 0.7  # Keep 70% of user messages
            else:  # assistant messages
                compression_ratio = 0.5  # Keep 50% of assistant messages

            # Set target token count for this message
            target_msg_tokens = max(min_tokens_to_summarize // 2, int(content_tokens * compression_ratio))

            # Don't bother with small reductions (less than 25 tokens)
            if content_tokens - target_msg_tokens < 25:
                summarized_messages.append(message)
                continue
            
            # Create parameters dictionary for caching
            # Note: summarizer still uses character-based length, so we estimate
            target_msg_length_chars = target_msg_tokens * 4  # Approximate chars from tokens
            params = {
                'target_length': target_msg_length_chars,
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

                # Calculate reduction in tokens
                summary_tokens = self.count_tokens(summary)
                reduction = content_tokens - summary_tokens
                reduced += reduction

                summarized_messages.append(new_message)
                logger.info(f"AI-summarized a {message['role']} message from {content_tokens} to {summary_tokens} tokens (cached={summary == cache.get(cache._hash_content(content, params))})")

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
            logger.info(f"AI summarization wasn't enough, need to remove {overage_remaining} more tokens")

            # Remove from oldest messages first, skipping the recent preserved ones
            while reduced < reduction_needed and len(final_messages) > preserve_recent:
                # Remove the oldest message
                removed_message = final_messages.pop(0)
                removed_tokens = self.count_tokens(removed_message.get('content', ''))
                reduced += removed_tokens
                logger.info(f"Removed a {removed_message['role']} message of {removed_tokens} tokens")

        return final_messages
