# aim/chat/strategy/xmlmemory.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
import copy
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
import random
from typing import Optional, List, Dict, Any, Tuple

from ..manager import ChatManager
from ..util import insert_at_fold
from ...utils.xml import XmlFormatter
from .base import ChatTurnStrategy
from ...utils.keywords import extract_semantic_keywords
from ...agents.persona import Persona
from ...nlp.summarize import TextSummarizer, get_default_summarizer
from ...utils.redis_cache import RedisCache
from ...utils.tokens import count_tokens as _count_tokens
from ...conversation.rerank import MemoryReranker, TaggedResult
from ...constants import (
    DOC_CONVERSATION, CHUNK_LEVEL_256, CHUNK_LEVEL_768,
    DOC_INSPIRATION, DOC_UNDERSTANDING, DOC_PONDERING, DOC_BRAINSTORM,
    DOC_MUD_AGENT, DOC_MUD_WORLD, DOC_MUD_ACTION, DOC_CODE_ACTION, DOC_CODE_FILE
)

# Insight documents - rich generative content that benefits from larger chunks (768)
INSIGHT_DOC_TYPES = [DOC_INSPIRATION, DOC_UNDERSTANDING, DOC_PONDERING, DOC_BRAINSTORM]
LONG_CONTEXT_DOC_TYPES = [
    DOC_CONVERSATION,
    DOC_MUD_AGENT,
    DOC_MUD_WORLD,
    DOC_MUD_ACTION,
    DOC_CODE_ACTION,
    DOC_CODE_FILE,
]

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONTEXT = 32768
DEFAULT_MAX_OUTPUT = 4096

class XMLMemoryTurnStrategy(ChatTurnStrategy):

    def count_tokens(self, text: str) -> int:
        """Count tokens using shared utility."""
        return _count_tokens(text)

    def __init__(self, chat : ChatManager):
        super().__init__(chat)
        self.hud_name = "HUD Display Output"

    def _calc_max_context_tokens(self, max_context_tokens: int, max_output_tokens: int) -> int:
        """Calculate usable context tokens (reserve output + system prompt + safety margin).

        The system prompt is prepended to messages in the LLM provider's stream_turns(),
        so we must account for it here to avoid exceeding context limits.
        """
        system_tokens = 0
        if hasattr(self.chat, 'config') and self.chat.config:
            system_message = getattr(self.chat.config, 'system_message', None)
            if system_message and isinstance(system_message, str):
                system_tokens = self.count_tokens(system_message)
        return max_context_tokens - max_output_tokens - system_tokens - 1024

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

    def _query_by_buckets(self,
                          queries: List[str],
                          source_tag: str,
                          seen_docs: set,
                          top_n: int,
                          length_boost: float = 0.0,
                          query_embedding: Optional[np.ndarray] = None) -> Tuple[List[TaggedResult], List[TaggedResult], List[TaggedResult]]:
        """
        Execute triple queries for optimal document distribution:
        - Conversations at chunk_768 (dialog context)
        - Insights at chunk_768 (rich generative content)
        - All types at chunk_256 (broad memory distribution)

        Args:
            queries: Query texts to search for
            source_tag: Tag for this source (memory_thought, memory_ws, etc.)
            seen_docs: Doc IDs to filter out
            top_n: Max results per bucket
            length_boost: Length boost factor
            query_embedding: Optional pre-computed embedding for FAISS reranking.
                If provided, uses this instead of computing from query text.

        Returns:
            (conversation_results, insight_results, broad_results) as lists of (source_tag, row)
        """
        conversation_results: List[TaggedResult] = []
        insight_results: List[TaggedResult] = []
        broad_results: List[TaggedResult] = []

        # Query 1: Conversations at chunk_768 (larger context for dialog)
        conv_df = self.chat.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            query_document_type=LONG_CONTEXT_DOC_TYPES,
            chunk_level=CHUNK_LEVEL_768,
            length_boost_factor=length_boost,
            query_embedding=query_embedding,
        )
        if not conv_df.empty:
            for _, row in conv_df.iterrows():
                conversation_results.append((source_tag, row))

        # Query 2: Insights at chunk_768 (rich content deserves longer context)
        insight_df = self.chat.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            query_document_type=INSIGHT_DOC_TYPES,
            chunk_level=CHUNK_LEVEL_768,
            length_boost_factor=length_boost,
            query_embedding=query_embedding,
        )
        if not insight_df.empty:
            for _, row in insight_df.iterrows():
                insight_results.append((source_tag + "_insight", row))

        # Query 3: Broad distribution at chunk_256 (all doc types for memory breadth)
        # Only NER, step, and MOTD excluded via filter_metadocs=True
        broad_df = self.chat.cvm.query(
            queries,
            filter_doc_ids=seen_docs,
            top_n=top_n,
            filter_metadocs=True,
            chunk_level=CHUNK_LEVEL_256,
            length_boost_factor=length_boost,
            query_embedding=query_embedding,
        )
        if not broad_df.empty:
            for _, row in broad_df.iterrows():
                broad_results.append((source_tag + "_broad", row))

        return conversation_results, insight_results, broad_results

    def get_consciousness_head(self, formatter: XmlFormatter) -> XmlFormatter:
        """Hook for subclasses to add content at the start of consciousness block.

        Called after formatter creation but before PraxOS header.
        Default implementation returns formatter unchanged.

        Args:
            formatter: XmlFormatter instance to extend

        Returns:
            Modified formatter
        """
        return formatter

    def get_consciousness_tail(self, formatter: XmlFormatter) -> XmlFormatter:
        """Hook for subclasses to add content at the end of consciousness block.

        Called after memory count but before render().
        Default implementation returns formatter unchanged.

        Args:
            formatter: XmlFormatter instance to extend

        Returns:
            Modified formatter
        """
        return formatter

    def get_conscious_memory(self, persona: Persona, query: Optional[str] = None, user_queries: list[str] = [], assistant_queries: list[str] = [], content_len: int = 0, thought_stream: list[str] = [], max_context_tokens: int = DEFAULT_MAX_CONTEXT, max_output_tokens: int = DEFAULT_MAX_OUTPUT, query_embedding: Optional[np.ndarray] = None) -> tuple[str, int]:
        """
        Retrieves the conscious memory content to be included in the chat response.

        The conscious memory content includes the persona's thoughts, as well as relevant memories from the conversation history. It also includes any relevant documents that have been revealed to the user.

        Args:
            query (Optional[str]): The current user query, used to filter the retrieved memories.
            user_queries (List[str]): The history of user queries, used to retrieve relevant memories.
            assistant_queries (List[str]): The history of assistant queries, used to retrieve relevant memories.
            thought_stream (List[str]): Prior reasoning from assistant turns to include in header.
            query_embedding (Optional[np.ndarray]): Pre-computed embedding for FAISS reranking.
                If provided, uses this instead of computing from query text. Typically
                from the current conversation entry's embedding.

        Returns:
            str: The conscious memory content, formatted as a string to be included in the chat response.
        """
        # Calculate usable context (reserve output tokens + safety margin)
        usable_context_tokens = self._calc_max_context_tokens(max_context_tokens, max_output_tokens)

        formatter = XmlFormatter()
        total_len = content_len
        aggregated_emotions = defaultdict(int)
        aggregated_keywords = defaultdict(int)
        seen_docs = set() # Keep track of doc_ids we've already included

        #logger.info(f"Initial Conscious Memory Length: {total_len}")

        # Hook: Allow subclasses to add content at head
        formatter = self.get_consciousness_head(formatter)

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
                             formatter.add_element(self.hud_name, "Active Memory", "memory_pinned",
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


        # 3. Journal/Conscious Entries (1 random + rest relevance-based)
        # Combine user and assistant queries for context, with user queries (more recent) last
        journal_query_context = assistant_queries + user_queries if (user_queries or assistant_queries) else None
        conscious = self.chat.cvm.get_conscious(
            persona.persona_id,
            top_n=self.chat.config.recall_size,
            query_texts=journal_query_context
        )
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
        # Estimate tokens for head/tail hooks
        # Create temporary formatter to get hook content for token counting
        temp_formatter = XmlFormatter()
        temp_formatter = self.get_consciousness_head(temp_formatter)
        head_tokens_estimate = self.count_tokens(temp_formatter.render())

        temp_formatter_tail = XmlFormatter()
        temp_formatter_tail = self.get_consciousness_tail(temp_formatter_tail)
        tail_tokens_estimate = self.count_tokens(temp_formatter_tail.render())

        available_tokens_for_dynamic_queries = (
            usable_context_tokens
            - current_tokens
            - thought_estimate_tokens
            - ws_tokens_estimate
            - scratch_tokens_estimate
            - thought_stream_estimate
            - head_tokens_estimate  # NEW
            - tail_tokens_estimate  # NEW
            - content_len  # External tokens: history, wakeup, user_input, etc.
        )
        logger.debug(f"Token budget: max={usable_context_tokens}, current={current_tokens}, thoughts={thought_estimate_tokens}, ws={ws_tokens_estimate}, scratch={scratch_tokens_estimate}, stream={thought_stream_estimate}, head={head_tokens_estimate}, tail={tail_tokens_estimate}, external={content_len}, available={available_tokens_for_dynamic_queries}")

        if available_tokens_for_dynamic_queries > 50:  # Min threshold for dynamic querying
            # Define query sources
            workspace_content_for_query = self.chat.current_workspace if self.chat.current_workspace and self.chat.current_workspace.strip() else None
            location_content_for_query = self.chat.current_location if self.chat.current_location and self.chat.current_location.strip() else None

            query_sources_data = []

            # 0. Explicit query (highest priority - from Phase 1 act args)
            if query and query.strip():
                query_sources_data.append({
                    "name": "ExplicitQuery",
                    "queries": [query.strip()],
                    "length_boost": 0.1,  # Boost explicit queries
                    "memory_type_tag": "memory_query"
                })

            # 1. Thought-based queries
            if self.thought_content and self.thought_content.strip():
                query_sources_data.append({
                    "name": "PersonaThoughts",
                    "queries": [self.thought_content.strip()],
                    "length_boost": 0.0,
                    "memory_type_tag": "memory_thought"
                })

            # 2. Workspace-based queries
            if workspace_content_for_query:
                query_sources_data.append({
                    "name": "Workspace",
                    "queries": [workspace_content_for_query],
                    "length_boost": 0.05,
                    "memory_type_tag": "memory_ws"
                })

            # 3. Location-based queries
            if location_content_for_query:
                query_sources_data.append({
                    "name": "Location",
                    "queries": [location_content_for_query],
                    "length_boost": 0.05,
                    "memory_type_tag": "memory_loc"
                })

            # 4. User history queries
            if user_queries:
                query_sources_data.append({
                    "name": "UserHistory",
                    "queries": user_queries,
                    "length_boost": 0.05,
                    "memory_type_tag": "memory_user"
                })

            # 5. Assistant history queries
            if assistant_queries:
                query_sources_data.append({
                    "name": "AssistantHistory",
                    "queries": assistant_queries,
                    "length_boost": 0.0,
                    "memory_type_tag": "memory_asst"
                })

            # Collect into three buckets: conversation, insight, and broad
            all_conversation_results: List[TaggedResult] = []
            all_insight_results: List[TaggedResult] = []
            all_broad_results: List[TaggedResult] = []

            if query_sources_data:
                top_n_per_source = self.chat.config.memory_window * 2
                if top_n_per_source == 0 and self.chat.config.memory_window > 0:
                    top_n_per_source = 2
                if self.chat.config.memory_window == 0:
                    top_n_per_source = 0

                for source_data in query_sources_data:
                    if not source_data["queries"] or top_n_per_source == 0:
                        logger.debug(f"Skipping query for {source_data['name']}")
                        continue

                    conv_results, insight_results, broad_results = self._query_by_buckets(
                        queries=source_data["queries"],
                        source_tag=source_data["memory_type_tag"],
                        seen_docs=seen_docs,
                        top_n=top_n_per_source,
                        length_boost=source_data["length_boost"],
                        query_embedding=query_embedding,
                    )
                    all_conversation_results.extend(conv_results)
                    all_insight_results.extend(insight_results)
                    all_broad_results.extend(broad_results)
                    logger.debug(f"Queried {source_data['name']}: {len(conv_results)} conv, {len(insight_results)} insight, {len(broad_results)} broad")

            # Pass to reranker - conversations + insights get 60% of budget (both chunk_768), broad gets 40%
            # Combine conversations and insights as primary content (both benefit from longer context)
            all_long_context = all_conversation_results + all_insight_results

            if all_long_context or all_broad_results:
                reranker = MemoryReranker(
                    token_counter=self.count_tokens,
                    lambda_param=0.7,
                    conversation_budget_ratio=0.6,
                )

                reranked_results = reranker.rerank(
                    conversation_results=all_long_context,  # Conversations + Insights (chunk_768)
                    other_results=all_broad_results,        # Broad distribution (chunk_256)
                    token_budget=available_tokens_for_dynamic_queries,
                    seen_parent_ids=seen_docs,
                )

                # Deduplicate by parent_doc_id/doc_id to avoid repeats across buckets
                deduped_results = []
                seen_ids = set(seen_docs)
                for source_tag, row in reranked_results:
                    doc_id = row.get('parent_doc_id', row.get('doc_id', ''))
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)
                    deduped_results.append((source_tag, row))

                # Add reranked results to formatter
                for source_tag, row in deduped_results:
                    formatter.add_element(
                        self.hud_name, "Active Memory", source_tag,
                        date=row.get('date', ''),
                        type=row.get('document_type', ''),
                        content=row.get('content', ''),
                        priority=1,
                        noindent=True
                    )
                    emotions, keywords = self.extract_memory_metadata(row)
                    for e in emotions: aggregated_emotions[e] += 1 if e else 0
                    for k in keywords: aggregated_keywords[k] += 1 if k else 0
                    seen_docs.add(row.get('parent_doc_id', row.get('doc_id', '')))

                logger.info(f"Added {len(deduped_results)} reranked results from {len(all_conversation_results)} conv + {len(all_insight_results)} insight + {len(all_broad_results)} broad candidates")
            else:
                logger.info("No dynamic query sources had results.")
        else:
            logger.info(f"Skipping dynamic memory queries: insufficient token budget ({available_tokens_for_dynamic_queries} available, need > 50)")

        # --- End Dynamic Memory Search ---


        # Add Persona Thoughts (as before)
        for thought in persona.thoughts:
            formatter.add_element(self.hud_name, "thought", content=thought, nowrap=True, priority=2)

        # Add Aggregated Emotions/Keywords (as before, but now includes dynamic results)
        # Only add emotions/keywords if there are actual non-None values
        emotions_content = ", ".join(e for e in aggregated_emotions.keys() if e is not None)
        if emotions_content:  # Only add if non-empty after filtering
            formatter.add_element(self.hud_name, "emotions", content=emotions_content, priority=1, nowrap=True)

        keywords_content = ", ".join(k for k in aggregated_keywords.keys() if k is not None)
        if keywords_content:  # Only add if non-empty after filtering
            formatter.add_element(self.hud_name, "keywords", content=keywords_content, priority=1, nowrap=True)

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

        # Add memory count at the end
        formatter.add_element(self.hud_name, "Memory Count", content=str(len(seen_docs)), nowrap=True, priority=1)

        # Hook: Allow subclasses to add content at tail
        formatter = self.get_consciousness_tail(formatter)

        final_output = formatter.render()
        final_tokens = self.count_tokens(final_output)
        logger.debug(f"Final Conscious Memory: Total Tokens: {final_tokens}/{usable_context_tokens}")

        # Return content and memory count (seen_docs minus non-memory items like workspace)
        memory_count = len(seen_docs)
        return final_output, memory_count
        
    def chat_turns_for(self, persona: Persona, user_input: str, history: list[dict[str, str]] = [], content_len: Optional[int] = None, max_context_tokens: int = DEFAULT_MAX_CONTEXT, max_output_tokens: int = DEFAULT_MAX_OUTPUT, query: str = "", query_embedding: Optional[np.ndarray] = None) -> list[dict[str, str]]:
        """
        Generate a chat session, augmenting the response with information from the database.

        This is what will be passed to the chat completion API.

        Args:
            user_input (str): The user input.
            history (List[Dict[str, str]]): The chat history (may include 'think' field).
            query (str): Optional explicit query for memory search. If provided, used
                instead of user_input for CVM retrieval.
            query_embedding (Optional[np.ndarray]): Pre-computed embedding for FAISS reranking.
                If provided, uses this instead of computing from query text. Typically
                from the current conversation entry's embedding.

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

        # Calculate usable context (reserve output tokens + safety margin)
        usable_context_tokens = self._calc_max_context_tokens(max_context_tokens, max_output_tokens)

        content_tokens_pct = content_tokens / usable_context_tokens
        history_tokens_pct = history_tokens / usable_context_tokens
        logger.info(f"Generating chat turns.  System: {content_tokens} tokens Thought : {thought_tokens} tokens Current History: {history_tokens} tokens ({len(history)} turns) | System: {content_tokens_pct:.2f} History: {history_tokens_pct:.2f}")

        fold_consciousness = 4
        history_cutoff_threshold = 0.5

        # if our history is over 50%, we need to reduce its size
        if history_tokens_pct >= history_cutoff_threshold:
            # Calculate overage in tokens
            overage_tokens = int((history_tokens_pct - history_cutoff_threshold) * usable_context_tokens)
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

        # Use explicit query if provided, otherwise fall back to user_input
        memory_query = query if query else user_input
        consciousness, memory_count = self.get_conscious_memory(
                persona=persona,
                query=memory_query,
                user_queries=user_turn_history,
                assistant_queries=assistant_turn_history,
                content_len=external_tokens,
                thought_stream=thought_stream,
                max_context_tokens=max_context_tokens,
                max_output_tokens=max_output_tokens,
                query_embedding=query_embedding,
                )

        consciousness_tokens = self.count_tokens(consciousness)
        logger.info(f"Consciousness Tokens: {consciousness_tokens}")

        consciousness_turn = {"role": "user", "content": consciousness}

        wakeup = persona.get_wakeup()
        # Template replacement for wakeup
        wakeup = wakeup.replace("{{memory_count}}", str(memory_count))
        wakeup_turn = {"role": "assistant", "content": wakeup}

        if len(history) > 0:
            if history[0]['role'] == 'assistant':
                turns = [consciousness_turn, *history]
            else:
                turns = [consciousness_turn, wakeup_turn, *history]
            
        else:
            turns = [consciousness_turn, wakeup_turn]

        # Only add user turn if there's actual content (avoids empty user turns)
        if user_input and user_input.strip():
            turns.append({"role": "user", "content": user_input + "\n\n"})

        if self.thought_content:
            # Insert current thought content above the fold
            turns = insert_at_fold(turns, f"{self.thought_content}\n\n<% End XML Thought Begin Action %>\n\n", fold_depth=4)
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
