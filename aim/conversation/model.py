# aimnversation/model.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
from datetime import datetime
import json
import faiss
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time
import pytz
from typing import Optional, Set, List, Dict, Any
from wonderwords import RandomWord

from ..config import ChatConfig
from ..constants import DOC_ANALYSIS, DOC_CODEX, DOC_CONVERSATION, DOC_JOURNAL, DOC_NER, DOC_STEP, LISTENER_ALL, DOC_MOTD
from .index import SearchIndex
from .message import ConversationMessage, VISIBLE_COLUMNS, QUERY_COLUMNS
from .loader import ConversationLoader

logger = logging.getLogger(__name__)


def sanitize_timestamp(timestamp: int) -> int:
    """
    Sanitizes a timestamp to ensure it's within valid datetime range.
    Max timestamp is set to year 9999 to stay within datetime limits.
    Min timestamp is set to year 1970 (Unix epoch start).
    """
    max_timestamp = 253402300799  # 9999-12-31 23:59:59
    min_timestamp = 0  # 1970-01-01 00:00:00

    if not isinstance(timestamp, (int, float)):
        return min_timestamp

    return max(min(int(timestamp), max_timestamp), min_timestamp)


def mmr_rerank(df: pd.DataFrame, score_col: str = 'score', embedding_col: str = 'index_a',
               top_n: int = 10, lambda_param: float = 0.7) -> pd.DataFrame:
    """
    Apply Maximum Marginal Relevance (MMR) reranking to balance relevance with diversity.

    MMR iteratively selects documents that are both relevant to the query and
    diverse from already-selected documents. This helps surface unique entries
    that might otherwise be suppressed by similar, higher-scoring documents.

    Args:
        df: DataFrame with scores and embeddings
        score_col: Column name containing relevance scores
        embedding_col: Column name containing document embeddings (numpy arrays)
        top_n: Number of documents to select
        lambda_param: Tradeoff between relevance (1.0) and diversity (0.0).
                      Default 0.7 favors relevance while still promoting diversity.

    Returns:
        DataFrame reordered by MMR selection order
    """
    if df.empty or len(df) <= 1:
        return df

    # Normalize scores to [0, 1] for fair MMR calculation
    scores = df[score_col].values
    if scores.max() > scores.min():
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        norm_scores = np.ones(len(scores))

    # Get embeddings as matrix
    embeddings = np.stack(df[embedding_col].values)

    # Precompute cosine similarities between all document pairs
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized_embeddings = embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # MMR selection
    n_docs = len(df)
    selected_indices = []
    remaining_indices = list(range(n_docs))

    # Select top_n documents (or all if fewer available)
    n_to_select = min(top_n, n_docs)

    for _ in range(n_to_select):
        if not remaining_indices:
            break

        best_idx = None
        best_mmr = float('-inf')

        for idx in remaining_indices:
            relevance = norm_scores[idx]

            # Calculate max similarity to already selected documents
            if selected_indices:
                max_sim = max(similarity_matrix[idx, sel_idx] for sel_idx in selected_indices)
            else:
                max_sim = 0  # First selection has no penalty

            # MMR score: balance relevance and diversity
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    # Return DataFrame reordered by MMR selection
    return df.iloc[selected_indices].reset_index(drop=True)


class ConversationModel:
    collection_name : str = 'memory'

    def __init__(self, memory_path: str, embedding_model: str, user_timezone: Optional[str] = None, embedding_device: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.index = SearchIndex(Path('.', memory_path, 'indices'), embedding_model=embedding_model, device=embedding_device)
        self.memory_path = memory_path
        self.loader = ConversationLoader(conversations_dir=os.path.join(memory_path, 'conversations'))
        self.user_timezone = pytz.timezone(user_timezone) if user_timezone is not None else None

    @classmethod
    def init_folders(cls, memory_path: str):
        """
        Creates the necessary folders for the conversation model.
        """
        collection_path = Path(f'./{memory_path}/conversations')
        if not collection_path.exists():
            collection_path.mkdir(parents=True)
        index_path = Path(f'./{memory_path}/indices')
        if not index_path.exists():
            index_path.mkdir(parents=True)

    @classmethod
    def from_config(cls, config: ChatConfig) -> 'ConversationModel':
        """
        Creates a new conversation model from the given config.
        """
        cls.init_folders(config.memory_path)
        return cls(memory_path=config.memory_path, embedding_model=config.embedding_model, user_timezone=config.user_timezone, embedding_device=config.embedding_device)

    @property
    def collection_path(self) -> Path:
        """
        Returns the path to the collection.
        """
        return Path(f'./{self.memory_path}/conversations')
    
    def refresh(self) -> None:
        """
        Refreshes the collection.
        """
        self.index.index.reload()
        
    def load_conversation(self, conversation_id: str) -> list[ConversationMessage]:
        """
        Loads a conversation from the collection.
        """
        return self.loader.load_or_new(conversation_id)

    def get_by_doc_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its doc_id.

        First tries the index, then falls back to searching JSONL files directly.
        """
        # Try index first (fast path)
        result = self.index.get_document(doc_id)
        if result is not None:
            return result

        # Fallback: search JSONL files directly
        for jsonl_file in self.collection_path.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        doc = json.loads(line)
                        if doc.get('doc_id') == doc_id:
                            logger.info(f"Found {doc_id} in {jsonl_file.name} (not in index)")
                            return doc
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

        return None

    def _append_message(self, message: ConversationMessage) -> None:
        """
        Appends a message to a conversation.
        """
        # Get our document
        document_name = self.collection_path / f"{message.conversation_id}.jsonl"
        create = True if not document_name.exists() else False
        
        if create:
            # Create an empty document
            with open(document_name, 'w') as f:
                f.write(json.dumps(message.to_dict()) + '\n')
        else:
            # Append the message
            with open(document_name, 'a') as f:
                f.write(json.dumps(message.to_dict()) + '\n')

    def insert(self, message: ConversationMessage) -> None:
        """
        Inserts a conversation in to the collection.
        """

        logger.info(f"Inserting {message.doc_id} into {self.collection_name}/{message.conversation_id}")
        # First, check if we already have the document
        document_name = self.collection_path / f"{message.conversation_id}.jsonl"
        if not document_name.exists():
            # Create an empty document
            with open(document_name, 'w') as f:
                f.write('')
        
        # Append the message
        self._append_message(message)
        self.index.add_document(message.to_dict())
        
    def update_document(self, conversation_id: str, document_id: str, update_data: dict[str, Any]) -> None:
        """
        Updates a document in the collection.
        """
        # Get our document
        document_name = self.collection_path / f"{conversation_id}.jsonl"
        if not document_name.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

        # find the message, and replace it
        new_document = []
        with open(document_name, 'r') as f:
            for line in f:
                line : dict = json.loads(line)
                if line['doc_id'] == document_id:
                    # This is unsafe
                    line.update(update_data)
                new_document.append(json.dumps(line) + "\n")

        # Write the new document
        with open(document_name, 'w') as f:
            for line in new_document:
                f.write(line)

    def delete_conversation(self, conversation_id: str, persona_id : Optional[str] = None, user_id : Optional[str] = None) -> None:
        """
        Deletes a conversation from the collection.
        """

        # Get our document
        document_name = self.collection_path / f"{conversation_id}.jsonl"
        if not document_name.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

        if persona_id is None and user_id is None:
            document_name.unlink()
            return
        
        new_document = []
        with open(document_name, 'r') as f:
            for line in f:
                line = json.loads(line)
                if line['persona_id'] == persona_id and line['user_id'] == user_id:
                    continue
                new_document.append(json.dumps(line) + "\n")

        # Write the new document
        with open(document_name, 'w') as f:
            for line in new_document:
                f.write(line)

    def delete_document(self, conversation_id: str, message_id: str) -> None:
        """
        Deletes a document from the collection.
        """

        # Get our document
        document_name = self.collection_path / f"{conversation_id}.jsonl"
        if not document_name.exists():
            raise FileNotFoundError(f"Conversation {conversation_id} not found")

        # find the message, and replace it
        new_document = []
        with open(document_name, 'r') as f:
            for line in f:
                line = json.loads(line)
                if line['doc_id'] == message_id:
                    continue
                new_document.append(json.dumps(line) + "\n")

        # Write the new document
        with open(document_name, 'w') as f:
            for line in new_document:
                f.write(line)
    
    def query(self, query_texts: List[str], filter_doc_ids: Optional[Set[str]] = None, top_n: Optional[int] = None,
              query_document_type: Optional[str | list[str]] = None, query_conversation_id: Optional[str] = None,
              max_length: Optional[int] = None, turn_decay: float = 0.7, temporal_decay: float = 0.5, length_boost_factor: float = 0.0,
              filter_metadocs: bool = True, chunk_size: int = -1, sort_by: str = 'relevance', diversity: float = 0.3,
              keyword_boost: float = 2.0, chunk_level: str = "full", **kwargs) -> pd.DataFrame:
        """
        Queries the conversation collection and returns a DataFrame containing the top `top_n` most relevant conversation entries based on the given query texts, filters, and decay factors.

        The query is performed using the collection's search functionality, with optional filters applied to exclude certain document types, document IDs, and text content. The relevance score for each entry is calculated as a combination of the text similarity to the query texts, the temporal decay based on the entry's timestamp, and the entry's weight.

        Args:
            query_texts: List of query strings to search for.
            filter_doc_ids: Document IDs to exclude from results.
            top_n: Maximum number of results to return.
            query_document_type: Filter to specific document type(s).
            query_conversation_id: Filter to specific conversation.
            max_length: Maximum cumulative content length.
            turn_decay: Decay factor for conversation turns (unused currently).
            temporal_decay: Exponential decay factor for recency (default 0.3, reduced for
                           memory search). Floor of 0.3 prevents old memories from being zeroed.
            length_boost_factor: Boost factor for content length.
            filter_metadocs: Whether to filter out NER/step documents.
            chunk_size: Size for chunking long queries (-1 to disable).
            sort_by: Sort order - 'relevance' (with MMR diversity) or 'recency'.
            diversity: MMR diversity factor (0.0-1.0). Higher values promote more
                       diverse results at the cost of pure relevance. Set to 0 to
                       disable MMR and use pure relevance ranking. Default 0.3.
            keyword_boost: Boost factor for capitalized terms/n-grams (proper nouns,
                          names, acronyms). These are detected and boosted in the
                          Tantivy query. Set to 1.0 to disable. Default 2.0.
            chunk_level: Filter to specific chunk level (chunk_256, chunk_768, full).
                        Defaults to "full" for backwards compatibility.

        Returns:
            DataFrame with columns: VISIBLE_COLUMNS + ['date', 'speaker', 'score', 'index_a'] + CHUNK_COLUMNS
        """

        # `query_limit` is how many results to fetch from the underlying search index.
        # `top_n` is how many results to return to the caller after processing.
        # Setting query_limit higher than top_n to ensure diverse candidates for reranking and MMR.
        # Using top_n * 3 as a balance between recall and performance.
        query_limit = top_n * 3 if top_n is not None else None # Allow top_n to be None for unlimited internal fetch

        original_query_texts = list(query_texts) if query_texts is not None else [] # Keep a copy for reranking

        if chunk_size > 0 and original_query_texts:
            # Define get_chunks helper inside this block as it's only used here
            def get_chunks(text: str, size: int) -> List[str]:
                if not isinstance(text, str) or not text:
                    return []
                if size <= 0 or len(text) <= size:
                    return [text]
                return [text[i:i+size] for i in range(0, len(text), size)]

            all_chunks = []
            for q_text in original_query_texts:
                if isinstance(q_text, str) and q_text:
                    all_chunks.extend(get_chunks(q_text, chunk_size))
            
            if not all_chunks:
                logger.warning("No valid chunks generated from query texts, returning empty DataFrame")
                return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])
            
            #logger.info(f"Generated {len(all_chunks)} chunks for searching.")
            query_texts_for_index_search = all_chunks
            # The rest of the function will now proceed as if these chunks were the original query_texts,
            # but reranking will use original_query_texts[-1]
            query_texts = query_texts_for_index_search # Modify query_texts for self.index.search call

        # Original logic starts here (or continues with chunked query_texts)
        # Filter out meta documents and MOTD (MOTD retrieved separately via get_motd)
        filter_document_type = [DOC_NER, DOC_STEP, DOC_MOTD] if filter_metadocs and not query_document_type else None

        if not query_texts: # Handles empty list and ensures it's not None for len()
            logger.warning("No query texts provided, returning empty DataFrame")
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])

        results = self.index.search(query_texts, query_document_type=query_document_type, filter_doc_ids=filter_doc_ids, filter_document_type=filter_document_type, query_conversation_id=query_conversation_id,
                                    query_limit=query_limit, keyword_boost=keyword_boost, chunk_level=chunk_level)

        if query_conversation_id is not None:
            conversation = self._query_conversation(conversation_id=query_conversation_id, query_document_type=query_document_type)

            # Sniff our results index_a vector, and create our zero-vector
            if results.empty:
                logger.warning("No results found for query conversation, returning empty DataFrame")
                return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])
            embedding_dim = results.iloc[0]['index_a']
            zero_vector  = np.zeros(embedding_dim.shape)

            # Results header is QUERY_COLUMNS + ['distance', 'hits']
            conversation['hits'] = 0
            conversation['distance'] = 1000
            conversation['index_a'] = [zero_vector] * len(conversation)

            # Merge the conversation history into the results
            # First, find our difference
            conversation_ids = set(conversation['doc_id'])
            results_ids = set(results['doc_id'])
            conversation_ids_to_add = conversation_ids - results_ids
            # Now we need to add the missing conversations to our results
            results = pd.concat([results, conversation[conversation['doc_id'].isin(conversation_ids_to_add)]])
        
        if len(results) == 0:
            logger.warning("No results found, returning empty DataFrame")
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])

        #print(results)
        #logger.info(f"Found {len(results)} results")
        
        # Our results come back with hits, representing the number of matches for a single document. We need to boost the score as the hits go up; but not as linearly as the hits do.
        # Cap at 2.0 to prevent keyword-heavy old docs from drowning out recency
        results['hits_score'] = np.minimum(np.log2(results['hits'] + 1), 2.0)

        # Vectorize our query texts and get the similarity scores
        # IMPORTANT: Use the LAST of the ORIGINAL query texts for reranking, even if chunks were used for search
        rerank_query_text = original_query_texts[-1] if original_query_texts and isinstance(original_query_texts[-1], str) and original_query_texts[-1] else None

        if rerank_query_text and not results.empty:
            query_vectors = np.array(self.index.vectorizer.transform([rerank_query_text]))
            
            # Check if query_vectors is empty or has zero dimension
            if query_vectors.size == 0 or query_vectors.shape[-1] == 0:
                logger.warning(f"Could not generate valid query vector for reranking from: {rerank_query_text}. Skipping FAISS reranking.")
                results['rerank'] = 1.0 # Neutral rerank score
            else:
                faiss_index = faiss.IndexFlatL2(query_vectors[0].shape[0])
                result_indices = np.stack(results['index_a'].to_numpy())
                faiss_index.add(result_indices)
                distance, index = faiss_index.search(query_vectors, results.shape[0])
                distance_index = zip(distance[0], index[0])

                distance_index = sorted(distance_index, key=lambda x: x[1])
                # Normalize rerank to [0, 1] range using 1/(1+d) formula
                # This prevents unbounded scores and gives d=0 (perfect match) a score of 1.0
                results['rerank'] = [1 / (1 + d) for d, _ in distance_index]
        elif results.empty:
            results['rerank'] = 1.0 # Neutral rerank score, or handle as appropriate if results is empty
        else: # No rerank_query_text or results is empty
            logger.warning("Skipping FAISS reranking due to no valid rerank_query_text or empty results.")
            results['rerank'] = 1.0 # Neutral rerank score

        results['length_score'] = (np.log2(results['content'].str.len() + 1) * length_boost_factor) + 1
        # Scale with the following rules - 
        # A slope of 7 days, and a y-intercept of 1
        # Beta binomial curve optimizing for recency

        current_time = int(time.time())
        thirty_days_in_seconds = 30 * 24 * 60 * 60 
        decay_length = thirty_days_in_seconds

        # Calculate decay factor with floor to prevent old memories from being zeroed out
        temporal_decay_floor = 0.3
        raw_decay = np.exp(-temporal_decay * (current_time - results['timestamp']) / decay_length)
        results['temporal_decay'] = np.maximum(raw_decay, temporal_decay_floor)
        
        # Now, we sum the scores, and multiply by the dscore
        results['score'] = results.filter(regex='score_').sum(axis=1).fillna(0) * \
                           results['hits_score'] * \
                           results['rerank'] * \
                           results['weight'] * \
                           results['temporal_decay'] + results['length_score']

        results['date'] = results['timestamp'].apply(lambda d: datetime.fromtimestamp(sanitize_timestamp(d), self.user_timezone).strftime('%Y-%m-%d %H:%M:%S'))

        # if the role is user, we use the user_id as the speaker, if the role is assistant, we use the persona_id
        results['speaker'] = results.apply(lambda row: row['user_id'] if row['role'] == 'user' else row['persona_id'], axis=1)

        # return our filtered results
        # Sort based on sort_by parameter, then apply top_n
        if sort_by == 'recency':
            results = results.sort_values(by='timestamp', ascending=False)
            if top_n is not None:
                results = results.head(top_n)
        elif sort_by == 'random':
            # Random sampling for exploration - no semantic ranking
            if top_n is not None and len(results) > top_n:
                results = results.sample(n=top_n)
            elif len(results) > 0:
                results = results.sample(frac=1)  # Shuffle all results
        elif sort_by == 'relevance':
            # Use MMR to balance relevance with diversity for unique entry recall
            # First sort by score to ensure MMR starts with the best candidates
            results = results.sort_values(by='score', ascending=False)
            if top_n is not None and 'index_a' in results.columns and diversity > 0:
                # Apply MMR reranking to get diverse results
                # lambda_param = 1 - diversity: higher diversity = lower lambda = more diversity penalty
                lambda_param = 1.0 - diversity
                results = mmr_rerank(results, score_col='score', embedding_col='index_a',
                                     top_n=top_n, lambda_param=lambda_param)
            elif top_n is not None:
                results = results.head(top_n)
        else:
            logger.warning(f"Unknown sort_by value: {sort_by}. Defaulting to 'relevance'")
            results = results.sort_values(by='score', ascending=False)
            if top_n is not None:
                results = results.head(top_n)
            
        if not results.empty and 'content' in results.columns:
            results['cumlen'] = results['content'].astype(str).str.len().cumsum()
            if max_length is not None:
                results = results[results['cumlen'] <= max_length]
        elif not results.empty: # content column missing
            logger.warning("'content' column missing for cumlen calculation. Setting cumlen to 0.")
            results['cumlen'] = 0
            if max_length is not None:
                 results = results[results['cumlen'] <= max_length]

        if results.empty: # If filters made results empty
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])

        # Return results with embeddings and chunk info for caller to handle reranking
        return_columns = VISIBLE_COLUMNS + ['date', 'speaker', 'score', 'index_a', 'parent_doc_id', 'chunk_level', 'chunk_index', 'chunk_start', 'chunk_end', 'chunk_count']
        # Only include columns that exist in the results
        available_columns = [col for col in return_columns if col in results.columns]
        return results[available_columns]

    def get_motd(self, top_n: int = 1) -> pd.DataFrame:
        results = self.index.search(query_document_type=DOC_MOTD, query_limit=top_n, descending=True)

        results = self._fix_dataframe(results)
        return results[VISIBLE_COLUMNS + ['date', 'speaker']].head(top_n)
        
    def get_conscious(self, persona_id: str, top_n: int, query_texts: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get conscious/journal entries for a persona.

        Strategy: Pull N random entries, search for relevant ones, replace randoms with relevant if found.
        """
        if top_n <= 0:
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker'])

        # 1. Get random journal entries
        results: pd.DataFrame = self.index.search(
            query_document_type=DOC_JOURNAL,
            query_persona_id=persona_id,
            query_limit=max(top_n * 3, 10)
        )

        if results.empty:
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker'])

        results = self._fix_dataframe(results)
        results['random_score'] = np.random.rand(len(results)) * results['weight']
        random_entries = results.nlargest(top_n, 'random_score')

        # 2. Search for relevant entries if we have context
        if query_texts and len(query_texts) > 0:
            relevance_results = self.index.search(
                query_texts=query_texts,
                query_document_type=DOC_JOURNAL,
                query_persona_id=persona_id,
                query_limit=top_n
            )

            if not relevance_results.empty:
                relevance_results = self._fix_dataframe(relevance_results)
                if 'distance' in relevance_results.columns:
                    relevance_results = relevance_results.nlargest(top_n, 'distance')

                # 3. Merge: relevant first, fill rest with randoms not already included
                relevant_ids = set(relevance_results['doc_id'])
                remaining_random = random_entries[~random_entries['doc_id'].isin(relevant_ids)]
                slots_for_random = top_n - len(relevance_results)

                if slots_for_random > 0:
                    combined = pd.concat([relevance_results, remaining_random.head(slots_for_random)], ignore_index=True)
                else:
                    combined = relevance_results.head(top_n)

                return combined.drop_duplicates(subset=['doc_id'])[VISIBLE_COLUMNS + ['date', 'speaker']].head(top_n)

        return random_entries[VISIBLE_COLUMNS + ['date', 'speaker']].head(top_n)

    def sample_by_type(
        self,
        doc_types: List[str],
        top_n: int = 20,
        persona_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Randomly sample documents by type without semantic search.

        Args:
            doc_types: List of document types to sample from
            top_n: Maximum number of documents to return
            persona_id: Optional persona ID filter

        Returns:
            DataFrame with randomly sampled documents
        """
        if not doc_types or top_n <= 0:
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])

        all_results = []

        # Query each doc type and collect results
        for doc_type in doc_types:
            results = self.index.search(
                query_document_type=doc_type,
                query_persona_id=persona_id,
                query_limit=top_n * 2,  # Get extra to sample from
            )
            if not results.empty:
                all_results.append(results)

        if not all_results:
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])

        # Combine all results
        combined = pd.concat(all_results, ignore_index=True)
        combined = self._fix_dataframe(combined)

        # Remove duplicates
        combined = combined.drop_duplicates(subset=['doc_id'])

        # Random sample with weight influence
        if len(combined) > top_n:
            combined['random_score'] = np.random.rand(len(combined)) * combined['weight']
            combined = combined.nlargest(top_n, 'random_score')
        else:
            combined = combined.sample(frac=1)  # Shuffle

        combined['score'] = 1.0  # Dummy score for compatibility

        return_columns = VISIBLE_COLUMNS + ['date', 'speaker', 'score']
        available_columns = [col for col in return_columns if col in combined.columns]
        return combined[available_columns]

    def _fix_dataframe(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Fixes the given DataFrame by adding the missing columns and removing the unnecessary ones.
        """
        results['date'] = results['timestamp'].apply(lambda d: datetime.fromtimestamp(sanitize_timestamp(d), self.user_timezone).strftime('%Y-%m-%d %H:%M:%S'))
        results['speaker'] = results.apply(lambda row: row['user_id'] if row['role'] == 'user' else row['persona_id'], axis=1)
        return results
        
    def get_documents(self, message_ids: list[str]) -> pd.DataFrame:
        """
        Returns a DataFrame containing the documents with the given `converation_id/document_id`.
        """

        results = [
            self.index.get_document(message_id) for message_id in message_ids
        ]
        
        results = [
            r for r in results if r is not None
        ]

        #logger.info(results)
        
        results = pd.DataFrame(results)
        #logger.info(results.columns)
        results = self._fix_dataframe(results)
        return results[VISIBLE_COLUMNS + ['date', 'speaker']]

    def next_conversation_id(self) -> str:
        """
        Returns a valid unique conversation ID for a given user.
        """

        # A converation id is three random words separated by dashes
        random_words = RandomWord()
        conversation_id = "-".join(random_words.word() for _ in range(2))

        # We need to check if the conversation ID already exists
        converation_file = self.collection_path / f"{conversation_id}.jsonl"
        
        if converation_file.exists():
            return self.next_conversation_id()

        return conversation_id

    def _query_conversation(self, conversation_id: str, query_document_type: Optional[str | list[str]] = None, filter_document_type: Optional[str | list[str]] = None, **kwargs) -> pd.DataFrame:
        """
        Returns the conversation history for a given user and conversation ID.
        """

        if query_document_type is not None and filter_document_type is not None:
            raise ValueError("You can't filter and query at the same time.")

        # Load our conversation file
        conversations = self.load_conversation(conversation_id)
        if len(conversations) == 0:
            return pd.DataFrame(columns=QUERY_COLUMNS)
        results = pd.DataFrame([c.to_dict() for c in conversations])

        if type(query_document_type) == str:
            results = results[results['document_type'] == query_document_type]
        elif type(query_document_type) == list:
            results = results[results['document_type'].isin(query_document_type)]

        if type(filter_document_type) == str:
            results = results[results['document_type'] != filter_document_type]
        elif type(filter_document_type) == list:
            results = results[~results['document_type'].isin(filter_document_type)]

        logger.info(f"Found {len(results)} results for {conversation_id}")

        # Filter to query columns, only including columns that exist in the dataframe
        available_columns = [col for col in QUERY_COLUMNS if col in results.columns]
        return results[available_columns]

    def get_conversation_history(self, conversation_id: str, query_document_type: Optional[str | list[str]] = None, filter_document_type: Optional[str | list[str]] = None, **kwargs) -> pd.DataFrame:
        """
        Returns the conversation history for a given user and conversation ID.
        """

        if query_document_type is not None and filter_document_type is not None:
            raise ValueError("You can't filter and query at the same time.")

        # Load our conversation file
        results : pd.DataFrame = self._query_conversation(conversation_id, query_document_type, filter_document_type, **kwargs)

        results['date'] = results['timestamp'].apply(lambda d: datetime.fromtimestamp(sanitize_timestamp(d), self.user_timezone).strftime('%Y-%m-%d %H:%M:%S'))
        results['speaker'] = results.apply(lambda row: row['user_id'] if row['role'] == 'user' else row['persona_id'], axis=1)

        # Filter to visible columns, only including columns that exist in the dataframe
        desired_columns = VISIBLE_COLUMNS + ['date', 'speaker']
        available_columns = [col for col in desired_columns if col in results.columns]
        return results[available_columns]

    def ner_query(self, query_text: str, filter_text: Optional[str] = None, top_n: int = -1, **kwargs) -> pd.DataFrame:
        """
        Returns the convesation id's for a given query.
        """
        filter_text = f"document_type = '{DOC_NER}'" if filter_text is None else f"document_type = '{DOC_NER}' and ({filter_text})"
        return self.query([query_text], filter_text, top_n, **kwargs)
    
    def get_next_branch(self, conversation_id: str) -> int:
        """
        Returns the maximum branch number for a given user and conversation ID.
        """
        conversation = self.get_conversation_history(conversation_id)

        if conversation.empty:
            logger.warning("No branches found")
            return 0
        
        next_branch = int(conversation['branch'].max()) + 1
        logger.info(f"Next branch: {next_branch}")
        return next_branch

    def get_conversation_report(self):
        all_df = self.to_pandas()
        # determine conversations without analysis:
        # we need to find all conversation ids by document type
        if all_df.empty:
            return pd.DataFrame(columns=['conversation_id', 'document_type', 'timestamp_max'])
        docs = all_df.groupby(['document_type', 'conversation_id']).size().reset_index()
        # reshape so docuemnt types are columns
        docs = docs.pivot(index='conversation_id', columns='document_type', values=0).fillna(0).reset_index()
        conversation_time = all_df.groupby('conversation_id').agg({'timestamp': 'max'}).reset_index()
        conversation_time.columns = ['conversation_id', 'timestamp_max']

        conversation_report = pd.merge(docs, conversation_time, on='conversation_id').sort_values('timestamp_max')
        return conversation_report

    @property
    def next_analysis(self) -> Optional[str]:
        cr = self.get_conversation_report()
        conversation_mask = cr[DOC_CONVERSATION] > 0
        if DOC_ANALYSIS not in cr.columns:
            next_analysis = cr[conversation_mask].sort_values(by='timestamp_max', ascending=True).head(1)
        else:
            analysis_mask = cr[DOC_ANALYSIS] == 0
            next_analysis = cr[conversation_mask & analysis_mask].sort_values(by='timestamp_max', ascending=True).head(1)
        if next_analysis.empty:
            return None
        else:
            return next_analysis['conversation_id'].values[0]

    def to_pandas(self) -> pd.DataFrame:
        """
        If you really need all of the data...
        """
        files = self.collection_path.glob('*.jsonl')
        results = []
        lineno = 0
        for file in files:
            try:
                with open(file, 'r') as f:
                    for i, line in enumerate(f):
                        lineno = i
                        if len(line) == 0:
                            continue
                        jline = json.loads(line)
                        # If we rename the file, we need to update the conversation_id
                        jline['conversation_id'] = file.stem
                        jline['lineno'] = lineno
                        results.append(jline)
            except json.decoder.JSONDecodeError as e:
                logger.warning(f"Could not read {file} line {lineno}: {e}")

        return pd.DataFrame(results)
