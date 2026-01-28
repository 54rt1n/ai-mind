# aim/conversation/index.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import pandas as pd
import re
import tiktoken
from tantivy import Index, Document as TantivyDocument, SchemaBuilder, Query, Occur, Order
from ..constants import (
    DOC_CONVERSATION,
    CHUNK_LEVEL_256, CHUNK_LEVEL_768, CHUNK_LEVEL_FULL,
    CHUNK_SIZE_256, CHUNK_SIZE_768, CHUNK_SLIDE_256, CHUNK_SLIDE_768
)
from .blacklist import STOPWORDS, BLACKLIST_WORDS, CONTRACTIONS
from .embedding import HuggingFaceEmbedding
from .message import VISIBLE_COLUMNS, QUERY_COLUMNS

logger = logging.getLogger(__name__)


def clean_query_text(text: str) -> str:
    """Clean text for Tantivy query parsing: remove contractions and special chars."""
    if not text:
        return text
    # Remove contractions entirely (they'll be filtered as stopwords anyway)
    words = text.split()
    filtered = [w for w in words if w.lower().rstrip(".,!?;:'\"") not in CONTRACTIONS]
    joined = ' '.join(filtered)
    # Remove special chars except word chars and whitespace, normalize whitespace
    cleaned = re.sub(r"\s+", ' ', re.sub(r'[^\w\s]', ' ', joined))
    return cleaned.strip()


def boost_query_terms(text: str, base_boost: float = 1.0, keyword_boost: float = 2.0, max_terms: int = 0) -> str:
    """
    Boost query terms with position-based and keyword-based weights.

    Filters out stopwords and blacklist words, then sorts by importance
    (length, capitalization) and keeps top N terms.

    Args:
        text: Original query text
        base_boost: Base multiplier for all terms (default 1.0)
        keyword_boost: Additional multiplier for capitalized terms (default 2.0)
        max_terms: Maximum number of terms to include (0 = unlimited)

    Returns:
        Query string with terms boosted.
        e.g., with base_boost=1.5, keyword_boost=2.0:
        "I met John at NASA" -> 'john^3.0 nasa^3.0 met^1.5'
    """
    if not text:
        return text

    words = text.split()
    term_info = []  # List of (clean_word, clean_lower, is_keyword, length)

    for i, word in enumerate(words):
        # Clean the word for analysis (remove punctuation at edges)
        clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)

        if not clean_word:
            continue

        clean_lower = clean_word.lower()

        # Filter out stopwords and blacklist words
        if clean_lower in STOPWORDS or clean_word in BLACKLIST_WORDS:
            continue

        # Check capitalization for keyword boost
        is_capitalized = clean_word[0].isupper()
        is_all_caps = clean_word.isupper() and len(clean_word) > 1
        is_first_word = (i == 0)

        # Keyword boost if: ALL CAPS (acronyms), or capitalized but not sentence-start
        is_keyword = is_all_caps or (is_capitalized and not is_first_word)

        term_info.append((clean_word, clean_lower, is_keyword, len(clean_word)))

    # Sort by (length desc, is_keyword desc) - longer words and keywords first
    term_info.sort(key=lambda x: (x[3], x[2]), reverse=True)

    # Limit terms if max_terms is set
    if max_terms > 0 and len(term_info) > max_terms:
        term_info = term_info[:max_terms]

    # Build result with boosts
    result_parts = []
    for clean_word, clean_lower, is_keyword, _ in term_info:
        # Calculate total boost: base * keyword_multiplier
        if is_keyword and keyword_boost > 1.0:
            total_boost = base_boost * keyword_boost
        else:
            total_boost = base_boost

        # Only add boost syntax if boost > 1.0
        if total_boost > 1.0:
            result_parts.append(f'{clean_lower}^{total_boost:.1f}')
        else:
            result_parts.append(clean_lower)

    return ' '.join(result_parts)


class SearchIndex:
    """Tantivy-based search index for conversations"""

    def __init__(self, index_path: Path, embedding_model: str = "arkohut/jina-embeddings-v3", device: str = "cpu", skip_vectorizer: bool = False, keep_warm: bool = True):
        """Initialize the search index.

        Args:
            index_path: Path to the Tantivy index directory.
            embedding_model: Name of the HuggingFace embedding model.
            device: Device for embedding computation ("cpu", "cuda:0", etc.).
            skip_vectorizer: If True, do not load the embedding model. Useful when
                embeddings are pre-computed externally (e.g., by a mediator).
                When True, add_document() requires pre-computed embeddings, and
                query() will skip FAISS reranking unless query_embedding is provided.
            keep_warm: If True, move vectorizer to CPU on release (faster reload).
                If False, fully unload vectorizer (frees all memory).
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.device = device
        self.skip_vectorizer = skip_vectorizer
        self.keep_warm = keep_warm

        if skip_vectorizer:
            self.vectorizer = None
            logger.info("SearchIndex initialized with skip_vectorizer=True (no embedding model loaded)")
        else:
            if embedding_model == "arkohut/jina-embeddings-v3":
                raise ValueError("You must specify an embedding model")
            self.vectorizer = HuggingFaceEmbedding(model_name=embedding_model, device=device)

        # Build schema
        builder = SchemaBuilder()
        builder.add_float_field("importance", stored=True)
        builder.add_float_field("sentiment_a", stored=True)
        builder.add_float_field("sentiment_d", stored=True)
        builder.add_float_field("sentiment_v", stored=True)
        builder.add_float_field("weight", stored=True)
        builder.add_integer_field("branch", stored=True)
        builder.add_integer_field("sequence_no", stored=True)
        builder.add_integer_field("status", stored=True)
        builder.add_integer_field("timestamp", stored=True, fast=True, indexed=True)
        builder.add_text_field("content", stored=True, tokenizer_name="en_stem")
        builder.add_text_field("conversation_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("document_type", stored=True, tokenizer_name="raw")
        builder.add_text_field("inference_model", stored=True, tokenizer_name="raw")
        builder.add_text_field("listener_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("metadata", stored=True, tokenizer_name="raw")
        builder.add_text_field("observer", stored=True, tokenizer_name="raw")
        builder.add_text_field("persona_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("role", stored=True, tokenizer_name="raw")
        builder.add_text_field("speaker_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("user_id", stored=True, tokenizer_name="raw")
        builder.add_bytes_field("index_a", stored=True)
        builder.add_bytes_field("index_b", stored=True)

        # Chunk fields for multi-resolution indexing
        builder.add_text_field("parent_doc_id", stored=True, tokenizer_name="raw")
        builder.add_text_field("chunk_level", stored=True, tokenizer_name="raw")
        builder.add_integer_field("chunk_index", stored=True)
        builder.add_integer_field("chunk_start", stored=True)
        builder.add_integer_field("chunk_end", stored=True)
        builder.add_integer_field("chunk_count", stored=True)

        # Build schema and create/open index
        self.schema = builder.build()
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index = Index(self.schema, str(self.index_path))

        # Tokenizer for chunk-level indexing
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def load_vectorizer(self) -> None:
        """Load vectorizer explicitly for batch write operations.

        Call before add_document() calls, then release_vectorizer() after.
        Only works when skip_vectorizer=True (otherwise vectorizer already loaded).
        """
        if self.vectorizer is None and self.skip_vectorizer:
            logger.info(f"Loading vectorizer for write batch: {self.embedding_model} on {self.device}")
            from aim.conversation.embedding import HuggingFaceEmbedding
            self.vectorizer = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                device=self.device,
            )
        elif self.vectorizer is not None and self.keep_warm:
            # Move from CPU to GPU if warming
            logger.debug("Moving vectorizer to GPU")
            self.vectorizer.to(self.device)

    def release_vectorizer(self) -> None:
        """Release explicitly loaded vectorizer to free GPU memory.

        If keep_warm=True: moves to CPU (faster reload)
        If keep_warm=False: fully unloads (frees all memory)

        Safe to call even if vectorizer wasn't explicitly loaded.
        Only operates when skip_vectorizer=True.
        """
        if self.skip_vectorizer and self.vectorizer is not None:
            if self.keep_warm:
                logger.info("Moving vectorizer to CPU (keep_warm=True)")
                self.vectorizer.to("cpu")
            else:
                logger.info("Releasing vectorizer (keep_warm=False)")
                del self.vectorizer
                self.vectorizer = None
                import gc
                gc.collect()

    def _vector_to_bytes(self, vector: np.ndarray) -> bytes:
        """Convert numpy vector to bytes, preserving shape and dtype."""
        return vector.astype(np.float32).tobytes()

    def _bytes_to_vector(self, byte_list: list[int]) -> np.ndarray:
        """Convert Tantivy's list[int] bytes representation back to numpy vector."""
        # Convert list of ints to bytes
        byte_data = bytes(byte_list)
        # Convert bytes back to float32 array
        vector = np.frombuffer(byte_data, dtype=np.float32)
        #logger.info(f"Converted vector shape: {vector.shape}")
        return vector

    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text using tiktoken cl100k_base encoding."""
        return self._tokenizer.encode(text)

    def _detokenize(self, tokens: list[int]) -> str:
        """Convert tokens back to text."""
        return self._tokenizer.decode(tokens)

    def _chunk_by_tokens(self, text: str, chunk_size: int, slide: int) -> list[tuple[str, int, int]]:
        """
        Split text into chunks based on token count.

        Returns: list of (chunk_text, start_token_idx, end_token_idx)
        """
        tokens = self._tokenize(text)
        if len(tokens) <= chunk_size:
            return [(text, 0, len(tokens))]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = self._detokenize(tokens[start:end])
            chunks.append((chunk_text, start, end))
            if end >= len(tokens):
                break
            start += slide
        return chunks

    def _expand_document_to_entries(self, doc: dict) -> list[dict]:
        """
        Expand a document into entries at all chunk levels.

        Returns entries for: full document, 768-token chunks, 256-token chunks.
        """
        entries = []
        parent_id = doc["doc_id"]
        content = doc.get("content", "")
        tokens = self._tokenize(content)

        # 1. Full document entry
        full_entry = doc.copy()
        full_entry["parent_doc_id"] = parent_id
        full_entry["chunk_level"] = CHUNK_LEVEL_FULL
        full_entry["chunk_index"] = 0
        full_entry["chunk_start"] = 0
        full_entry["chunk_end"] = len(tokens)
        full_entry["chunk_count"] = 1
        entries.append(full_entry)

        # 2. chunk_768 entries (768 tokens, 384 slide for 50% overlap)
        chunks_768 = self._chunk_by_tokens(content, CHUNK_SIZE_768, CHUNK_SLIDE_768)
        for idx, (chunk_text, start, end) in enumerate(chunks_768):
            entry = doc.copy()
            entry["doc_id"] = f"{parent_id}_{CHUNK_LEVEL_768}_{idx}"
            entry["parent_doc_id"] = parent_id
            entry["chunk_level"] = CHUNK_LEVEL_768
            entry["chunk_index"] = idx
            entry["chunk_start"] = start
            entry["chunk_end"] = end
            entry["chunk_count"] = len(chunks_768)
            entry["content"] = chunk_text
            entries.append(entry)

        # 3. chunk_256 entries (256 tokens, non-overlapping)
        chunks_256 = self._chunk_by_tokens(content, CHUNK_SIZE_256, CHUNK_SLIDE_256)
        for idx, (chunk_text, start, end) in enumerate(chunks_256):
            entry = doc.copy()
            entry["doc_id"] = f"{parent_id}_{CHUNK_LEVEL_256}_{idx}"
            entry["parent_doc_id"] = parent_id
            entry["chunk_level"] = CHUNK_LEVEL_256
            entry["chunk_index"] = idx
            entry["chunk_start"] = start
            entry["chunk_end"] = end
            entry["chunk_count"] = len(chunks_256)
            entry["content"] = chunk_text
            entries.append(entry)

        return entries

    def to_doc(self, doc: dict, index_a : np.ndarray) -> TantivyDocument:
        """Convert a dictionary to a tantivy document"""
        index_a_bytes = self._vector_to_bytes(index_a)
        #logger.info(f"Index has shape {index_a.shape}")
        return TantivyDocument(
            doc_id=doc["doc_id"],
            content=doc["content"],
            conversation_id=doc["conversation_id"],
            user_id=doc["user_id"],
            persona_id=doc["persona_id"],
            speaker=doc.get("speaker_id", ""),
            listener=doc.get("listener_id", ""),
            role=doc["role"],
            document_type=doc.get("document_type", DOC_CONVERSATION),
            timestamp=doc["timestamp"],
            sequence_no=doc["sequence_no"],
            branch=doc["branch"],
            sentiment_v=doc.get("sentiment_v", 0.0),
            sentiment_a=doc.get("sentiment_a", 0.0),
            sentiment_d=doc.get("sentiment_d", 0.0),
            importance=doc.get("importance", 1.0),
            weight=doc.get("weight", 1.0),
            observer=doc.get("observer", ""),
            inference_model=doc.get("inference_model", ""),
            metadata=doc.get("metadata", ""),
            status=doc.get("status", 0),
            index_a=index_a_bytes,
            # Chunk fields for multi-resolution indexing
            parent_doc_id=doc.get("parent_doc_id", doc["doc_id"]),
            chunk_level=doc.get("chunk_level", CHUNK_LEVEL_FULL),
            chunk_index=doc.get("chunk_index", 0),
            chunk_start=doc.get("chunk_start", 0),
            chunk_end=doc.get("chunk_end", 0),
            chunk_count=doc.get("chunk_count", 1),
        )

    def add_document(self, doc: dict, embedding: Optional[np.ndarray] = None) -> None:
        """Add a single document to the index at all chunk levels.

        Args:
            doc: Document dictionary with required fields.
            embedding: DEPRECATED - ignored, all embeddings computed internally.

        Raises:
            RuntimeError: If vectorizer is not loaded when skip_vectorizer=True.
        """
        if embedding is not None:
            logger.debug("Ignoring pre-computed embedding - computing all embeddings internally")

        entries = self._expand_document_to_entries(doc)

        if self.vectorizer is None:
            raise RuntimeError(
                "Cannot add document without vectorizer. "
                "Call load_vectorizer() before batch writes when skip_vectorizer=True."
            )

        writer = self.index.writer()
        for entry in entries:
            # Compute embedding for this entry (full or chunk)
            index_a = self.vectorizer(entry["content"])

            tantivy_doc = self.to_doc(entry, index_a)
            writer.add_document(tantivy_doc)

        writer.commit()
        self.index.reload()

    def add_documents(self, documents: list[dict], use_tqdm: bool = False, batch_size: int = 64) -> None:
        """Add multiple documents to the index efficiently at all chunk levels.

        Raises:
            RuntimeError: If skip_vectorizer=True. Batch document addition requires
                the vectorizer to compute embeddings for all entries. Use add_document()
                with pre-computed embeddings when skip_vectorizer=True.
        """
        if self.vectorizer is None:
            raise RuntimeError(
                "add_documents() requires vectorizer. Cannot batch-add documents when "
                "skip_vectorizer=True. Use add_document() with pre-computed embeddings instead."
            )

        # First expand all documents to entries (full + chunks)
        all_entries = []
        for doc in documents:
            all_entries.extend(self._expand_document_to_entries(doc))

        writer = self.index.writer()
        num_entries = len(all_entries)

        if use_tqdm:
            from tqdm import tqdm
            num_batches = (num_entries + batch_size - 1) // batch_size

            for i in tqdm(range(num_batches), total=num_batches, desc="Adding Entries in Batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_entries)
                batch_entries = all_entries[start_idx:end_idx]

                if not batch_entries:
                    continue

                # Vectorize the content of the current batch
                contents = [entry["content"] for entry in batch_entries]
                indices = self.vectorizer.transform(contents)

                # Add entries from the batch to the writer
                for j, entry in enumerate(batch_entries):
                    index_a = indices[j]
                    tantivy_doc = self.to_doc(entry, index_a=index_a)
                    writer.add_document(tantivy_doc)
        else:
            # Vectorize all entries first
            indices = self.vectorizer.transform([entry["content"] for entry in all_entries])
            for i, entry in enumerate(all_entries):
                index_a = indices[i]
                tantivy_doc = self.to_doc(entry, index_a=index_a)
                writer.add_document(tantivy_doc)

        writer.commit()
        self.index.reload()
        logger.info(f"Added {len(documents)} documents expanded to {num_entries} entries")

    def search(self, query_texts: list[str] = [],
               query_document_type: Optional[str | list[str]] = None, filter_document_type: Optional[str | list[str]] = None,
               query_persona_id: Optional[str] = None, query_conversation_id: Optional[str] = None,
               filter_doc_ids: Optional[list[str]] = None, query_limit: int = 20,
               descending: Optional[bool] = None, keyword_boost: float = 2.0,
               chunk_level: Optional[str] = None) -> pd.DataFrame:
        """Search the index and return scored results.

        Args:
            query_texts: List of query strings to search for.
            query_document_type: Filter to specific document type(s).
            filter_document_type: Exclude specific document type(s).
            query_persona_id: Filter to specific persona.
            query_conversation_id: Filter to specific conversation.
            filter_doc_ids: Exclude specific document IDs.
            query_limit: Maximum number of results to return.
            descending: Sort by timestamp descending (True) or ascending (False). None for relevance.
            keyword_boost: Boost factor for capitalized terms (proper nouns, acronyms).
                          Set to 1.0 to disable boosting. Default 2.0.
            chunk_level: Filter to specific chunk level (chunk_256, chunk_768, full).
                        None returns all levels.
        """
        searcher = self.index.searcher()

        # We need to build our subqueries
        subqueries = []

        # Build base text query if `query_texts` is provided
        if len(query_texts) > 0:
            text_subqueries = []
            n_queries = len(query_texts)

            # Recency window: boost decays over last 12 messages, then levels out
            # Most recent gets 4.0x, 12 messages back gets 1.0x, older than 12 stays at 1.0x
            recency_window = 12
            max_boost = 4.0  # Recent messages get 4x weight over older ones

            for idx, query_text in enumerate(query_texts):
                # Calculate position-based boost within recency window
                if n_queries <= recency_window:
                    # All messages fit in window: scale 1.0 (first) to max_boost (last)
                    if n_queries > 1:
                        position_boost = 1.0 + (max_boost - 1.0) * (idx / (n_queries - 1))
                    else:
                        position_boost = max_boost  # Single message gets full boost
                else:
                    # More messages than window: only last 12 get boosted
                    window_start = n_queries - recency_window
                    if idx >= window_start:
                        # In the active window: scale 1.0 to max_boost
                        position_in_window = idx - window_start  # 0 to 11
                        position_boost = 1.0 + (max_boost - 1.0) * (position_in_window / (recency_window - 1))
                    else:
                        # Outside window: leveled out at 1.0
                        position_boost = 1.0

                # Clean first (expand contractions, remove special chars), then boost
                cleaned_text = clean_query_text(query_text)
                boosted_query = boost_query_terms(cleaned_text, base_boost=position_boost, keyword_boost=keyword_boost, max_terms=64)

                # logger.debug(f"Query {idx+1}/{n_queries} (boost={position_boost:.1f}): {boosted_query[:60]}...")
                text_query = self.index.parse_query(query=boosted_query, default_field_names=["content"])
                text_subqueries.append((Occur.Should, text_query))
            subqueries.append((Occur.Must, Query.boolean_query(text_subqueries)))
        else:
            subqueries.append((Occur.Must, Query.all_query()))

        # Add conditions based on provided arguments
        if query_conversation_id:
            conversation_query = self.index.parse_query(query=query_conversation_id, default_field_names=["conversation_id"])
            subqueries.append((Occur.Must, conversation_query))

        if query_document_type:
            doc_type_queries = [(Occur.Should, self.index.parse_query(query=dt, default_field_names=["document_type"]))
                                for dt in (query_document_type if isinstance(query_document_type, list) else [query_document_type])]
            subqueries.append((Occur.Must, Query.boolean_query(subqueries=doc_type_queries)))

        if filter_document_type:
            filter_type_queries = [(Occur.Should, self.index.parse_query(query=ft, default_field_names=["document_type"]))
                                for ft in (filter_document_type if isinstance(filter_document_type, list) else [filter_document_type])]
            subqueries.append((Occur.MustNot, Query.boolean_query(subqueries=filter_type_queries)))

        if query_persona_id:
            persona_query = self.index.parse_query(query=query_persona_id, default_field_names=["persona_id"])
            subqueries.append((Occur.Must, persona_query))

        if filter_doc_ids:
            doc_id_queries = [(Occur.Should, self.index.parse_query(query=doc_id, default_field_names=["doc_id"]))
                            for doc_id in filter_doc_ids]
            subqueries.append((Occur.MustNot, Query.boolean_query(subqueries=doc_id_queries)))

        # Filter by chunk level if specified
        if chunk_level:
            chunk_query = self.index.parse_query(query=chunk_level, default_field_names=["chunk_level"])
            subqueries.append((Occur.Must, chunk_query))

        # Combine all subqueries into a single query
        query = Query.boolean_query(subqueries=subqueries)

        if descending is not None:
            search_args = {
                'order_by_field': 'timestamp',
                'order': Order.Desc if descending else Order.Asc
            }
        else:
            search_args = {}

        # Execute the search and process results
        search_results = searcher.search(query, query_limit, **search_args)
                
        #logger.info(f"Found {search_results} for {query}")

        results = {}
        doc_hits = defaultdict(int)
        doc_ref = {}

        # First pass: get doc_id from each hit and use it as the unique key
        for score, doc_addr in search_results.hits:
            doc = searcher.doc(doc_addr)
            doc_id = doc.get_first("doc_id")

            if doc_id is None:
                continue

            # Keep the highest score for each unique doc_id
            if doc_id not in doc_ref or score > doc_ref[doc_id][0]:
                doc_ref[doc_id] = (score, doc_addr)
            doc_hits[doc_id] += 1

        # Second pass: build results from unique documents
        for doc_id, (score, doc_addr) in doc_ref.items():
            doc = searcher.doc(doc_addr)
            byte_list : list[int] = doc.get_first("index_a")
            index_a = self._bytes_to_vector(byte_list)
            result = {
                k: doc.get_first(k) for k in QUERY_COLUMNS
            }
            result["index_a"] = index_a
            result["hits"] = doc_hits[doc_id]

            results[doc_id] = (score, result)

        if len(results.keys()) == 0:
            return pd.DataFrame(columns=QUERY_COLUMNS + ['distance', 'hits'])

        results = pd.DataFrame([{**d, 'distance': ts} for ts, d in results.values()])
        return results

    def rebuild(self, documents: list[dict], use_tqdm: bool = True) -> None:
        """Clear and rebuild the entire index.

        Raises:
            RuntimeError: If skip_vectorizer=True. Rebuild requires the vectorizer
                to compute embeddings for all documents.
        """
        # Clear existing index
        if self.index_path.exists():
            import shutil

            shutil.rmtree(self.index_path)

        # Reinitialize with stored config (preserves skip_vectorizer setting)
        self.__init__(self.index_path, embedding_model=self.embedding_model, device=self.device, skip_vectorizer=self.skip_vectorizer)

        # Add all documents (will raise if skip_vectorizer=True)
        self.add_documents(documents, use_tqdm=use_tqdm)
        logger.info(f"Rebuilt index with {len(documents)} documents")

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Retrieve a specific document by ID"""
        searcher = self.index.searcher()
        query = Query.term_query(schema=self.schema, field_name="doc_id", field_value=doc_id)
        results = searcher.search(query, limit=1)
        
        if not results.hits:
            logger.warning(f"No results found for {doc_id} {query} {results}")
            return None
            
        _, doc_addr = results.hits[0]
        doc = searcher.doc(doc_addr)
        
        return {
            k : doc.get_first(k) for k in QUERY_COLUMNS
        }

    def get_all_document_ids(self) -> set[str]:
        """Get all document IDs currently in the index"""
        searcher = self.index.searcher()
        query = Query.all_query()
        results = searcher.search(query, limit=1000000)  # Large limit to get all docs
        
        doc_ids = set()
        for _, doc_addr in results.hits:
            doc = searcher.doc(doc_addr)
            doc_id = doc.get_first("doc_id")
            if doc_id:
                doc_ids.add(doc_id)
        
        return doc_ids

    def get_all_documents_with_content(self) -> dict[str, dict]:
        """Get all documents with their content for comparison"""
        searcher = self.index.searcher()
        query = Query.all_query()
        results = searcher.search(query, limit=1000000)  # Large limit to get all docs

        documents = {}
        for _, doc_addr in results.hits:
            doc = searcher.doc(doc_addr)
            doc_id = doc.get_first("doc_id")
            if doc_id:
                documents[doc_id] = {
                    k: doc.get_first(k) for k in QUERY_COLUMNS
                }

        return documents

    def get_all_parent_documents(self) -> dict[str, dict]:
        """Get all parent documents (full chunk level only) for comparison."""
        searcher = self.index.searcher()
        # Only get full documents (parent level)
        chunk_query = self.index.parse_query(query=CHUNK_LEVEL_FULL, default_field_names=["chunk_level"])
        results = searcher.search(chunk_query, limit=1000000)

        documents = {}
        for _, doc_addr in results.hits:
            doc = searcher.doc(doc_addr)
            parent_id = doc.get_first("parent_doc_id")
            if parent_id:
                documents[parent_id] = {
                    k: doc.get_first(k) for k in QUERY_COLUMNS
                }

        return documents

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the index"""
        writer = self.index.writer()
        deleted_count = writer.delete_documents("doc_id", doc_id)
        writer.commit()
        self.index.reload()
        logger.debug(f"Deleted {deleted_count} documents with doc_id: {doc_id}")

    def delete_by_parent_doc_id(self, parent_doc_id: str) -> int:
        """Delete all chunk entries for a parent document."""
        writer = self.index.writer()
        deleted_count = writer.delete_documents("parent_doc_id", parent_doc_id)
        writer.commit()
        self.index.reload()
        logger.debug(f"Deleted {deleted_count} entries with parent_doc_id: {parent_doc_id}")
        return deleted_count

    def update_document(self, doc: dict) -> None:
        """Update a document in the index (delete old chunks, add new chunks)."""
        # Delete all chunks for this parent document
        self.delete_by_parent_doc_id(doc["doc_id"])

        # Then add the new version (which expands to all chunk levels)
        self.add_document(doc)

    def incremental_update(self, new_documents: list[dict], use_tqdm: bool = False, batch_size: int = 64) -> tuple[int, int, int]:
        """
        Perform incremental update of the index at parent document level.
        Compares parent documents and updates all chunk levels when content changes.
        Returns: (added_count, updated_count, deleted_count)
        """
        # Get current index state at parent level only
        existing_docs = self.get_all_parent_documents()
        existing_parent_ids = set(existing_docs.keys())
        new_doc_ids = {doc["doc_id"] for doc in new_documents}

        # Determine what needs to be done
        docs_to_add = []
        docs_to_update = []
        parent_ids_to_delete = existing_parent_ids - new_doc_ids

        for doc in new_documents:
            doc_id = doc["doc_id"]
            if doc_id not in existing_docs:
                # New document
                docs_to_add.append(doc)
            else:
                # Check if content has changed
                existing_doc = existing_docs[doc_id]
                if (doc.get("content", "") != existing_doc.get("content", "") or
                    doc.get("timestamp", 0) != existing_doc.get("timestamp", 0)):
                    # Document has changed - need to re-chunk
                    docs_to_update.append(doc)

        # Perform updates
        added_count = len(docs_to_add)
        updated_count = len(docs_to_update)
        deleted_count = len(parent_ids_to_delete)

        if use_tqdm:
            from tqdm import tqdm

            # Delete documents (all chunks for each parent)
            if parent_ids_to_delete:
                for parent_id in tqdm(parent_ids_to_delete, desc="Deleting outdated documents"):
                    self.delete_by_parent_doc_id(parent_id)

            # Add new documents (expands to all chunk levels)
            if docs_to_add:
                self.add_documents(docs_to_add, use_tqdm=True, batch_size=batch_size)

            # Update changed documents (re-chunks at all levels)
            if docs_to_update:
                for doc in tqdm(docs_to_update, desc="Updating changed documents"):
                    self.update_document(doc)
        else:
            # Delete documents (all chunks for each parent)
            for parent_id in parent_ids_to_delete:
                self.delete_by_parent_doc_id(parent_id)
            
            # Add new documents
            if docs_to_add:
                self.add_documents(docs_to_add, use_tqdm=False, batch_size=batch_size)
            
            # Update changed documents
            for doc in docs_to_update:
                self.update_document(doc)
        
        logger.info(f"Incremental update complete: {added_count} added, {updated_count} updated, {deleted_count} deleted")
        return added_count, updated_count, deleted_count
