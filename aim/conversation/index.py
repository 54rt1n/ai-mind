# aim/conversation/index.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0 

from collections import defaultdict
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import pandas as pd
import re
from tantivy import Index, Document as TantivyDocument, SchemaBuilder, Query, Occur, Order
from ..constants import DOC_CONVERSATION
from .embedding import HuggingFaceEmbedding
from .message import VISIBLE_COLUMNS, QUERY_COLUMNS

logger = logging.getLogger(__name__)

class SearchIndex:
    """Tantivy-based search index for conversations"""

    def __init__(self, index_path: Path, embedding_model: str = "arkohut/jina-embeddings-v3", device: str = "cpu"):
        self.index_path = index_path
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

        # Build schema and create/open index
        self.schema = builder.build()
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.index = Index(self.schema, str(self.index_path))

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
        )

    def add_document(self, doc: dict) -> None:
        """Add a single document to the index"""
        index_a = self.vectorizer(doc["content"])
        tantivy_doc = self.to_doc(doc, index_a)
        writer = self.index.writer()
        writer.add_document(tantivy_doc)
        writer.commit()
        self.index.reload()

    def add_documents(self, documents: list[dict], use_tqdm: bool = False, batch_size: int = 64) -> None:
        """Add multiple documents to the index efficiently"""
        writer = self.index.writer()

        if use_tqdm:
            # Assume tqdm is installed if use_tqdm is True
            from tqdm import tqdm 

            num_documents = len(documents)
            num_batches = (num_documents + batch_size - 1) // batch_size
            
            # Iterate over batch indices with tqdm
            for i in tqdm(range(num_batches), total=num_batches, desc="Adding Documents in Batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_documents)
                batch_docs = documents[start_idx:end_idx]

                if not batch_docs:
                    continue 

                # Vectorize the content of the current batch
                contents = [doc["content"] for doc in batch_docs]
                indices = self.vectorizer.transform(contents)

                # Add documents from the batch to the writer
                for j, doc in enumerate(batch_docs):
                    index_a = indices[j]
                    tantivy_doc = self.to_doc(doc, index_a=index_a)
                    writer.add_document(tantivy_doc)
        else:
            # vectorize all documents first
            indices = self.vectorizer.transform([doc["content"] for doc in documents])
            for i, doc in enumerate(documents):
                index_a = indices[i]
                tantivy_doc = self.to_doc(doc, index_a=index_a)
                writer.add_document(tantivy_doc)

        writer.commit()
        self.index.reload()

    def search(self, query_texts: list[str] = [],
               query_document_type: Optional[str | list[str]] = None, filter_document_type: Optional[str | list[str]] = None,
               query_persona_id: Optional[str] = None, query_conversation_id: Optional[str] = None,
               filter_doc_ids: Optional[list[str]] = None, query_limit: int = 20,
               descending: Optional[bool] = None) -> pd.DataFrame:
        """Search the index and return scored results"""

        searcher = self.index.searcher()

        # We need to build our subqueries
        subqueries = []

        # Build base text query if `query_texts` is provided
        if len(query_texts) > 0:
            text_subqueries = []
            for query_text in query_texts:
                query_text = re.sub(r"\s+", ' ', re.sub(r'[^\w\s]|\n', ' ', query_text)).lower()
                #logger.info(f"Parsing query: {len(query_text)}")
                text_query = self.index.parse_query(query=query_text, default_field_names=["content"])
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

    def rebuild(self, documents: list[dict]) -> None:
        """Clear and rebuild the entire index"""
        # Clear existing index
        if self.index_path.exists():
            import shutil

            shutil.rmtree(self.index_path)

        # Reinitialize
        self.__init__(self.index_path)

        # Add all documents
        self.add_documents(documents)
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

    def delete_document(self, doc_id: str) -> None:
        """Delete a document from the index"""
        writer = self.index.writer()
        deleted_count = writer.delete_documents("doc_id", doc_id)
        writer.commit()
        self.index.reload()
        logger.debug(f"Deleted {deleted_count} documents with doc_id: {doc_id}")

    def update_document(self, doc: dict) -> None:
        """Update a document in the index (delete old, add new)"""
        # First delete the old document
        self.delete_document(doc["doc_id"])
        
        # Then add the new version
        self.add_document(doc)

    def incremental_update(self, new_documents: list[dict], use_tqdm: bool = False, batch_size: int = 64) -> tuple[int, int, int]:
        """
        Perform incremental update of the index
        Returns: (added_count, updated_count, deleted_count)
        """
        # Get current index state
        existing_docs = self.get_all_documents_with_content()
        existing_doc_ids = set(existing_docs.keys())
        new_doc_ids = {doc["doc_id"] for doc in new_documents}
        
        # Determine what needs to be done
        docs_to_add = []
        docs_to_update = []
        docs_to_delete = existing_doc_ids - new_doc_ids
        
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
                    # Document has changed
                    docs_to_update.append(doc)
        
        # Perform updates
        added_count = len(docs_to_add)
        updated_count = len(docs_to_update)
        deleted_count = len(docs_to_delete)
        
        if use_tqdm:
            from tqdm import tqdm
            
            # Delete documents
            if docs_to_delete:
                for doc_id in tqdm(docs_to_delete, desc="Deleting outdated documents"):
                    self.delete_document(doc_id)
            
            # Add new documents
            if docs_to_add:
                self.add_documents(docs_to_add, use_tqdm=True, batch_size=batch_size)
            
            # Update changed documents
            if docs_to_update:
                for doc in tqdm(docs_to_update, desc="Updating changed documents"):
                    self.update_document(doc)
        else:
            # Delete documents
            for doc_id in docs_to_delete:
                self.delete_document(doc_id)
            
            # Add new documents
            if docs_to_add:
                self.add_documents(docs_to_add, use_tqdm=False, batch_size=batch_size)
            
            # Update changed documents
            for doc in docs_to_update:
                self.update_document(doc)
        
        logger.info(f"Incremental update complete: {added_count} added, {updated_count} updated, {deleted_count} deleted")
        return added_count, updated_count, deleted_count
