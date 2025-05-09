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
from ..constants import DOC_ANALYSIS, DOC_CONVERSATION, DOC_JOURNAL, DOC_NER, DOC_STEP, LISTENER_ALL, DOC_MOTD
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


class ConversationModel:
    collection_name : str = 'memory'

    def __init__(self, memory_path: str, embedding_model: str, user_timezone: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)

        self.index = SearchIndex(Path('.', memory_path, 'indices'), embedding_model=embedding_model)
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
        return cls(memory_path=config.memory_path, embedding_model=config.embedding_model, user_timezone=config.user_timezone)

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
              max_length: Optional[int] = None, turn_decay: float = 0.7, temporal_decay: float = 0.99, length_boost_factor: float = 0.0,
              filter_metadocs: bool = True, chunk_size: int = -1, **kwargs) -> pd.DataFrame:
        """
        Queries the conversation collection and returns a DataFrame containing the top `top_n` most relevant conversation entries based on the given query texts, filters, and decay factors.
        
        The query is performed using the collection's search functionality, with optional filters applied to exclude certain document types, document IDs, and text content. The relevance score for each entry is calculated as a combination of the text similarity to the query texts, the temporal decay based on the entry's timestamp, and the entry's weight.
        
        The returned DataFrame includes the following columns:
            - `query_text`: The query text.
            - `filter_doc_ids`: The document IDs to filter.
            - `top_n`: The number of results to return.
            - `query_document_type`: The document type to query.
            - `query_conversation_id`: The conversation ID to query.
            - `max_length`: The maximum length of the results.
            - `turn_decay`: The decay factor for the turn.
            - `temporal_decay`: The decay factor for the temporal.
            - `length_boost_factor`: The boost factor for the length.
            - `filter_metadocs`: Whether to filter the metadocs.
            - `chunk_size`: The size of the chunk to query.
        """

        # `query_limit` is how many results to fetch from the underlying search index.
        # `top_n` is how many results to return to the caller after processing.
        # Setting query_limit slightly higher than top_n can be beneficial if post-filtering or re-ranking occurs.
        # Using top_n * 2 here as a heuristic.
        query_limit = top_n * 2 if top_n is not None else None # Allow top_n to be None for unlimited internal fetch

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
        filter_document_type = [DOC_NER, DOC_STEP] if filter_metadocs and not query_document_type else None

        if not query_texts: # Handles empty list and ensures it's not None for len()
            logger.warning("No query texts provided, returning empty DataFrame")
            return pd.DataFrame(columns=VISIBLE_COLUMNS + ['date', 'speaker', 'score'])

        results = self.index.search(query_texts, query_document_type=query_document_type, filter_doc_ids=filter_doc_ids, filter_document_type=filter_document_type, query_conversation_id=query_conversation_id,
                                    query_limit=query_limit)

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
        results['hits_score'] = np.log2(results['hits'] + 1)

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
                results['rerank'] = [1 / d if d > 0 else 0 for d, _ in distance_index]
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

        # Calculate decay factor
        results['temporal_decay'] = np.exp(-temporal_decay * (current_time - results['timestamp']) / decay_length)
        
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
        # Sort by score first, then apply top_n
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
            
        return results[VISIBLE_COLUMNS + ['date', 'speaker', 'score']]

    def get_motd(self, top_n: int = 1) -> pd.DataFrame:
        results = self.index.search(query_document_type=DOC_MOTD, query_limit=top_n, descending=True)

        results = self._fix_dataframe(results)
        return results[VISIBLE_COLUMNS + ['date', 'speaker']].head(top_n)
        
    def get_conscious(self, persona_id: str, top_n: int) -> pd.DataFrame:
        results : pd.DataFrame = self.index.search(query_document_type=DOC_JOURNAL, query_persona_id=persona_id, query_limit=int(top_n * 1.5))

        # our score will be stochastic, to bring in a variety of entries
        results['score'] = np.random.rand(len(results)) * results['weight']
        results = self._fix_dataframe(results)

        return results.sort_values(by='score', ascending=False).reset_index(drop=True)[VISIBLE_COLUMNS + ['date', 'speaker']].head(top_n)

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

        return results[QUERY_COLUMNS]

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

        return results[VISIBLE_COLUMNS + ['date', 'speaker']]

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
