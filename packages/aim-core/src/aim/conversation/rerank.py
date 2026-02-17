# aim/conversation/rerank.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0

from typing import List, Tuple, Set, Optional, Callable, Dict
import numpy as np
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

# Type alias: (source_tag, row_data, optional turn_index)
TaggedResult = Tuple[str, pd.Series, Optional[int]]


class MemoryReranker:
    """
    Reranks and merges memory results from conversation and other buckets
    while enforcing token budget and diversity constraints.

    Conversations get priority with a larger share of the budget, then
    remaining budget is filled with other document types.
    """

    def __init__(self,
                 token_counter: Callable[[str], int],
                 lambda_param: float = 0.7,
                 conversation_budget_ratio: float = 0.6):
        """
        Args:
            token_counter: Function to count tokens in a string
            lambda_param: MMR parameter (0=max diversity, 1=pure relevance)
            conversation_budget_ratio: Portion of budget reserved for conversations (default 0.6)
        """
        self.token_counter = token_counter
        self.lambda_param = lambda_param
        self.conversation_budget_ratio = conversation_budget_ratio

    def rerank(self,
               conversation_results: List[TaggedResult],
               other_results: List[TaggedResult],
               token_budget: int,
               seen_parent_ids: Optional[Set[str]] = None) -> List[TaggedResult]:
        """
        Merge and rerank results from both buckets within token budget.

        Conversations get first priority up to conversation_budget_ratio of budget,
        then remaining budget is filled with other docs.

        Args:
            conversation_results: List of (source_tag, row) for DOC_CONVERSATION
            other_results: List of (source_tag, row) for other doc types
            token_budget: Maximum tokens for all returned results
            seen_parent_ids: Parent doc IDs already added (priority items)

        Returns:
            Merged, reranked list of (source_tag, row) tuples that fit in budget
        """
        if seen_parent_ids is None:
            seen_parent_ids = set()
        rerank_start = time.perf_counter()

        # 1. Deduplicate each bucket by parent_doc_id (keep highest score)
        conv_deduped = self._deduplicate_by_parent(conversation_results)
        other_deduped = self._deduplicate_by_parent(other_results)
        dedupe_ms = int((time.perf_counter() - rerank_start) * 1000)

        # 2. Filter out already-seen parent_doc_ids
        filter_start = time.perf_counter()
        conv_filtered = self._filter_seen(conv_deduped, seen_parent_ids)
        other_filtered = self._filter_seen(other_deduped, seen_parent_ids)
        filter_ms = int((time.perf_counter() - filter_start) * 1000)

        # 3. Normalize scores within each bucket
        normalize_start = time.perf_counter()
        conv_filtered = self._normalize_scores(conv_filtered)
        other_filtered = self._normalize_scores(other_filtered)
        normalize_ms = int((time.perf_counter() - normalize_start) * 1000)

        # 4. Prepare candidate data for each bucket
        prepare_start = time.perf_counter()
        conv_candidates = self._prepare_candidates(conv_filtered)
        other_candidates = self._prepare_candidates(other_filtered)
        prepare_ms = int((time.perf_counter() - prepare_start) * 1000)

        selected: List[TaggedResult] = []
        selected_embeddings: List[np.ndarray] = []
        selected_parent_ids: Set[str] = set()
        tokens_used = 0

        # 5. First pass: Fill from conversations up to their budget share
        conv_fill_start = time.perf_counter()
        conv_budget = int(token_budget * self.conversation_budget_ratio)
        conv_selected, conv_tokens, conv_embeddings, conv_parents = self._fill_from_bucket(
            conv_candidates, conv_budget, selected_embeddings, selected_parent_ids
        )
        conv_fill_ms = int((time.perf_counter() - conv_fill_start) * 1000)
        selected.extend(conv_selected)
        selected_embeddings.extend(conv_embeddings)
        selected_parent_ids.update(conv_parents)
        tokens_used += conv_tokens

        # 6. Second pass: Fill remaining budget from other docs
        other_fill_start = time.perf_counter()
        remaining_budget = token_budget - tokens_used
        other_selected, other_tokens, other_embeddings, other_parents = self._fill_from_bucket(
            other_candidates, remaining_budget, selected_embeddings, selected_parent_ids
        )
        other_fill_ms = int((time.perf_counter() - other_fill_start) * 1000)
        selected.extend(other_selected)
        selected_embeddings.extend(other_embeddings)
        selected_parent_ids.update(other_parents)
        tokens_used += other_tokens

        # 7. If conversations didn't use their full budget, backfill with more other docs
        backfill_ms = 0
        backfill_selected = 0
        remaining_budget = token_budget - tokens_used
        if remaining_budget > 50 and other_candidates:
            # Filter out already-selected from other_candidates
            remaining_other = [c for c in other_candidates if c['parent_id'] not in selected_parent_ids]
            backfill_start = time.perf_counter()
            more_selected, more_tokens, more_embeddings, more_parents = self._fill_from_bucket(
                remaining_other, remaining_budget, selected_embeddings, selected_parent_ids
            )
            backfill_ms = int((time.perf_counter() - backfill_start) * 1000)
            backfill_selected = len(more_selected)
            selected.extend(more_selected)
            tokens_used += more_tokens

        total_ms = int((time.perf_counter() - rerank_start) * 1000)
        logger.info(
            "MemoryReranker timing: total=%dms dedupe=%dms filter=%dms normalize=%dms prepare=%dms conv_fill=%dms other_fill=%dms backfill=%dms candidates=%d/%d selected=%d (%d conv + %d other + %d backfill) tokens=%d/%d",
            total_ms,
            dedupe_ms,
            filter_ms,
            normalize_ms,
            prepare_ms,
            conv_fill_ms,
            other_fill_ms,
            backfill_ms,
            len(conv_candidates),
            len(other_candidates),
            len(selected),
            len(conv_selected),
            len(other_selected),
            backfill_selected,
            tokens_used,
            token_budget,
        )
        return selected

    def _prepare_candidates(self, results: List[TaggedResult]) -> List[Dict]:
        """Convert TaggedResults to candidate dicts with precomputed data."""
        candidates = []
        for item in results:
            # Handle both 2-tuple (legacy) and 3-tuple (with turn_index)
            if len(item) == 3:
                source_tag, row, turn_index = item
            else:
                source_tag, row = item
                turn_index = None

            embedding = row.get('index_a')
            # Allow candidates without embeddings (skip MMR diversity for them)
            token_cost = self._estimate_tokens(row)
            parent_id = row.get('parent_doc_id', row.get('doc_id', ''))
            candidates.append({
                'source_tag': source_tag,
                'row': row,
                'embedding': embedding if isinstance(embedding, np.ndarray) and embedding.size > 0 else None,
                'token_cost': token_cost,
                'parent_id': parent_id,
                'norm_score': row.get('norm_score', 0.0),
                'turn_index': turn_index,
            })
        return candidates

    def _fill_from_bucket(self,
                          candidates: List[Dict],
                          budget: int,
                          existing_embeddings: List[np.ndarray],
                          existing_parent_ids: Set[str]
                          ) -> Tuple[List[TaggedResult], int, List[np.ndarray], Set[str]]:
        """
        Fill from a bucket using MMR until budget exhausted.

        Returns: (selected_results, tokens_used, new_embeddings, new_parent_ids)
        """
        selected: List[TaggedResult] = []
        selected_embeddings: List[np.ndarray] = list(existing_embeddings)
        selected_parent_ids: Set[str] = set(existing_parent_ids)
        new_embeddings: List[np.ndarray] = []
        new_parent_ids: Set[str] = set()
        tokens_used = 0
        fill_start = time.perf_counter()
        loop_count = 0
        candidate_evals = 0
        similarity_calls = 0

        remaining = list(range(len(candidates)))

        while remaining:
            loop_count += 1
            best_idx = None
            best_mmr = float('-inf')
            best_remaining_idx = None

            for i, cand_idx in enumerate(remaining):
                cand = candidates[cand_idx]

                # Skip if parent already selected
                if cand['parent_id'] in selected_parent_ids:
                    continue

                # Skip if doesn't fit in remaining budget
                if tokens_used + cand['token_cost'] > budget:
                    continue

                candidate_evals += 1
                # Calculate MMR score
                relevance = cand['norm_score']

                # Apply turn recency boost if available
                if cand.get('turn_index') is not None:
                    turn_index = cand['turn_index']
                    if turn_index == 0:
                        recency_multiplier = 2.0   # Current turn
                    elif turn_index == 1:
                        recency_multiplier = 1.5   # One turn ago
                    elif turn_index == 2:
                        recency_multiplier = 1.2   # Two turns ago
                    else:
                        recency_multiplier = 1.0   # Three+ turns ago (no boost)

                    relevance = relevance * recency_multiplier
                    logger.debug(f"Applied turn recency boost: turn_index={turn_index}, multiplier={recency_multiplier}, boosted_relevance={relevance:.4f}")

                # Apply source tag boost (explicit query gets highest priority)
                source_tag = cand.get('source_tag', '')
                if source_tag:
                    if source_tag == 'memory_query':
                        source_multiplier = 3.0  # ExplicitQuery - agent's explicit intent
                    elif source_tag in ('memory_user', 'memory_asst'):
                        source_multiplier = 1.5  # Recent conversation turns
                    elif source_tag in ('memory_thought', 'memory_ws', 'memory_loc'):
                        source_multiplier = 1.2  # Context-based queries
                    else:
                        source_multiplier = 1.0  # Other sources

                    if source_multiplier > 1.0:
                        relevance = relevance * source_multiplier
                        logger.debug(f"Applied source boost: source_tag={source_tag}, multiplier={source_multiplier}, boosted_relevance={relevance:.4f}")

                if selected_embeddings and cand['embedding'] is not None:
                    similarity_calls += 1
                    max_sim = self._max_similarity(cand['embedding'], selected_embeddings)
                else:
                    max_sim = 0.0

                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = cand_idx
                    best_remaining_idx = i

            if best_idx is None:
                break

            cand = candidates[best_idx]
            # Preserve turn_index if present
            if cand['turn_index'] is not None:
                selected.append((cand['source_tag'], cand['row'], cand['turn_index']))
            else:
                selected.append((cand['source_tag'], cand['row']))
            if cand['embedding'] is not None:
                selected_embeddings.append(cand['embedding'])
                new_embeddings.append(cand['embedding'])
            selected_parent_ids.add(cand['parent_id'])
            new_parent_ids.add(cand['parent_id'])
            tokens_used += cand['token_cost']

            remaining.pop(best_remaining_idx)

        elapsed_ms = int((time.perf_counter() - fill_start) * 1000)
        logger.info(
            "MemoryReranker fill: input=%d selected=%d loops=%d evals=%d sim_calls=%d tokens=%d/%d took=%dms",
            len(candidates),
            len(selected),
            loop_count,
            candidate_evals,
            similarity_calls,
            tokens_used,
            budget,
            elapsed_ms,
        )
        return selected, tokens_used, new_embeddings, new_parent_ids

    def _safe_score(self, value) -> float:
        """Convert score to safe float, handling NaN/None."""
        if value is None:
            return 0.0
        try:
            f = float(value)
            return 0.0 if np.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0

    def _deduplicate_by_parent(self, results: List[TaggedResult]) -> List[TaggedResult]:
        """Keep only highest-scored result per parent_doc_id."""
        if not results:
            return []

        best_by_parent: Dict[str, TaggedResult] = {}

        for item in results:
            # Handle both 2-tuple (legacy) and 3-tuple (with turn_index)
            if len(item) == 3:
                source_tag, row, turn_index = item
            else:
                source_tag, row = item
                turn_index = None

            parent_id = row.get('parent_doc_id', row.get('doc_id', ''))
            score = self._safe_score(row.get('score', 0.0))

            if parent_id not in best_by_parent:
                best_by_parent[parent_id] = (source_tag, row, turn_index) if turn_index is not None else (source_tag, row)
            else:
                existing_score = self._safe_score(best_by_parent[parent_id][1].get('score', 0.0))
                if score > existing_score:
                    best_by_parent[parent_id] = (source_tag, row, turn_index) if turn_index is not None else (source_tag, row)

        return list(best_by_parent.values())

    def _filter_seen(self, results: List[TaggedResult], seen_parent_ids: Set[str]) -> List[TaggedResult]:
        """Filter out results whose parent_doc_id is in seen_parent_ids."""
        filtered = []
        for item in results:
            # Handle both 2-tuple (legacy) and 3-tuple (with turn_index)
            if len(item) == 3:
                source_tag, row, turn_index = item
            else:
                source_tag, row = item
                turn_index = None

            parent_id = row.get('parent_doc_id', row.get('doc_id', ''))
            if parent_id not in seen_parent_ids:
                filtered.append((source_tag, row, turn_index) if turn_index is not None else (source_tag, row))
        return filtered

    def _normalize_scores(self, results: List[TaggedResult]) -> List[TaggedResult]:
        """Normalize scores to [0, 1] range for fair MMR calculation."""
        if not results:
            return []

        # Extract scores - handle both 2-tuple and 3-tuple formats
        scores = []
        for item in results:
            if len(item) == 3:
                _, row, _ = item
            else:
                _, row = item
            scores.append(self._safe_score(row.get('score', 0.0)))

        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0

        if max_score > min_score:
            for i, item in enumerate(results):
                if len(item) == 3:
                    _, row, _ = item
                else:
                    _, row = item
                row['norm_score'] = (scores[i] - min_score) / (max_score - min_score)
        else:
            for item in results:
                if len(item) == 3:
                    _, row, _ = item
                else:
                    _, row = item
                row['norm_score'] = 1.0

        return results

    def _max_similarity(self, embedding: np.ndarray, selected_embeddings: List[np.ndarray]) -> float:
        """Calculate max cosine similarity between embedding and selected embeddings."""
        if not selected_embeddings:
            return 0.0

        # Normalize query embedding
        emb_norm = np.linalg.norm(embedding)
        if emb_norm == 0:
            return 0.0
        normalized_emb = embedding / emb_norm

        max_sim = 0.0
        for sel_emb in selected_embeddings:
            sel_norm = np.linalg.norm(sel_emb)
            if sel_norm == 0:
                continue
            normalized_sel = sel_emb / sel_norm
            sim = np.dot(normalized_emb, normalized_sel)
            if sim > max_sim:
                max_sim = sim

        return max_sim

    def _estimate_tokens(self, row: pd.Series) -> int:
        """Estimate token cost of a result (content + XML wrapper overhead)."""
        content = row.get('content', '')
        if not isinstance(content, str):
            content = str(content) if content else ''
        # Add ~50 tokens for XML wrapper overhead
        return self.token_counter(content) + 50
