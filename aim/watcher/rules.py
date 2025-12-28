# aim/watcher/rules.py
# AI-Mind © 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""
Rule definitions for the watcher.

Rules define conditions that trigger pipeline execution.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from aim.config import ChatConfig
from aim.conversation.model import ConversationModel
from aim.llm.models import LanguageModelV2
from aim.utils.tokens import count_tokens

logger = logging.getLogger(__name__)


@dataclass
class RuleMatch:
    """Result of a rule evaluation."""

    conversation_id: str
    scenario: str
    persona_id: Optional[str] = None
    model: Optional[str] = None
    guidance: Optional[str] = None
    message_count: int = 0  # For stability tracking
    token_count: int = 0  # For token-based threshold

    def __repr__(self) -> str:
        return f"RuleMatch({self.conversation_id} -> {self.scenario})"


class Rule(ABC):
    """Base class for watcher rules."""

    name: str = "base_rule"
    description: str = "Base rule class"

    @abstractmethod
    def evaluate(self, cvm: ConversationModel) -> list[RuleMatch]:
        """
        Evaluate the rule against the conversation model.

        Args:
            cvm: ConversationModel to query

        Returns:
            List of RuleMatch objects for conversations that match
        """
        pass


class UnanalyzedConversationRule(Rule):
    """
    Rule: Find conversations with document_type 'conversation'
    but no document_type 'analysis'.

    Triggers the analyst pipeline for unanalyzed conversations.
    """

    name = "unanalyzed_conversation"
    description = "Find conversations that haven't been analyzed yet"

    def __init__(
        self,
        scenario: str = "analyst",
        persona_id: Optional[str] = None,
        model: Optional[str] = None,
        min_messages: int = 1,
    ):
        """
        Args:
            scenario: Pipeline scenario to trigger (default: analyst)
            persona_id: Persona to use for the pipeline
            model: Model override for the pipeline
            min_messages: Minimum conversation messages required
        """
        self.scenario = scenario
        self.persona_id = persona_id
        self.model = model
        self.min_messages = min_messages

    def evaluate(self, cvm: ConversationModel) -> list[RuleMatch]:
        """Find conversations with 'conversation' docs but no 'analysis' docs."""
        matches = []

        # Query for all documents with document_type='conversation'
        conv_df = cvm.index.search(
            query_document_type='conversation',
            query_limit=10000,
        )

        if conv_df.empty:
            logger.debug("No conversation documents in index")
            return matches

        # Get unique conversation IDs with 'conversation' type
        has_conversation = set(conv_df['conversation_id'].unique())

        # Query for documents with document_type='analysis'
        analysis_df = cvm.index.search(
            query_document_type='analysis',
            query_limit=10000,
        )

        # Get conversation IDs that have analysis
        has_analysis = set()
        if not analysis_df.empty:
            has_analysis = set(analysis_df['conversation_id'].unique())

        # Find conversations that have conversation but no analysis
        unanalyzed = has_conversation - has_analysis

        if unanalyzed:
            logger.info(f"Found {len(unanalyzed)} unanalyzed conversations")
        else:
            logger.debug("No unanalyzed conversations found")

        for conversation_id in unanalyzed:
            # Check minimum message count
            msg_count = len(conv_df[conv_df['conversation_id'] == conversation_id])

            if msg_count >= self.min_messages:
                matches.append(RuleMatch(
                    conversation_id=conversation_id,
                    scenario=self.scenario,
                    persona_id=self.persona_id,
                    model=self.model,
                    message_count=msg_count,
                ))

        return matches


class StaleConversationRule(Rule):
    """
    Rule: Find conversations that haven't been updated in a while
    but have recent activity.

    Useful for triggering summarization or journaling.
    """

    name = "stale_conversation"
    description = "Find conversations with no recent pipeline activity"

    def __init__(
        self,
        scenario: str = "summarizer",
        persona_id: Optional[str] = None,
        model: Optional[str] = None,
        stale_hours: int = 24,
        required_doc_type: str = "conversation",
        missing_doc_type: str = "summary",
    ):
        self.scenario = scenario
        self.persona_id = persona_id
        self.model = model
        self.stale_hours = stale_hours
        self.required_doc_type = required_doc_type
        self.missing_doc_type = missing_doc_type

    def evaluate(self, cvm: ConversationModel) -> list[RuleMatch]:
        """Find stale conversations missing the target document type."""
        from datetime import datetime, timedelta

        matches = []

        # Query for documents with required doc type
        required_df = cvm.index.search(
            query_document_type=self.required_doc_type,
            query_limit=10000,
        )

        if required_df.empty:
            return matches

        has_required = set(required_df['conversation_id'].unique())

        # Query for documents with target doc type
        target_df = cvm.index.search(
            query_document_type=self.missing_doc_type,
            query_limit=10000,
        )

        has_target = set()
        if not target_df.empty:
            has_target = set(target_df['conversation_id'].unique())

        # Conversations missing the target doc type
        candidates = has_required - has_target

        cutoff_ts = int((datetime.now() - timedelta(hours=self.stale_hours)).timestamp())

        for conversation_id in candidates:
            # Check if last activity is older than cutoff
            conv_docs = required_df[required_df['conversation_id'] == conversation_id]

            if 'timestamp' in conv_docs.columns:
                last_timestamp = conv_docs['timestamp'].max()

                if last_timestamp < cutoff_ts:
                    matches.append(RuleMatch(
                        conversation_id=conversation_id,
                        scenario=self.scenario,
                        persona_id=self.persona_id,
                        model=self.model,
                    ))

        return matches


class AnalysisWithSummaryRule(Rule):
    """
    Rule: Find unanalyzed conversations and determine if they need
    summarization first based on token count.

    Logic:
    - Conversations under token threshold: trigger analyst directly
    - Conversations over token threshold without summary: trigger summarizer first
    - Conversations over token threshold with summary: trigger analyst
    """

    name = "analysis_with_summary"
    description = "Analyze conversations, summarizing long ones first"

    def __init__(
        self,
        config: ChatConfig,
        persona_id: Optional[str] = None,
        model: Optional[str] = None,
        token_threshold_ratio: float = 0.8,
        min_messages: int = 1,
    ):
        """
        Args:
            config: ChatConfig for model lookup
            persona_id: Persona to use for the pipeline
            model: Model override for the pipeline
            token_threshold_ratio: Fraction of model context (0.0-1.0) above which summary is required
            min_messages: Minimum messages required to trigger analysis
        """
        self.config = config
        self.persona_id = persona_id
        self.model = model
        self.threshold_ratio = token_threshold_ratio
        self.min_messages = min_messages

        # Look up model's max_tokens
        model_name = model or config.default_model
        if model_name:
            models = LanguageModelV2.index_models(config)
            self._model_max_tokens = models[model_name].max_tokens if model_name in models else 32768
        else:
            self._model_max_tokens = 32768  # Default fallback

        self._token_threshold = int(self._model_max_tokens * self.threshold_ratio)
        logger.info(
            f"AnalysisWithSummaryRule: token threshold = {self._token_threshold} "
            f"({self.threshold_ratio:.0%} of {self._model_max_tokens})"
        )

    def evaluate(self, cvm: ConversationModel) -> list[RuleMatch]:
        """Find unanalyzed conversations, routing long ones to summarizer first."""
        matches = []

        # Query for all documents with document_type='conversation'
        conv_df = cvm.index.search(
            query_document_type='conversation',
            query_limit=10000,
        )

        if conv_df.empty:
            logger.debug("No conversation documents in index")
            return matches

        # Get unique conversation IDs with 'conversation' type
        has_conversation = set(conv_df['conversation_id'].unique())

        # Query for documents with document_type='analysis'
        analysis_df = cvm.index.search(
            query_document_type='analysis',
            query_limit=10000,
        )
        has_analysis = set()
        if not analysis_df.empty:
            has_analysis = set(analysis_df['conversation_id'].unique())

        # Query for documents with document_type='summary'
        summary_df = cvm.index.search(
            query_document_type='summary',
            query_limit=10000,
        )
        has_summary = set()
        if not summary_df.empty:
            has_summary = set(summary_df['conversation_id'].unique())

        # Find conversations that have conversation but no analysis
        unanalyzed = has_conversation - has_analysis

        if unanalyzed:
            logger.info(f"Found {len(unanalyzed)} unanalyzed conversations")
        else:
            logger.debug("No unanalyzed conversations found")

        for conversation_id in unanalyzed:
            # Get messages for this conversation
            conv_messages = conv_df[conv_df['conversation_id'] == conversation_id]
            msg_count = len(conv_messages)

            if msg_count < self.min_messages:
                continue

            # Count total tokens
            total_tokens = sum(
                count_tokens(str(content))
                for content in conv_messages['content'].tolist()
            )

            # Determine which scenario to trigger based on token count
            if total_tokens > self._token_threshold and conversation_id not in has_summary:
                # Over token threshold without summary -> summarize first
                scenario = "summarizer"
                guidance = f"Summarize this conversation ({total_tokens} tokens) before analysis."
                logger.debug(
                    f"Conversation {conversation_id}: {total_tokens} tokens > {self._token_threshold} threshold, needs summary"
                )
            else:
                # Under threshold OR already has summary -> analyze
                scenario = "analyst"
                guidance = None

            matches.append(RuleMatch(
                conversation_id=conversation_id,
                scenario=scenario,
                persona_id=self.persona_id,
                model=self.model,
                guidance=guidance,
                message_count=msg_count,
                token_count=total_tokens,
            ))

        return matches


class PostSummaryAnalysisRule(Rule):
    """
    Rule: Find conversations that have been summarized but not analyzed.

    This catches the second stage of the summarizer->analyst chain.
    Self-healing: if summarizer completes but analyst wasn't triggered,
    the next watcher cycle picks it up.
    """

    name = "post_summary_analysis"
    description = "Analyze conversations that have summaries but no analysis"

    def __init__(
        self,
        persona_id: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.persona_id = persona_id
        self.model = model

    def evaluate(self, cvm: ConversationModel) -> list[RuleMatch]:
        """Find conversations with summary but no analysis."""
        matches = []

        # Query for documents with document_type='summary'
        summary_df = cvm.index.search(
            query_document_type='summary',
            query_limit=10000,
        )

        if summary_df.empty:
            return matches

        has_summary = set(summary_df['conversation_id'].unique())

        # Query for documents with document_type='analysis'
        analysis_df = cvm.index.search(
            query_document_type='analysis',
            query_limit=10000,
        )
        has_analysis = set()
        if not analysis_df.empty:
            has_analysis = set(analysis_df['conversation_id'].unique())

        # Find conversations with summary but no analysis
        needs_analysis = has_summary - has_analysis

        if needs_analysis:
            logger.info(f"Found {len(needs_analysis)} summarized conversations needing analysis")
        else:
            logger.debug("No summarized conversations needing analysis")

        for conversation_id in needs_analysis:
            matches.append(RuleMatch(
                conversation_id=conversation_id,
                scenario="analyst",
                persona_id=self.persona_id,
                model=self.model,
            ))

        return matches
