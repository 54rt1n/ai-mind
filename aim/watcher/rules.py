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

from aim.conversation.model import ConversationModel

logger = logging.getLogger(__name__)


@dataclass
class RuleMatch:
    """Result of a rule evaluation."""

    conversation_id: str
    scenario: str
    persona_id: Optional[str] = None
    model: Optional[str] = None
    guidance: Optional[str] = None

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

        logger.info(f"Found {len(unanalyzed)} unanalyzed conversations")

        for conversation_id in unanalyzed:
            # Check minimum message count
            msg_count = len(conv_df[conv_df['conversation_id'] == conversation_id])

            if msg_count >= self.min_messages:
                matches.append(RuleMatch(
                    conversation_id=conversation_id,
                    scenario=self.scenario,
                    persona_id=self.persona_id,
                    model=self.model,
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
