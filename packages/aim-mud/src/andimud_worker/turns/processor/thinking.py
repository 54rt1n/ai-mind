# andimud_worker/turns/processor/thinking.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Thinking turn processor: processes turns with externally injected thought content."""

import logging
from typing import TYPE_CHECKING

from aim_mud_types import MUDAction, MUDTurnRequest, MUDEvent
from .phased import PhasedTurnProcessor

if TYPE_CHECKING:
    from ...mixins.turns import TurnsMixin


logger = logging.getLogger(__name__)


class ThinkingTurnProcessor(PhasedTurnProcessor):
    """Turn processor that emphasizes externally injected thought content.

    Extends PhasedTurnProcessor with thought injection:
    - Reads thought_content from constructor
    - Injects thought as guidance for Phase 1 decision
    - Sets thought_content on response strategy for Phase 2
    - Otherwise follows normal phased processing

    The injected thought influences:
    1. Decision guidance (what action to take)
    2. Memory retrieval (thought becomes query context)
    3. Response generation (thought_content in strategy)
    """

    def __init__(self, worker: "TurnsMixin", thought_content: str = ""):
        """Initialize with worker and thought content.

        Args:
            worker: MUDAgentWorker instance
            thought_content: External thought to inject into processing
        """
        super().__init__(worker)
        self.thought_content = thought_content

    async def _decide_action(
        self, turn_request: MUDTurnRequest, events: list[MUDEvent]
    ) -> tuple[list[MUDAction], str]:
        """Execute phased decision strategy with thought injection.

        Injects thought_content into:
        1. user_guidance (for Phase 1 decision)
        2. response_strategy.thought_content (for Phase 2 memory/context)

        Args:
            turn_request: Current turn request
            events: List of events to process

        Returns:
            Tuple of (actions_taken, thinking)
        """
        # Inject thought into guidance if present
        if self.thought_content:
            thought_guidance = f"[Thought injection: {self.thought_content}]"

            if self.user_guidance:
                self.user_guidance = f"{thought_guidance}\n{self.user_guidance}"
            else:
                self.user_guidance = thought_guidance

            logger.info(
                "ThinkingTurnProcessor: injected thought (%d chars) into guidance",
                len(self.thought_content),
            )

            # Also set thought_content on response strategy for Phase 2
            if self.worker._response_strategy:
                self.worker._response_strategy.thought_content = self.thought_content
                logger.debug("Set thought_content on response strategy")

        # Use parent implementation for actual processing
        return await super()._decide_action(turn_request, events)
