# aim/app/mud/worker/mixins/datastore/report.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Conversation report management for the MUD worker.

Handles generating and caching conversation reports in Redis.
Extracted from worker.py lines 841-860
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...worker import MUDAgentWorker


logger = logging.getLogger(__name__)


class ReportMixin:
    """Mixin for conversation report management methods.

    These methods are mixed into MUDAgentWorker in worker.py.
    """

    async def _update_conversation_report(self: "MUDAgentWorker") -> None:
        """Generate conversation report and save to Redis.

        Originally from worker.py lines 841-860

        Calls cvm.get_conversation_report() and stores result in Redis
        for fast access by @list command.
        """
        try:
            report_df = self.cvm.get_conversation_report()

            if not report_df.empty:
                report_dict = report_df.set_index('conversation_id').T.to_dict()
            else:
                report_dict = {}

            from aim_mud_types.client import RedisMUDClient
            client = RedisMUDClient(self.redis)
            await client.set_conversation_report(self.config.agent_id, report_dict)
            logger.debug(f"Updated conversation report: {len(report_dict)} conversations")
        except Exception as e:
            logger.error(f"Failed to update conversation report: {e}", exc_info=True)
