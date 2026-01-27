# aim/tool/impl/container_mcp.py
# AI-Mind (C) 2025 by Martin Bukowski is licensed under CC BY-NC-SA 4.0
"""Container-MCP proxy tool implementation.

This module provides a tool implementation that forwards calls to
a container-mcp server via the MCP protocol over SSE.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from .base import ToolImplementation

logger = logging.getLogger(__name__)


class ContainerMCPTool(ToolImplementation):
    """Tool implementation that proxies calls to container-mcp.

    This class acts as an MCP client, forwarding tool calls to a
    container-mcp server for secure sandboxed execution.
    """

    def __init__(self, endpoint: Optional[str] = None):
        """Initialize the container-mcp proxy.

        Args:
            endpoint: The container-mcp SSE endpoint URL.
                     If None, uses ChatConfig.mcp_endpoint from environment.
        """
        if endpoint is None:
            from aim.config import ChatConfig
            config = ChatConfig.from_env()
            endpoint = config.mcp_endpoint
        self.endpoint = endpoint
        self._session: Optional[Any] = None

    async def _call_mcp_async(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a container-mcp tool asynchronously.

        Args:
            tool_name: Name of the MCP tool to call.
            args: Arguments to pass to the tool.

        Returns:
            Result dictionary from the tool execution.
        """
        try:
            from mcp.client.sse import sse_client
            from mcp import ClientSession

            async with sse_client(self.endpoint) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, args)

                    # Check for MCP error response
                    if hasattr(result, "isError") and result.isError:
                        # Extract error message from content
                        error_msg = "Tool execution failed"
                        if hasattr(result, "content") and result.content:
                            content = result.content
                            if isinstance(content, list) and len(content) > 0:
                                first = content[0]
                                if hasattr(first, "text"):
                                    error_msg = first.text
                        return {"success": False, "error": error_msg}

                    # Convert MCP result to dict
                    if hasattr(result, "content"):
                        # Handle text content
                        content = result.content
                        if isinstance(content, list) and len(content) > 0:
                            first = content[0]
                            if hasattr(first, "text"):
                                import json
                                try:
                                    parsed = json.loads(first.text)
                                    # Ensure output key exists for terminal commands
                                    if "output" not in parsed and "stdout" in parsed:
                                        parsed["output"] = parsed["stdout"]
                                    return parsed
                                except json.JSONDecodeError:
                                    return {"output": first.text, "success": True}
                        return {"output": str(content), "success": True}
                    return {"output": str(result), "success": True}
        except ImportError:
            logger.error("MCP package not installed. Install with: pip install 'mcp[cli]>=1.0.0'")
            return {
                "success": False,
                "error": "MCP client not available. Container-MCP integration requires the mcp package."
            }
        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            return {"success": False, "error": str(e)}

    def _call_mcp(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a container-mcp tool synchronously.

        Args:
            tool_name: Name of the MCP tool to call.
            args: Arguments to pass to the tool.

        Returns:
            Result dictionary from the tool execution.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If we're in an async context, create a new thread to run the call
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._call_mcp_async(tool_name, args))
                return future.result()
        else:
            return asyncio.run(self._call_mcp_async(tool_name, args))

    # =========================================================================
    # Code Terminal Tools
    # =========================================================================

    def py_exec(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute Python code in a secure sandbox.

        Args:
            code: Python code to execute.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with execution results.
        """
        return self._call_mcp("system_run_python", {"code": code})

    def bash_exec(self, command: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a bash command in a secure sandbox.

        Args:
            command: Bash command to execute.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with command output.
        """
        return self._call_mcp("system_run_command", {"command": command})

    # =========================================================================
    # Web Terminal Tools
    # =========================================================================

    def web_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Search the web.

        Args:
            query: Search query.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with search results.
        """
        return self._call_mcp("web_search", {"query": query})

    def web_scrape(self, url: str, selector: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Scrape content from a URL.

        Args:
            url: URL to scrape.
            selector: Optional CSS selector to target specific content.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with scraped content.
        """
        args: Dict[str, Any] = {"url": url}
        if selector:
            args["selector"] = selector
        return self._call_mcp("web_scrape", args)

    def web_browse(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Browse a website with Playwright.

        Args:
            url: URL to browse.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with page content.
        """
        return self._call_mcp("web_browse", {"url": url})

    # =========================================================================
    # Market Terminal Tools
    # =========================================================================

    def market_query(self, symbol: str, period: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        """Query stock/crypto prices.

        Args:
            symbol: Stock/crypto symbol (e.g., "AAPL", "BTC-USD").
            period: Historical period. If None, returns full data including
                    fundamentals, news, and trend analysis.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with market data.
        """
        args: Dict[str, Any] = {"symbol": symbol}
        if period is not None:
            args["period"] = period
        return self._call_mcp("market_query", args)

    def rss_fetch(self, url: str, limit: int = 10, **kwargs: Any) -> Dict[str, Any]:
        """Fetch and parse an RSS feed.

        Args:
            url: RSS feed URL.
            limit: Maximum number of items to return.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with feed items.
        """
        return self._call_mcp("rss_fetch", {"url": url, "limit": limit})

    # =========================================================================
    # Research Terminal Tools
    # =========================================================================

    def kb_search(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Search the knowledge base.

        Args:
            query: Search query string.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with search results.
        """
        return self._call_mcp("kb_search", {"query": query})

    def kb_read(self, doc_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Read a document from the knowledge base.

        Args:
            doc_id: Document identifier.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with document content.
        """
        return self._call_mcp("kb_read", {"doc_id": doc_id})

    # =========================================================================
    # List Terminal Tools
    # =========================================================================

    def list_get(self, list_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Get items from a list.

        Args:
            list_id: List identifier.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with list items.
        """
        return self._call_mcp("list_get", {"name": list_id})

    def list_modify(
        self,
        list_id: str,
        action: str,
        item_text: Optional[str] = None,
        item_index: Optional[int] = None,
        status: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Modify a list (add item, update item, remove item).

        Args:
            list_id: List identifier.
            action: Action to perform ("add", "update", "remove").
            item_text: Text for items (required for "add" action).
            item_index: Item index (required for "update"/"remove" actions).
            status: Optional status for the item.
            tags: Optional list of tags.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dictionary with operation result.
        """
        args: Dict[str, Any] = {"list_name": list_id, "action": action}
        if item_text is not None:
            args["item_text"] = item_text
        if item_index is not None:
            args["item_index"] = item_index
        if status is not None:
            args["status"] = status
        if tags is not None:
            args["tags"] = tags
        return self._call_mcp("list_modify", args)

    # =========================================================================
    # Ledger/Portfolio Tools
    # =========================================================================

    def ledger_get(self, ledger_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Get all ledger entries.

        Args:
            ledger_id: The ledger list identifier.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dict with 'items' list of ledger entries.
        """
        return self.list_get(ledger_id)

    def ledger_add(
        self,
        ledger_id: str,
        entry_type: str,
        symbol: Optional[str] = None,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        amount: Optional[float] = None,
        actor: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Add a transaction entry to the ledger.

        Args:
            ledger_id: The ledger list identifier.
            entry_type: Transaction type (DEPOSIT, WITHDRAW, BUY, SELL, DIVIDEND).
            symbol: Stock symbol (for BUY, SELL, DIVIDEND).
            quantity: Number of shares (for BUY, SELL).
            price: Price per share (for BUY, SELL).
            amount: Total amount (for DEPOSIT, WITHDRAW, DIVIDEND).
            actor: Name of the player/agent who made the transaction.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dict with 'success' bool and 'entry' data.
        """
        from datetime import datetime, timezone

        # Build entry as pipe-delimited string: TYPE|TIMESTAMP|ACTOR|SYMBOL|QUANTITY|PRICE|AMOUNT|TOTAL
        timestamp = datetime.now(timezone.utc).isoformat()
        actor_str = actor or ""

        if entry_type == "DEPOSIT":
            total = amount
            item_text = f"DEPOSIT|{timestamp}|{actor_str}|||{amount}|{total}"
        elif entry_type == "WITHDRAW":
            total = amount
            item_text = f"WITHDRAW|{timestamp}|{actor_str}|||{amount}|{total}"
        elif entry_type == "BUY":
            total = quantity * price
            item_text = f"BUY|{timestamp}|{actor_str}|{symbol}|{quantity}|{price}||{total}"
        elif entry_type == "SELL":
            total = quantity * price
            item_text = f"SELL|{timestamp}|{actor_str}|{symbol}|{quantity}|{price}||{total}"
        elif entry_type == "DIVIDEND":
            total = amount
            item_text = f"DIVIDEND|{timestamp}|{actor_str}|{symbol}||{amount}|{total}"
        else:
            return {"success": False, "error": f"Unknown entry_type: {entry_type}"}

        # Add to ledger using list_modify
        result = self.list_modify(
            list_id=ledger_id,
            action="add",
            item_text=item_text,
            status="DONE"
        )

        return result

    def portfolio_calculate(self, ledger_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Calculate portfolio state from ledger entries.

        Aggregates all FILLED transactions to derive:
        - Current cash balance
        - Positions with quantity and average cost basis

        Args:
            ledger_id: The ledger list identifier.
            **kwargs: Additional parameters (ignored).

        Returns:
            Dict with 'cash' float and 'positions' dict
            {symbol: {shares, cost_basis, total_cost}}.
        """
        ledger = self.ledger_get(ledger_id)
        entries = ledger.get("items", [])

        cash = 0.0
        positions: Dict[str, Dict[str, float]] = {}

        for entry in entries:
            if entry.get("status") != "DONE":
                continue

            # Parse pipe-delimited format: TYPE|TIMESTAMP|ACTOR|SYMBOL|QUANTITY|PRICE|AMOUNT|TOTAL
            text = entry.get("text", "")
            parts = text.split("|")
            if len(parts) < 7:
                continue

            entry_type = parts[0]
            # parts[1] = timestamp, parts[2] = actor (not needed for calculation)
            symbol = parts[3] if len(parts) > 3 else ""
            quantity = float(parts[4]) if parts[4] else 0
            price = float(parts[5]) if parts[5] else 0
            amount = float(parts[6]) if parts[6] else 0
            total = float(parts[7]) if len(parts) > 7 and parts[7] else 0

            if entry_type == "DEPOSIT":
                cash += amount
            elif entry_type == "WITHDRAW":
                cash -= amount
            elif entry_type == "BUY":
                cash -= total
                if symbol and symbol not in positions:
                    positions[symbol] = {"shares": 0, "total_cost": 0}
                if symbol:
                    positions[symbol]["shares"] += quantity
                    positions[symbol]["total_cost"] += total
            elif entry_type == "SELL":
                cash += total
                if symbol in positions:
                    current = positions[symbol]
                    if current["shares"] > 0:
                        cost_per_share = current["total_cost"] / current["shares"]
                        current["shares"] -= quantity
                        current["total_cost"] = current["shares"] * cost_per_share
            elif entry_type == "DIVIDEND":
                cash += amount

        # Build result with cost_basis per share for each position
        result_positions: Dict[str, Dict[str, float]] = {}
        for symbol, data in positions.items():
            if data["shares"] > 0:
                result_positions[symbol] = {
                    "shares": data["shares"],
                    "cost_basis": data["total_cost"] / data["shares"],
                    "total_cost": data["total_cost"]
                }

        return {
            "cash": cash,
            "positions": result_positions
        }

    # =========================================================================
    # Execute Dispatcher
    # =========================================================================

    def execute(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a container-mcp tool.

        Args:
            function_name: Name of the tool function.
            parameters: Tool parameters.

        Returns:
            Dictionary with tool results.

        Raises:
            ValueError: If function is unknown or required parameters are missing.
        """
        # Code terminal tools
        if function_name == "py_exec":
            if "code" not in parameters:
                raise ValueError("Code parameter is required")
            return self.py_exec(code=parameters["code"])

        elif function_name == "bash_exec":
            if "command" not in parameters:
                raise ValueError("Command parameter is required")
            return self.bash_exec(command=parameters["command"])

        # Web terminal tools
        elif function_name == "web_search":
            if "query" not in parameters:
                raise ValueError("Query parameter is required")
            return self.web_search(query=parameters["query"])

        elif function_name == "web_scrape":
            if "url" not in parameters:
                raise ValueError("URL parameter is required")
            return self.web_scrape(
                url=parameters["url"],
                selector=parameters.get("selector")
            )

        elif function_name == "web_browse":
            if "url" not in parameters:
                raise ValueError("URL parameter is required")
            return self.web_browse(url=parameters["url"])

        # Market terminal tools
        elif function_name == "market_query":
            if "symbol" not in parameters:
                raise ValueError("Symbol parameter is required")
            return self.market_query(
                symbol=parameters["symbol"],
                period=parameters.get("period")
            )

        elif function_name == "rss_fetch":
            if "url" not in parameters:
                raise ValueError("URL parameter is required")
            return self.rss_fetch(
                url=parameters["url"],
                limit=parameters.get("limit", 10)
            )

        # Web terminal tools (additional)
        elif function_name == "visit_webpage":
            if "url" not in parameters:
                raise ValueError("URL parameter is required")
            return self.web_scrape(url=parameters["url"])

        # News terminal tools
        elif function_name == "get_feed":
            if "url" not in parameters:
                raise ValueError("URL parameter is required")
            return self.rss_fetch(
                url=parameters["url"],
                limit=parameters.get("limit", 10)
            )

        # Research terminal tools
        elif function_name == "research":
            if "query" not in parameters:
                raise ValueError("Query parameter is required")
            return self.kb_search(query=parameters["query"])

        elif function_name == "read_doc":
            if "doc_id" not in parameters:
                raise ValueError("doc_id parameter is required")
            return self.kb_read(doc_id=parameters["doc_id"])

        # List terminal tools
        elif function_name == "show_list":
            if "list_id" not in parameters:
                raise ValueError("list_id parameter is required")
            return self.list_get(list_id=parameters["list_id"])

        elif function_name == "add_item":
            if "list_id" not in parameters:
                raise ValueError("list_id parameter is required")
            if "text" not in parameters:
                raise ValueError("text parameter is required")
            return self.list_modify(
                list_id=parameters["list_id"],
                action=parameters.get("action", "add"),
                item_text=parameters["text"]
            )

        elif function_name == "check_item":
            if "list_id" not in parameters:
                raise ValueError("list_id parameter is required")
            if "item_id" not in parameters:
                raise ValueError("item_id parameter is required")
            return self.list_modify(
                list_id=parameters["list_id"],
                action=parameters.get("action", "update"),
                item_index=int(parameters["item_id"]),
                status="DONE"
            )

        # Market terminal tools
        elif function_name == "stock_quote":
            if "symbol" not in parameters:
                raise ValueError("Symbol parameter is required")
            return self.market_query(
                symbol=parameters["symbol"],
                period=parameters.get("period")
            )

        # Ledger/Portfolio tools
        elif function_name == "ledger_get":
            if "ledger_id" not in parameters:
                raise ValueError("ledger_id parameter is required")
            return self.ledger_get(ledger_id=parameters["ledger_id"])

        elif function_name == "ledger_add":
            if "ledger_id" not in parameters:
                raise ValueError("ledger_id parameter is required")
            if "entry_type" not in parameters:
                raise ValueError("entry_type parameter is required")
            return self.ledger_add(
                ledger_id=parameters["ledger_id"],
                entry_type=parameters["entry_type"],
                symbol=parameters.get("symbol"),
                quantity=parameters.get("quantity"),
                price=parameters.get("price"),
                amount=parameters.get("amount"),
                actor=parameters.get("actor")
            )

        elif function_name == "portfolio_calculate":
            if "ledger_id" not in parameters:
                raise ValueError("ledger_id parameter is required")
            return self.portfolio_calculate(ledger_id=parameters["ledger_id"])

        else:
            return {
                "success": False,
                "error": f"Unknown container-mcp tool: {function_name}"
            }
