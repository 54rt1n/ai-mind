# tests/core_tests/unit/tool/test_container_mcp.py
"""Tests for ContainerMCPTool ledger and portfolio functionality."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from aim.tool.impl.container_mcp import ContainerMCPTool


@pytest.fixture
def mcp_tool():
    """Create a ContainerMCPTool instance for testing."""
    # Use a dummy endpoint since we mock _call_mcp
    return ContainerMCPTool(endpoint="http://localhost:8080/sse")


@pytest.fixture
def sample_ledger_entries():
    """Sample ledger entries for testing portfolio calculation."""
    return {
        "items": [
            {
                "id": "1",
                "text": "DEPOSIT $10000.00",
                "status": "FILLED",
                "metadata": {
                    "type": "DEPOSIT",
                    "amount": 10000.0,
                    "timestamp": "2025-01-01T00:00:00+00:00"
                }
            },
            {
                "id": "2",
                "text": "BUY 10 AAPL @ $185.50",
                "status": "FILLED",
                "metadata": {
                    "type": "BUY",
                    "symbol": "AAPL",
                    "quantity": 10,
                    "price": 185.50,
                    "total": 1855.0,
                    "timestamp": "2025-01-02T00:00:00+00:00"
                }
            },
            {
                "id": "3",
                "text": "BUY 5 AAPL @ $190.00",
                "status": "FILLED",
                "metadata": {
                    "type": "BUY",
                    "symbol": "AAPL",
                    "quantity": 5,
                    "price": 190.0,
                    "total": 950.0,
                    "timestamp": "2025-01-03T00:00:00+00:00"
                }
            },
            {
                "id": "4",
                "text": "SELL 5 AAPL @ $200.00",
                "status": "FILLED",
                "metadata": {
                    "type": "SELL",
                    "symbol": "AAPL",
                    "quantity": 5,
                    "price": 200.0,
                    "total": 1000.0,
                    "timestamp": "2025-01-04T00:00:00+00:00"
                }
            },
            {
                "id": "5",
                "text": "DIVIDEND AAPL $25.00",
                "status": "FILLED",
                "metadata": {
                    "type": "DIVIDEND",
                    "symbol": "AAPL",
                    "amount": 25.0,
                    "timestamp": "2025-01-05T00:00:00+00:00"
                }
            }
        ]
    }


class TestLedgerGet:
    """Tests for ledger_get method."""

    def test_ledger_get_calls_list_get(self, mcp_tool):
        """ledger_get should delegate to list_get."""
        with patch.object(mcp_tool, "list_get") as mock_list_get:
            mock_list_get.return_value = {"items": []}

            result = mcp_tool.ledger_get("test-ledger")

            mock_list_get.assert_called_once_with("test-ledger")
            assert result == {"items": []}

    def test_ledger_get_returns_items(self, mcp_tool, sample_ledger_entries):
        """ledger_get should return ledger entries."""
        with patch.object(mcp_tool, "list_get") as mock_list_get:
            mock_list_get.return_value = sample_ledger_entries

            result = mcp_tool.ledger_get("portfolio-ledger")

            assert "items" in result
            assert len(result["items"]) == 5


class TestLedgerAdd:
    """Tests for ledger_add method."""

    def test_ledger_add_deposit(self, mcp_tool):
        """ledger_add should create DEPOSIT entry correctly."""
        with patch.object(mcp_tool, "list_modify") as mock_list_modify:
            mock_list_modify.return_value = {"success": True, "item_id": "1"}

            result = mcp_tool.ledger_add(
                ledger_id="test-ledger",
                entry_type="DEPOSIT",
                amount=5000.0
            )

            mock_list_modify.assert_called_once()
            call_kwargs = mock_list_modify.call_args[1]

            assert call_kwargs["list_id"] == "test-ledger"
            assert call_kwargs["action"] == "add"
            assert call_kwargs["text"] == "DEPOSIT $5000.00"
            assert call_kwargs["status"] == "FILLED"
            assert call_kwargs["metadata"]["type"] == "DEPOSIT"
            assert call_kwargs["metadata"]["amount"] == 5000.0
            assert call_kwargs["metadata"]["total"] == 5000.0
            assert "timestamp" in call_kwargs["metadata"]

    def test_ledger_add_withdraw(self, mcp_tool):
        """ledger_add should create WITHDRAW entry correctly."""
        with patch.object(mcp_tool, "list_modify") as mock_list_modify:
            mock_list_modify.return_value = {"success": True}

            result = mcp_tool.ledger_add(
                ledger_id="test-ledger",
                entry_type="WITHDRAW",
                amount=1000.0
            )

            call_kwargs = mock_list_modify.call_args[1]

            assert call_kwargs["text"] == "WITHDRAW $1000.00"
            assert call_kwargs["metadata"]["type"] == "WITHDRAW"
            assert call_kwargs["metadata"]["amount"] == 1000.0

    def test_ledger_add_buy(self, mcp_tool):
        """ledger_add should create BUY entry correctly."""
        with patch.object(mcp_tool, "list_modify") as mock_list_modify:
            mock_list_modify.return_value = {"success": True}

            result = mcp_tool.ledger_add(
                ledger_id="test-ledger",
                entry_type="BUY",
                symbol="AAPL",
                quantity=10,
                price=185.50
            )

            call_kwargs = mock_list_modify.call_args[1]

            assert call_kwargs["text"] == "BUY 10 AAPL @ $185.50"
            assert call_kwargs["metadata"]["type"] == "BUY"
            assert call_kwargs["metadata"]["symbol"] == "AAPL"
            assert call_kwargs["metadata"]["quantity"] == 10
            assert call_kwargs["metadata"]["price"] == 185.50
            assert call_kwargs["metadata"]["total"] == 1855.0

    def test_ledger_add_sell(self, mcp_tool):
        """ledger_add should create SELL entry correctly."""
        with patch.object(mcp_tool, "list_modify") as mock_list_modify:
            mock_list_modify.return_value = {"success": True}

            result = mcp_tool.ledger_add(
                ledger_id="test-ledger",
                entry_type="SELL",
                symbol="AAPL",
                quantity=5,
                price=200.0
            )

            call_kwargs = mock_list_modify.call_args[1]

            assert call_kwargs["text"] == "SELL 5 AAPL @ $200.00"
            assert call_kwargs["metadata"]["type"] == "SELL"
            assert call_kwargs["metadata"]["symbol"] == "AAPL"
            assert call_kwargs["metadata"]["quantity"] == 5
            assert call_kwargs["metadata"]["price"] == 200.0
            assert call_kwargs["metadata"]["total"] == 1000.0

    def test_ledger_add_dividend(self, mcp_tool):
        """ledger_add should create DIVIDEND entry correctly."""
        with patch.object(mcp_tool, "list_modify") as mock_list_modify:
            mock_list_modify.return_value = {"success": True}

            result = mcp_tool.ledger_add(
                ledger_id="test-ledger",
                entry_type="DIVIDEND",
                symbol="AAPL",
                amount=25.0
            )

            call_kwargs = mock_list_modify.call_args[1]

            assert call_kwargs["text"] == "DIVIDEND AAPL $25.00"
            assert call_kwargs["metadata"]["type"] == "DIVIDEND"
            assert call_kwargs["metadata"]["symbol"] == "AAPL"
            assert call_kwargs["metadata"]["amount"] == 25.0

    def test_ledger_add_unknown_type(self, mcp_tool):
        """ledger_add should return error for unknown entry_type."""
        result = mcp_tool.ledger_add(
            ledger_id="test-ledger",
            entry_type="UNKNOWN"
        )

        assert result["success"] is False
        assert "Unknown entry_type" in result["error"]


class TestPortfolioCalculate:
    """Tests for portfolio_calculate method."""

    def test_portfolio_calculate_full_scenario(self, mcp_tool, sample_ledger_entries):
        """portfolio_calculate should correctly aggregate transactions."""
        with patch.object(mcp_tool, "ledger_get") as mock_ledger_get:
            mock_ledger_get.return_value = sample_ledger_entries

            result = mcp_tool.portfolio_calculate("portfolio-ledger")

            # Initial deposit: $10,000
            # BUY 10 AAPL @ $185.50 = -$1,855 (cash: $8,145)
            # BUY 5 AAPL @ $190 = -$950 (cash: $7,195)
            # SELL 5 AAPL @ $200 = +$1,000 (cash: $8,195)
            # DIVIDEND = +$25 (cash: $8,220)
            assert result["cash"] == 8220.0

            # Positions: Started with 15 AAPL, sold 5, have 10 left
            # Before sale: total_cost = 1855 + 950 = 2805, shares = 15
            # Average cost per share = 2805 / 15 = 187.0
            # After selling 5: shares = 10, total_cost = 10 * 187.0 = 1870.0
            # Cost basis = 1870 / 10 = 187.0
            assert "AAPL" in result["positions"]
            aapl = result["positions"]["AAPL"]
            assert aapl["shares"] == 10
            assert aapl["cost_basis"] == 187.0
            assert aapl["total_cost"] == 1870.0

    def test_portfolio_calculate_empty_ledger(self, mcp_tool):
        """portfolio_calculate should return zero for empty ledger."""
        with patch.object(mcp_tool, "ledger_get") as mock_ledger_get:
            mock_ledger_get.return_value = {"items": []}

            result = mcp_tool.portfolio_calculate("empty-ledger")

            assert result["cash"] == 0.0
            assert result["positions"] == {}

    def test_portfolio_calculate_skips_non_filled(self, mcp_tool):
        """portfolio_calculate should skip entries that are not FILLED."""
        entries = {
            "items": [
                {
                    "id": "1",
                    "text": "DEPOSIT $10000.00",
                    "status": "FILLED",
                    "metadata": {"type": "DEPOSIT", "amount": 10000.0}
                },
                {
                    "id": "2",
                    "text": "BUY 100 AAPL @ $100.00",
                    "status": "PENDING",  # Should be skipped
                    "metadata": {
                        "type": "BUY",
                        "symbol": "AAPL",
                        "quantity": 100,
                        "total": 10000.0
                    }
                }
            ]
        }

        with patch.object(mcp_tool, "ledger_get") as mock_ledger_get:
            mock_ledger_get.return_value = entries

            result = mcp_tool.portfolio_calculate("test-ledger")

            # Only the deposit should be counted
            assert result["cash"] == 10000.0
            assert result["positions"] == {}

    def test_portfolio_calculate_deposit_and_withdraw(self, mcp_tool):
        """portfolio_calculate should handle deposits and withdrawals."""
        entries = {
            "items": [
                {
                    "id": "1",
                    "status": "FILLED",
                    "metadata": {"type": "DEPOSIT", "amount": 10000.0}
                },
                {
                    "id": "2",
                    "status": "FILLED",
                    "metadata": {"type": "WITHDRAW", "amount": 3000.0}
                },
                {
                    "id": "3",
                    "status": "FILLED",
                    "metadata": {"type": "DEPOSIT", "amount": 500.0}
                }
            ]
        }

        with patch.object(mcp_tool, "ledger_get") as mock_ledger_get:
            mock_ledger_get.return_value = entries

            result = mcp_tool.portfolio_calculate("test-ledger")

            # 10000 - 3000 + 500 = 7500
            assert result["cash"] == 7500.0

    def test_portfolio_calculate_excludes_zero_positions(self, mcp_tool):
        """portfolio_calculate should exclude positions with zero shares."""
        entries = {
            "items": [
                {
                    "id": "1",
                    "status": "FILLED",
                    "metadata": {"type": "DEPOSIT", "amount": 5000.0}
                },
                {
                    "id": "2",
                    "status": "FILLED",
                    "metadata": {
                        "type": "BUY",
                        "symbol": "AAPL",
                        "quantity": 10,
                        "total": 1000.0
                    }
                },
                {
                    "id": "3",
                    "status": "FILLED",
                    "metadata": {
                        "type": "SELL",
                        "symbol": "AAPL",
                        "quantity": 10,
                        "total": 1200.0
                    }
                }
            ]
        }

        with patch.object(mcp_tool, "ledger_get") as mock_ledger_get:
            mock_ledger_get.return_value = entries

            result = mcp_tool.portfolio_calculate("test-ledger")

            # Cash: 5000 - 1000 + 1200 = 5200
            assert result["cash"] == 5200.0
            # AAPL position should be excluded (0 shares)
            assert "AAPL" not in result["positions"]

    def test_portfolio_calculate_multiple_symbols(self, mcp_tool):
        """portfolio_calculate should handle multiple stock symbols."""
        entries = {
            "items": [
                {
                    "id": "1",
                    "status": "FILLED",
                    "metadata": {"type": "DEPOSIT", "amount": 50000.0}
                },
                {
                    "id": "2",
                    "status": "FILLED",
                    "metadata": {
                        "type": "BUY",
                        "symbol": "AAPL",
                        "quantity": 10,
                        "total": 2000.0
                    }
                },
                {
                    "id": "3",
                    "status": "FILLED",
                    "metadata": {
                        "type": "BUY",
                        "symbol": "GOOGL",
                        "quantity": 5,
                        "total": 5000.0
                    }
                },
                {
                    "id": "4",
                    "status": "FILLED",
                    "metadata": {
                        "type": "BUY",
                        "symbol": "MSFT",
                        "quantity": 20,
                        "total": 8000.0
                    }
                }
            ]
        }

        with patch.object(mcp_tool, "ledger_get") as mock_ledger_get:
            mock_ledger_get.return_value = entries

            result = mcp_tool.portfolio_calculate("test-ledger")

            # Cash: 50000 - 2000 - 5000 - 8000 = 35000
            assert result["cash"] == 35000.0

            assert len(result["positions"]) == 3
            assert result["positions"]["AAPL"]["shares"] == 10
            assert result["positions"]["AAPL"]["cost_basis"] == 200.0  # 2000/10
            assert result["positions"]["GOOGL"]["shares"] == 5
            assert result["positions"]["GOOGL"]["cost_basis"] == 1000.0  # 5000/5
            assert result["positions"]["MSFT"]["shares"] == 20
            assert result["positions"]["MSFT"]["cost_basis"] == 400.0  # 8000/20


class TestExecuteDispatcher:
    """Tests for execute() dispatcher with ledger/portfolio tools."""

    def test_execute_ledger_get(self, mcp_tool):
        """execute should dispatch ledger_get correctly."""
        with patch.object(mcp_tool, "ledger_get") as mock_method:
            mock_method.return_value = {"items": []}

            result = mcp_tool.execute("ledger_get", {"ledger_id": "test-ledger"})

            mock_method.assert_called_once_with(ledger_id="test-ledger")

    def test_execute_ledger_get_requires_ledger_id(self, mcp_tool):
        """execute ledger_get should require ledger_id parameter."""
        with pytest.raises(ValueError, match="ledger_id parameter is required"):
            mcp_tool.execute("ledger_get", {})

    def test_execute_ledger_add(self, mcp_tool):
        """execute should dispatch ledger_add correctly."""
        with patch.object(mcp_tool, "ledger_add") as mock_method:
            mock_method.return_value = {"success": True}

            result = mcp_tool.execute("ledger_add", {
                "ledger_id": "test-ledger",
                "entry_type": "BUY",
                "symbol": "AAPL",
                "quantity": 10,
                "price": 185.50
            })

            mock_method.assert_called_once_with(
                ledger_id="test-ledger",
                entry_type="BUY",
                symbol="AAPL",
                quantity=10,
                price=185.50,
                amount=None
            )

    def test_execute_ledger_add_requires_ledger_id(self, mcp_tool):
        """execute ledger_add should require ledger_id parameter."""
        with pytest.raises(ValueError, match="ledger_id parameter is required"):
            mcp_tool.execute("ledger_add", {"entry_type": "BUY"})

    def test_execute_ledger_add_requires_entry_type(self, mcp_tool):
        """execute ledger_add should require entry_type parameter."""
        with pytest.raises(ValueError, match="entry_type parameter is required"):
            mcp_tool.execute("ledger_add", {"ledger_id": "test-ledger"})

    def test_execute_portfolio_calculate(self, mcp_tool):
        """execute should dispatch portfolio_calculate correctly."""
        with patch.object(mcp_tool, "portfolio_calculate") as mock_method:
            mock_method.return_value = {"cash": 1000.0, "positions": {}}

            result = mcp_tool.execute("portfolio_calculate", {"ledger_id": "test-ledger"})

            mock_method.assert_called_once_with(ledger_id="test-ledger")

    def test_execute_portfolio_calculate_requires_ledger_id(self, mcp_tool):
        """execute portfolio_calculate should require ledger_id parameter."""
        with pytest.raises(ValueError, match="ledger_id parameter is required"):
            mcp_tool.execute("portfolio_calculate", {})


class TestListModifyWithMetadata:
    """Tests for extended list_modify with status and metadata."""

    def test_list_modify_passes_status_and_metadata(self, mcp_tool):
        """list_modify should pass status and metadata to MCP."""
        with patch.object(mcp_tool, "_call_mcp") as mock_call:
            mock_call.return_value = {"success": True}

            result = mcp_tool.list_modify(
                list_id="test-list",
                action="add",
                text="Test entry",
                status="FILLED",
                metadata={"type": "TEST", "value": 123}
            )

            mock_call.assert_called_once_with("list_modify", {
                "list_id": "test-list",
                "action": "add",
                "text": "Test entry",
                "status": "FILLED",
                "metadata": {"type": "TEST", "value": 123}
            })

    def test_list_modify_omits_none_values(self, mcp_tool):
        """list_modify should not include None parameters."""
        with patch.object(mcp_tool, "_call_mcp") as mock_call:
            mock_call.return_value = {"success": True}

            result = mcp_tool.list_modify(
                list_id="test-list",
                action="add",
                text="Test entry"
            )

            mock_call.assert_called_once_with("list_modify", {
                "list_id": "test-list",
                "action": "add",
                "text": "Test entry"
            })
