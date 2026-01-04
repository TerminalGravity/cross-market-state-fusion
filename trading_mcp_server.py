#!/usr/bin/env python3
"""
Trading MCP Server - Monitor and control the Polymarket trading bot.

This FastMCP server provides tools for:
- Viewing current trading status and positions
- Checking recent trades and PnL
- Validating live trades against Polymarket API
- Monitoring wallet balance
- Viewing unverified trades

Run: fastmcp run trading_mcp_server.py
"""
import os
import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastmcp import FastMCP
from dotenv import load_dotenv

# Import our modules
from db.connection import DatabaseConnection
from helpers.wallet_balance import WalletBalanceTracker
from helpers.polymarket_validator import PolymarketValidator, run_validation_check

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Polymarket Trading Monitor")

# Global state
db: Optional[DatabaseConnection] = None
balance_tracker: Optional[WalletBalanceTracker] = None
validator: Optional[PolymarketValidator] = None


@mcp.tool()
async def get_trading_status() -> dict:
    """
    Get current trading status including active sessions, positions, and PnL.

    Returns comprehensive overview of the trading system state.
    """
    global db

    if not db:
        return {"error": "Database not connected"}

    try:
        # Get active session
        session = await db.get_active_session()

        if not session:
            return {
                "status": "No active session",
                "sessions": "All sessions stopped"
            }

        session_id = str(session["id"])

        # Get session stats
        stats = await db.get_session_stats(session_id)

        # Get open trades/positions
        open_trades = await db.get_open_trades(session_id)

        # Get recent trades
        recent = await db.get_recent_trades(session_id, limit=10)

        return {
            "status": "Active",
            "session": {
                "id": session_id,
                "mode": session["mode"],
                "started_at": session["started_at"].isoformat(),
                "trade_size": float(session["trade_size"]),
                "model_version": session["model_version"]
            },
            "performance": {
                "total_pnl": float(stats.get("total_pnl", 0)),
                "total_trades": stats.get("total_trades", 0),
                "win_rate": float(stats.get("win_rate", 0)),
                "avg_pnl": float(stats.get("avg_pnl", 0)),
                "best_trade": float(stats.get("best_trade", 0)),
                "worst_trade": float(stats.get("worst_trade", 0))
            },
            "open_positions": len(open_trades),
            "recent_trades": [
                {
                    "asset": t["asset"],
                    "side": t["side"],
                    "pnl": float(t["pnl"]) if t["pnl"] else 0,
                    "duration_sec": t["duration_seconds"],
                    "exit_time": t["exit_time"].isoformat() if t["exit_time"] else None
                }
                for t in recent[:5]
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_wallet_balance() -> dict:
    """
    Get current USDC balance from the Polymarket wallet on Polygon.

    This is the REAL on-chain balance, not what the system reports.
    """
    global balance_tracker

    if not balance_tracker:
        return {"error": "Balance tracker not initialized"}

    try:
        balance = await balance_tracker.get_balance()

        return {
            "wallet_address": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "usdc_balance": balance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "chain": "Polygon",
            "source": "polygon_rpc"
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def validate_live_trades(max_age_hours: int = 24) -> dict:
    """
    Validate all unverified live trades against Polymarket CLOB API.

    Args:
        max_age_hours: Only validate trades from the last N hours (default: 24)

    Returns:
        Validation summary showing how many trades were verified
    """
    global db, validator

    if not db:
        return {"error": "Database not connected"}

    if not validator:
        validator = PolymarketValidator(db, None, balance_tracker)

    try:
        results = await validator.validate_unverified_trades(max_age_hours)

        return {
            "validation_time": datetime.now(timezone.utc).isoformat(),
            "total_trades": results["total"],
            "verified": results["verified"],
            "failed": results["failed"],
            "errors": results["errors"][:10]  # Limit to first 10 errors
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_unverified_trades() -> dict:
    """
    Get list of live trades that haven't been verified yet.

    These are trades the system claims are "live" but haven't been validated
    against the Polymarket API.
    """
    global db

    if not db:
        return {"error": "Database not connected"}

    try:
        rows = await db.fetch("""
            SELECT
                id, asset, side, entry_price, size_dollars,
                entry_time, order_id, fill_status, execution_type
            FROM trades
            WHERE execution_type = 'live'
              AND (verified IS NULL OR verified = FALSE)
            ORDER BY entry_time DESC
            LIMIT 50
        """)

        return {
            "count": len(rows),
            "trades": [
                {
                    "trade_id": str(row["id"]),
                    "asset": row["asset"],
                    "side": row["side"],
                    "entry_price": float(row["entry_price"]),
                    "size": float(row["size_dollars"]),
                    "entry_time": row["entry_time"].isoformat(),
                    "order_id": row["order_id"],
                    "fill_status": row["fill_status"]
                }
                for row in rows
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_balance_discrepancies() -> dict:
    """
    Check for discrepancies between reported PnL and actual wallet balance.

    This reveals if the system is lying about profits.
    """
    global db

    if not db:
        return {"error": "Database not connected"}

    try:
        rows = await db.fetch("""
            SELECT *
            FROM balance_discrepancies
            ORDER BY snapshot_at DESC
            LIMIT 20
        """)

        return {
            "count": len(rows),
            "snapshots": [
                {
                    "timestamp": row["snapshot_at"].isoformat(),
                    "actual_balance": float(row["usdc_balance"]),
                    "reported_pnl": float(row["reported_pnl"]) if row["reported_pnl"] else 0,
                    "discrepancy": float(row["discrepancy"]) if row["discrepancy"] else 0,
                    "severity": row["severity"]
                }
                for row in rows
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_recent_trades(limit: int = 20, mode: Optional[str] = None) -> dict:
    """
    Get recent completed trades.

    Args:
        limit: Number of trades to return (default: 20, max: 100)
        mode: Filter by mode ('paper' or 'live'), or None for all

    Returns:
        List of recent trades with full details
    """
    global db

    if not db:
        return {"error": "Database not connected"}

    try:
        limit = min(limit, 100)  # Cap at 100

        if mode:
            rows = await db.fetch("""
                SELECT
                    t.id, t.asset, t.side, t.entry_price, t.exit_price,
                    t.pnl, t.size_dollars, t.duration_seconds,
                    t.entry_time, t.exit_time, t.order_id,
                    t.execution_type, t.verified, s.mode as session_mode
                FROM trades t
                JOIN sessions s ON t.session_id = s.id
                WHERE t.pnl IS NOT NULL AND s.mode = $1
                ORDER BY t.exit_time DESC
                LIMIT $2
            """, mode, limit)
        else:
            rows = await db.fetch("""
                SELECT
                    t.id, t.asset, t.side, t.entry_price, t.exit_price,
                    t.pnl, t.size_dollars, t.duration_seconds,
                    t.entry_time, t.exit_time, t.order_id,
                    t.execution_type, t.verified, s.mode as session_mode
                FROM trades t
                JOIN sessions s ON t.session_id = s.id
                WHERE t.pnl IS NOT NULL
                ORDER BY t.exit_time DESC
                LIMIT $1
            """, limit)

        return {
            "count": len(rows),
            "trades": [
                {
                    "trade_id": str(row["id"]),
                    "asset": row["asset"],
                    "side": row["side"],
                    "entry_price": float(row["entry_price"]),
                    "exit_price": float(row["exit_price"]) if row["exit_price"] else 0,
                    "pnl": float(row["pnl"]) if row["pnl"] else 0,
                    "size": float(row["size_dollars"]),
                    "duration_sec": row["duration_seconds"],
                    "entry_time": row["entry_time"].isoformat(),
                    "exit_time": row["exit_time"].isoformat() if row["exit_time"] else None,
                    "execution_type": row["execution_type"],
                    "verified": row["verified"],
                    "order_id": row["order_id"]
                }
                for row in rows
            ]
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def run_full_validation() -> dict:
    """
    Run comprehensive validation check on the entire system.

    This validates:
    1. All unverified live trades
    2. Balance reconciliation
    3. Database integrity

    Returns detailed validation report.
    """
    global db, balance_tracker

    if not db:
        return {"error": "Database not connected"}

    try:
        results = await run_validation_check(db, None, balance_tracker)

        return {
            "validation_time": results["timestamp"],
            "trade_validation": results["trade_validation"],
            "balance_validation": results["balance_validation"],
            "summary": {
                "trades_validated": results["trade_validation"]["verified"],
                "trades_failed": results["trade_validation"]["failed"],
                "balance_ok": results["balance_validation"] is not None
            }
        }

    except Exception as e:
        return {"error": str(e)}


# Server lifecycle
@mcp.on_startup
async def startup():
    """Initialize database and balance tracker on server startup."""
    global db, balance_tracker, validator

    print("🚀 Starting Polymarket Trading MCP Server...")

    # Initialize database
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("⚠️  DATABASE_URL not set - database tools will be unavailable")
        return

    db = DatabaseConnection(database_url=database_url)
    await db.connect()
    print("✓ Database connected")

    # Initialize balance tracker
    rpc_url = os.getenv("POLYGON_RPC_URL")
    wallet_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if rpc_url and wallet_address:
        balance_tracker = WalletBalanceTracker(rpc_url, wallet_address)
        print("✓ Balance tracker initialized")
    else:
        print("⚠️  POLYGON_RPC_URL or POLYMARKET_FUNDER_ADDRESS not set - balance tools unavailable")

    # Initialize validator
    if db:
        validator = PolymarketValidator(db, None, balance_tracker)
        print("✓ Validator initialized")

    print("✅ MCP Server ready!")


@mcp.on_shutdown
async def shutdown():
    """Cleanup on server shutdown."""
    global db

    if db:
        await db.close()
        print("✓ Database connection closed")


if __name__ == "__main__":
    # Run the server
    mcp.run()
