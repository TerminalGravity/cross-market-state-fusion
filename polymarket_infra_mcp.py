#!/usr/bin/env python3
"""
Polymarket Infrastructure MCP Server - Database & System Records

This is for INTERNAL SYSTEM DATA, not the source of truth.
Use polymarket_trader_mcp.py for live on-chain data.

Data Sources:
- PostgreSQL database (sessions, trades, PnL records)
- Balance discrepancy tracking
- Trade validation against live data
- Proxy health & network status
- WebSocket connection monitoring
- Safety systems status
- Fly.io deployment status
- Discord alert configuration

All responses include "source" field indicating exact data origin.

Run: uv run python polymarket_infra_mcp.py
"""
import os
import json
import asyncio
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, AsyncIterator, Dict, Any

from fastmcp import FastMCP
from dotenv import load_dotenv

# Import database module
from db.connection import DatabaseConnection
from helpers.polymarket_validator import PolymarketValidator

# Lazy-loaded infrastructure modules (initialized on demand)
_proxy_manager = None
_discord_webhook = None

load_dotenv()

# Global state (initialized in lifespan)
db: Optional[DatabaseConnection] = None
validator: Optional[PolymarketValidator] = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize database on startup, cleanup on shutdown."""
    global db, validator

    print("=" * 60)
    print("POLYMARKET INFRA MCP - Database & System Records")
    print("=" * 60)
    print("NOTE: This is INTERNAL data, not source of truth.")
    print("      Use polymarket_trader_mcp.py for live data.")
    print("-" * 60)

    # Initialize database
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[WARN] DATABASE_URL not set - all tools unavailable")
    else:
        try:
            db = DatabaseConnection(database_url=database_url)
            await db.connect()
            print("[OK] Database connected")

            # Initialize validator
            validator = PolymarketValidator(db, None, None)
            print("[OK] Validator initialized")
        except Exception as e:
            print(f"[ERROR] Database connection failed: {e}")

    print("=" * 60)

    yield {}

    # Cleanup
    if db:
        await db.close()
        print("Database connection closed")


# Initialize server
mcp = FastMCP("Polymarket Infrastructure", lifespan=lifespan)


# ============================================================================
# SECTION 1: SESSION & TRADE RECORDS (Database)
# Source: postgresql_database
# ============================================================================

@mcp.tool()
async def get_trading_status() -> dict:
    """
    Get current trading status from INTERNAL DATABASE.

    Shows active sessions, positions, and PnL from system records.
    NOTE: This may differ from live on-chain data.
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    try:
        session = await db.get_active_session()

        if not session:
            return {
                "source": "postgresql_database",
                "status": "No active session",
                "note": "This is internal system state, not live data"
            }

        session_id = str(session["id"])
        stats = await db.get_session_stats(session_id)
        open_trades = await db.get_open_trades(session_id)
        recent = await db.get_recent_trades(session_id, limit=10)

        return {
            "source": "postgresql_database",
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
            ],
            "note": "Internal database records - verify against live data"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


@mcp.tool()
async def get_recent_trades(limit: int = 20, mode: Optional[str] = None) -> dict:
    """
    Get recent completed trades from INTERNAL DATABASE.

    Args:
        limit: Number of trades to return (default: 20, max: 100)
        mode: Filter by mode ('paper' or 'live'), or None for all

    NOTE: These are system records. Verify live trades with polymarket_trader_mcp.
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    try:
        limit = min(limit, 100)

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
            "source": "postgresql_database",
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
            ],
            "note": "Internal database records - use polymarket_trader_mcp for live data"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


@mcp.tool()
async def get_unverified_trades() -> dict:
    """
    Get trades that haven't been verified against live CLOB API.

    These are trades the system claims executed but haven't been
    validated against Polymarket's actual records.
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

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
            "source": "postgresql_database",
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
            ],
            "action_needed": "Run validate_live_trades() to verify against CLOB API"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


# ============================================================================
# SECTION 2: VALIDATION & RECONCILIATION
# Source: postgresql_database + polymarket_clob_api (for validation)
# ============================================================================

@mcp.tool()
async def validate_live_trades(max_age_hours: int = 24) -> dict:
    """
    Validate unverified live trades against Polymarket CLOB API.

    Compares internal database records against actual live trades.

    Args:
        max_age_hours: Only validate trades from the last N hours (default: 24)

    Returns:
        Validation summary showing how many trades were verified
    """
    global db, validator

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    if not validator:
        validator = PolymarketValidator(db, None, None)

    try:
        results = await validator.validate_unverified_trades(max_age_hours)

        return {
            "source": "postgresql_database + polymarket_clob_api",
            "validation_time": datetime.now(timezone.utc).isoformat(),
            "total_trades": results["total"],
            "verified": results["verified"],
            "failed": results["failed"],
            "errors": results["errors"][:10],
            "note": "Compares database records against live CLOB API"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


@mcp.tool()
async def get_balance_discrepancies() -> dict:
    """
    Check for discrepancies between reported PnL and actual wallet balance.

    This reveals if the system is lying about profits. Compares:
    - Internal PnL records (what system says)
    - Actual on-chain balance (what blockchain shows)
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    try:
        rows = await db.fetch("""
            SELECT *
            FROM balance_discrepancies
            ORDER BY snapshot_at DESC
            LIMIT 20
        """)

        return {
            "source": "postgresql_database",
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
            ],
            "interpretation": {
                "positive_discrepancy": "Actual balance > reported (underreported gains)",
                "negative_discrepancy": "Actual balance < reported (overreported gains or loss)"
            }
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


@mcp.tool()
async def run_full_validation() -> dict:
    """
    Run comprehensive validation check on the entire system.

    Validates:
    1. All unverified live trades against CLOB API
    2. Balance reconciliation against on-chain
    3. Database integrity

    Use this to detect system issues or discrepancies.
    """
    global db, validator

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    try:
        from helpers.polymarket_validator import run_validation_check
        results = await run_validation_check(db, None, None)

        return {
            "source": "postgresql_database + polymarket_clob_api",
            "validation_time": results["timestamp"],
            "trade_validation": results["trade_validation"],
            "balance_validation": results["balance_validation"],
            "summary": {
                "trades_validated": results["trade_validation"]["verified"],
                "trades_failed": results["trade_validation"]["failed"],
                "balance_ok": results["balance_validation"] is not None
            },
            "recommendation": "Use polymarket_trader_mcp for source-of-truth data"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


# ============================================================================
# SECTION 3: SESSION MANAGEMENT
# Source: postgresql_database
# ============================================================================

@mcp.tool()
async def get_all_sessions(limit: int = 20) -> dict:
    """
    Get all trading sessions from database.

    Args:
        limit: Maximum sessions to return (default: 20)
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    try:
        rows = await db.fetch("""
            SELECT
                id, mode, started_at, ended_at, trade_size,
                model_version, status
            FROM sessions
            ORDER BY started_at DESC
            LIMIT $1
        """, limit)

        return {
            "source": "postgresql_database",
            "count": len(rows),
            "sessions": [
                {
                    "id": str(row["id"]),
                    "mode": row["mode"],
                    "started_at": row["started_at"].isoformat(),
                    "ended_at": row["ended_at"].isoformat() if row["ended_at"] else None,
                    "trade_size": float(row["trade_size"]) if row["trade_size"] else None,
                    "model_version": row["model_version"],
                    "status": row["status"]
                }
                for row in rows
            ]
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


@mcp.tool()
async def get_session_pnl(session_id: Optional[str] = None) -> dict:
    """
    Get PnL summary for a specific session or all sessions.

    Args:
        session_id: Specific session ID, or None for aggregate
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    try:
        if session_id:
            rows = await db.fetch("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE session_id = $1 AND pnl IS NOT NULL
            """, session_id)
        else:
            rows = await db.fetch("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades
                WHERE pnl IS NOT NULL
            """)

        if not rows or rows[0]["total_trades"] == 0:
            return {
                "source": "postgresql_database",
                "error": "No completed trades found",
                "session_id": session_id
            }

        row = rows[0]
        total = row["total_trades"]
        wins = row["wins"] or 0

        return {
            "source": "postgresql_database",
            "session_id": session_id or "all_sessions",
            "summary": {
                "total_trades": total,
                "wins": wins,
                "losses": row["losses"] or 0,
                "win_rate": round(wins / total, 3) if total > 0 else 0,
                "total_pnl": round(float(row["total_pnl"] or 0), 2),
                "avg_pnl": round(float(row["avg_pnl"] or 0), 4),
                "best_trade": round(float(row["best_trade"] or 0), 2),
                "worst_trade": round(float(row["worst_trade"] or 0), 2)
            },
            "note": "Database records - compare with polymarket_trader_mcp for live data"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


# ============================================================================
# SECTION 4: TRANSFER HISTORY
# Source: postgresql_database
# ============================================================================

@mcp.tool()
async def get_transfer_history(limit: int = 20) -> dict:
    """
    Get profit transfer history from database.

    Shows automated transfers to cold wallet.

    Args:
        limit: Maximum transfers to return (default: 20)
    """
    global db

    if not db:
        return {"error": "Database not connected", "source": "postgresql_database"}

    # Get addresses from environment (fixed for this system)
    from_address = os.getenv("POLYMARKET_FUNDER_ADDRESS", "unknown")
    to_address = os.getenv("COLD_WALLET_ADDRESS", "unknown")

    try:
        rows = await db.fetch("""
            SELECT
                id, amount, tx_hash, status, created_at,
                confirmed_at, error_message, trigger_reason
            FROM transfers
            ORDER BY created_at DESC
            LIMIT $1
        """, limit)

        return {
            "source": "postgresql_database",
            "count": len(rows),
            "from_address": from_address,
            "to_address": to_address,
            "transfers": [
                {
                    "id": str(row["id"]),
                    "amount": float(row["amount"]),
                    "tx_hash": row["tx_hash"],
                    "status": row["status"],
                    "trigger_reason": row["trigger_reason"],
                    "created_at": row["created_at"].isoformat(),
                    "confirmed_at": row["confirmed_at"].isoformat() if row["confirmed_at"] else None,
                    "error_message": row["error_message"],
                    "polygonscan_url": f"https://polygonscan.com/tx/{row['tx_hash']}" if row["tx_hash"] else None
                }
                for row in rows
            ],
            "note": "Verify on-chain with polymarket_trader_mcp.get_usdc_transfers()"
        }

    except Exception as e:
        return {"error": str(e), "source": "postgresql_database"}


# ============================================================================
# HELPER FUNCTIONS (lazy initialization)
# ============================================================================

def _get_proxy_manager():
    """Lazy-load proxy manager singleton."""
    global _proxy_manager
    if _proxy_manager is None:
        try:
            from helpers.hifi_proxy import get_proxy_manager
            _proxy_manager = get_proxy_manager()
        except ImportError:
            return None
    return _proxy_manager


def _get_discord_webhook():
    """Lazy-load Discord webhook instance."""
    global _discord_webhook
    if _discord_webhook is None:
        try:
            from helpers.discord import get_webhook
            _discord_webhook = get_webhook()
        except ImportError:
            return None
    return _discord_webhook


async def _run_fly_command(args: list, timeout: int = 30) -> Dict[str, Any]:
    """Run fly CLI command and parse JSON output."""
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["fly"] + args + ["--json", "-a", "cross-market-state-fusion"],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0:
            try:
                return {"success": True, "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"success": True, "data": result.stdout.strip()}
        return {"success": False, "error": result.stderr.strip()}
    except FileNotFoundError:
        return {"success": False, "error": "fly CLI not installed"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# SECTION 5: PROXY & NETWORK HEALTH
# Source: hifi_proxy_manager + live_test
# ============================================================================

@mcp.tool()
async def get_proxy_health() -> dict:
    """
    Get HiFi proxy health status and metrics.

    Shows:
    - Success rate and request counts
    - Consecutive failures and Cloudflare blocks
    - Last success/error timestamps
    - Whether proxy is configured

    Source: HiFiProxyManager singleton state
    """
    pm = _get_proxy_manager()

    if pm is None:
        return {
            "source": "hifi_proxy_manager",
            "error": "HiFi proxy module not available",
            "proxy_configured": False
        }

    try:
        health = pm.get_health_status()

        return {
            "source": "hifi_proxy_manager",
            "proxy_configured": health["proxy_configured"],
            "is_healthy": health["is_healthy"],
            "metrics": {
                "success_rate": health["success_rate"],
                "total_requests": health["total_requests"],
                "consecutive_failures": health["consecutive_failures"],
                "cloudflare_blocks": health["cloudflare_blocks"]
            },
            "timestamps": {
                "last_success": health["last_success"],
                "last_error": health["last_error"]
            },
            "note": "Health tracks all requests through Decodo SOCKS5 proxy"
        }

    except Exception as e:
        return {"error": str(e), "source": "hifi_proxy_manager"}


@mcp.tool()
async def test_proxy_connection(target: str = "polymarket") -> dict:
    """
    Test proxy connection with live latency measurement.

    Args:
        target: Target to test - "polymarket", "binance", or "ipinfo"

    Returns:
        Connection status, latency, and IP information
    """
    import time

    targets = {
        "polymarket": "https://clob.polymarket.com/",
        "binance": "https://fapi.binance.com/fapi/v1/ping",
        "ipinfo": "https://ipinfo.io/json"
    }

    url = targets.get(target, targets["polymarket"])

    pm = _get_proxy_manager()

    if pm is None:
        return {
            "source": "live_proxy_test",
            "error": "HiFi proxy module not available"
        }

    try:
        # Get async httpx client from proxy manager
        client = pm.get_async_httpx_client()

        start = time.time()
        async with client:
            response = await client.get(url)
            latency_ms = (time.time() - start) * 1000

        # Record success
        pm.record_success()

        result = {
            "source": "live_proxy_test",
            "target": target,
            "url": url,
            "status": "connected",
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "proxy_used": pm.health.total_requests > 0
        }

        # Include IP info if testing ipinfo
        if target == "ipinfo" and response.status_code == 200:
            try:
                ip_data = response.json()
                result["ip_info"] = {
                    "ip": ip_data.get("ip"),
                    "city": ip_data.get("city"),
                    "region": ip_data.get("region"),
                    "country": ip_data.get("country"),
                    "org": ip_data.get("org")
                }
            except:
                pass

        return result

    except Exception as e:
        # Record failure
        is_cloudflare = "cloudflare" in str(e).lower() or "403" in str(e)
        pm.record_failure(str(e), is_cloudflare=is_cloudflare)

        return {
            "source": "live_proxy_test",
            "target": target,
            "status": "failed",
            "error": str(e),
            "is_cloudflare_block": is_cloudflare
        }


@mcp.tool()
async def get_proxy_geo_verification() -> dict:
    """
    Verify proxy IP is actually from Canada (expected geo-location).

    Uses ipinfo.io to check:
    - IP address being used
    - Geographic location (should be Canada)
    - ISP/organization (should be residential)

    This validates Decodo proxy is working correctly.
    """
    pm = _get_proxy_manager()

    if pm is None:
        return {
            "source": "geo_verification",
            "error": "HiFi proxy module not available"
        }

    try:
        client = pm.get_async_httpx_client()

        async with client:
            response = await client.get("https://ipinfo.io/json")

        if response.status_code != 200:
            return {
                "source": "geo_verification",
                "status": "failed",
                "error": f"ipinfo.io returned {response.status_code}"
            }

        data = response.json()

        # Expected geo for Canada proxy
        expected_country = "CA"
        actual_country = data.get("country", "unknown")
        geo_verified = actual_country == expected_country

        # Check if residential (non-datacenter)
        org = data.get("org", "").lower()
        is_residential = not any(
            dc in org for dc in ["amazon", "google", "microsoft", "digitalocean", "vultr", "linode", "ovh"]
        )

        return {
            "source": "geo_verification",
            "verified": geo_verified and is_residential,
            "ip": data.get("ip"),
            "location": {
                "city": data.get("city"),
                "region": data.get("region"),
                "country": actual_country,
                "country_name": "Canada" if actual_country == "CA" else data.get("country")
            },
            "network": {
                "org": data.get("org"),
                "is_residential": is_residential
            },
            "expectations": {
                "expected_country": expected_country,
                "country_match": actual_country == expected_country,
                "residential_expected": True,
                "residential_match": is_residential
            },
            "note": "Decodo proxy should show Canadian residential IP"
        }

    except Exception as e:
        return {"error": str(e), "source": "geo_verification"}


# ============================================================================
# SECTION 6: WEBSOCKET CONNECTION STATUS
# Source: orderbook_wss + futures_streamer
# ============================================================================

@mcp.tool()
async def get_websocket_status() -> dict:
    """
    Get WebSocket connection status for all data streams.

    Shows status of:
    - Polymarket orderbook WebSocket
    - Binance futures WebSocket
    - Connection state and failure counts

    Note: This reflects runtime state only when trading worker is running.
    """
    try:
        # Try to get orderbook WebSocket state
        wss_state = {
            "polymarket_orderbook": {"status": "unknown", "note": "Module not loaded"},
            "binance_futures": {"status": "unknown", "note": "Module not loaded"}
        }

        try:
            from helpers.orderbook_wss import OrderbookStreamer
            # Check if global instance exists (set by trading worker)
            import helpers.orderbook_wss as obs_module
            if hasattr(obs_module, '_global_streamer') and obs_module._global_streamer:
                streamer = obs_module._global_streamer
                wss_state["polymarket_orderbook"] = {
                    "status": "connected" if streamer._connected else "disconnected",
                    "subscribed_assets": list(streamer._subscriptions.keys()) if hasattr(streamer, '_subscriptions') else [],
                    "failure_count": getattr(streamer, '_failure_count', 0),
                    "using_rest_fallback": getattr(streamer, '_use_rest_fallback', False)
                }
            else:
                wss_state["polymarket_orderbook"]["note"] = "Streamer not initialized (worker not running)"
        except ImportError:
            pass

        try:
            from helpers.binance_futures import FuturesStreamer
            import helpers.binance_futures as bf_module
            if hasattr(bf_module, '_global_streamer') and bf_module._global_streamer:
                streamer = bf_module._global_streamer
                wss_state["binance_futures"] = {
                    "status": "connected" if streamer._connected else "disconnected",
                    "subscribed_symbols": list(streamer._subscriptions.keys()) if hasattr(streamer, '_subscriptions') else [],
                    "reconnect_count": getattr(streamer, '_reconnect_count', 0)
                }
            else:
                wss_state["binance_futures"]["note"] = "Streamer not initialized (worker not running)"
        except ImportError:
            pass

        return {
            "source": "websocket_modules",
            "streams": wss_state,
            "note": "WebSocket state is only available when trading worker is actively running"
        }

    except Exception as e:
        return {"error": str(e), "source": "websocket_modules"}


@mcp.tool()
async def get_orderbook_health() -> dict:
    """
    Get orderbook staleness and health status per asset.

    Health states:
    - HEALTHY: Data < 30 seconds old
    - DEGRADED: Data 30-60 seconds old
    - CRITICAL: Data > 60 seconds old

    Critical state triggers REST API fallback.
    """
    try:
        from helpers.orderbook_health import OrderbookHealthMonitor, HealthState
    except ImportError:
        return {
            "source": "orderbook_health_monitor",
            "error": "OrderbookHealthMonitor module not available"
        }

    try:
        # Try to get the global health monitor if it exists
        import helpers.orderbook_health as oh_module
        if hasattr(oh_module, '_global_monitor') and oh_module._global_monitor:
            monitor = oh_module._global_monitor
            summary = monitor.get_summary()

            return {
                "source": "orderbook_health_monitor",
                "assets": summary.get("assets", {}),
                "overall_health": summary.get("overall_health", "unknown"),
                "degraded_count": summary.get("degraded_count", 0),
                "critical_count": summary.get("critical_count", 0),
                "health_thresholds": {
                    "healthy": "<30s",
                    "degraded": "30-60s",
                    "critical": ">60s (triggers REST fallback)"
                }
            }
        else:
            return {
                "source": "orderbook_health_monitor",
                "status": "not_initialized",
                "note": "Health monitor only runs when trading worker is active",
                "health_thresholds": {
                    "healthy": "<30s",
                    "degraded": "30-60s",
                    "critical": ">60s"
                }
            }

    except Exception as e:
        return {"error": str(e), "source": "orderbook_health_monitor"}


# ============================================================================
# SECTION 7: SAFETY SYSTEMS STATUS
# Source: safety_manager + components
# ============================================================================

@mcp.tool()
async def get_safety_status() -> dict:
    """
    Get comprehensive safety dashboard with all risk controls.

    Aggregates:
    - Kill switch state
    - Daily loss limits
    - Consecutive loss tracking
    - Position and exposure limits

    Note: Returns configuration if SafetyManager not actively running.
    """
    try:
        from helpers.safety_manager import SafetyManager
    except ImportError:
        return {
            "source": "safety_manager",
            "error": "SafetyManager module not available"
        }

    try:
        # Try to get global safety manager if running
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            status = await manager.get_status()

            return {
                "source": "safety_manager",
                "kill_switch": {
                    "active": status["kill_switch_active"],
                    "reason": status.get("kill_switch_reason"),
                    "activated_at": status.get("kill_switch_activated_at")
                },
                "loss_tracking": {
                    "daily_pnl": status["daily_pnl"],
                    "daily_limit": status["daily_loss_limit"],
                    "remaining_allowance": status["remaining_loss_allowance"],
                    "consecutive_losses": status["consecutive_losses"],
                    "consecutive_limit": status["consecutive_loss_limit"]
                },
                "position_limits": {
                    "open_positions": status["open_positions"],
                    "max_positions": status["max_positions"],
                    "available_slots": status["available_slots"],
                    "total_exposure": status["total_exposure"],
                    "max_exposure": status["max_total_exposure"],
                    "remaining_capacity": status["remaining_capacity"]
                },
                "trading_allowed": not status["kill_switch_active"]
            }
        else:
            # Return default configuration
            return {
                "source": "safety_manager",
                "status": "not_initialized",
                "default_config": {
                    "daily_loss_limit": 100.0,
                    "consecutive_loss_limit": 3,
                    "max_positions": 4,
                    "max_position_size": 100.0,
                    "max_total_exposure": 400.0
                },
                "note": "Safety manager only runs when trading worker is active"
            }

    except Exception as e:
        return {"error": str(e), "source": "safety_manager"}


@mcp.tool()
async def get_kill_switch_status() -> dict:
    """
    Get kill switch state and activation history.

    Kill switch halts ALL trading when activated by:
    - Daily loss limit breach
    - Consecutive loss limit breach
    - Manual emergency halt
    - System errors

    Returns current state and recent activations.
    """
    global db

    result = {
        "source": "kill_switch",
        "current_state": {"active": False, "reason": None},
        "recent_activations": []
    }

    # Try to get live state from safety manager
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            state = await manager.kill_switch.get_state()
            result["current_state"] = {
                "active": state["is_active"],
                "reason": state.get("activation_reason"),
                "activated_at": state.get("activated_at"),
                "activated_by": state.get("activated_by")
            }
    except Exception:
        pass

    # Get history from database
    if db:
        try:
            rows = await db.fetch("""
                SELECT reason, activated_at, deactivated_at, source
                FROM kill_switch_history
                ORDER BY activated_at DESC
                LIMIT 10
            """)
            result["recent_activations"] = [
                {
                    "reason": row["reason"],
                    "activated_at": row["activated_at"].isoformat() if row["activated_at"] else None,
                    "deactivated_at": row["deactivated_at"].isoformat() if row["deactivated_at"] else None,
                    "source": row["source"]
                }
                for row in rows
            ]
        except Exception:
            result["note"] = "Kill switch history table may not exist"

    return result


@mcp.tool()
async def get_loss_tracker_status() -> dict:
    """
    Get daily P&L and consecutive loss tracking status.

    Monitors:
    - Today's P&L vs daily loss limit
    - Consecutive losing trades vs limit
    - Remaining loss allowance before kill switch

    Auto-triggers kill switch if limits breached.
    """
    result = {
        "source": "loss_tracker",
        "daily_pnl": 0.0,
        "daily_limit": 100.0,
        "consecutive_losses": 0,
        "consecutive_limit": 3
    }

    # Try to get live state
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            state = await manager.loss_tracker.get_state()
            result.update({
                "daily_pnl": state["daily_pnl"],
                "daily_limit": state["daily_loss_limit"],
                "remaining_allowance": state["remaining_allowance"],
                "consecutive_losses": state["consecutive_losses"],
                "consecutive_limit": state["consecutive_loss_limit"],
                "limits_ok": not state.get("limit_exceeded", False)
            })
            return result
    except Exception:
        pass

    # Fall back to database for today's PnL
    global db
    if db:
        try:
            row = await db.fetchrow("""
                SELECT
                    COALESCE(SUM(pnl), 0) as daily_pnl,
                    COUNT(*) as trade_count
                FROM trades
                WHERE DATE(exit_time) = CURRENT_DATE
                  AND pnl IS NOT NULL
            """)
            if row:
                result["daily_pnl"] = float(row["daily_pnl"])
                result["trade_count_today"] = row["trade_count"]
                result["remaining_allowance"] = result["daily_limit"] + result["daily_pnl"]
        except Exception:
            pass

    result["note"] = "Live tracking only available when trading worker is running"
    return result


@mcp.tool()
async def get_position_monitor_status() -> dict:
    """
    Get position monitoring and timeout tracking status.

    Monitors:
    - Open position count vs limit
    - Total exposure vs maximum
    - Position ages and timeout warnings
    - Stuck position detection

    Positions older than timeout threshold get force-closed.
    """
    result = {
        "source": "position_monitor",
        "open_positions": 0,
        "max_positions": 4,
        "total_exposure": 0.0,
        "max_exposure": 400.0
    }

    # Try to get live state
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            state = await manager.position_limiter.get_state()
            result.update({
                "open_positions": state["open_positions"],
                "max_positions": state["max_positions"],
                "available_slots": state["available_slots"],
                "total_exposure": state["total_exposure"],
                "max_exposure": state["max_total_exposure"],
                "remaining_capacity": state["remaining_capacity"],
                "positions_by_asset": state.get("positions_by_asset", {})
            })
            return result
    except Exception:
        pass

    # Fall back to database
    global db
    if db:
        try:
            rows = await db.fetch("""
                SELECT asset, side, size_dollars, entry_time
                FROM trades
                WHERE exit_time IS NULL
                ORDER BY entry_time DESC
            """)
            result["open_positions"] = len(rows)
            result["total_exposure"] = sum(float(r["size_dollars"]) for r in rows)
            result["positions"] = [
                {
                    "asset": r["asset"],
                    "side": r["side"],
                    "size": float(r["size_dollars"]),
                    "age_seconds": (datetime.now(timezone.utc) - r["entry_time"]).total_seconds()
                }
                for r in rows
            ]
        except Exception:
            pass

    result["note"] = "Live monitoring only available when trading worker is running"
    return result


# ============================================================================
# SECTION 7b: SAFETY SYSTEM CONTROLS
# Source: safety_manager + database
# ============================================================================

@mcp.tool()
async def activate_kill_switch(reason: str) -> dict:
    """
    Activate kill switch to halt ALL trading immediately.

    This is an emergency action that:
    - Stops all new trades from being placed
    - Does NOT close existing positions (use cancel_all_orders for that)
    - Logs the activation to database
    - Sends Discord alert if configured

    Args:
        reason: Why the kill switch is being activated (required for audit)

    Returns:
        Confirmation of kill switch activation
    """
    global db

    result = {
        "source": "kill_switch_control",
        "action": "activate",
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Try to activate via live safety manager
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            await manager.kill_switch.activate(reason, source="mcp_manual")
            result["success"] = True
            result["method"] = "live_safety_manager"

            # Send Discord alert
            webhook = _get_discord_webhook()
            if webhook and webhook.enabled:
                try:
                    embed = {
                        "title": "KILL SWITCH ACTIVATED",
                        "description": f"**Reason:** {reason}\n\nAll trading has been halted.",
                        "color": 0xff0000,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await webhook._send(embed)
                    result["discord_notified"] = True
                except Exception:
                    result["discord_notified"] = False

            return result
    except Exception as e:
        result["live_manager_error"] = str(e)

    # Fall back to database flag
    if db:
        try:
            await db.execute("""
                INSERT INTO kill_switch_history (reason, activated_at, source)
                VALUES ($1, NOW(), 'mcp_manual')
            """, reason)

            # Also set a runtime flag if table exists
            await db.execute("""
                INSERT INTO runtime_flags (key, value, updated_at)
                VALUES ('kill_switch_active', 'true', NOW())
                ON CONFLICT (key) DO UPDATE SET value = 'true', updated_at = NOW()
            """)

            result["success"] = True
            result["method"] = "database_flag"
            result["note"] = "Kill switch recorded. Worker will pick up on next check."
            return result
        except Exception as e:
            result["database_error"] = str(e)

    result["success"] = False
    result["error"] = "Could not activate kill switch - neither live manager nor database available"
    return result


@mcp.tool()
async def deactivate_kill_switch() -> dict:
    """
    Deactivate kill switch to resume trading.

    Clears the kill switch and allows the trading worker to resume operations.

    Returns:
        Confirmation of kill switch deactivation
    """
    global db

    result = {
        "source": "kill_switch_control",
        "action": "deactivate",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Try to deactivate via live safety manager
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            await manager.kill_switch.deactivate()
            result["success"] = True
            result["method"] = "live_safety_manager"

            # Send Discord alert
            webhook = _get_discord_webhook()
            if webhook and webhook.enabled:
                try:
                    embed = {
                        "title": "Kill Switch Deactivated",
                        "description": "Trading has been resumed.",
                        "color": 0x00ff00,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    await webhook._send(embed)
                    result["discord_notified"] = True
                except Exception:
                    result["discord_notified"] = False

            return result
    except Exception as e:
        result["live_manager_error"] = str(e)

    # Fall back to database flag
    if db:
        try:
            # Update history record
            await db.execute("""
                UPDATE kill_switch_history
                SET deactivated_at = NOW()
                WHERE deactivated_at IS NULL
            """)

            # Clear runtime flag
            await db.execute("""
                INSERT INTO runtime_flags (key, value, updated_at)
                VALUES ('kill_switch_active', 'false', NOW())
                ON CONFLICT (key) DO UPDATE SET value = 'false', updated_at = NOW()
            """)

            result["success"] = True
            result["method"] = "database_flag"
            result["note"] = "Kill switch cleared. Worker will resume on next check."
            return result
        except Exception as e:
            result["database_error"] = str(e)

    result["success"] = False
    result["error"] = "Could not deactivate kill switch"
    return result


@mcp.tool()
async def update_trade_size(size: float) -> dict:
    """
    Update the trade size for future trades.

    Changes the dollar amount used for each trade.
    Does NOT affect existing positions.

    Args:
        size: New trade size in dollars (e.g., 25.0 for $25 trades)

    Returns:
        Confirmation of trade size update
    """
    global db

    if size <= 0:
        return {
            "source": "trade_size_control",
            "success": False,
            "error": "Trade size must be positive"
        }

    if size > 500:
        return {
            "source": "trade_size_control",
            "success": False,
            "error": "Trade size exceeds safety limit of $500"
        }

    result = {
        "source": "trade_size_control",
        "action": "update_trade_size",
        "new_size": size,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Try to update via live safety manager
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            manager = sm_module._global_manager
            manager.trade_size = size
            result["success"] = True
            result["method"] = "live_safety_manager"
            return result
    except Exception as e:
        result["live_manager_error"] = str(e)

    # Store in database for worker to pick up
    if db:
        try:
            await db.execute("""
                INSERT INTO runtime_flags (key, value, updated_at)
                VALUES ('trade_size', $1, NOW())
                ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = NOW()
            """, str(size))

            # Also update active session if exists
            await db.execute("""
                UPDATE sessions
                SET trade_size = $1
                WHERE ended_at IS NULL
            """, size)

            result["success"] = True
            result["method"] = "database_flag"
            result["note"] = "Trade size updated. Worker will use on next trade."
            return result
        except Exception as e:
            result["database_error"] = str(e)

    result["success"] = False
    result["error"] = "Could not update trade size"
    return result


@mcp.tool()
async def set_trading_mode(mode: str) -> dict:
    """
    Switch between paper and live trading modes.

    Args:
        mode: Either 'paper' or 'live'
            - paper: Simulated trades, no real money
            - live: Real trades on Polymarket

    Returns:
        Confirmation of mode change

    IMPORTANT: Switching to live mode will use real funds!
    """
    global db

    mode = mode.lower().strip()
    if mode not in ["paper", "live"]:
        return {
            "source": "trading_mode_control",
            "success": False,
            "error": "Mode must be 'paper' or 'live'"
        }

    result = {
        "source": "trading_mode_control",
        "action": "set_trading_mode",
        "new_mode": mode,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Store in database
    if db:
        try:
            await db.execute("""
                INSERT INTO runtime_flags (key, value, updated_at)
                VALUES ('trading_mode', $1, NOW())
                ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = NOW()
            """, mode)

            result["success"] = True
            result["method"] = "database_flag"

            if mode == "live":
                result["warning"] = "LIVE MODE: Real funds will be used for trades!"
            else:
                result["note"] = "Paper mode: Trades are simulated, no real funds used."

            result["note"] = (result.get("note", "") + " Worker restart required for mode change to take effect.").strip()
            return result

        except Exception as e:
            result["database_error"] = str(e)

    result["success"] = False
    result["error"] = "Could not update trading mode - database not connected"
    return result


# ============================================================================
# SECTION 8: FLY.IO DEPLOYMENT
# Source: fly_cli
# ============================================================================

@mcp.tool()
async def get_fly_deployment_status() -> dict:
    """
    Get Fly.io deployment status for cross-market-state-fusion app.

    Shows:
    - Machine status (started/stopped)
    - Region and instance type
    - Memory and CPU allocation
    - Deployment health

    Requires: fly CLI installed and authenticated.
    """
    result = await _run_fly_command(["status"])

    if not result["success"]:
        return {
            "source": "fly_cli",
            "error": result["error"],
            "note": "Ensure 'fly' CLI is installed and you're authenticated (fly auth login)"
        }

    return {
        "source": "fly_cli",
        "app": "cross-market-state-fusion",
        "status": result["data"]
    }


@mcp.tool()
async def get_fly_logs(lines: int = 50, machine: Optional[str] = None) -> dict:
    """
    Get recent logs from Fly.io deployment (buffered, non-streaming).

    Args:
        lines: Number of most recent log lines to return (default: 50, max: 500)
        machine: Specific machine ID to filter (optional)

    Returns:
        Recent log entries from the trading worker

    Note: Uses --no-tail to fetch buffer instead of streaming.
    """
    lines = min(lines, 500)

    # Build command - use --no-tail to get batch instead of streaming
    cmd = ["fly", "logs", "-a", "cross-market-state-fusion", "--no-tail"]

    if machine:
        cmd.extend(["--machine", machine])

    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            return {
                "source": "fly_cli",
                "error": result.stderr.strip() or "Failed to fetch logs"
            }

        # Parse log lines and return the most recent ones
        all_lines = result.stdout.strip().split("\n")
        log_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

        return {
            "source": "fly_cli",
            "app": "cross-market-state-fusion",
            "total_in_buffer": len(all_lines),
            "returned_lines": len(log_lines),
            "logs": log_lines
        }

    except FileNotFoundError:
        return {"source": "fly_cli", "error": "fly CLI not installed"}
    except subprocess.TimeoutExpired:
        return {"source": "fly_cli", "error": "Log fetch timed out"}
    except Exception as e:
        return {"source": "fly_cli", "error": str(e)}


@mcp.tool()
async def get_fly_secrets_list() -> dict:
    """
    List configured secrets in Fly.io (names only, not values).

    Shows which environment variables are configured for the worker.
    Critical secrets: POLYMARKET_PRIVATE_KEY, DATABASE_URL, etc.

    Note: Values are never exposed, only secret names.
    """
    result = await _run_fly_command(["secrets", "list"])

    if not result["success"]:
        return {
            "source": "fly_cli",
            "error": result["error"]
        }

    # Parse secrets list
    secrets_data = result["data"]

    # Expected secrets for trading worker
    expected_secrets = [
        "DATABASE_URL",
        "POLYMARKET_PRIVATE_KEY",
        "POLYMARKET_FUNDER_ADDRESS",
        "POLYMARKET_SIGNATURE_TYPE",
        "POLYMARKET_API_KEY",
        "POLYMARKET_API_SECRET",
        "POLYMARKET_API_PASSPHRASE",
        "DISCORD_WEBHOOK_URL",
        "COLD_WALLET_ADDRESS",
        "POLYGON_RPC_URL",
        "RESIDENTIAL_SOCKS5_URL",
        "TRADING_MODE"
    ]

    configured = []
    if isinstance(secrets_data, list):
        configured = [s.get("Name") or s.get("name") for s in secrets_data if isinstance(s, dict)]
    elif isinstance(secrets_data, str):
        # Parse text output
        for line in secrets_data.split("\n"):
            if line.strip() and not line.startswith("NAME"):
                parts = line.split()
                if parts:
                    configured.append(parts[0])

    missing = [s for s in expected_secrets if s not in configured]

    return {
        "source": "fly_cli",
        "app": "cross-market-state-fusion",
        "configured_secrets": configured,
        "expected_secrets": expected_secrets,
        "missing_secrets": missing,
        "all_expected_configured": len(missing) == 0
    }


# ============================================================================
# SECTION 9: DISCORD ALERTS
# Source: discord_webhook
# ============================================================================

@mcp.tool()
async def get_alert_config() -> dict:
    """
    Get Discord webhook configuration status.

    Shows:
    - Whether webhook URL is configured
    - Webhook enabled/disabled state
    - Rate limiting configuration

    Does NOT expose the actual webhook URL.
    """
    webhook = _get_discord_webhook()

    if webhook is None:
        return {
            "source": "discord_webhook",
            "error": "Discord module not available",
            "enabled": False
        }

    return {
        "source": "discord_webhook",
        "enabled": webhook.enabled,
        "configured": bool(webhook.webhook_url),
        "rate_limit_delay_seconds": 0.5,
        "note": "Webhook URL is configured via DISCORD_WEBHOOK_URL env var"
    }


@mcp.tool()
async def send_test_alert(message: str = "Test alert from MCP infrastructure tools") -> dict:
    """
    Send a test Discord notification to verify webhook is working.

    Args:
        message: Custom test message (optional)

    Returns:
        Success/failure status of the test alert
    """
    webhook = _get_discord_webhook()

    if webhook is None:
        return {
            "source": "discord_webhook",
            "success": False,
            "error": "Discord module not available"
        }

    if not webhook.enabled:
        return {
            "source": "discord_webhook",
            "success": False,
            "error": "Discord webhook not configured (DISCORD_WEBHOOK_URL not set)"
        }

    try:
        # Send test embed
        embed = {
            "title": "Test Alert from Infrastructure MCP",
            "description": message,
            "color": 0x3498db,  # Blue
            "fields": [
                {"name": "Source", "value": "polymarket_infra_mcp.py", "inline": True},
                {"name": "Type", "value": "Test", "inline": True}
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        success = await webhook._send(embed)

        return {
            "source": "discord_webhook",
            "success": success,
            "message": message,
            "note": "Check your Discord channel for the test message"
        }

    except Exception as e:
        return {
            "source": "discord_webhook",
            "success": False,
            "error": str(e)
        }


# ============================================================================
# SECTION 10: SYSTEM HEALTH DASHBOARD
# Source: aggregated
# ============================================================================

@mcp.tool()
async def get_system_health() -> dict:
    """
    Get comprehensive system health dashboard aggregating all components.

    Combines:
    - Database connection status
    - Proxy health summary
    - WebSocket connection states
    - Safety system status
    - Discord alert status

    Use this for a quick overview of entire infrastructure health.
    """
    global db

    health = {
        "source": "aggregated",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "healthy",
        "components": {}
    }

    issues = []

    # 1. Database
    health["components"]["database"] = {
        "status": "connected" if db else "disconnected",
        "healthy": db is not None
    }
    if not db:
        issues.append("Database not connected")

    # 2. Proxy Health
    pm = _get_proxy_manager()
    if pm:
        proxy_health = pm.get_health_status()
        health["components"]["proxy"] = {
            "status": "healthy" if proxy_health["is_healthy"] else "degraded",
            "configured": proxy_health["proxy_configured"],
            "success_rate": proxy_health["success_rate"],
            "consecutive_failures": proxy_health["consecutive_failures"]
        }
        if not proxy_health["is_healthy"]:
            issues.append(f"Proxy unhealthy: {proxy_health['consecutive_failures']} consecutive failures")
    else:
        health["components"]["proxy"] = {"status": "unavailable", "healthy": False}
        issues.append("Proxy manager not available")

    # 3. Safety Systems
    try:
        import helpers.safety_manager as sm_module
        if hasattr(sm_module, '_global_manager') and sm_module._global_manager:
            status = await sm_module._global_manager.get_status()
            health["components"]["safety"] = {
                "status": "active" if not status["kill_switch_active"] else "kill_switch_engaged",
                "kill_switch": status["kill_switch_active"],
                "daily_pnl": status["daily_pnl"],
                "positions": status["open_positions"]
            }
            if status["kill_switch_active"]:
                issues.append(f"Kill switch active: {status.get('kill_switch_reason')}")
        else:
            health["components"]["safety"] = {"status": "not_running", "note": "Worker not active"}
    except Exception:
        health["components"]["safety"] = {"status": "unavailable"}

    # 4. Discord
    webhook = _get_discord_webhook()
    health["components"]["discord"] = {
        "status": "configured" if webhook and webhook.enabled else "not_configured",
        "enabled": webhook.enabled if webhook else False
    }
    if not webhook or not webhook.enabled:
        issues.append("Discord alerts not configured")

    # Determine overall status
    if len(issues) >= 3:
        health["overall_status"] = "critical"
    elif len(issues) >= 1:
        health["overall_status"] = "degraded"
    else:
        health["overall_status"] = "healthy"

    health["issues"] = issues
    health["issue_count"] = len(issues)

    return health


if __name__ == "__main__":
    mcp.run()
