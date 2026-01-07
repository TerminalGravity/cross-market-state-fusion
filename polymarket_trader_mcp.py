#!/usr/bin/env python3
"""
Polymarket Trader MCP Server - Live On-Chain Data & Analytics

This is the SOURCE OF TRUTH for all live trading data. Every tool queries
real blockchain/API data, NOT internal database records.

Data Sources:
- Polymarket CLOB API (authenticated trades, orders)
- Polymarket Data API (positions, activity, redeemable)
- Polygon RPC (USDC/MATIC balances)
- Polygonscan API (transactions, token transfers)

All responses include "source" field indicating exact data origin.

Run: uv run python polymarket_trader_mcp.py
"""
import os
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, AsyncIterator
from collections import defaultdict

from fastmcp import FastMCP
from dotenv import load_dotenv

# Import clients
from helpers.polymarket_data_client import PolymarketDataClient, create_polymarket_client
from helpers.polygon_chain_client import PolygonChainClient, create_polygon_client

load_dotenv()

# Global clients (initialized in lifespan)
polymarket_client: Optional[PolymarketDataClient] = None
polygon_client: Optional[PolygonChainClient] = None

# Lazy CLOB client for analytics
_clob_client = None


def _get_clob_client():
    """Get authenticated CLOB client for analytics (lazy init)."""
    global _clob_client
    if _clob_client is not None:
        return _clob_client

    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        raise RuntimeError("py-clob-client required. Run: pip install py-clob-client")

    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
    funder_address = os.getenv("POLYMARKET_FUNDER_ADDRESS")

    if not private_key or not funder_address:
        raise ValueError("Missing POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS")

    _clob_client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=137,
        signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
        funder=funder_address
    )
    _clob_client.set_api_creds(_clob_client.create_or_derive_api_creds())
    return _clob_client


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize clients on startup, cleanup on shutdown."""
    global polymarket_client, polygon_client

    print("=" * 60)
    print("POLYMARKET TRADER MCP - Live On-Chain Data & Analytics")
    print("=" * 60)

    # Initialize Polymarket client
    polymarket_client = create_polymarket_client()
    if await polymarket_client.initialize():
        print("[OK] Polymarket client (CLOB + Data API)")
    else:
        print("[WARN] Polymarket client failed - check credentials")

    # Initialize Polygon client
    polygon_client = create_polygon_client()
    if polygon_client.is_configured():
        wallet = polygon_client.wallet_address
        print(f"[OK] Polygon client (wallet: {wallet[:8]}...{wallet[-6:]})")
    else:
        print("[WARN] Polygon client not configured")

    print("-" * 60)
    print("Ready! All data is LIVE from blockchain/APIs")
    print("=" * 60)

    yield {}

    print("Shutting down Polymarket Trader MCP...")


# Initialize server
mcp = FastMCP("Polymarket Trader", lifespan=lifespan)


# Health check endpoint for Fly.io deployment
from starlette.requests import Request
from starlette.responses import JSONResponse


@mcp.custom_route("/health", methods=["GET", "HEAD"])
async def health_check(request: Request) -> JSONResponse:
    """Health check for load balancer and monitoring."""
    return JSONResponse({"status": "healthy", "server": "polymarket-trader"})


# ============================================================================
# SECTION 1: POLYMARKET ACCOUNT DATA (CLOB + Data API)
# Source: polymarket_clob_api, polymarket_data_api
# ============================================================================

@mcp.tool()
async def get_clob_trades(limit: int = 50) -> dict:
    """
    Get authenticated trade history from Polymarket CLOB API.

    This is REAL trade data directly from the orderbook, not database records.

    Args:
        limit: Maximum trades to return (default: 50)

    Returns:
        List of trades with price, size, side, status, timestamp
    """
    global polymarket_client

    if not polymarket_client:
        return {"error": "Polymarket client not initialized", "source": "polymarket_clob_api"}

    try:
        trades = await polymarket_client.get_user_trades(limit=limit)
        return {
            "source": "polymarket_clob_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(trades),
            "trades": trades[:limit],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def get_order_status(order_id: str) -> dict:
    """
    Look up a specific order by ID from Polymarket CLOB API.

    Args:
        order_id: The order ID to look up

    Returns:
        Order details including status, fill amount, price
    """
    global polymarket_client

    if not polymarket_client:
        return {"error": "Polymarket client not initialized", "source": "polymarket_clob_api"}

    try:
        order = await polymarket_client.get_order_status(order_id)
        if not order:
            return {"error": f"Order not found: {order_id}", "source": "polymarket_clob_api"}
        return {
            "source": "polymarket_clob_api",
            "order": order,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def get_open_orders() -> dict:
    """
    Fetch current open orders from Polymarket CLOB API.

    Returns list of unfilled limit orders.
    """
    try:
        client = _get_clob_client()
        orders = client.get_orders()

        if not orders:
            return {"source": "polymarket_clob_api", "open_orders": 0, "orders": []}

        formatted = []
        for o in orders:
            formatted.append({
                "order_id": o.get("id"),
                "side": o.get("side"),
                "price": float(o.get("price", 0)),
                "original_size": float(o.get("original_size", 0)),
                "remaining_size": float(o.get("size_matched", 0)),
                "status": o.get("status"),
                "created_at": o.get("created_at")
            })

        return {
            "source": "polymarket_clob_api",
            "open_orders": len(formatted),
            "orders": formatted,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def get_open_positions() -> dict:
    """
    Get current open positions from Polymarket Data API.

    Returns list of open positions with token details and values.
    """
    global polymarket_client

    if not polymarket_client:
        return {"error": "Polymarket client not initialized", "source": "polymarket_data_api"}

    try:
        positions = await polymarket_client.get_user_positions()
        return {
            "source": "polymarket_data_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(positions),
            "positions": positions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_data_api"}


@mcp.tool()
async def get_redeemable_positions() -> dict:
    """
    Get positions that can be redeemed (resolved markets).

    These are positions in markets that have resolved and can be
    converted to USDC.
    """
    global polymarket_client

    if not polymarket_client:
        return {"error": "Polymarket client not initialized", "source": "polymarket_data_api"}

    try:
        positions = await polymarket_client.get_redeemable_positions()
        return {
            "source": "polymarket_data_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(positions),
            "redeemable": positions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_data_api"}


@mcp.tool()
async def get_activity_history(limit: int = 50) -> dict:
    """
    Get user activity history from Polymarket Data API.

    Includes trades, deposits, withdrawals, and other account activity.

    Args:
        limit: Maximum records to return (default: 50)
    """
    global polymarket_client

    if not polymarket_client:
        return {"error": "Polymarket client not initialized", "source": "polymarket_data_api"}

    try:
        activity = await polymarket_client.get_activity(limit=limit)
        return {
            "source": "polymarket_data_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(activity),
            "activity": activity,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_data_api"}


@mcp.tool()
async def get_market_info(condition_id: str) -> dict:
    """
    Get market information from Polymarket CLOB API.

    Args:
        condition_id: Market condition ID
    """
    global polymarket_client

    if not polymarket_client:
        return {"error": "Polymarket client not initialized", "source": "polymarket_clob_api"}

    try:
        market = await polymarket_client.get_market_info(condition_id)
        if not market:
            return {"error": f"Market not found: {condition_id}", "source": "polymarket_clob_api"}
        return {
            "source": "polymarket_clob_api",
            "market": market,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


# ============================================================================
# SECTION 2: POLYGON BLOCKCHAIN (On-Chain Data)
# Source: polygonscan_api, polygon_rpc
# ============================================================================

@mcp.tool()
async def get_wallet_balances() -> dict:
    """
    Get current wallet balances (USDC and MATIC) from Polygon RPC.

    This is the REAL on-chain balance, not what any system reports.
    """
    global polygon_client

    if not polygon_client:
        return {"error": "Polygon client not initialized", "source": "polygon_rpc"}

    try:
        usdc = await polygon_client.get_usdc_balance()
        matic = await polygon_client.get_matic_balance()
        return {
            "source": "polygon_rpc",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "balances": {"usdc": usdc, "matic": matic},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polygon_rpc"}


@mcp.tool()
async def get_polygon_transactions(limit: int = 20) -> dict:
    """
    Get on-chain transaction history from Polygonscan.

    Shows all transactions including contract interactions, transfers, approvals.

    Args:
        limit: Maximum transactions to return (default: 20)
    """
    global polygon_client

    if not polygon_client:
        return {"error": "Polygon client not initialized", "source": "polygonscan_api"}

    try:
        transactions = await polygon_client.get_wallet_transactions(limit=limit)
        return {
            "source": "polygonscan_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(transactions),
            "transactions": transactions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polygonscan_api"}


@mcp.tool()
async def get_usdc_transfers(limit: int = 20) -> dict:
    """
    Get USDC token transfer history from Polygonscan.

    Shows all USDC transfers in/out of the wallet.
    Useful for tracking deposits, withdrawals, and settlements.

    Args:
        limit: Maximum transfers to return (default: 20)
    """
    global polygon_client

    if not polygon_client:
        return {"error": "Polygon client not initialized", "source": "polygonscan_api"}

    try:
        transfers = await polygon_client.get_usdc_transfers(limit=limit)
        return {
            "source": "polygonscan_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(transfers),
            "transfers": transfers,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polygonscan_api"}


@mcp.tool()
async def get_internal_transactions(limit: int = 20) -> dict:
    """
    Get internal transactions (contract calls) from Polygonscan.

    Shows contract-to-contract calls involving the wallet.

    Args:
        limit: Maximum transactions to return (default: 20)
    """
    global polygon_client

    if not polygon_client:
        return {"error": "Polygon client not initialized", "source": "polygonscan_api"}

    try:
        transactions = await polygon_client.get_internal_transactions(limit=limit)
        return {
            "source": "polygonscan_api",
            "wallet": os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            "count": len(transactions),
            "internal_transactions": transactions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polygonscan_api"}


@mcp.tool()
async def get_transaction_details(tx_hash: str) -> dict:
    """
    Get detailed information about a specific transaction.

    Args:
        tx_hash: Transaction hash to look up
    """
    global polygon_client

    if not polygon_client:
        return {"error": "Polygon client not initialized", "source": "polygonscan_api"}

    try:
        details = await polygon_client.get_transaction_details(tx_hash)
        if not details:
            return {"error": f"Transaction not found: {tx_hash}", "source": "polygonscan_api"}
        return {
            "source": "polygonscan_api",
            "transaction": details,
            "polygonscan_url": f"https://polygonscan.com/tx/{tx_hash}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polygonscan_api"}


@mcp.tool()
async def get_wallet_summary() -> dict:
    """
    Get comprehensive wallet summary with balances and recent activity.

    Combines data from Polygon RPC and Polygonscan API.
    """
    global polygon_client

    if not polygon_client:
        return {"error": "Polygon client not initialized", "source": "polygon_rpc+polygonscan_api"}

    try:
        summary = await polygon_client.get_wallet_summary()
        return {
            "source": "polygon_rpc+polygonscan_api",
            **summary
        }
    except Exception as e:
        return {"error": str(e), "source": "polygon_rpc+polygonscan_api"}


# ============================================================================
# SECTION 3: TRADING ANALYTICS (Analysis of Live CLOB Data)
# Source: polymarket_clob_api (all analytics query live trades)
# ============================================================================

def _wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple:
    """Wilson score interval for binomial proportion (95% CI)."""
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (max(0, center - margin), min(1, center + margin))


@mcp.tool()
async def get_trade_summary() -> dict:
    """
    Calculate comprehensive trading statistics from LIVE CLOB data.

    Returns PnL estimates, win rate, trade counts, execution analysis.
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        if not trades:
            return {"error": "No trades found", "source": "polymarket_clob_api"}

        buy_trades = [t for t in trades if t.get("side") == "BUY"]
        sell_trades = [t for t in trades if t.get("side") == "SELL"]
        taker_trades = [t for t in trades if t.get("trader_side") == "TAKER"]
        maker_trades = [t for t in trades if t.get("trader_side") == "MAKER"]

        buy_prices = [float(t.get("price", 0)) for t in buy_trades]
        sell_prices = [float(t.get("price", 0)) for t in sell_trades]

        total_buy_size = sum(float(t.get("size", 0)) for t in buy_trades)
        total_sell_size = sum(float(t.get("size", 0)) for t in sell_trades)

        buy_volume = sum(float(t.get("price", 0)) * float(t.get("size", 0)) for t in buy_trades)
        sell_volume = sum(float(t.get("price", 0)) * float(t.get("size", 0)) for t in sell_trades)

        # Match trades for PnL
        market_trades = defaultdict(list)
        for t in trades:
            market_trades[(t.get("market"), t.get("asset_id"))].append(t)

        estimated_pnl = 0.0
        matched_trades = 0
        for (market, asset), trades_list in market_trades.items():
            buys = sorted([t for t in trades_list if t["side"] == "BUY"],
                         key=lambda x: int(x.get("match_time", 0)))
            sells = sorted([t for t in trades_list if t["side"] == "SELL"],
                          key=lambda x: int(x.get("match_time", 0)))

            for sell in sells:
                sell_time = int(sell.get("match_time", 0))
                sell_price = float(sell.get("price", 0))
                sell_size = float(sell.get("size", 0))

                for buy in reversed(buys):
                    if int(buy.get("match_time", 0)) < sell_time:
                        buy_price = float(buy.get("price", 0))
                        estimated_pnl += (sell_price - buy_price) * sell_size
                        matched_trades += 1
                        break

        unique_markets = len(set(t.get("market") for t in trades))

        timestamps = [int(t.get("match_time", 0)) for t in trades if t.get("match_time")]
        first_trade = datetime.fromtimestamp(min(timestamps), tz=timezone.utc) if timestamps else None
        last_trade = datetime.fromtimestamp(max(timestamps), tz=timezone.utc) if timestamps else None

        return {
            "source": "polymarket_clob_api",
            "summary": {
                "total_trades": len(trades),
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "taker_trades": len(taker_trades),
                "maker_trades": len(maker_trades),
                "taker_ratio": round(len(taker_trades) / len(trades), 3) if trades else 0,
                "unique_markets": unique_markets
            },
            "volume": {
                "total_buy_shares": round(total_buy_size, 2),
                "total_sell_shares": round(total_sell_size, 2),
                "buy_dollar_volume": round(buy_volume, 2),
                "sell_dollar_volume": round(sell_volume, 2),
                "net_position_shares": round(total_buy_size - total_sell_size, 2)
            },
            "pnl_estimate": {
                "matched_trade_pnl": round(estimated_pnl, 2),
                "matched_trades": matched_trades,
                "note": "PnL from matched BUY->SELL pairs. Does not include positions held to resolution."
            },
            "price_analysis": {
                "avg_buy_price": round(sum(buy_prices) / len(buy_prices), 3) if buy_prices else 0,
                "avg_sell_price": round(sum(sell_prices) / len(sell_prices), 3) if sell_prices else 0,
                "buy_price_range": [round(min(buy_prices), 2), round(max(buy_prices), 2)] if buy_prices else [0, 0],
                "sell_price_range": [round(min(sell_prices), 2), round(max(sell_prices), 2)] if sell_prices else [0, 0]
            },
            "time_range": {
                "first_trade": first_trade.isoformat() if first_trade else None,
                "last_trade": last_trade.isoformat() if last_trade else None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def calculate_expected_value() -> dict:
    """
    Calculate Expected Value by entry price bucket with statistical confidence.

    For binary markets: EV = win_rate * (1 - entry_price) - (1 - win_rate) * entry_price
    Break-even win rate = entry_price (e.g., 0.40 entry needs >40% win rate)

    Identifies which price zones have positive mathematical edge.
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        market_trades = defaultdict(list)
        for t in trades:
            if t.get("side") == "BUY":
                market_trades[(t.get("market"), t.get("asset_id"))].append(t)

        buy_results = []
        for (market, asset), buys in market_trades.items():
            all_trades = [t for t in trades if t.get("market") == market and t.get("asset_id") == asset]
            sells = sorted([t for t in all_trades if t.get("side") == "SELL"],
                          key=lambda x: int(x.get("match_time", 0)))

            for buy in buys:
                buy_price = float(buy.get("price", 0))
                buy_time = int(buy.get("match_time", 0))
                buy_size = float(buy.get("size", 0))

                exit_price = None
                for sell in sells:
                    if int(sell.get("match_time", 0)) > buy_time:
                        exit_price = float(sell.get("price", 0))
                        break

                if exit_price is not None:
                    win = exit_price > buy_price
                    pnl = (exit_price - buy_price) * buy_size
                else:
                    win = buy_price < 0.5
                    pnl = (1.0 - buy_price) * buy_size if win else -buy_price * buy_size

                buy_results.append({"entry_price": buy_price, "size": buy_size, "win": win, "pnl": pnl})

        # 5% bucket analysis
        buckets = defaultdict(lambda: {"wins": 0, "total": 0, "pnl": 0.0, "size": 0.0})
        for r in buy_results:
            bucket = int(r["entry_price"] * 20) * 5
            if bucket >= 100:
                bucket = 95
            buckets[bucket]["wins"] += 1 if r["win"] else 0
            buckets[bucket]["total"] += 1
            buckets[bucket]["pnl"] += r["pnl"]
            buckets[bucket]["size"] += r["size"]

        analysis = []
        for bucket in sorted(buckets.keys()):
            data = buckets[bucket]
            if data["total"] == 0:
                continue

            win_rate = data["wins"] / data["total"]
            entry_price = (bucket + 2.5) / 100
            ev_per_share = win_rate * (1 - entry_price) - (1 - win_rate) * entry_price
            ci_low, ci_high = _wilson_ci(data["wins"], data["total"])
            ev_low = ci_low * (1 - entry_price) - (1 - ci_low) * entry_price

            analysis.append({
                "price_range": f"{bucket}-{bucket+5}%",
                "trades": data["total"],
                "win_rate": round(win_rate, 3),
                "breakeven_rate": round(entry_price, 3),
                "ev_per_share": round(ev_per_share, 4),
                "ev_95ci": [round(ev_low, 4), round(ci_high * (1 - entry_price) - (1 - ci_high) * entry_price, 4)],
                "total_pnl": round(data["pnl"], 2),
                "edge_significant": ev_low > 0,
                "edge_status": "POSITIVE" if ev_low > 0 else ("MARGINAL" if ev_per_share > 0 else "NEGATIVE")
            })

        positive_ev = [a for a in analysis if a["ev_per_share"] > 0]
        significant = [a for a in analysis if a["edge_significant"]]

        return {
            "source": "polymarket_clob_api",
            "total_buys_analyzed": len(buy_results),
            "buckets": analysis,
            "summary": {
                "positive_ev_zones": len(positive_ev),
                "statistically_significant_zones": len(significant),
                "best_bucket": max(analysis, key=lambda x: x["ev_per_share"])["price_range"] if analysis else None,
                "worst_bucket": min(analysis, key=lambda x: x["ev_per_share"])["price_range"] if analysis else None
            },
            "key_insight": "Focus on price zones where ev_95ci[0] > 0 (statistically significant edge)",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def analyze_spread_impact(assumed_spread_pct: float = 1.5) -> dict:
    """
    Decompose execution costs to reveal hidden alpha.

    Estimates spread cost from taker executions to separate:
    - Gross Alpha (strategy edge before costs)
    - Execution Drag (spread + fees paid)
    - Net PnL (what you actually made/lost)

    Args:
        assumed_spread_pct: Estimated bid-ask spread percentage (default 1.5%)
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        taker_trades = [t for t in trades if t.get("trader_side") == "TAKER"]
        maker_trades = [t for t in trades if t.get("trader_side") == "MAKER"]

        spread_rate = assumed_spread_pct / 100
        taker_buys = [t for t in taker_trades if t.get("side") == "BUY"]
        taker_sells = [t for t in taker_trades if t.get("side") == "SELL"]

        entry_spread_cost = sum(
            float(t.get("price", 0)) * float(t.get("size", 0)) * (spread_rate / 2)
            for t in taker_buys
        )
        exit_spread_cost = sum(
            float(t.get("price", 0)) * float(t.get("size", 0)) * (spread_rate / 2)
            for t in taker_sells
        )
        total_spread_cost = entry_spread_cost + exit_spread_cost

        # Matched PnL
        market_trades = defaultdict(list)
        for t in trades:
            market_trades[(t.get("market"), t.get("asset_id"))].append(t)

        matched_pnl = 0.0
        for (market, asset), trades_list in market_trades.items():
            buys = sorted([t for t in trades_list if t["side"] == "BUY"],
                         key=lambda x: int(x.get("match_time", 0)))
            sells = sorted([t for t in trades_list if t["side"] == "SELL"],
                          key=lambda x: int(x.get("match_time", 0)))

            for sell in sells:
                sell_time = int(sell.get("match_time", 0))
                sell_price = float(sell.get("price", 0))
                sell_size = float(sell.get("size", 0))

                for buy in reversed(buys):
                    if int(buy.get("match_time", 0)) < sell_time:
                        buy_price = float(buy.get("price", 0))
                        matched_pnl += (sell_price - buy_price) * sell_size
                        break

        gross_alpha = matched_pnl + total_spread_cost
        total_volume = sum(float(t.get("price", 0)) * float(t.get("size", 0)) for t in trades)

        return {
            "source": "polymarket_clob_api",
            "cost_decomposition": {
                "net_pnl": round(matched_pnl, 2),
                "spread_cost_estimate": round(total_spread_cost, 2),
                "gross_alpha": round(gross_alpha, 2),
                "alpha_positive": gross_alpha > 0
            },
            "execution_quality": {
                "taker_trades": len(taker_trades),
                "maker_trades": len(maker_trades),
                "taker_ratio": round(len(taker_trades) / len(trades), 3) if trades else 0,
                "total_volume": round(total_volume, 2),
                "spread_as_pct_of_volume": round(total_spread_cost / total_volume * 100, 2) if total_volume > 0 else 0
            },
            "recommendation": {
                "switch_to_maker": total_spread_cost > abs(matched_pnl),
                "potential_savings": round(total_spread_cost * 0.8, 2),
                "note": "If gross_alpha > 0 but net_pnl < 0, execution costs are destroying edge."
            },
            "assumptions": {"spread_pct": assumed_spread_pct},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def get_risk_metrics() -> dict:
    """
    Calculate risk-adjusted performance metrics from LIVE trade data.

    Returns:
    - Sharpe Ratio (risk-adjusted return)
    - Sortino Ratio (downside-adjusted return)
    - Maximum Drawdown
    - Win/Loss Streaks
    - Value at Risk (95% VaR)
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        market_trades = defaultdict(list)
        for t in trades:
            market_trades[(t.get("market"), t.get("asset_id"))].append(t)

        pnl_series = []
        for (market, asset), trades_list in market_trades.items():
            buys = sorted([t for t in trades_list if t["side"] == "BUY"],
                         key=lambda x: int(x.get("match_time", 0)))
            sells = sorted([t for t in trades_list if t["side"] == "SELL"],
                          key=lambda x: int(x.get("match_time", 0)))

            for sell in sells:
                sell_time = int(sell.get("match_time", 0))
                sell_price = float(sell.get("price", 0))
                sell_size = float(sell.get("size", 0))

                for buy in reversed(buys):
                    if int(buy.get("match_time", 0)) < sell_time:
                        buy_price = float(buy.get("price", 0))
                        pnl = (sell_price - buy_price) * sell_size
                        pnl_series.append({"pnl": pnl, "time": sell_time, "win": pnl > 0})
                        break

        if len(pnl_series) < 2:
            return {"error": "Not enough matched trades for risk analysis", "source": "polymarket_clob_api"}

        pnls = [p["pnl"] for p in pnl_series]
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / len(pnls)
        std_dev = math.sqrt(variance) if variance > 0 else 0.001

        sharpe = mean_pnl / std_dev if std_dev > 0 else 0

        downside_pnls = [p for p in pnls if p < 0]
        if downside_pnls:
            downside_var = sum(p ** 2 for p in downside_pnls) / len(downside_pnls)
            downside_dev = math.sqrt(downside_var)
            sortino = mean_pnl / downside_dev if downside_dev > 0 else 0
        else:
            sortino = float('inf')

        cumulative = 0
        peak = 0
        max_drawdown = 0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        for p in pnl_series:
            if p["win"]:
                current_streak = current_streak + 1 if current_streak > 0 else 1
                max_win_streak = max(max_win_streak, current_streak)
            else:
                current_streak = current_streak - 1 if current_streak < 0 else -1
                max_loss_streak = max(max_loss_streak, abs(current_streak))

        sorted_pnls = sorted(pnls)
        var_index = int(len(sorted_pnls) * 0.05)
        var_95 = sorted_pnls[var_index] if var_index < len(sorted_pnls) else sorted_pnls[0]
        worst_5_pct = sorted_pnls[:max(1, var_index + 1)]
        expected_shortfall = sum(worst_5_pct) / len(worst_5_pct)

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)

        return {
            "source": "polymarket_clob_api",
            "performance": {
                "total_trades": len(pnls),
                "wins": wins,
                "losses": losses,
                "win_rate": round(wins / len(pnls), 3),
                "total_pnl": round(sum(pnls), 2),
                "mean_pnl": round(mean_pnl, 4),
                "std_dev": round(std_dev, 4)
            },
            "risk_adjusted": {
                "sharpe_ratio": round(sharpe, 3),
                "sortino_ratio": round(sortino, 3) if sortino != float('inf') else "inf",
                "interpretation": "Sharpe > 1 is good, > 2 is excellent"
            },
            "drawdown": {
                "max_drawdown": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown / abs(sum(pnls)) * 100, 1) if sum(pnls) != 0 else 0
            },
            "streaks": {
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
                "current_streak": current_streak
            },
            "var": {
                "var_95": round(var_95, 2),
                "expected_shortfall": round(expected_shortfall, 2),
                "interpretation": "95% VaR: 5% chance of losing more than this per trade"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def analyze_hold_duration() -> dict:
    """
    Analyze relationship between hold duration and profitability.

    For 15-min markets, determines optimal entry/exit timing.
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        market_trades = defaultdict(list)
        for t in trades:
            market_trades[(t.get("market"), t.get("asset_id"))].append(t)

        duration_data = []
        for (market, asset), trades_list in market_trades.items():
            buys = sorted([t for t in trades_list if t["side"] == "BUY"],
                         key=lambda x: int(x.get("match_time", 0)))
            sells = sorted([t for t in trades_list if t["side"] == "SELL"],
                          key=lambda x: int(x.get("match_time", 0)))

            for sell in sells:
                sell_time = int(sell.get("match_time", 0))
                sell_price = float(sell.get("price", 0))
                sell_size = float(sell.get("size", 0))

                for buy in reversed(buys):
                    buy_time = int(buy.get("match_time", 0))
                    if buy_time < sell_time:
                        buy_price = float(buy.get("price", 0))
                        duration_sec = sell_time - buy_time
                        pnl = (sell_price - buy_price) * sell_size
                        duration_data.append({
                            "duration_sec": duration_sec,
                            "duration_min": duration_sec / 60,
                            "pnl": pnl,
                            "win": pnl > 0
                        })
                        break

        if not duration_data:
            return {"error": "No matched trades for duration analysis", "source": "polymarket_clob_api"}

        buckets = {
            "0-1min": {"wins": 0, "total": 0, "pnl": 0},
            "1-3min": {"wins": 0, "total": 0, "pnl": 0},
            "3-5min": {"wins": 0, "total": 0, "pnl": 0},
            "5-10min": {"wins": 0, "total": 0, "pnl": 0},
            "10-15min": {"wins": 0, "total": 0, "pnl": 0},
            "15min+": {"wins": 0, "total": 0, "pnl": 0}
        }

        for d in duration_data:
            mins = d["duration_min"]
            if mins < 1:
                bucket = "0-1min"
            elif mins < 3:
                bucket = "1-3min"
            elif mins < 5:
                bucket = "3-5min"
            elif mins < 10:
                bucket = "5-10min"
            elif mins < 15:
                bucket = "10-15min"
            else:
                bucket = "15min+"

            buckets[bucket]["total"] += 1
            buckets[bucket]["wins"] += 1 if d["win"] else 0
            buckets[bucket]["pnl"] += d["pnl"]

        analysis = []
        for bucket_name in ["0-1min", "1-3min", "3-5min", "5-10min", "10-15min", "15min+"]:
            data = buckets[bucket_name]
            if data["total"] > 0:
                analysis.append({
                    "duration": bucket_name,
                    "trades": data["total"],
                    "win_rate": round(data["wins"] / data["total"], 3),
                    "total_pnl": round(data["pnl"], 2),
                    "avg_pnl": round(data["pnl"] / data["total"], 4)
                })

        best = max(analysis, key=lambda x: x["avg_pnl"]) if analysis else None

        return {
            "source": "polymarket_clob_api",
            "duration_analysis": analysis,
            "summary": {
                "total_matched_trades": len(duration_data),
                "avg_duration_sec": round(sum(d["duration_sec"] for d in duration_data) / len(duration_data), 1),
                "optimal_duration": best["duration"] if best else None,
                "recommendation": f"Best performance in {best['duration']} holds" if best else "Insufficient data"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def counterfactual_analysis() -> dict:
    """
    What-if analysis to test alternative strategies on historical CLOB data.

    Tests:
    1. What if we only traded 40-60% entry prices?
    2. What if we avoided contrarian bets (<20%)?
    3. What if we only traded >30% entries?
    4. What if we only traded low entries (0-30%)?
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        buy_trades = [t for t in trades if t.get("side") == "BUY"]

        scenarios = [
            ([t for t in buy_trades if 0.40 <= float(t.get("price", 0)) <= 0.60], "Only 40-60% entries"),
            ([t for t in buy_trades if float(t.get("price", 0)) >= 0.20], "Avoid contrarian (<20%)"),
            ([t for t in buy_trades if 0.30 <= float(t.get("price", 0)) <= 0.70], "Only 30-70% entries"),
            ([t for t in buy_trades if float(t.get("price", 0)) <= 0.30], "Only low entries (0-30%)")
        ]

        results = []
        for scenario_trades, name in scenarios:
            if not scenario_trades:
                results.append({"scenario": name, "trades": 0, "error": "No trades match filter"})
                continue

            total_size = sum(float(t.get("size", 0)) for t in scenario_trades)
            avg_price = sum(float(t.get("price", 0)) for t in scenario_trades) / len(scenario_trades)

            results.append({
                "scenario": name,
                "trades": len(scenario_trades),
                "pct_of_total": round(len(scenario_trades) / len(buy_trades) * 100, 1),
                "total_shares": round(total_size, 1),
                "avg_entry_price": round(avg_price, 3),
                "breakeven_win_rate": round(avg_price * 100, 1)
            })

        baseline_avg = sum(float(t.get("price", 0)) for t in buy_trades) / len(buy_trades) if buy_trades else 0

        return {
            "source": "polymarket_clob_api",
            "baseline": {
                "total_buy_trades": len(buy_trades),
                "avg_entry_price": round(baseline_avg, 3)
            },
            "scenarios": results,
            "recommendation": "Filter trades to zones with demonstrated positive EV",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


@mcp.tool()
async def get_taker_maker_analysis() -> dict:
    """
    Analyze taker vs maker execution to understand trading costs.

    Takers pay the spread, makers earn the spread. High taker ratio = higher costs.
    """
    try:
        client = _get_clob_client()
        trades = client.get_trades()

        taker_trades = [t for t in trades if t.get("trader_side") == "TAKER"]
        maker_trades = [t for t in trades if t.get("trader_side") == "MAKER"]

        taker_volume = sum(float(t.get("size", 0)) * float(t.get("price", 0)) for t in taker_trades)
        maker_volume = sum(float(t.get("size", 0)) * float(t.get("price", 0)) for t in maker_trades)

        taker_buys = [t for t in taker_trades if t.get("side") == "BUY"]
        taker_sells = [t for t in taker_trades if t.get("side") == "SELL"]
        maker_buys = [t for t in maker_trades if t.get("side") == "BUY"]
        maker_sells = [t for t in maker_trades if t.get("side") == "SELL"]

        return {
            "source": "polymarket_clob_api",
            "execution_summary": {
                "taker_trades": len(taker_trades),
                "maker_trades": len(maker_trades),
                "taker_ratio": round(len(taker_trades) / len(trades), 3) if trades else 0,
                "taker_volume": round(taker_volume, 2),
                "maker_volume": round(maker_volume, 2)
            },
            "breakdown": {
                "taker_buys": len(taker_buys),
                "taker_sells": len(taker_sells),
                "maker_buys": len(maker_buys),
                "maker_sells": len(maker_sells)
            },
            "cost_insight": {
                "note": "Takers cross the spread (~0.5-2%). Makers earn spread.",
                "recommendation": "For time-sensitive signals, taker OK. For scalping, use maker."
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


# ============================================================================
# SECTION 4: POSITION MANAGEMENT & ACTIONS
# Source: polymarket_clob_api + position_redeemer
# ============================================================================

@mcp.tool()
async def redeem_winning_positions() -> dict:
    """
    Redeem all winning positions from resolved markets.

    Converts winning outcome tokens back to USDC.
    Only positions with value > $0 are redeemed.

    Returns:
        Redemption results with tx hashes and amounts
    """
    try:
        from helpers.position_redeemer import PositionRedeemer

        rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        wallet = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")

        if not wallet or not private_key:
            return {
                "source": "position_redeemer",
                "error": "Missing POLYMARKET_FUNDER_ADDRESS or POLYMARKET_PRIVATE_KEY"
            }

        redeemer = PositionRedeemer(
            rpc_url=rpc_url,
            wallet_address=wallet,
            private_key=private_key,
            discord_webhook=discord_webhook
        )

        # Get positions first for preview
        positions = await redeemer.get_redeemable_positions()
        winning = [p for p in positions if p.current_value > 0 and p.is_winning]

        if not winning:
            return {
                "source": "position_redeemer",
                "message": "No winning positions to redeem",
                "positions_checked": len(positions)
            }

        total_value = sum(p.current_value for p in winning)

        # Execute redemptions
        results = await redeemer.redeem_all_winning()

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_redeemed = sum(r.amount_redeemed for r in successful)

        return {
            "source": "position_redeemer",
            "summary": {
                "positions_found": len(winning),
                "expected_value": round(total_value, 2),
                "successful": len(successful),
                "failed": len(failed),
                "total_redeemed": round(total_redeemed, 2)
            },
            "redemptions": [
                {
                    "condition_id": r.condition_id[:16] + "...",
                    "success": r.success,
                    "amount": round(r.amount_redeemed, 2),
                    "tx_hash": r.tx_hash,
                    "polygonscan": f"https://polygonscan.com/tx/{r.tx_hash}" if r.tx_hash else None,
                    "error": r.error
                }
                for r in results
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except ImportError:
        return {"error": "position_redeemer module not available", "source": "position_redeemer"}
    except Exception as e:
        return {"error": str(e), "source": "position_redeemer"}


@mcp.tool()
async def cleanup_dead_positions() -> dict:
    """
    Clean up dead (losing) positions worth $0.

    These positions clutter the account after losing bets.
    Calling redeemPositions removes them (returns $0 USDC).

    Returns:
        Cleanup results with count of positions removed
    """
    try:
        from helpers.position_redeemer import PositionRedeemer

        rpc_url = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
        wallet = os.getenv("POLYMARKET_FUNDER_ADDRESS")
        private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if not wallet or not private_key:
            return {
                "source": "position_redeemer",
                "error": "Missing POLYMARKET_FUNDER_ADDRESS or POLYMARKET_PRIVATE_KEY"
            }

        redeemer = PositionRedeemer(
            rpc_url=rpc_url,
            wallet_address=wallet,
            private_key=private_key
        )

        # Get positions first for preview
        positions = await redeemer.get_redeemable_positions()
        dead = [p for p in positions if p.current_value == 0 or not p.is_winning]

        if not dead:
            return {
                "source": "position_redeemer",
                "message": "No dead positions to clean up",
                "positions_checked": len(positions)
            }

        # Execute cleanup
        results = await redeemer.cleanup_dead_positions()

        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        return {
            "source": "position_redeemer",
            "summary": {
                "dead_positions_found": len(dead),
                "cleaned_up": successful,
                "failed": failed
            },
            "details": [
                {
                    "condition_id": r.condition_id[:16] + "...",
                    "success": r.success,
                    "tx_hash": r.tx_hash,
                    "error": r.error
                }
                for r in results
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except ImportError:
        return {"error": "position_redeemer module not available", "source": "position_redeemer"}
    except Exception as e:
        return {"error": str(e), "source": "position_redeemer"}


@mcp.tool()
async def cancel_all_orders() -> dict:
    """
    Cancel all open orders. Emergency action.

    Cancels all unfilled limit orders in one operation.
    Use when you need to immediately halt all pending trades.

    Returns:
        List of cancelled order IDs
    """
    try:
        client = _get_clob_client()

        # Get current open orders
        orders = client.get_orders()

        if not orders:
            return {
                "source": "polymarket_clob_api",
                "message": "No open orders to cancel",
                "cancelled": 0
            }

        # Cancel all orders
        cancelled = []
        failed = []

        for order in orders:
            order_id = order.get("id")
            try:
                client.cancel(order_id)
                cancelled.append(order_id)
            except Exception as e:
                failed.append({"order_id": order_id, "error": str(e)})

        return {
            "source": "polymarket_clob_api",
            "summary": {
                "total_orders": len(orders),
                "cancelled": len(cancelled),
                "failed": len(failed)
            },
            "cancelled_order_ids": cancelled,
            "failures": failed,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {"error": str(e), "source": "polymarket_clob_api"}


if __name__ == "__main__":
    mcp.run()
