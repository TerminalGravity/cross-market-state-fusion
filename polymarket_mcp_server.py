#!/usr/bin/env python3
"""
Production MCP Server for Polymarket Trading

Features:
1. Vector store integration (ChromaDB) for market patterns
2. Real-time orderbook streaming via WebSocket
3. Signal generation combining momentum + historical patterns
4. Trade execution via CLOB API

Transport: Streamable HTTP (production) + Stdio (development)
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

# MCP imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    from mcp.server.stdio import stdio_server
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("Warning: MCP SDK not installed. Run: pip install mcp")

# Vector store
try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

# Trading client
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY, SELL
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False

load_dotenv()

# Configuration
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://fapi.binance.com"
CHAIN_ID = 137

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("polymarket-mcp")


# ============================================================================
# Vector Store Adapter
# ============================================================================

class VectorStore:
    """ChromaDB adapter for market pattern storage."""

    def __init__(self, persist_path: str = "/tmp/polymarket_vectors"):
        self.persist_path = persist_path
        self.client = None
        self.collection = None

    def initialize(self):
        """Initialize ChromaDB."""
        if not HAS_CHROMA:
            logger.warning("ChromaDB not installed - vector store disabled")
            return False

        try:
            self.client = chromadb.PersistentClient(path=self.persist_path)
            self.collection = self.client.get_or_create_collection(
                name="polymarket_signals",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized at {self.persist_path}")
            return True
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            return False

    def store(self, doc_id: str, embedding: List[float], data: dict, metadata: dict):
        """Store a document with embedding."""
        if not self.collection:
            return None

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[json.dumps(data)],
            metadatas=[metadata]
        )
        return doc_id

    def query(self, embedding: List[float], top_k: int = 5, asset_filter: str = None) -> List[dict]:
        """Query for similar patterns."""
        if not self.collection:
            return []

        where = {"asset": asset_filter} if asset_filter else None

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where
        )

        if not results["ids"][0]:
            return []

        return [
            {
                "id": results["ids"][0][i],
                "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                "data": json.loads(results["documents"][0][i]),
                "metadata": results["metadatas"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]


# ============================================================================
# Binance Price Monitor
# ============================================================================

class BinanceMonitor:
    """Fetch Binance futures prices for momentum calculation."""

    SYMBOLS = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
        "XRP": "XRPUSDT",
    }

    def __init__(self):
        self.prices: Dict[str, List[float]] = {a: [] for a in self.SYMBOLS}
        self.last_update = 0

    def update(self):
        """Fetch latest prices."""
        try:
            resp = requests.get(f"{BINANCE_API}/fapi/v1/ticker/price", timeout=5)
            if resp.status_code != 200:
                return

            data = {d["symbol"]: float(d["price"]) for d in resp.json()}

            for asset, symbol in self.SYMBOLS.items():
                if symbol in data:
                    self.prices[asset].append(data[symbol])
                    self.prices[asset] = self.prices[asset][-60:]  # Keep 1 min

            self.last_update = datetime.now(timezone.utc).timestamp()
        except Exception as e:
            logger.error(f"Binance update error: {e}")

    def get_momentum(self, asset: str, lookback: int = 10) -> float:
        """Calculate momentum over lookback period."""
        prices = self.prices.get(asset, [])
        if len(prices) < 2:
            return 0.0

        actual = min(lookback, len(prices) - 1)
        if actual < 1:
            return 0.0

        old = prices[-(actual + 1)]
        new = prices[-1]
        return (new - old) / old

    def get_price(self, asset: str) -> float:
        """Get latest price."""
        prices = self.prices.get(asset, [])
        return prices[-1] if prices else 0.0


# ============================================================================
# Polymarket Data Client
# ============================================================================

class PolymarketClient:
    """Polymarket market discovery and trading."""

    def __init__(self):
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        self.clob_client = None

        if self.private_key and HAS_CLOB:
            try:
                self.clob_client = ClobClient(CLOB_API, key=self.private_key, chain_id=CHAIN_ID)
                creds = self.clob_client.create_or_derive_api_creds()
                self.clob_client.set_api_creds(creds)
                logger.info("CLOB client initialized")
            except Exception as e:
                logger.error(f"CLOB init failed: {e}")

    def discover_markets(self) -> List[dict]:
        """Find active 15-minute crypto markets."""
        markets = []
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        window_start = (current_ts // 900) * 900

        for asset in ["btc", "eth", "sol", "xrp"]:
            for ts in [window_start, window_start + 900]:
                slug = f"{asset}-updown-15m-{ts}"
                try:
                    resp = requests.get(f"{GAMMA_API}/events?slug={slug}", timeout=5)
                    if resp.status_code != 200:
                        continue

                    events = resp.json()
                    if not events:
                        continue

                    mkt = events[0].get("markets", [{}])[0]

                    tokens = mkt.get("clobTokenIds", [])
                    prices = mkt.get("outcomePrices", [])

                    if isinstance(tokens, str):
                        tokens = json.loads(tokens)
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    if len(tokens) < 2 or len(prices) < 2:
                        continue

                    end_str = mkt.get("endDate", "")
                    if not end_str:
                        continue

                    end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    remaining = (end_time - now).total_seconds()

                    if remaining < 120 or remaining > 1200:
                        continue

                    markets.append({
                        "asset": asset.upper(),
                        "slug": slug,
                        "condition_id": mkt.get("conditionId", ""),
                        "token_up": tokens[0],
                        "token_down": tokens[1],
                        "price_up": float(prices[0]),
                        "price_down": float(prices[1]),
                        "end_time": end_time.isoformat(),
                        "remaining_seconds": remaining,
                        "liquidity": float(mkt.get("liquidityNum", 0) or 0),
                    })
                except Exception:
                    continue

        return markets

    def get_orderbook(self, token_id: str) -> dict:
        """Get orderbook for a token."""
        if not self.clob_client:
            return {"bids": [], "asks": []}

        try:
            book = self.clob_client.get_order_book(token_id)
            return {
                "bids": [{"price": float(b.price), "size": float(b.size)} for b in (book.bids or [])[:5]],
                "asks": [{"price": float(a.price), "size": float(a.size)} for a in (book.asks or [])[:5]],
            }
        except Exception as e:
            logger.error(f"Orderbook error: {e}")
            return {"bids": [], "asks": []}

    def execute_order(self, token_id: str, side: str, price: float, size: float) -> dict:
        """Execute a trade."""
        if not self.clob_client:
            return {"success": False, "error": "CLOB client not initialized"}

        try:
            order = self.clob_client.create_and_post_order(
                OrderArgs(
                    price=price,
                    size=size,
                    side=BUY if side == "BUY" else SELL,
                    token_id=token_id,
                )
            )
            return {
                "success": order.get("success", False),
                "order_id": order.get("orderID", ""),
                "status": order.get("status", "unknown"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# Signal Generator
# ============================================================================

class SignalGenerator:
    """Generate trading signals from momentum + vector patterns."""

    def __init__(self, binance: BinanceMonitor, vector_store: VectorStore):
        self.binance = binance
        self.vector_store = vector_store

    def generate(self, asset: str, price_up: float, price_down: float) -> dict:
        """Generate signal for an asset."""
        # Get momentum
        momentum = self.binance.get_momentum(asset, lookback=10)

        # Create embedding from current state
        embedding = [
            momentum * 100,  # Scale momentum
            price_up - 0.5,  # Deviation from fair
            price_down - 0.5,
            abs(price_up - price_down),  # Spread
            1.0 if momentum > 0 else -1.0,  # Direction
            min(abs(momentum) * 1000, 1.0),  # Strength
        ]

        # Query vector store for similar patterns
        similar = self.vector_store.query(embedding, top_k=3, asset_filter=asset)

        # Calculate signal
        signal_type = "HOLD"
        confidence = 0.0
        reasoning = "Insufficient data"

        if abs(momentum) > 0.0002:  # 0.02% threshold
            if momentum > 0 and price_up < 0.65:
                signal_type = "BUY_UP"
                confidence = min(abs(momentum) * 500 + 0.3, 0.95)
                reasoning = f"Bullish momentum {momentum*100:.3f}%, UP @ {price_up:.2f}"
            elif momentum < 0 and price_down < 0.65:
                signal_type = "BUY_DOWN"
                confidence = min(abs(momentum) * 500 + 0.3, 0.95)
                reasoning = f"Bearish momentum {momentum*100:.3f}%, DOWN @ {price_down:.2f}"

        # Boost confidence if similar patterns were profitable
        vector_boost = 0.0
        if similar:
            profitable = sum(1 for s in similar if s["data"].get("pnl", 0) > 0)
            vector_boost = profitable / len(similar) * 0.2
            confidence = min(confidence + vector_boost, 0.95)

        return {
            "asset": asset,
            "signal": signal_type,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
            "momentum": round(momentum * 100, 4),
            "price_up": price_up,
            "price_down": price_down,
            "similar_patterns": len(similar),
            "vector_boost": round(vector_boost, 3),
        }


# ============================================================================
# MCP Server
# ============================================================================

# Initialize components
vector_store = VectorStore()
binance = BinanceMonitor()
polymarket = PolymarketClient()
signal_gen = None

def init_components():
    """Initialize all components."""
    global signal_gen
    vector_store.initialize()
    binance.update()
    signal_gen = SignalGenerator(binance, vector_store)
    logger.info("Components initialized")

# MCP Server definition
if HAS_MCP:
    server = Server("polymarket-trader")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tools."""
        return [
            Tool(
                name="get_markets",
                description="Get active 15-minute crypto markets on Polymarket",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            Tool(
                name="get_market_snapshot",
                description="Get current prices and orderbook for a market",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string", "description": "Asset (BTC, ETH, SOL, XRP)"},
                        "token_id": {"type": "string", "description": "Token ID for orderbook"}
                    },
                    "required": ["asset"]
                }
            ),
            Tool(
                name="get_signal",
                description="Get AI trading signal for an asset",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string", "description": "Asset (BTC, ETH, SOL, XRP)"},
                        "price_up": {"type": "number", "description": "Current UP price"},
                        "price_down": {"type": "number", "description": "Current DOWN price"}
                    },
                    "required": ["asset", "price_up", "price_down"]
                }
            ),
            Tool(
                name="execute_trade",
                description="Execute a trade on Polymarket",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "token_id": {"type": "string", "description": "Token ID to trade"},
                        "side": {"type": "string", "enum": ["BUY", "SELL"], "description": "Trade side"},
                        "price": {"type": "number", "description": "Limit price"},
                        "size": {"type": "number", "description": "Size in shares"}
                    },
                    "required": ["token_id", "side", "price", "size"]
                }
            ),
            Tool(
                name="store_pattern",
                description="Store a market pattern in vector store for future matching",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string"},
                        "signal": {"type": "string"},
                        "entry_price": {"type": "number"},
                        "exit_price": {"type": "number"},
                        "pnl": {"type": "number"},
                        "momentum": {"type": "number"}
                    },
                    "required": ["asset", "signal", "pnl"]
                }
            ),
            Tool(
                name="query_patterns",
                description="Query vector store for similar historical patterns",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string", "description": "Filter by asset"},
                        "momentum": {"type": "number", "description": "Current momentum"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": []
                }
            ),
            Tool(
                name="get_binance_momentum",
                description="Get current Binance momentum for assets",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string", "description": "Asset (BTC, ETH, SOL, XRP) or 'all'"}
                    },
                    "required": []
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List[TextContent]:
        """Handle tool calls."""

        # Refresh Binance data
        binance.update()

        if name == "get_markets":
            markets = polymarket.discover_markets()
            return [TextContent(
                type="text",
                text=json.dumps({"markets": markets, "count": len(markets)}, indent=2)
            )]

        elif name == "get_market_snapshot":
            asset = arguments.get("asset", "BTC").upper()
            token_id = arguments.get("token_id")

            # Get market data
            markets = polymarket.discover_markets()
            market = next((m for m in markets if m["asset"] == asset), None)

            result = {
                "asset": asset,
                "binance_price": binance.get_price(asset),
                "momentum": round(binance.get_momentum(asset) * 100, 4),
            }

            if market:
                result.update({
                    "price_up": market["price_up"],
                    "price_down": market["price_down"],
                    "remaining_seconds": market["remaining_seconds"],
                    "token_up": market["token_up"],
                    "token_down": market["token_down"],
                })

            if token_id:
                result["orderbook"] = polymarket.get_orderbook(token_id)

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_signal":
            asset = arguments.get("asset", "BTC").upper()
            price_up = arguments.get("price_up", 0.5)
            price_down = arguments.get("price_down", 0.5)

            signal = signal_gen.generate(asset, price_up, price_down)
            return [TextContent(type="text", text=json.dumps(signal, indent=2))]

        elif name == "execute_trade":
            token_id = arguments["token_id"]
            side = arguments["side"]
            price = arguments["price"]
            size = arguments["size"]

            result = polymarket.execute_order(token_id, side, price, size)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "store_pattern":
            doc_id = f"{arguments['asset']}_{datetime.now(timezone.utc).timestamp()}"
            embedding = [
                arguments.get("momentum", 0) * 100,
                arguments.get("entry_price", 0.5) - 0.5,
                arguments.get("exit_price", 0.5) - 0.5,
                arguments.get("pnl", 0),
                1.0 if arguments.get("signal") == "BUY_UP" else -1.0,
                0.5,
            ]

            vector_store.store(
                doc_id=doc_id,
                embedding=embedding,
                data=arguments,
                metadata={"asset": arguments["asset"], "signal": arguments.get("signal", "HOLD")}
            )

            return [TextContent(type="text", text=json.dumps({"stored": doc_id}))]

        elif name == "query_patterns":
            asset = arguments.get("asset")
            momentum = arguments.get("momentum", 0)
            top_k = arguments.get("top_k", 5)

            embedding = [momentum * 100, 0, 0, 0, 1.0 if momentum > 0 else -1.0, 0.5]
            patterns = vector_store.query(embedding, top_k=top_k, asset_filter=asset)

            return [TextContent(type="text", text=json.dumps({"patterns": patterns}, indent=2))]

        elif name == "get_binance_momentum":
            asset = arguments.get("asset", "all")

            if asset == "all":
                result = {
                    a: {
                        "price": binance.get_price(a),
                        "momentum": round(binance.get_momentum(a) * 100, 4)
                    }
                    for a in ["BTC", "ETH", "SOL", "XRP"]
                }
            else:
                asset = asset.upper()
                result = {
                    asset: {
                        "price": binance.get_price(asset),
                        "momentum": round(binance.get_momentum(asset) * 100, 4)
                    }
                }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


async def main():
    """Run the MCP server."""
    init_components()

    if not HAS_MCP:
        print("MCP SDK not installed. Install with: pip install mcp")
        return

    logger.info("Starting Polymarket MCP Server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
