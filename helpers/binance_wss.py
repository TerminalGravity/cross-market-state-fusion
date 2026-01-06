"""
Binance WebSocket helpers for real-time crypto price data.

Uses HiFi proxy system for zero Cloudflare blocks via residential SOCKS5.
Features REST API fallback when WebSocket is blocked.
"""
import asyncio
import json
import logging
import os
import websockets
import httpx
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional

logger = logging.getLogger(__name__)

# Try to import SOCKS proxy support
try:
    from python_socks.async_.asyncio.v2 import Proxy
    PROXY_AVAILABLE = True
except ImportError:
    PROXY_AVAILABLE = False

BINANCE_WSS = "wss://stream.binance.com:9443"
BINANCE_REST = "https://api.binance.com/api/v3"

# Import proxy URL from centralized HiFi proxy manager
from helpers.hifi_proxy import PROXY_SOCKS5_URL as PROXY_URL

# REST fallback configuration
REST_POLL_INTERVAL = 0.2  # 200ms polling when WSS unavailable
WSS_FAILURE_THRESHOLD = 5  # Switch to REST after this many WSS failures

# Asset to Binance symbol mapping
SYMBOLS = {
    "BTC": "btcusdt",
    "ETH": "ethusdt",
    "SOL": "solusdt",
    "XRP": "xrpusdt",
}


@dataclass
class PriceState:
    """Real-time price state for an asset."""
    asset: str
    price: float = 0.0
    last_update: Optional[datetime] = None
    history: List[float] = field(default_factory=list)
    max_history: int = 1000

    def update(self, price: float):
        self.price = price
        self.last_update = datetime.now(timezone.utc)
        self.history.append(price)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]


class BinanceStreamer:
    """Stream real-time prices from Binance for multiple assets.

    Automatically falls back to REST API polling when WebSocket is blocked.
    """

    def __init__(self, assets: List[str] = None):
        """
        Initialize streamer.

        Args:
            assets: List of assets to track (e.g., ["BTC", "ETH", "SOL"])
        """
        self.assets = assets or ["BTC", "ETH", "SOL"]
        self.states: Dict[str, PriceState] = {}
        self.running = False
        self.callbacks: List[Callable] = []

        # Fallback tracking
        self._wss_failures = 0
        self._using_rest_fallback = False
        self._rest_client: Optional[httpx.AsyncClient] = None

        for asset in self.assets:
            self.states[asset] = PriceState(asset=asset)

    def on_price(self, callback: Callable):
        """Register a callback for price updates."""
        self.callbacks.append(callback)

    def get_price(self, asset: str) -> float:
        """Get current price for an asset."""
        state = self.states.get(asset)
        return state.price if state else 0.0

    def get_history(self, asset: str, n: int = 100) -> List[float]:
        """Get price history for an asset."""
        state = self.states.get(asset)
        return state.history[-n:] if state else []

    async def _poll_rest_prices(self) -> None:
        """Poll prices via REST API (fallback mode).

        Called when WebSocket connection fails repeatedly.
        Polls every REST_POLL_INTERVAL seconds.
        """
        if not self._rest_client:
            self._rest_client = httpx.AsyncClient(timeout=5.0)

        symbols = [SYMBOLS[a].upper() for a in self.assets if a in SYMBOLS]
        # Use batch endpoint for efficiency
        url = f"{BINANCE_REST}/ticker/price"

        poll_count = 0
        while self.running and self._using_rest_fallback:
            try:
                resp = await self._rest_client.get(url)
                if resp.status_code == 200:
                    all_prices = resp.json()
                    price_map = {p["symbol"]: float(p["price"]) for p in all_prices}

                    for asset, sym in SYMBOLS.items():
                        if asset in self.assets:
                            price = price_map.get(sym.upper())
                            if price:
                                state = self.states.get(asset)
                                if state:
                                    state.update(price)
                                    for cb in self.callbacks:
                                        try:
                                            cb(asset, price)
                                        except:
                                            pass

                    # Log periodically (every 5 minutes = 1500 polls at 200ms)
                    poll_count += 1
                    if poll_count % 1500 == 0:
                        prices_str = ", ".join(f"{a}=${price_map.get(SYMBOLS[a].upper(), 0):,.0f}" for a in self.assets)
                        logger.info(f"[BINANCE] REST fallback active ({poll_count} polls): {prices_str}")

            except Exception as e:
                logger.debug(f"[BINANCE] REST poll error: {e}")

            await asyncio.sleep(REST_POLL_INTERVAL)

    async def _connect_ws(self, url: str):
        """Connect to WebSocket, optionally through proxy."""
        if PROXY_URL and PROXY_AVAILABLE:
            proxy = Proxy.from_url(PROXY_URL)
            sock = await proxy.connect(dest_host="stream.binance.com", dest_port=9443)
            return await websockets.connect(url, sock=sock, server_hostname="stream.binance.com")
        return await websockets.connect(url)

    async def stream(self):
        """Start streaming prices.

        Attempts WebSocket first. Falls back to REST API polling after
        WSS_FAILURE_THRESHOLD consecutive failures.
        """
        self.running = True
        self._wss_failures = 0
        self._using_rest_fallback = False
        rest_task = None

        # Build stream URL
        symbols = [SYMBOLS[a] for a in self.assets if a in SYMBOLS]
        streams = "/".join([f"{s}@trade" for s in symbols])
        url = f"{BINANCE_WSS}/stream?streams={streams}"

        proxy_status = "via proxy" if (PROXY_URL and PROXY_AVAILABLE) else "direct"
        logger.info(f"[BINANCE] Connecting to WSS for {', '.join(self.assets)} ({proxy_status})...")

        backoff = 1
        max_backoff = 120  # Max 2 minutes between retries

        while self.running:
            try:
                async with await self._connect_ws(url) as ws:
                    logger.info("[BINANCE] ✓ Connected to Binance WSS")
                    backoff = 1  # Reset on success
                    self._wss_failures = 0

                    # Stop REST fallback if WSS reconnects
                    if self._using_rest_fallback:
                        logger.info("[BINANCE] WSS recovered, stopping REST fallback")
                        self._using_rest_fallback = False
                        if rest_task:
                            rest_task.cancel()
                            rest_task = None

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            data = json.loads(msg)

                            if "data" in data:
                                trade = data["data"]
                                symbol = trade["s"].upper()
                                price = float(trade["p"])

                                # Map to asset
                                for asset, sym in SYMBOLS.items():
                                    if sym.upper() == symbol:
                                        state = self.states.get(asset)
                                        if state:
                                            state.update(price)

                                            # Call callbacks
                                            for cb in self.callbacks:
                                                try:
                                                    cb(asset, price)
                                                except:
                                                    pass
                                        break

                        except asyncio.TimeoutError:
                            pass
                        except json.JSONDecodeError:
                            pass

            except Exception as e:
                self._wss_failures += 1

                # Check if we should fall back to REST
                if self._wss_failures >= WSS_FAILURE_THRESHOLD and not self._using_rest_fallback:
                    logger.warning(
                        f"[BINANCE] WSS failed {self._wss_failures}x, "
                        f"switching to REST fallback (polling every {REST_POLL_INTERVAL}s)"
                    )
                    self._using_rest_fallback = True
                    rest_task = asyncio.create_task(self._poll_rest_prices())

                if not self._using_rest_fallback and backoff < 10:
                    logger.warning(f"[BINANCE] WSS error: {e}, retrying in {backoff}s...")

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def stop(self):
        """Stop streaming and cleanup resources."""
        self.running = False
        self._using_rest_fallback = False
        if self._rest_client:
            # Close async client (will happen on next event loop tick)
            asyncio.create_task(self._close_rest_client())

    async def _close_rest_client(self):
        """Async cleanup for REST client."""
        if self._rest_client:
            await self._rest_client.aclose()
            self._rest_client = None


async def get_current_prices(assets: List[str] = None) -> Dict[str, float]:
    """
    Get current prices for assets (one-shot, not streaming).

    Args:
        assets: List of assets (default: BTC, ETH, SOL)

    Returns:
        Dict mapping asset to price
    """
    if assets is None:
        assets = ["BTC", "ETH", "SOL"]

    prices = {}

    for asset in assets:
        symbol = SYMBOLS.get(asset)
        if not symbol:
            continue

        try:
            import requests
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                prices[asset] = float(data["price"])
        except:
            pass

    return prices


if __name__ == "__main__":
    import asyncio

    async def test():
        prices = await get_current_prices(["BTC", "ETH", "SOL", "XRP"])
        print("Current prices:")
        for asset, price in prices.items():
            print(f"  {asset}: ${price:,.2f}")

    asyncio.run(test())
