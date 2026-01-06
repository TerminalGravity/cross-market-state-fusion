#!/usr/bin/env python3
"""
OKX Futures data - datacenter-friendly alternative to Binance.

Provides: funding rate, open interest, liquidations, mark price, CVD.
"""
import asyncio
import json
import logging
import os
import aiohttp
import requests
import websockets
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger(__name__)

OKX_API = "https://www.okx.com"
OKX_WSS = "wss://ws.okx.com:8443/ws/v5/public"

# Asset to OKX instrument mapping (perpetual swaps)
OKX_INSTRUMENTS = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
    "XRP": "XRP-USDT-SWAP",
}


@dataclass
class OKXFuturesState:
    """Futures market state for an asset from OKX."""
    asset: str

    # Funding & Premium
    funding_rate: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0

    # Open Interest
    open_interest: float = 0.0
    oi_history: List[float] = field(default_factory=list)

    # Trade flow (CVD)
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    cvd: float = 0.0
    trade_count: int = 0

    # Trade intensity
    trade_timestamps: List[float] = field(default_factory=list)
    large_trade_threshold: float = 0.0
    large_trade_flag: float = 0.0
    recent_trade_sizes: List[float] = field(default_factory=list)

    # Liquidations
    recent_long_liqs: float = 0.0
    recent_short_liqs: float = 0.0

    # Multi-timeframe returns
    returns_1m: float = 0.0
    returns_5m: float = 0.0
    returns_10m: float = 0.0
    returns_15m: float = 0.0
    returns_1h: float = 0.0

    # Volatility
    realized_vol_1h: float = 0.0

    # Volume
    volume_24h: float = 0.0
    volume_1h: float = 0.0

    last_update: Optional[datetime] = None

    @property
    def basis(self) -> float:
        if self.index_price > 0:
            return (self.mark_price - self.index_price) / self.index_price
        return 0.0

    @property
    def oi_change_1h(self) -> float:
        if len(self.oi_history) < 2:
            return 0.0
        return (self.open_interest - self.oi_history[0]) / max(1, self.oi_history[0])

    @property
    def trade_flow_imbalance(self) -> float:
        total = self.buy_volume + self.sell_volume
        if total == 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total

    @property
    def trade_intensity(self) -> float:
        import time
        now = time.time()
        recent = [t for t in self.trade_timestamps if now - t < 10]
        return len(recent) / 10.0

    @property
    def liquidation_pressure(self) -> float:
        total = self.recent_long_liqs + self.recent_short_liqs
        if total == 0:
            return 0.0
        return (self.recent_long_liqs - self.recent_short_liqs) / total

    @property
    def vol_ratio(self) -> float:
        """Recent volume vs 24h average."""
        avg_hourly = self.volume_24h / 24 if self.volume_24h > 0 else 1
        return self.volume_1h / max(1, avg_hourly)


def fetch_okx_funding(asset: str) -> Optional[Dict]:
    """Fetch funding rate from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/public/funding-rate?instId={inst_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                item = data["data"][0]
                # Handle empty string or missing values
                funding_str = item.get("fundingRate", "0") or "0"
                next_str = item.get("nextFundingRate", "0") or "0"
                return {
                    "funding_rate": float(funding_str),
                    "next_funding_rate": float(next_str),
                }
    except Exception as e:
        print(f"OKX funding error for {asset}: {e}")
    return None


def fetch_okx_ticker(asset: str) -> Optional[Dict]:
    """Fetch ticker (mark price, index, volume) from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/market/ticker?instId={inst_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                item = data["data"][0]
                return {
                    "last_price": float(item.get("last", 0)),
                    "volume_24h": float(item.get("volCcy24h", 0)),
                }
    except Exception as e:
        print(f"OKX ticker error for {asset}: {e}")
    return None


def fetch_okx_mark_price(asset: str) -> Optional[Dict]:
    """Fetch mark and index price from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/public/mark-price?instId={inst_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                item = data["data"][0]
                return {
                    "mark_price": float(item.get("markPx", 0)),
                }
    except Exception as e:
        print(f"OKX mark price error for {asset}: {e}")
    return None


def fetch_okx_open_interest(asset: str) -> Optional[Dict]:
    """Fetch open interest from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/public/open-interest?instId={inst_id}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                item = data["data"][0]
                return {
                    "open_interest": float(item.get("oi", 0)),
                    "open_interest_value": float(item.get("oiCcy", 0)),
                }
    except Exception as e:
        print(f"OKX OI error for {asset}: {e}")
    return None


def fetch_okx_candles(asset: str, bar: str = "1m", limit: int = 100) -> Optional[List]:
    """Fetch candles from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("data"):
                # OKX returns newest first, reverse for chronological
                return list(reversed(data["data"]))
    except Exception as e:
        print(f"OKX candles error for {asset}: {e}")
    return None


def compute_returns_from_candles(candles: List) -> Dict[str, float]:
    """Compute multi-timeframe returns from 1m candles."""
    if not candles or len(candles) < 15:
        return {"1m": 0.0, "5m": 0.0, "10m": 0.0, "15m": 0.0, "1h": 0.0, "realized_vol_1h": 0.0}

    # OKX candle format: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
    current_close = float(candles[-1][4])

    def get_close(idx):
        if idx < len(candles):
            return float(candles[-(idx+1)][4])
        return current_close

    close_1m = get_close(1)
    close_5m = get_close(5)
    close_10m = get_close(10)
    close_15m = get_close(15)
    close_1h = get_close(60) if len(candles) >= 61 else current_close

    ret_1m = (current_close - close_1m) / close_1m if close_1m > 0 else 0.0
    ret_5m = (current_close - close_5m) / close_5m if close_5m > 0 else 0.0
    ret_10m = (current_close - close_10m) / close_10m if close_10m > 0 else 0.0
    ret_15m = (current_close - close_15m) / close_15m if close_15m > 0 else 0.0
    ret_1h = (current_close - close_1h) / close_1h if close_1h > 0 else 0.0

    # Realized volatility
    realized_vol_1h = 0.0
    if len(candles) >= 60:
        import numpy as np
        closes = [float(c[4]) for c in candles[-60:]]
        returns = [(closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] > 0 else 0.0
                   for i in range(1, len(closes))]
        if returns:
            realized_vol_1h = float(np.std(returns) * np.sqrt(60))

    return {
        "1m": ret_1m, "5m": ret_5m, "10m": ret_10m,
        "15m": ret_15m, "1h": ret_1h, "realized_vol_1h": realized_vol_1h
    }


# ============ ASYNC FETCH FUNCTIONS ============
# These non-blocking versions use aiohttp to avoid event loop starvation.
# The sync versions above are kept for backward compatibility.

async def async_fetch_okx_funding(session: aiohttp.ClientSession, asset: str) -> Optional[Dict]:
    """Async fetch funding rate from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/public/funding-rate?instId={inst_id}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("data"):
                    item = data["data"][0]
                    funding_str = item.get("fundingRate", "0") or "0"
                    next_str = item.get("nextFundingRate", "0") or "0"
                    return {
                        "funding_rate": float(funding_str),
                        "next_funding_rate": float(next_str),
                    }
    except asyncio.TimeoutError:
        print(f"Async timeout fetching OKX funding for {asset}")
    except Exception as e:
        print(f"Async OKX funding error for {asset}: {e}")
    return None


async def async_fetch_okx_ticker(session: aiohttp.ClientSession, asset: str) -> Optional[Dict]:
    """Async fetch ticker from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/market/ticker?instId={inst_id}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("data"):
                    item = data["data"][0]
                    return {
                        "last_price": float(item.get("last", 0)),
                        "volume_24h": float(item.get("volCcy24h", 0)),
                    }
    except asyncio.TimeoutError:
        print(f"Async timeout fetching OKX ticker for {asset}")
    except Exception as e:
        print(f"Async OKX ticker error for {asset}: {e}")
    return None


async def async_fetch_okx_mark_price(session: aiohttp.ClientSession, asset: str) -> Optional[Dict]:
    """Async fetch mark price from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/public/mark-price?instId={inst_id}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("data"):
                    item = data["data"][0]
                    return {
                        "mark_price": float(item.get("markPx", 0)),
                    }
    except asyncio.TimeoutError:
        print(f"Async timeout fetching OKX mark price for {asset}")
    except Exception as e:
        print(f"Async OKX mark price error for {asset}: {e}")
    return None


async def async_fetch_okx_open_interest(session: aiohttp.ClientSession, asset: str) -> Optional[Dict]:
    """Async fetch open interest from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/public/open-interest?instId={inst_id}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("data"):
                    item = data["data"][0]
                    return {
                        "open_interest": float(item.get("oi", 0)),
                        "open_interest_value": float(item.get("oiCcy", 0)),
                    }
    except asyncio.TimeoutError:
        print(f"Async timeout fetching OKX OI for {asset}")
    except Exception as e:
        print(f"Async OKX OI error for {asset}: {e}")
    return None


async def async_fetch_okx_candles(session: aiohttp.ClientSession, asset: str, bar: str = "1m", limit: int = 100) -> Optional[List]:
    """Async fetch candles from OKX."""
    inst_id = OKX_INSTRUMENTS.get(asset)
    if not inst_id:
        return None

    try:
        url = f"{OKX_API}/api/v5/market/candles?instId={inst_id}&bar={bar}&limit={limit}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status == 200:
                data = await resp.json()
                if data.get("data"):
                    # OKX returns newest first, reverse for chronological
                    return list(reversed(data["data"]))
    except asyncio.TimeoutError:
        print(f"Async timeout fetching OKX candles for {asset}")
    except Exception as e:
        print(f"Async OKX candles error for {asset}: {e}")
    return None


class OKXFuturesStreamer:
    """Stream futures data from OKX."""

    def __init__(self, assets: List[str] = None):
        self.assets = assets or ["BTC", "ETH", "SOL", "XRP"]
        self.states: Dict[str, OKXFuturesState] = {}
        self.running = False

        for asset in self.assets:
            self.states[asset] = OKXFuturesState(asset=asset)

    def get_state(self, asset: str) -> Optional[OKXFuturesState]:
        return self.states.get(asset)

    async def _poll_rest_data(self):
        """Periodically fetch REST data using async HTTP.

        Uses aiohttp for non-blocking HTTP requests to avoid event loop starvation.
        All requests for all assets run concurrently, completing in ~1-2 seconds
        instead of 20+ seconds with sequential sync requests.
        """
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=8)
        timeout = aiohttp.ClientTimeout(total=10, connect=5)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            logger.info("[OKX] REST polling started (async aiohttp)")

            while self.running:
                poll_start = datetime.now(timezone.utc)

                # Build list of all async tasks for all assets
                # Each task returns (type, asset, result)
                tasks = []
                for asset in self.assets:
                    tasks.append(("funding", asset, async_fetch_okx_funding(session, asset)))
                    tasks.append(("ticker", asset, async_fetch_okx_ticker(session, asset)))
                    tasks.append(("mark", asset, async_fetch_okx_mark_price(session, asset)))
                    tasks.append(("oi", asset, async_fetch_okx_open_interest(session, asset)))
                    tasks.append(("candles", asset, async_fetch_okx_candles(session, asset, "1m", 65)))

                # Run ALL requests concurrently (20 requests for 4 assets)
                results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

                poll_elapsed = (datetime.now(timezone.utc) - poll_start).total_seconds()

                # Process results
                success_count = 0
                error_count = 0
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        error_count += 1
                        continue

                    task_type, asset, _ = tasks[i]
                    state = self.states.get(asset)
                    if not state:
                        continue

                    if result is None:
                        error_count += 1
                        continue

                    success_count += 1

                    if task_type == "funding":
                        state.funding_rate = result.get("funding_rate", 0.0)
                    elif task_type == "ticker":
                        state.volume_24h = result.get("volume_24h", 0.0)
                    elif task_type == "mark":
                        state.mark_price = result.get("mark_price", 0.0)
                    elif task_type == "oi":
                        state.open_interest = result.get("open_interest", 0.0)
                        state.oi_history.append(result.get("open_interest", 0.0))
                        if len(state.oi_history) > 60:
                            state.oi_history = state.oi_history[-60:]
                    elif task_type == "candles":
                        candles = result
                        if candles:
                            returns = compute_returns_from_candles(candles)
                            state.returns_1m = returns["1m"]
                            state.returns_5m = returns["5m"]
                            state.returns_10m = returns["10m"]
                            state.returns_15m = returns["15m"]
                            state.returns_1h = returns["1h"]
                            state.realized_vol_1h = returns["realized_vol_1h"]
                            # Sum volume from last 60 candles (1 hour)
                            state.volume_1h = sum(float(c[6]) for c in candles[-60:])

                    state.last_update = datetime.now(timezone.utc)

                # Log performance periodically
                if poll_elapsed > 3.0 or success_count < len(tasks) // 2:
                    logger.info(f"[OKX] REST poll: {success_count}/{len(tasks)} in {poll_elapsed:.1f}s")

                await asyncio.sleep(10)

    async def _stream_trades(self):
        """Stream trades for CVD calculation."""
        backoff = 1  # Start with 1 second backoff
        max_backoff = 60  # Max 60 seconds between retries

        while self.running:
            try:
                async with websockets.connect(OKX_WSS) as ws:
                    backoff = 1  # Reset backoff on successful connection
                    # Subscribe to trades for all instruments
                    sub_msg = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "trades", "instId": inst_id}
                            for inst_id in OKX_INSTRUMENTS.values()
                        ]
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info("[OKX] ✓ Connected to OKX trades WebSocket")

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = json.loads(msg)

                            if data.get("data"):
                                import time
                                for trade in data["data"]:
                                    inst_id = trade.get("instId", "")
                                    price = float(trade.get("px", 0))
                                    qty = float(trade.get("sz", 0))
                                    side = trade.get("side", "")  # buy or sell
                                    trade_value = qty * price

                                    # Find asset
                                    for asset, oid in OKX_INSTRUMENTS.items():
                                        if oid == inst_id:
                                            state = self.states.get(asset)
                                            if state:
                                                if side == "buy":
                                                    state.buy_volume += trade_value
                                                else:
                                                    state.sell_volume += trade_value
                                                state.cvd = state.buy_volume - state.sell_volume
                                                state.trade_count += 1

                                                # Track timestamps
                                                now = time.time()
                                                state.trade_timestamps.append(now)
                                                state.trade_timestamps = [t for t in state.trade_timestamps if now - t < 30]

                                                # Large trade detection
                                                state.recent_trade_sizes.append(trade_value)
                                                if len(state.recent_trade_sizes) > 100:
                                                    state.recent_trade_sizes = state.recent_trade_sizes[-100:]

                                                if len(state.recent_trade_sizes) >= 20:
                                                    import numpy as np
                                                    median = np.median(state.recent_trade_sizes)
                                                    state.large_trade_threshold = median * 3
                                                    if trade_value > state.large_trade_threshold:
                                                        state.large_trade_flag = 1.0
                                            break

                        except asyncio.TimeoutError:
                            # Send ping to keep alive
                            await ws.send("ping")

            except Exception as e:
                logger.warning(f"[OKX] Trade stream error: {e}, retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)  # Exponential backoff

    async def _stream_liquidations(self):
        """Stream liquidation orders from OKX."""
        backoff = 1
        max_backoff = 60

        while self.running:
            try:
                async with websockets.connect(OKX_WSS) as ws:
                    backoff = 1
                    # Subscribe to liquidation orders
                    sub_msg = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "liquidation-orders", "instType": "SWAP"}
                        ]
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info("[OKX] ✓ Connected to OKX liquidations WebSocket")

                    while self.running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = json.loads(msg)

                            if data.get("data"):
                                for liq in data["data"]:
                                    inst_id = liq.get("instId", "")
                                    side = liq.get("side", "")  # buy = short liq, sell = long liq
                                    sz = float(liq.get("sz", 0))
                                    px = float(liq.get("bkPx", 0))
                                    value = sz * px

                                    for asset, oid in OKX_INSTRUMENTS.items():
                                        if oid == inst_id:
                                            state = self.states.get(asset)
                                            if state:
                                                if side == "sell":
                                                    state.recent_long_liqs += value
                                                else:
                                                    state.recent_short_liqs += value
                                            break

                        except asyncio.TimeoutError:
                            await ws.send("ping")

            except Exception as e:
                # Liquidations stream might not always be available
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    async def _decay_volumes(self):
        """Periodically decay counters."""
        while self.running:
            await asyncio.sleep(5)
            for asset in self.assets:
                state = self.states.get(asset)
                if state:
                    state.large_trade_flag *= 0.7

            await asyncio.sleep(55)
            for asset in self.assets:
                state = self.states.get(asset)
                if state:
                    state.buy_volume *= 0.9
                    state.sell_volume *= 0.9
                    state.recent_long_liqs *= 0.9
                    state.recent_short_liqs *= 0.9
                    state.cvd = state.buy_volume - state.sell_volume

    async def stream(self):
        """Start all OKX data streams."""
        self.running = True
        logger.info("[OKX] Starting OKX Futures streams...")

        # Initial data fetch
        for asset in self.assets:
            state = self.states.get(asset)
            if state:
                funding = fetch_okx_funding(asset)
                mark = fetch_okx_mark_price(asset)
                if funding and mark:
                    state.funding_rate = funding["funding_rate"]
                    state.mark_price = mark["mark_price"]
                    logger.info(f"[OKX] {asset}: funding={state.funding_rate:.4%}, mark=${state.mark_price:,.2f}")

        await asyncio.gather(
            self._poll_rest_data(),
            self._stream_trades(),
            self._stream_liquidations(),
            self._decay_volumes(),
        )

    def stop(self):
        self.running = False


def get_okx_snapshot(asset: str) -> Optional[OKXFuturesState]:
    """Get a snapshot of OKX futures data (non-streaming)."""
    state = OKXFuturesState(asset=asset)

    funding = fetch_okx_funding(asset)
    if funding:
        state.funding_rate = funding["funding_rate"]

    mark = fetch_okx_mark_price(asset)
    if mark:
        state.mark_price = mark["mark_price"]

    oi = fetch_okx_open_interest(asset)
    if oi:
        state.open_interest = oi["open_interest"]

    ticker = fetch_okx_ticker(asset)
    if ticker:
        state.volume_24h = ticker["volume_24h"]

    candles = fetch_okx_candles(asset, "1m", 65)
    if candles:
        returns = compute_returns_from_candles(candles)
        state.returns_1m = returns["1m"]
        state.returns_5m = returns["5m"]
        state.returns_10m = returns["10m"]
        state.returns_15m = returns["15m"]
        state.returns_1h = returns["1h"]
        state.realized_vol_1h = returns["realized_vol_1h"]

    state.last_update = datetime.now(timezone.utc)
    return state


if __name__ == "__main__":
    print("Fetching OKX futures data...")

    for asset in ["BTC", "ETH", "SOL", "XRP"]:
        state = get_okx_snapshot(asset)
        if state:
            print(f"\n{asset}:")
            print(f"  Funding: {state.funding_rate:.4%}")
            print(f"  Mark: ${state.mark_price:,.2f}")
            print(f"  OI: {state.open_interest:,.0f}")
            print(f"  Returns: 1m={state.returns_1m:.3%} 5m={state.returns_5m:.3%} 10m={state.returns_10m:.3%}")
