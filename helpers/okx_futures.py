#!/usr/bin/env python3
"""
OKX Futures data - datacenter-friendly alternative to Binance.

Provides: funding rate, open interest, liquidations, mark price, CVD.
"""
import asyncio
import json
import os
import requests
import websockets
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import deque

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
        """Periodically fetch REST data."""
        while self.running:
            for asset in self.assets:
                state = self.states.get(asset)
                if not state:
                    continue

                # Funding rate
                funding = fetch_okx_funding(asset)
                if funding:
                    state.funding_rate = funding["funding_rate"]

                # Mark price
                mark = fetch_okx_mark_price(asset)
                if mark:
                    state.mark_price = mark["mark_price"]

                # Ticker (volume)
                ticker = fetch_okx_ticker(asset)
                if ticker:
                    state.volume_24h = ticker["volume_24h"]

                # Open interest
                oi = fetch_okx_open_interest(asset)
                if oi:
                    state.open_interest = oi["open_interest"]
                    state.oi_history.append(oi["open_interest"])
                    if len(state.oi_history) > 60:
                        state.oi_history = state.oi_history[-60:]

                # Candles for returns and volume_1h
                candles = fetch_okx_candles(asset, "1m", 65)
                if candles:
                    returns = compute_returns_from_candles(candles)
                    state.returns_1m = returns["1m"]
                    state.returns_5m = returns["5m"]
                    state.returns_10m = returns["10m"]
                    state.returns_15m = returns["15m"]
                    state.returns_1h = returns["1h"]
                    state.realized_vol_1h = returns["realized_vol_1h"]
                    # Sum volume from last 60 candles (1 hour)
                    # OKX candle: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                    state.volume_1h = sum(float(c[6]) for c in candles[-60:])  # volCcy = base volume

                state.last_update = datetime.now(timezone.utc)

            await asyncio.sleep(10)

    async def _stream_trades(self):
        """Stream trades for CVD calculation."""
        while self.running:
            try:
                async with websockets.connect(OKX_WSS) as ws:
                    # Subscribe to trades for all instruments
                    sub_msg = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "trades", "instId": inst_id}
                            for inst_id in OKX_INSTRUMENTS.values()
                        ]
                    }
                    await ws.send(json.dumps(sub_msg))
                    print("✓ Connected to OKX trades WebSocket")

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
                print(f"OKX trade stream error: {e}")
                await asyncio.sleep(2)

    async def _stream_liquidations(self):
        """Stream liquidation orders from OKX."""
        while self.running:
            try:
                async with websockets.connect(OKX_WSS) as ws:
                    # Subscribe to liquidation orders
                    sub_msg = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "liquidation-orders", "instType": "SWAP"}
                        ]
                    }
                    await ws.send(json.dumps(sub_msg))
                    print("✓ Connected to OKX liquidations WebSocket")

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
                await asyncio.sleep(5)

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
        print("Starting OKX Futures streams...")

        # Initial data fetch
        for asset in self.assets:
            state = self.states.get(asset)
            if state:
                funding = fetch_okx_funding(asset)
                mark = fetch_okx_mark_price(asset)
                if funding and mark:
                    state.funding_rate = funding["funding_rate"]
                    state.mark_price = mark["mark_price"]
                    print(f"  {asset}: funding={state.funding_rate:.4%}, mark=${state.mark_price:,.2f}")

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
