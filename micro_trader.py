#!/usr/bin/env python3
"""
Micro Capital Polymarket Trader

Designed for small balances ($1-50). Key principles:
1. Trade binary 15-minute markets (BTC, ETH, SOL, XRP)
2. Use Binance futures as leading indicator
3. Simple momentum: if Binance moves, Polymarket follows
4. Size trades to meet 5-share minimum while preserving capital

Based on patterns from Polymarket/agents repo.
"""
import os
import sys
import time
import asyncio
import signal
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from decimal import Decimal

import requests
from dotenv import load_dotenv

# Poly CLOB client
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

# Configuration
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://fapi.binance.com"

CHAIN_ID = 137  # Polygon
ASSETS = ["btc", "eth", "sol", "xrp"]

# Trading parameters for micro capital
MIN_SHARES = 5  # Polymarket minimum
MAX_POSITION_PCT = 0.50  # Max 50% of balance per trade
MOMENTUM_THRESHOLD = 0.0005  # 0.05% move triggers signal (more sensitive)
SIGNAL_STRENGTH_MIN = 0.3  # Lower threshold for more signals
EDGE_THRESHOLD = 0.03  # 3% edge minimum


@dataclass
class Market:
    """15-min market data."""
    condition_id: str
    question: str
    asset: str
    token_up: str
    token_down: str
    end_time: datetime
    price_up: float = 0.5
    price_down: float = 0.5
    slug: str = ""


@dataclass
class Signal:
    """Trading signal."""
    asset: str
    direction: str  # "UP" or "DOWN"
    strength: float  # 0-1
    binance_return: float
    polymarket_price: float
    edge: float  # Expected edge


@dataclass
class Position:
    """Open position tracker."""
    asset: str
    token_id: str
    side: str  # "UP" or "DOWN"
    shares: float
    entry_price: float
    entry_time: datetime


class BinanceData:
    """Simple Binance price data."""

    SYMBOLS = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "sol": "SOLUSDT",
        "xrp": "XRPUSDT",
    }

    def __init__(self):
        self.prices: Dict[str, List[float]] = {a: [] for a in ASSETS}
        self.last_update = 0

    def update(self):
        """Fetch latest prices from Binance futures."""
        try:
            url = f"{BINANCE_API}/fapi/v1/ticker/price"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return

            data = {d["symbol"]: float(d["price"]) for d in resp.json()}

            for asset, symbol in self.SYMBOLS.items():
                if symbol in data:
                    self.prices[asset].append(data[symbol])
                    # Keep last 60 prices (1 minute at 1/sec)
                    self.prices[asset] = self.prices[asset][-60:]

            self.last_update = time.time()
        except Exception as e:
            print(f"[BINANCE] Error: {e}")

    def get_momentum(self, asset: str, lookback: int = 30) -> float:
        """Calculate momentum (return) over lookback period."""
        prices = self.prices.get(asset, [])
        if len(prices) < 2:
            return 0.0

        # Use available lookback if we don't have enough data
        actual_lookback = min(lookback, len(prices) - 1)
        if actual_lookback < 1:
            return 0.0

        old_price = prices[-(actual_lookback + 1)]
        new_price = prices[-1]
        return (new_price - old_price) / old_price


class MicroTrader:
    """Minimal trader for small balances."""

    def __init__(self, live: bool = False):
        self.live = live
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY not set")

        # Initialize CLOB client
        self.client = ClobClient(
            CLOB_API,
            key=self.private_key,
            chain_id=CHAIN_ID
        )
        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(self.creds)

        self.binance = BinanceData()
        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.balance: float = 0

        self.running = False
        self.trade_count = 0
        self.pnl = 0.0

    def get_balance(self) -> float:
        """Get USDC balance."""
        try:
            # Use web3 to get balance from Polygon USDC contract
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))

            USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            USDC_ABI = '[{"constant":true,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}]'

            account = w3.eth.account.from_key(self.private_key)
            usdc = w3.eth.contract(address=USDC_ADDRESS, abi=USDC_ABI)
            balance = usdc.functions.balanceOf(account.address).call()
            return balance / 1e6  # USDC has 6 decimals
        except Exception as e:
            print(f"[BALANCE] Error: {e}")
            return 0

    def discover_markets(self) -> Dict[str, Market]:
        """Find active 15-minute markets."""
        markets = {}
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        window_start = (current_ts // 900) * 900

        # Check current and next 2 windows
        timestamps = [window_start, window_start + 900, window_start + 1800]

        for asset in ASSETS:
            for ts in timestamps:
                slug = f"{asset}-updown-15m-{ts}"
                try:
                    resp = requests.get(f"{GAMMA_API}/events?slug={slug}", timeout=5)
                    if resp.status_code != 200:
                        continue

                    events = resp.json()
                    if not events:
                        continue

                    event = events[0]
                    mkt_list = event.get("markets", [])
                    if not mkt_list:
                        continue

                    mkt = mkt_list[0]
                    tokens = mkt.get("clobTokenIds", [])
                    prices = mkt.get("outcomePrices", [])

                    # Parse JSON strings if needed
                    import json
                    if isinstance(tokens, str):
                        tokens = json.loads(tokens)
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    if len(tokens) < 2:
                        continue

                    end_str = mkt.get("endDate", "")
                    if not end_str:
                        continue

                    end_time = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    remaining = (end_time - now).total_seconds()

                    # Only take markets with >2 minutes remaining
                    if remaining < 120:
                        continue

                    # Skip markets that are too far out (>20 min)
                    if remaining > 1200:
                        continue

                    market = Market(
                        condition_id=mkt.get("conditionId", ""),
                        question=mkt.get("question", ""),
                        asset=asset.upper(),
                        token_up=tokens[0],
                        token_down=tokens[1],
                        end_time=end_time,
                        price_up=float(prices[0]) if prices else 0.5,
                        price_down=float(prices[1]) if len(prices) > 1 else 0.5,
                        slug=slug,
                    )

                    # Only keep closest expiring market per asset
                    if asset not in markets or market.end_time < markets[asset].end_time:
                        markets[asset] = market

                except Exception as e:
                    print(f"[DISCOVER] Error for {slug}: {e}")
                    continue

        return markets

    def get_orderbook_prices(self, market: Market) -> tuple[float, float]:
        """Get best bid/ask from orderbook."""
        try:
            book = self.client.get_order_book(market.token_up)
            best_bid = float(book.bids[0].price) if book.bids else 0.01
            best_ask = float(book.asks[0].price) if book.asks else 0.99
            return best_bid, best_ask
        except Exception as e:
            return market.price_up, market.price_up

    def generate_signal(self, asset: str) -> Optional[Signal]:
        """Generate trading signal based on Binance momentum vs Polymarket mispricing.

        Strategy: If Binance shows momentum in one direction but Polymarket has
        extreme prices, there's potential edge.

        Example: BTC UP priced at 0.95 (market expects UP), but Binance showing
        slight down momentum - contrarian bet on DOWN at 0.05 has huge upside.
        """
        market = self.markets.get(asset.lower())
        if not market:
            return None

        # Get Binance momentum (30-second lookback)
        momentum = self.binance.get_momentum(asset.lower(), lookback=30)

        # Get Polymarket prices
        price_up = market.price_up
        price_down = market.price_down

        # Strategy 1: Follow strong momentum
        if abs(momentum) >= MOMENTUM_THRESHOLD:
            direction = "UP" if momentum > 0 else "DOWN"
            strength = min(abs(momentum) / 0.005, 1.0)  # Saturates at 0.5% move

            if direction == "UP":
                poly_price = price_up
                # Edge = how much cheaper UP is vs our implied probability
                implied_prob = 0.5 + (strength * 0.35)  # Up to 85% for strong moves
                edge = implied_prob - poly_price
            else:
                poly_price = price_down
                implied_prob = 0.5 + (strength * 0.35)
                edge = implied_prob - poly_price

            if edge > EDGE_THRESHOLD and strength >= SIGNAL_STRENGTH_MIN:
                return Signal(
                    asset=asset.upper(),
                    direction=direction,
                    strength=strength,
                    binance_return=momentum,
                    polymarket_price=poly_price,
                    edge=edge,
                )

        # Strategy 2: Value betting on extreme prices
        # If DOWN is priced very cheap (<0.15) but no strong UP momentum,
        # consider a contrarian DOWN bet for asymmetric upside
        if price_down < 0.15 and momentum < 0.001:  # Slight bearish or flat
            # Buying DOWN at 0.10 gives 10x if it wins
            direction = "DOWN"
            strength = 0.4  # Moderate confidence
            implied_prob = 0.25  # We think DOWN has ~25% chance
            edge = implied_prob - price_down

            if edge > 0.10:  # Need big edge for contrarian
                return Signal(
                    asset=asset.upper(),
                    direction=direction,
                    strength=strength,
                    binance_return=momentum,
                    polymarket_price=price_down,
                    edge=edge,
                )

        if price_up < 0.15 and momentum > -0.001:  # Slight bullish or flat
            direction = "UP"
            strength = 0.4
            implied_prob = 0.25
            edge = implied_prob - price_up

            if edge > 0.10:
                return Signal(
                    asset=asset.upper(),
                    direction=direction,
                    strength=strength,
                    binance_return=momentum,
                    polymarket_price=price_up,
                    edge=edge,
                )

        return None

    def calculate_size(self, signal: Signal) -> float:
        """Calculate position size in dollars."""
        # Available capital
        available = self.balance * MAX_POSITION_PCT

        # Price we'd pay
        price = signal.polymarket_price

        # Minimum to get 5 shares
        min_dollars = MIN_SHARES * price

        # If we can't afford minimum, skip
        if min_dollars > available:
            return 0

        # Scale by signal strength
        size = available * signal.strength

        # Ensure minimum
        size = max(size, min_dollars)

        # Cap at available
        size = min(size, available)

        return round(size, 2)

    def execute_trade(self, signal: Signal, size: float) -> bool:
        """Execute a trade on Polymarket."""
        market = self.markets.get(signal.asset.lower())
        if not market:
            return False

        token_id = market.token_up if signal.direction == "UP" else market.token_down

        print(f"\n[TRADE] {signal.asset} {signal.direction}")
        print(f"  Size: ${size:.2f} @ {signal.polymarket_price:.3f}")
        print(f"  Binance return: {signal.binance_return*100:.3f}%")
        print(f"  Edge: {signal.edge*100:.1f}%")

        if not self.live:
            print("  [PAPER MODE - no execution]")
            return True

        try:
            order = self.client.create_and_post_order(
                OrderArgs(
                    price=signal.polymarket_price,
                    size=size / signal.polymarket_price,  # Convert to shares
                    side=BUY,
                    token_id=token_id,
                )
            )
            print(f"  Order placed: {order}")
            self.trade_count += 1

            # Track position
            self.positions[signal.asset] = Position(
                asset=signal.asset,
                token_id=token_id,
                side=signal.direction,
                shares=size / signal.polymarket_price,
                entry_price=signal.polymarket_price,
                entry_time=datetime.now(timezone.utc),
            )

            return True

        except Exception as e:
            print(f"  Error: {e}")
            return False

    def run_loop(self):
        """Main trading loop."""
        print("\n" + "="*60)
        print("MICRO TRADER")
        print(f"Mode: {'LIVE' if self.live else 'PAPER'}")
        print("="*60)

        self.running = True
        last_discovery = 0
        last_signal = {a: 0 for a in ASSETS}

        while self.running:
            try:
                now = time.time()

                # Update Binance prices every second
                self.binance.update()

                # Refresh markets every 60 seconds
                if now - last_discovery > 60:
                    self.markets = self.discover_markets()
                    self.balance = self.get_balance()
                    last_discovery = now

                    print(f"\n[STATUS] Balance: ${self.balance:.2f} | Markets: {len(self.markets)} | Trades: {self.trade_count}")
                    for asset, mkt in self.markets.items():
                        remaining = (mkt.end_time - datetime.now(timezone.utc)).total_seconds()
                        print(f"  {asset.upper()}: {mkt.price_up:.2f}/{mkt.price_down:.2f} ({remaining/60:.1f}m left)")

                # Check for signals
                for asset in ASSETS:
                    if asset not in self.markets:
                        continue

                    # Cooldown between signals per asset (20 sec)
                    if now - last_signal[asset] < 20:
                        continue

                    # Skip if already have position
                    if asset.upper() in self.positions:
                        continue

                    signal = self.generate_signal(asset)
                    if signal and signal.edge > EDGE_THRESHOLD:
                        size = self.calculate_size(signal)
                        if size > 0:
                            if self.execute_trade(signal, size):
                                last_signal[asset] = now

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Stopping...")
                self.running = False
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        print(f"\n[FINAL] Trades: {self.trade_count} | PnL: ${self.pnl:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Micro Capital Polymarket Trader")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    trader = MicroTrader(live=args.live)

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        trader.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trader.run_loop()


if __name__ == "__main__":
    main()
