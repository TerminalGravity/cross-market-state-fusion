#!/usr/bin/env python3
"""
Polymarket Auto-Scalper

Actively trades 15-min crypto markets by buying cheap positions
and selling on price movement - NOT holding to resolution.

Strategy:
1. Buy when price < ENTRY_MAX (e.g., 15%)
2. Sell when price increases by TP_PCT (e.g., +50%)
3. Cut losses when price drops by SL_PCT (e.g., -30%)
4. Never hold to resolution - always exit before

This captures consistent small gains vs binary resolution outcomes.
"""
import os
import sys
import time
import json
import signal
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Dict, List
from decimal import Decimal

import requests
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

load_dotenv()

# Configuration
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://fapi.binance.com"
CHAIN_ID = 137

# Trading parameters
ENTRY_MAX = 0.55       # Entry up to 55% (momentum-based)
ENTRY_CHEAP = 0.35     # "Cheap" entry threshold (more aggressive)
ENTRY_MIN = 0.02       # Don't buy below 2% (too illiquid)
TP_PCT = 0.15          # Take profit at +15% gain (quicker exits)
SL_PCT = 0.10          # Stop loss at -10% loss (tighter risk)
MIN_SHARES = 5         # Polymarket minimum
MAX_POSITION_PCT = 0.50  # Max 50% of balance per trade
POSITION_TIMEOUT = 480   # Exit after 8 min regardless (before resolution)
MOMENTUM_THRESHOLD = 0.0002  # 0.02% move triggers signal (more sensitive)


@dataclass
class Position:
    """Active position being managed."""
    asset: str
    direction: str  # "UP" or "DOWN"
    token_id: str
    shares: float
    entry_price: float
    entry_time: float
    tp_price: float
    sl_price: float


@dataclass
class Market:
    """15-min market info."""
    asset: str
    slug: str
    condition_id: str
    token_up: str
    token_down: str
    price_up: float
    price_down: float
    end_time: datetime


class AutoScalper:
    """Automated scalping trader."""

    ASSETS = ["btc", "eth", "sol", "xrp"]
    BINANCE_SYMBOLS = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "sol": "SOLUSDT",
        "xrp": "XRPUSDT",
    }

    def __init__(self, live: bool = False, size: float = 0.5):
        self.live = live
        self.trade_size = size  # Dollar amount per trade
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY not set")

        self.client = ClobClient(CLOB_API, key=self.private_key, chain_id=CHAIN_ID)
        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(self.creds)

        self.markets: Dict[str, Market] = {}
        self.positions: Dict[str, Position] = {}
        self.balance = 0.0
        self.running = False

        # Stats
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0

    def get_balance(self) -> float:
        """Get USDC balance."""
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))

            USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            ABI = '[{"constant":true,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}]'

            account = w3.eth.account.from_key(self.private_key)
            contract = w3.eth.contract(address=USDC, abi=ABI)
            bal = contract.functions.balanceOf(account.address).call()
            return bal / 1e6
        except Exception as e:
            print(f"[BALANCE] Error: {e}")
            return 0

    def discover_markets(self) -> Dict[str, Market]:
        """Find active 15-minute markets."""
        markets = {}
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        window_start = (current_ts // 900) * 900

        # Check current and next windows
        timestamps = [window_start, window_start + 900]

        for asset in self.ASSETS:
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
                    mkt = event.get("markets", [])[0]

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

                    # Need >3 min to enter and exit before resolution
                    if remaining < 180 or remaining > 1200:
                        continue

                    market = Market(
                        asset=asset.upper(),
                        slug=slug,
                        condition_id=mkt.get("conditionId", ""),
                        token_up=tokens[0],
                        token_down=tokens[1],
                        price_up=float(prices[0]),
                        price_down=float(prices[1]),
                        end_time=end_time,
                    )

                    # Keep closest market per asset
                    if asset not in markets or market.end_time < markets[asset].end_time:
                        markets[asset] = market

                except Exception:
                    continue

        return markets

    def get_binance_momentum(self, asset: str) -> float:
        """Get 1-minute momentum from Binance."""
        try:
            symbol = self.BINANCE_SYMBOLS.get(asset.lower())
            if not symbol:
                return 0.0

            resp = requests.get(
                f"{BINANCE_API}/fapi/v1/klines?symbol={symbol}&interval=1m&limit=3",
                timeout=5
            )
            if resp.status_code != 200:
                return 0.0

            klines = resp.json()
            if len(klines) < 2:
                return 0.0

            old = float(klines[-2][4])
            new = float(klines[-1][4])
            return (new - old) / old

        except Exception:
            return 0.0

    def get_current_price(self, token_id: str) -> float:
        """Get current best bid for a token."""
        try:
            book = self.client.get_order_book(token_id)
            if book.bids:
                return float(book.bids[0].price)
            return 0.0
        except Exception:
            return 0.0

    def find_entry(self) -> Optional[tuple]:
        """Find an entry opportunity based on momentum or value."""
        best_entry = None
        best_score = 0

        for asset, market in self.markets.items():
            # Skip if already have position in this asset
            if asset in self.positions:
                continue

            # Check time remaining (need >3 min)
            now = datetime.now(timezone.utc)
            remaining = (market.end_time - now).total_seconds()
            if remaining < 180:
                print(f"  [{asset}] Skip: only {remaining:.0f}s remaining")
                continue

            # Get momentum
            momentum = self.get_binance_momentum(asset)

            # Debug: show entry evaluation
            if abs(momentum) > MOMENTUM_THRESHOLD * 0.5:  # Show near-threshold too
                print(f"  [{asset}] Eval: mom={momentum*100:+.4f}% thresh={MOMENTUM_THRESHOLD*100:.3f}% UP={market.price_up:.2f} DOWN={market.price_down:.2f}")

            # Strategy 1: Momentum-based entry (follow Binance)
            if abs(momentum) > MOMENTUM_THRESHOLD:
                if momentum > 0 and market.price_up <= ENTRY_MAX:
                    # Bullish momentum - buy UP
                    score = abs(momentum) * 1000 + (0.5 - market.price_up)
                    print(f"  [{asset}] 📈 SIGNAL: BUY UP @ {market.price_up:.2f} score={score:.4f}")
                    if score > best_score:
                        best_entry = (asset, "UP", market.token_up, market.price_up)
                        best_score = score
                elif momentum < 0 and market.price_down <= ENTRY_MAX:
                    # Bearish momentum - buy DOWN
                    score = abs(momentum) * 1000 + (0.5 - market.price_down)
                    print(f"  [{asset}] 📉 SIGNAL: BUY DOWN @ {market.price_down:.2f} score={score:.4f}")
                    if score > best_score:
                        best_entry = (asset, "DOWN", market.token_down, market.price_down)
                        best_score = score

            # Strategy 2: Value entries (cheap prices)
            if market.price_down <= ENTRY_CHEAP and market.price_down >= ENTRY_MIN:
                score = (ENTRY_CHEAP - market.price_down) * 10
                if momentum < 0:  # Extra points if momentum confirms
                    score += 0.5
                if score > best_score:
                    best_entry = (asset, "DOWN", market.token_down, market.price_down)
                    best_score = score

            if market.price_up <= ENTRY_CHEAP and market.price_up >= ENTRY_MIN:
                score = (ENTRY_CHEAP - market.price_up) * 10
                if momentum > 0:
                    score += 0.5
                if score > best_score:
                    best_entry = (asset, "UP", market.token_up, market.price_up)
                    best_score = score

        return best_entry

    def enter_position(self, asset: str, direction: str, token_id: str, price: float) -> bool:
        """Enter a new position."""
        shares = self.trade_size / price
        if shares < MIN_SHARES:
            return False

        tp_price = price * (1 + TP_PCT)
        sl_price = price * (1 - SL_PCT)

        print(f"\n[ENTRY] {asset} {direction} @ {price:.3f}")
        print(f"  Size: ${self.trade_size:.2f} ({shares:.1f} shares)")
        print(f"  TP: {tp_price:.3f} (+{TP_PCT*100:.0f}%)")
        print(f"  SL: {sl_price:.3f} (-{SL_PCT*100:.0f}%)")

        if not self.live:
            print("  [PAPER MODE]")
            self.positions[asset] = Position(
                asset=asset,
                direction=direction,
                token_id=token_id,
                shares=shares,
                entry_price=price,
                entry_time=time.time(),
                tp_price=tp_price,
                sl_price=sl_price,
            )
            return True

        try:
            order = self.client.create_and_post_order(
                OrderArgs(
                    price=price,
                    size=shares,
                    side=BUY,
                    token_id=token_id,
                )
            )
            print(f"  Order: {order.get('orderID', 'unknown')[:16]}...")

            if order.get("success"):
                self.positions[asset] = Position(
                    asset=asset,
                    direction=direction,
                    token_id=token_id,
                    shares=shares,
                    entry_price=price,
                    entry_time=time.time(),
                    tp_price=tp_price,
                    sl_price=sl_price,
                )
                self.trades += 1
                return True

        except Exception as e:
            print(f"  Error: {e}")

        return False

    def check_exits(self):
        """Check all positions for exit conditions."""
        now = time.time()
        to_close = []

        for asset, pos in self.positions.items():
            current_price = self.get_current_price(pos.token_id)
            if current_price <= 0:
                continue

            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
            hold_time = now - pos.entry_time

            # Check market end time
            market = self.markets.get(asset.lower())
            time_to_end = 999
            if market:
                time_to_end = (market.end_time - datetime.now(timezone.utc)).total_seconds()

            reason = None

            # Take profit
            if current_price >= pos.tp_price:
                reason = f"TP hit ({pnl_pct*100:+.1f}%)"
            # Stop loss
            elif current_price <= pos.sl_price:
                reason = f"SL hit ({pnl_pct*100:+.1f}%)"
            # Timeout - exit before resolution
            elif hold_time > POSITION_TIMEOUT:
                reason = f"Timeout ({pnl_pct*100:+.1f}%)"
            # Exit 2 min before resolution
            elif time_to_end < 120:
                reason = f"Pre-resolution exit ({pnl_pct*100:+.1f}%)"

            if reason:
                to_close.append((asset, pos, current_price, reason))

        for asset, pos, exit_price, reason in to_close:
            self.exit_position(asset, pos, exit_price, reason)

    def exit_position(self, asset: str, pos: Position, exit_price: float, reason: str):
        """Exit a position."""
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        pnl_dollars = pos.shares * (exit_price - pos.entry_price)

        print(f"\n[EXIT] {asset} {pos.direction} @ {exit_price:.3f}")
        print(f"  Reason: {reason}")
        print(f"  PnL: ${pnl_dollars:+.2f} ({pnl_pct*100:+.1f}%)")

        if self.live:
            try:
                # Sell the position
                order = self.client.create_and_post_order(
                    OrderArgs(
                        price=exit_price,
                        size=pos.shares,
                        side=SELL,
                        token_id=pos.token_id,
                    )
                )
                print(f"  Sell order: {order.get('orderID', 'unknown')[:16]}...")
            except Exception as e:
                print(f"  Sell error: {e}")

        # Update stats
        self.total_pnl += pnl_dollars
        if pnl_dollars > 0:
            self.wins += 1
        else:
            self.losses += 1

        del self.positions[asset]

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        print(f"\n{'='*60}")
        print(f"[{now.strftime('%H:%M:%S')} UTC] Balance: ${self.balance:.2f} | PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trades} | W/L: {self.wins}/{self.losses}")

        if self.positions:
            print("\nPositions:")
            for asset, pos in self.positions.items():
                current = self.get_current_price(pos.token_id)
                pnl_pct = (current - pos.entry_price) / pos.entry_price if current > 0 else 0
                hold_time = time.time() - pos.entry_time
                print(f"  {asset} {pos.direction}: entry={pos.entry_price:.3f} now={current:.3f} ({pnl_pct*100:+.1f}%) {hold_time:.0f}s")

        print("\nMarkets:")
        for asset, mkt in self.markets.items():
            remaining = (mkt.end_time - now).total_seconds()
            momentum = self.get_binance_momentum(asset)
            flag = "📍" if asset in self.positions else ""
            cheap = "🔥" if mkt.price_up < ENTRY_CHEAP or mkt.price_down < ENTRY_CHEAP else ""
            mom_icon = "📈" if momentum > MOMENTUM_THRESHOLD else ("📉" if momentum < -MOMENTUM_THRESHOLD else "➖")
            print(f"  {asset}: UP={mkt.price_up:.2f} DOWN={mkt.price_down:.2f} ({remaining/60:.1f}m) {mom_icon}{momentum*100:+.3f}% {flag}{cheap}")

    def run(self):
        """Main trading loop."""
        print("\n" + "="*60)
        print("POLYMARKET AUTO-SCALPER")
        print(f"Mode: {'LIVE' if self.live else 'PAPER'}")
        print(f"Trade size: ${self.trade_size:.2f}")
        print(f"Entry: {ENTRY_MIN*100:.0f}-{ENTRY_MAX*100:.0f}%")
        print(f"TP: +{TP_PCT*100:.0f}% | SL: -{SL_PCT*100:.0f}%")
        print("="*60)

        self.running = True
        last_discovery = 0
        last_status = 0

        while self.running:
            try:
                now = time.time()

                # Refresh markets every 30 sec
                if now - last_discovery > 30:
                    self.markets = self.discover_markets()
                    self.balance = self.get_balance()
                    last_discovery = now

                # Check exits first
                self.check_exits()

                # Look for entries
                if len(self.positions) < 2:  # Max 2 concurrent positions
                    entry = self.find_entry()
                    if entry:
                        asset, direction, token_id, price = entry
                        if self.balance >= self.trade_size:
                            self.enter_position(asset, direction, token_id, price)

                # Print status every 30 sec
                if now - last_status > 30:
                    self.print_status()
                    last_status = now

                time.sleep(2)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN]")
                self.running = False
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        # Final stats
        print(f"\n{'='*60}")
        print("FINAL STATS")
        print(f"Total PnL: ${self.total_pnl:+.2f}")
        print(f"Trades: {self.trades} | Wins: {self.wins} | Losses: {self.losses}")
        if self.trades > 0:
            print(f"Win rate: {self.wins/self.trades*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Auto-Scalper")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--size", type=float, default=0.5, help="Trade size in dollars")
    args = parser.parse_args()

    scalper = AutoScalper(live=args.live, size=args.size)

    def sig_handler(sig, frame):
        scalper.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    scalper.run()


if __name__ == "__main__":
    main()
