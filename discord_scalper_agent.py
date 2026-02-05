#!/usr/bin/env python3
"""
Discord Scalper Agent

High-frequency Polymarket scalper with Discord alerts.
Watches markets like a hawk and catches price movements in real-time.

Based on polymarket-agents patterns - adapted for scalping.

Features:
1. Real-time price monitoring via WebSocket
2. Instant entry on momentum signals
3. Take-profit and stop-loss exits
4. Discord alerts for all trades
5. Focus on liquid markets only

Usage:
    python discord_scalper_agent.py --live --webhook YOUR_DISCORD_WEBHOOK_URL
"""
import os
import sys
import time
import json
import signal
import asyncio
import argparse
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from decimal import Decimal
from queue import Queue

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

# Agent Parameters - tuned for aggressive scalping
MIN_LIQUIDITY = 100       # Lower for 15-min markets (they're less liquid)
MOMENTUM_THRESHOLD = 0.0001  # 0.01% triggers signal (very sensitive)
ENTRY_CHEAP = 0.40        # Consider "cheap" below 40%
TP_PCT = 0.10             # Take profit at +10%
SL_PCT = 0.08             # Stop loss at -8%
MAX_HOLD_SECONDS = 300    # Exit after 5 min max
MIN_SHARES = 5
SCAN_INTERVAL = 1.0       # Check every 1 second


@dataclass
class AgentState:
    """Agent's current state."""
    balance: float = 0.0
    positions: Dict[str, dict] = field(default_factory=dict)
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    last_trade_time: float = 0.0


@dataclass
class MarketOpportunity:
    """Detected trading opportunity."""
    asset: str
    direction: str  # "UP" or "DOWN"
    token_id: str
    price: float
    momentum: float
    score: float
    reasoning: str


class DiscordNotifier:
    """Send alerts to Discord webhook."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.enabled = bool(self.webhook_url)
        self.queue: Queue = Queue()

        if self.enabled:
            # Start background sender thread
            self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
            self.sender_thread.start()

    def _sender_loop(self):
        """Background thread to send Discord messages."""
        while True:
            try:
                msg = self.queue.get()
                if msg is None:
                    break
                self._send(msg)
            except Exception as e:
                print(f"[DISCORD] Error: {e}")
            time.sleep(0.5)  # Rate limit

    def _send(self, content: str):
        """Send message to Discord."""
        if not self.enabled:
            return
        try:
            requests.post(
                self.webhook_url,
                json={"content": content},
                timeout=5
            )
        except Exception:
            pass

    def alert(self, msg: str, emoji: str = "🤖"):
        """Queue an alert for sending."""
        formatted = f"{emoji} **Scalper Agent** | {msg}"
        if self.enabled:
            self.queue.put(formatted)
        print(f"[DISCORD] {msg}")

    def trade_entry(self, asset: str, direction: str, price: float, size: float, reasoning: str):
        """Alert for trade entry."""
        msg = f"📥 **ENTRY** {asset} {direction} @ ${price:.3f} (${size:.2f})\n> {reasoning}"
        self.alert(msg, "📈")

    def trade_exit(self, asset: str, direction: str, entry_price: float, exit_price: float, pnl: float, reason: str):
        """Alert for trade exit."""
        emoji = "✅" if pnl > 0 else "❌"
        msg = f"📤 **EXIT** {asset} {direction}: ${entry_price:.3f} → ${exit_price:.3f} | PnL: ${pnl:+.2f} ({reason})"
        self.alert(msg, emoji)

    def status(self, balance: float, pnl: float, trades: int, win_rate: float):
        """Periodic status update."""
        msg = f"📊 **STATUS** | Balance: ${balance:.2f} | PnL: ${pnl:+.2f} | Trades: {trades} | Win: {win_rate:.0%}"
        self.alert(msg, "📊")


class BinanceMonitor:
    """Monitor Binance futures for momentum signals."""

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
            resp = requests.get(f"{BINANCE_API}/fapi/v1/ticker/price", timeout=3)
            if resp.status_code != 200:
                return

            data = {d["symbol"]: float(d["price"]) for d in resp.json()}

            for asset, symbol in self.SYMBOLS.items():
                if symbol in data:
                    self.prices[asset].append(data[symbol])
                    self.prices[asset] = self.prices[asset][-120:]  # Keep 2 min

            self.last_update = time.time()
        except Exception:
            pass

    def get_momentum(self, asset: str, lookback: int = 10) -> float:
        """Get momentum over lookback period (in seconds of data)."""
        prices = self.prices.get(asset, [])
        if len(prices) < 2:
            return 0.0

        actual = min(lookback, len(prices) - 1)
        if actual < 1:
            return 0.0

        old = prices[-(actual + 1)]
        new = prices[-1]
        return (new - old) / old


class ScalperAgent:
    """
    Autonomous scalping agent.

    Based on polymarket-agents architecture:
    - Perceive: Monitor markets and prices
    - Decide: Generate signals based on momentum
    - Act: Execute trades via CLOB
    - Learn: Track performance and adapt
    """

    def __init__(
        self,
        live: bool = False,
        trade_size: float = 0.5,
        discord_webhook: Optional[str] = None
    ):
        self.live = live
        self.trade_size = trade_size
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY not set")

        # Initialize CLOB client
        self.client = ClobClient(CLOB_API, key=self.private_key, chain_id=CHAIN_ID)
        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(self.creds)

        # Components
        self.binance = BinanceMonitor()
        self.discord = DiscordNotifier(discord_webhook)
        self.state = AgentState()

        # Market data
        self.markets: Dict[str, dict] = {}

        # Control
        self.running = False

    # =========== PERCEPTION ===========

    def perceive_markets(self):
        """Perceive: Discover and update market data."""
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        window_start = (current_ts // 900) * 900

        new_markets = {}

        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            for ts in [window_start, window_start + 900]:
                slug = f"{asset.lower()}-updown-15m-{ts}"
                try:
                    resp = requests.get(f"{GAMMA_API}/events?slug={slug}", timeout=3)
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

                    # Need > 3 min remaining
                    if remaining < 180 or remaining > 1200:
                        continue

                    liquidity = float(mkt.get("liquidityNum", 0) or 0)

                    market_data = {
                        "asset": asset,
                        "slug": slug,
                        "token_up": tokens[0],
                        "token_down": tokens[1],
                        "price_up": float(prices[0]),
                        "price_down": float(prices[1]),
                        "end_time": end_time,
                        "remaining": remaining,
                        "liquidity": liquidity,
                    }

                    # Keep closest market per asset
                    if asset not in new_markets or remaining < new_markets[asset]["remaining"]:
                        new_markets[asset] = market_data

                except Exception:
                    continue

        self.markets = new_markets

    def perceive_balance(self):
        """Perceive: Get current USDC balance."""
        try:
            from web3 import Web3
            w3 = Web3(Web3.HTTPProvider("https://polygon-rpc.com"))

            USDC = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
            ABI = '[{"constant":true,"inputs":[{"name":"account","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}]'

            account = w3.eth.account.from_key(self.private_key)
            contract = w3.eth.contract(address=USDC, abi=ABI)
            bal = contract.functions.balanceOf(account.address).call()
            self.state.balance = bal / 1e6
        except Exception:
            pass

    def perceive_prices(self):
        """Perceive: Update Binance prices."""
        self.binance.update()

    def get_current_price(self, token_id: str) -> float:
        """Get current best bid for a token."""
        try:
            book = self.client.get_order_book(token_id)
            if book.bids:
                return float(book.bids[0].price)
            return 0.0
        except Exception:
            return 0.0

    # =========== DECISION ===========

    def decide(self) -> Optional[MarketOpportunity]:
        """Decide: Analyze markets and generate best opportunity."""
        best_opp = None
        best_score = 0

        for asset, market in self.markets.items():
            # Skip if already have position
            if asset in self.state.positions:
                continue

            # Skip low liquidity (but log it)
            if market["liquidity"] < MIN_LIQUIDITY:
                print(f"  [{asset}] Skip: liquidity ${market['liquidity']:.0f} < ${MIN_LIQUIDITY}")
                continue

            # Get momentum
            momentum = self.binance.get_momentum(asset, lookback=10)

            price_up = market["price_up"]
            price_down = market["price_down"]

            # Signal logic
            if abs(momentum) > MOMENTUM_THRESHOLD:
                if momentum > 0:
                    # Bullish - buy UP if not too expensive
                    if price_up < 0.65:  # Don't pay more than 65%
                        score = abs(momentum) * 1000 + (0.5 - price_up)
                        if score > best_score:
                            best_opp = MarketOpportunity(
                                asset=asset,
                                direction="UP",
                                token_id=market["token_up"],
                                price=price_up,
                                momentum=momentum,
                                score=score,
                                reasoning=f"Binance +{momentum*100:.3f}% momentum"
                            )
                            best_score = score
                else:
                    # Bearish - buy DOWN if not too expensive
                    if price_down < 0.65:
                        score = abs(momentum) * 1000 + (0.5 - price_down)
                        if score > best_score:
                            best_opp = MarketOpportunity(
                                asset=asset,
                                direction="DOWN",
                                token_id=market["token_down"],
                                price=price_down,
                                momentum=momentum,
                                score=score,
                                reasoning=f"Binance {momentum*100:.3f}% momentum"
                            )
                            best_score = score

            # Value opportunity (cheap prices)
            if price_up <= ENTRY_CHEAP and price_up > 0.02:
                score = (ENTRY_CHEAP - price_up) * 5
                if momentum >= 0:  # Neutral or bullish
                    score += 0.3
                if score > best_score:
                    best_opp = MarketOpportunity(
                        asset=asset,
                        direction="UP",
                        token_id=market["token_up"],
                        price=price_up,
                        momentum=momentum,
                        score=score,
                        reasoning=f"Value entry at {price_up:.0%}"
                    )
                    best_score = score

            if price_down <= ENTRY_CHEAP and price_down > 0.02:
                score = (ENTRY_CHEAP - price_down) * 5
                if momentum <= 0:  # Neutral or bearish
                    score += 0.3
                if score > best_score:
                    best_opp = MarketOpportunity(
                        asset=asset,
                        direction="DOWN",
                        token_id=market["token_down"],
                        price=price_down,
                        momentum=momentum,
                        score=score,
                        reasoning=f"Value entry at {price_down:.0%}"
                    )
                    best_score = score

        return best_opp

    def decide_exits(self) -> List[tuple]:
        """Decide: Check which positions should exit."""
        exits = []
        now = time.time()

        for asset, pos in self.state.positions.items():
            current = self.get_current_price(pos["token_id"])
            if current <= 0:
                continue

            pnl_pct = (current - pos["entry_price"]) / pos["entry_price"]
            hold_time = now - pos["entry_time"]

            # Check market end time
            market = self.markets.get(asset)
            time_to_end = 999
            if market:
                time_to_end = (market["end_time"] - datetime.now(timezone.utc)).total_seconds()

            reason = None

            if current >= pos["tp_price"]:
                reason = "Take Profit"
            elif current <= pos["sl_price"]:
                reason = "Stop Loss"
            elif hold_time > MAX_HOLD_SECONDS:
                reason = "Timeout"
            elif time_to_end < 120:
                reason = "Pre-Resolution"

            if reason:
                exits.append((asset, pos, current, reason))

        return exits

    # =========== ACTION ===========

    def act_entry(self, opp: MarketOpportunity) -> bool:
        """Act: Enter a position."""
        shares = self.trade_size / opp.price
        if shares < MIN_SHARES:
            return False

        tp_price = opp.price * (1 + TP_PCT)
        sl_price = opp.price * (1 - SL_PCT)

        print(f"\n🎯 ENTRY: {opp.asset} {opp.direction} @ {opp.price:.3f}")
        print(f"   Size: ${self.trade_size:.2f} ({shares:.1f} shares)")
        print(f"   TP: {tp_price:.3f} | SL: {sl_price:.3f}")
        print(f"   Reason: {opp.reasoning}")

        self.discord.trade_entry(
            opp.asset, opp.direction, opp.price,
            self.trade_size, opp.reasoning
        )

        if not self.live:
            print("   [PAPER MODE]")
            self.state.positions[opp.asset] = {
                "direction": opp.direction,
                "token_id": opp.token_id,
                "shares": shares,
                "entry_price": opp.price,
                "entry_time": time.time(),
                "tp_price": tp_price,
                "sl_price": sl_price,
            }
            self.state.trades += 1
            self.state.last_trade_time = time.time()
            return True

        try:
            order = self.client.create_and_post_order(
                OrderArgs(
                    price=opp.price,
                    size=shares,
                    side=BUY,
                    token_id=opp.token_id,
                )
            )

            if order.get("success"):
                self.state.positions[opp.asset] = {
                    "direction": opp.direction,
                    "token_id": opp.token_id,
                    "shares": shares,
                    "entry_price": opp.price,
                    "entry_time": time.time(),
                    "tp_price": tp_price,
                    "sl_price": sl_price,
                }
                self.state.trades += 1
                self.state.last_trade_time = time.time()
                return True
        except Exception as e:
            print(f"   Error: {e}")

        return False

    def act_exit(self, asset: str, pos: dict, exit_price: float, reason: str):
        """Act: Exit a position."""
        pnl = pos["shares"] * (exit_price - pos["entry_price"])

        print(f"\n🚪 EXIT: {asset} {pos['direction']} @ {exit_price:.3f}")
        print(f"   Entry: {pos['entry_price']:.3f} | PnL: ${pnl:+.2f}")
        print(f"   Reason: {reason}")

        self.discord.trade_exit(
            asset, pos["direction"], pos["entry_price"],
            exit_price, pnl, reason
        )

        if self.live:
            try:
                self.client.create_and_post_order(
                    OrderArgs(
                        price=exit_price,
                        size=pos["shares"],
                        side=SELL,
                        token_id=pos["token_id"],
                    )
                )
            except Exception as e:
                print(f"   Sell error: {e}")

        # Update stats
        self.state.total_pnl += pnl
        if pnl > 0:
            self.state.wins += 1
        else:
            self.state.losses += 1

        del self.state.positions[asset]

    # =========== MAIN LOOP ===========

    def run(self):
        """Main agent loop."""
        print("\n" + "="*60)
        print("🤖 DISCORD SCALPER AGENT")
        print(f"Mode: {'🔴 LIVE' if self.live else '📝 PAPER'}")
        print(f"Trade Size: ${self.trade_size:.2f}")
        print(f"Discord: {'✅ Enabled' if self.discord.enabled else '❌ Disabled'}")
        print("="*60)

        self.discord.alert(
            f"Agent started in {'LIVE' if self.live else 'PAPER'} mode | Size: ${self.trade_size:.2f}",
            "🚀"
        )

        self.running = True
        last_market_refresh = 0
        last_status = 0

        while self.running:
            try:
                now = time.time()

                # Update Binance prices every cycle
                self.perceive_prices()

                # Refresh markets every 20 sec
                if now - last_market_refresh > 20:
                    self.perceive_markets()
                    self.perceive_balance()
                    last_market_refresh = now

                # Check exits first
                for exit_data in self.decide_exits():
                    asset, pos, exit_price, reason = exit_data
                    self.act_exit(asset, pos, exit_price, reason)

                # Look for entries (if we have balance and not too many positions)
                if len(self.state.positions) < 2 and self.state.balance >= self.trade_size:
                    # Cooldown between trades
                    if now - self.state.last_trade_time > 10:
                        opp = self.decide()
                        if opp and opp.score > 0.1:
                            self.act_entry(opp)

                # Status update every 60 sec
                if now - last_status > 60:
                    win_rate = self.state.wins / max(self.state.trades, 1)
                    self.print_status()
                    if self.state.trades > 0:
                        self.discord.status(
                            self.state.balance, self.state.total_pnl,
                            self.state.trades, win_rate
                        )
                    last_status = now

                time.sleep(SCAN_INTERVAL)

            except KeyboardInterrupt:
                print("\n⛔ Shutdown requested")
                self.running = False
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(5)

        # Final stats
        self.print_final()
        self.discord.alert(
            f"Agent stopped | PnL: ${self.state.total_pnl:+.2f} | W/L: {self.state.wins}/{self.state.losses}",
            "🛑"
        )

    def print_status(self):
        """Print current status."""
        now = datetime.now(timezone.utc)
        print(f"\n{'─'*60}")
        print(f"[{now.strftime('%H:%M:%S')} UTC] Bal: ${self.state.balance:.2f} | PnL: ${self.state.total_pnl:+.2f}")

        if self.state.positions:
            print("Positions:")
            for asset, pos in self.state.positions.items():
                current = self.get_current_price(pos["token_id"])
                pnl_pct = (current - pos["entry_price"]) / pos["entry_price"] if current > 0 else 0
                hold = time.time() - pos["entry_time"]
                print(f"  {asset} {pos['direction']}: {pos['entry_price']:.3f}→{current:.3f} ({pnl_pct*100:+.1f}%) {hold:.0f}s")

        print("Markets:")
        for asset, mkt in self.markets.items():
            mom = self.binance.get_momentum(asset, 10)
            pos_flag = "📍" if asset in self.state.positions else ""
            mom_icon = "📈" if mom > MOMENTUM_THRESHOLD else ("📉" if mom < -MOMENTUM_THRESHOLD else "➖")
            print(f"  {asset}: UP={mkt['price_up']:.2f} DN={mkt['price_down']:.2f} {mom_icon}{mom*100:+.3f}% {pos_flag}")

    def print_final(self):
        """Print final stats."""
        print(f"\n{'='*60}")
        print("FINAL STATS")
        print(f"PnL: ${self.state.total_pnl:+.2f}")
        print(f"Trades: {self.state.trades} | W: {self.state.wins} | L: {self.state.losses}")
        if self.state.trades > 0:
            print(f"Win Rate: {self.state.wins/self.state.trades*100:.1f}%")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Discord Scalper Agent")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--size", type=float, default=0.5, help="Trade size in dollars")
    parser.add_argument("--webhook", type=str, help="Discord webhook URL")
    args = parser.parse_args()

    agent = ScalperAgent(
        live=args.live,
        trade_size=args.size,
        discord_webhook=args.webhook
    )

    def sig_handler(sig, frame):
        agent.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    agent.run()


if __name__ == "__main__":
    main()
