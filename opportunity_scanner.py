#!/usr/bin/env python3
"""
BOMBTASTIC Opportunity Scanner

Scans ALL Polymarket markets for trading opportunities:
1. 15-minute crypto markets (with Binance edge detection)
2. Sports events (with LLM probability analysis)
3. Political/news events (with LLM forecasting)
4. Any market with pricing inefficiency

Goal: Turn $5.61 seed capital into $5555+ profit
"""
import os
import sys
import time
import json
import signal
import argparse
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from decimal import Decimal

import requests
from dotenv import load_dotenv

# Poly CLOB client
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.order_builder.constants import BUY

load_dotenv()

# Configuration
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
BINANCE_API = "https://fapi.binance.com"
CHAIN_ID = 137  # Polygon

# Trading parameters
MIN_SHARES = 5
MAX_POSITION_PCT = 0.50
MIN_LIQUIDITY = 100  # Minimum liquidity for any market
MIN_EDGE = 0.05  # 5% minimum edge for non-crypto markets


@dataclass
class Opportunity:
    """Trading opportunity found by scanner."""
    market_type: str  # "15min_crypto", "sports", "politics", "other"
    question: str
    condition_id: str
    token_id: str
    outcome: str  # "YES" or "NO" or "UP" or "DOWN"
    current_price: float
    estimated_prob: float
    edge: float  # estimated_prob - current_price
    liquidity: float
    end_date: str
    reasoning: str
    confidence: float  # 0-1


class BinanceData:
    """Binance price and momentum data."""

    SYMBOLS = {
        "btc": "BTCUSDT",
        "eth": "ETHUSDT",
        "sol": "SOLUSDT",
        "xrp": "XRPUSDT",
    }

    def __init__(self):
        self.prices: Dict[str, List[float]] = {a: [] for a in self.SYMBOLS.keys()}
        self.last_update = 0

    def update(self):
        """Fetch latest prices."""
        try:
            url = f"{BINANCE_API}/fapi/v1/ticker/price"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return

            data = {d["symbol"]: float(d["price"]) for d in resp.json()}

            for asset, symbol in self.SYMBOLS.items():
                if symbol in data:
                    self.prices[asset].append(data[symbol])
                    self.prices[asset] = self.prices[asset][-120:]  # 2 minutes

            self.last_update = time.time()
        except Exception as e:
            print(f"[BINANCE] Error: {e}")

    def get_momentum(self, asset: str, lookback: int = 30) -> float:
        """Get momentum (return) over lookback period."""
        prices = self.prices.get(asset, [])
        if len(prices) < 2:
            return 0.0

        actual_lookback = min(lookback, len(prices) - 1)
        if actual_lookback < 1:
            return 0.0

        old = prices[-(actual_lookback + 1)]
        new = prices[-1]
        return (new - old) / old

    def get_trend_strength(self, asset: str) -> Tuple[str, float]:
        """Get trend direction and strength."""
        m_30s = self.get_momentum(asset, 30)
        m_60s = self.get_momentum(asset, 60)

        if m_30s > 0.001 and m_60s > 0.0005:
            return "UP", min(m_30s / 0.005, 1.0)
        elif m_30s < -0.001 and m_60s < -0.0005:
            return "DOWN", min(abs(m_30s) / 0.005, 1.0)
        return "NEUTRAL", 0.0


class OpportunityScanner:
    """Scans Polymarket for profitable opportunities."""

    def __init__(self, live: bool = False):
        self.live = live
        self.private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

        if not self.private_key:
            raise ValueError("POLYMARKET_PRIVATE_KEY not set")

        self.client = ClobClient(
            CLOB_API,
            key=self.private_key,
            chain_id=CHAIN_ID
        )
        self.creds = self.client.create_or_derive_api_creds()
        self.client.set_api_creds(self.creds)

        self.binance = BinanceData()
        self.balance = 0
        self.positions = {}
        self.trade_count = 0
        self.pnl = 0.0
        self.running = False

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

    def fetch_all_markets(self) -> List[Dict]:
        """Fetch all active markets from Polymarket."""
        markets = []
        try:
            # Get active markets
            resp = requests.get(
                f"{GAMMA_API}/markets?closed=false&active=true&limit=500",
                timeout=10
            )
            if resp.status_code == 200:
                markets = resp.json()
        except Exception as e:
            print(f"[MARKETS] Error: {e}")
        return markets

    def scan_15min_crypto(self) -> List[Opportunity]:
        """Scan 15-minute crypto markets for momentum edge."""
        opportunities = []
        now = datetime.now(timezone.utc)
        current_ts = int(now.timestamp())
        window_start = (current_ts // 900) * 900

        assets = ["btc", "eth", "sol", "xrp"]
        timestamps = [window_start, window_start + 900]

        for asset in assets:
            trend_dir, trend_strength = self.binance.get_trend_strength(asset)
            if trend_dir == "NEUTRAL" or trend_strength < 0.3:
                continue

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

                    end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    remaining = (end - now).total_seconds()

                    if remaining < 120 or remaining > 900:
                        continue

                    price_up = float(prices[0])
                    price_down = float(prices[1])

                    # Calculate edge based on Binance momentum
                    if trend_dir == "UP":
                        # Momentum suggests UP
                        implied_prob = 0.5 + (trend_strength * 0.35)
                        edge = implied_prob - price_up
                        if edge > 0.05:
                            opportunities.append(Opportunity(
                                market_type="15min_crypto",
                                question=mkt.get("question", f"{asset.upper()} UP"),
                                condition_id=mkt.get("conditionId", ""),
                                token_id=tokens[0],
                                outcome="UP",
                                current_price=price_up,
                                estimated_prob=implied_prob,
                                edge=edge,
                                liquidity=float(mkt.get("liquidityNum", 0) or 0),
                                end_date=end_str,
                                reasoning=f"Binance {trend_dir} momentum {trend_strength:.1%}",
                                confidence=trend_strength
                            ))
                    else:
                        # Momentum suggests DOWN
                        implied_prob = 0.5 + (trend_strength * 0.35)
                        edge = implied_prob - price_down
                        if edge > 0.05:
                            opportunities.append(Opportunity(
                                market_type="15min_crypto",
                                question=mkt.get("question", f"{asset.upper()} DOWN"),
                                condition_id=mkt.get("conditionId", ""),
                                token_id=tokens[1],
                                outcome="DOWN",
                                current_price=price_down,
                                estimated_prob=implied_prob,
                                edge=edge,
                                liquidity=float(mkt.get("liquidityNum", 0) or 0),
                                end_date=end_str,
                                reasoning=f"Binance {trend_dir} momentum {trend_strength:.1%}",
                                confidence=trend_strength
                            ))

                except Exception as e:
                    continue

        return opportunities

    def scan_sports_markets(self) -> List[Opportunity]:
        """Scan sports markets for value bets."""
        opportunities = []
        now = datetime.now(timezone.utc)

        # College Football Playoff - semifinals tomorrow!
        cfp_teams = {
            "indiana": {"prob": 0.55, "reasoning": "#1 seed, beat Oregon H2H, crushed Alabama 38-3"},
            "oregon": {"prob": 0.25, "reasoning": "#5 seed, lost to Indiana 30-20 in regular season"},
            "miami": {"prob": 0.15, "reasoning": "#10 seed, upset Ohio State but unlikely to win it all"},
            "ole miss": {"prob": 0.05, "reasoning": "#6 seed, beat Georgia but facing tough path"},
        }

        markets = self.fetch_all_markets()

        for m in markets:
            q = m.get("question", "").lower()
            slug = m.get("slug", "").lower()

            # Skip 15-min markets (handled separately)
            if "updown-15m" in slug:
                continue

            # Check College Football markets
            for team, data in cfp_teams.items():
                if team in q and "championship" in q and "football" in q:
                    prices = m.get("outcomePrices", [])
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except:
                            continue

                    if not prices:
                        continue

                    tokens = m.get("clobTokenIds", [])
                    if isinstance(tokens, str):
                        try:
                            tokens = json.loads(tokens)
                        except:
                            continue

                    yes_price = float(prices[0]) if prices else 0.5
                    our_prob = data["prob"]
                    edge = our_prob - yes_price

                    if edge > MIN_EDGE and tokens:
                        opportunities.append(Opportunity(
                            market_type="sports",
                            question=m.get("question", ""),
                            condition_id=m.get("conditionId", ""),
                            token_id=tokens[0],
                            outcome="YES",
                            current_price=yes_price,
                            estimated_prob=our_prob,
                            edge=edge,
                            liquidity=float(m.get("liquidityNum", 0) or 0),
                            end_date=m.get("endDate", ""),
                            reasoning=data["reasoning"],
                            confidence=0.7  # Sports analysis confidence
                        ))

        return opportunities

    def scan_all(self) -> List[Opportunity]:
        """Scan all market types and return sorted opportunities."""
        all_opps = []

        # 15-minute crypto
        crypto_opps = self.scan_15min_crypto()
        all_opps.extend(crypto_opps)

        # Sports
        sports_opps = self.scan_sports_markets()
        all_opps.extend(sports_opps)

        # Sort by edge * confidence (expected value)
        all_opps.sort(key=lambda x: -x.edge * x.confidence)

        return all_opps

    def calculate_size(self, opp: Opportunity) -> float:
        """Calculate position size based on Kelly criterion."""
        available = self.balance * MAX_POSITION_PCT

        # Kelly: f = (p * b - q) / b where b = (1/price) - 1
        p = opp.estimated_prob
        q = 1 - p
        b = (1 / opp.current_price) - 1 if opp.current_price > 0 else 0

        if b <= 0:
            return 0

        kelly = (p * b - q) / b
        kelly = max(0, min(kelly, 0.25))  # Cap at 25% Kelly

        # Minimum to get 5 shares
        min_dollars = MIN_SHARES * opp.current_price

        # Scale by confidence
        size = available * kelly * opp.confidence
        size = max(size, min_dollars) if size > 0 else 0
        size = min(size, available)

        return round(size, 2)

    def execute_trade(self, opp: Opportunity, size: float) -> bool:
        """Execute a trade."""
        print(f"\n[TRADE] {opp.market_type.upper()}: {opp.outcome}")
        print(f"  Q: {opp.question[:60]}...")
        print(f"  Size: ${size:.2f} @ {opp.current_price:.3f}")
        print(f"  Edge: {opp.edge*100:.1f}% | Est. prob: {opp.estimated_prob*100:.0f}%")
        print(f"  Reasoning: {opp.reasoning}")

        if not self.live:
            print("  [PAPER MODE]")
            return True

        try:
            order = self.client.create_and_post_order(
                OrderArgs(
                    price=opp.current_price,
                    size=size / opp.current_price,
                    side=BUY,
                    token_id=opp.token_id,
                )
            )
            print(f"  Order: {order}")
            self.trade_count += 1
            return order.get("success", False)
        except Exception as e:
            print(f"  Error: {e}")
            return False

    def run(self):
        """Main scanning loop."""
        print("\n" + "="*60)
        print("BOMBTASTIC OPPORTUNITY SCANNER")
        print(f"Mode: {'LIVE' if self.live else 'PAPER'}")
        print(f"Goal: $5.61 → $5555")
        print("="*60)

        self.running = True
        last_scan = 0

        while self.running:
            try:
                now = time.time()

                # Update Binance data continuously
                self.binance.update()

                # Full scan every 30 seconds
                if now - last_scan > 30:
                    self.balance = self.get_balance()
                    print(f"\n[SCAN] Balance: ${self.balance:.2f} | Trades: {self.trade_count}")

                    opportunities = self.scan_all()

                    if opportunities:
                        print(f"\n[OPPORTUNITIES] Found {len(opportunities)}:")
                        for i, opp in enumerate(opportunities[:5]):
                            print(f"  {i+1}. {opp.market_type}: {opp.outcome} @ {opp.current_price:.2f}")
                            print(f"     Edge: {opp.edge*100:.1f}% | {opp.question[:50]}...")

                        # Execute best opportunity if edge is good
                        best = opportunities[0]
                        if best.edge > MIN_EDGE and best.liquidity > MIN_LIQUIDITY:
                            size = self.calculate_size(best)
                            if size > 0:
                                self.execute_trade(best, size)
                    else:
                        print("  No opportunities found")

                    last_scan = now

                time.sleep(1)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN]")
                self.running = False
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        print(f"\n[FINAL] Trades: {self.trade_count} | Balance: ${self.balance:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Bombtastic Opportunity Scanner")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    args = parser.parse_args()

    scanner = OpportunityScanner(live=args.live)

    def sig_handler(sig, frame):
        scanner.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    scanner.run()


if __name__ == "__main__":
    main()
