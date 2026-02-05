#!/usr/bin/env python3
"""
XRP Arbitrage Bot - Reverse Engineered from AIRules247 Strategy

Scans 15-minute crypto markets for arbitrage opportunities where:
    UP_price + DOWN_price < $1.00

When detected, buys BOTH sides to lock in risk-free profit.

Example:
    UP = $0.48, DOWN = $0.48 → Cost = $0.96
    One side resolves to $1.00 → Profit = $0.04 (4.2%)
    After 0.4% fees → Net ~3.8%

Usage:
    # Scan only (no trades)
    python arbitrage_bot.py --scan

    # Paper trade mode
    python arbitrage_bot.py --paper

    # Live trading (requires funds)
    python arbitrage_bot.py --live --size 100

Author: Reverse-engineered from AIRules247 X post
"""
import os
import sys
import asyncio
import logging
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from decimal import Decimal
import aiohttp

# Load environment
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ArbConfig:
    """Arbitrage bot configuration."""
    # Minimum edge after fees to execute (0.5% = 0.005)
    min_edge: float = 0.005

    # Polymarket fees (taker = 0.2% each side)
    taker_fee: float = 0.002

    # Minimum sum threshold (buy both if sum < this)
    # 1.0 - min_edge - (2 * taker_fee) = breakeven
    # For 0.5% edge: 1.0 - 0.005 - 0.004 = 0.991
    max_sum_threshold: float = 0.99

    # Trade size per opportunity (USD)
    trade_size: float = 50.0

    # Maximum concurrent positions
    max_positions: int = 4

    # Scan interval (milliseconds)
    scan_interval_ms: int = 100

    # REST polling interval when WSS unavailable
    rest_interval_s: float = 0.5

    # Assets to monitor
    assets: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "XRP"])


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketPrices:
    """Real-time prices for a market."""
    asset: str
    condition_id: str
    up_token_id: str
    down_token_id: str

    # Best prices from orderbook
    up_best_bid: float = 0.0
    up_best_ask: float = 0.0
    down_best_bid: float = 0.0
    down_best_ask: float = 0.0

    # Calculated
    up_mid: float = 0.0
    down_mid: float = 0.0

    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def sum_asks(self) -> float:
        """Cost to buy both sides (asks)."""
        return self.up_best_ask + self.down_best_ask

    @property
    def sum_bids(self) -> float:
        """Revenue from selling both sides (bids)."""
        return self.up_best_bid + self.down_best_bid

    @property
    def buy_arb_edge(self) -> float:
        """Edge if we buy both sides. Positive = profitable."""
        if self.sum_asks <= 0:
            return 0
        return 1.0 - self.sum_asks

    @property
    def sell_arb_edge(self) -> float:
        """Edge if we sell both sides. Positive = profitable."""
        if self.sum_bids <= 0:
            return 0
        return self.sum_bids - 1.0


@dataclass
class ArbOpportunity:
    """A detected arbitrage opportunity."""
    asset: str
    condition_id: str
    direction: str  # "BUY_BOTH" or "SELL_BOTH"
    up_price: float
    down_price: float
    total_cost: float
    gross_edge: float  # Before fees
    net_edge: float    # After fees
    expected_profit: float  # Dollar profit for trade_size
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArbPosition:
    """An active arbitrage position."""
    asset: str
    condition_id: str
    up_token_id: str
    down_token_id: str
    up_shares: float
    down_shares: float
    entry_cost: float
    expected_payout: float
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def expected_profit(self) -> float:
        return self.expected_payout - self.entry_cost


# ============================================================================
# PRICE SCANNER
# ============================================================================

class PriceScanner:
    """
    High-frequency price scanner for 15-minute markets.

    Uses REST polling (WebSocket blocked by Cloudflare on most infra).
    """

    CLOB_REST = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"

    def __init__(self, config: ArbConfig):
        self.config = config
        self.markets: Dict[str, MarketPrices] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self._running = False

    async def start(self):
        """Start the scanner."""
        self.session = aiohttp.ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0",
                "Accept": "application/json",
            }
        )
        await self._discover_markets()
        self._running = True

    async def stop(self):
        """Stop the scanner."""
        self._running = False
        if self.session:
            await self.session.close()

    async def _discover_markets(self):
        """Find active 15-minute markets."""
        from helpers.polymarket_api import get_15m_markets

        markets = get_15m_markets()
        logger.info(f"Discovered {len(markets)} active 15-min markets")

        for m in markets:
            if m.asset in self.config.assets:
                self.markets[m.asset] = MarketPrices(
                    asset=m.asset,
                    condition_id=m.condition_id,
                    up_token_id=m.token_up,
                    down_token_id=m.token_down,
                )
                logger.info(f"  {m.asset}: {m.condition_id[:16]}...")

    async def fetch_orderbook(self, token_id: str) -> Tuple[float, float]:
        """Fetch best bid/ask for a token."""
        try:
            url = f"{self.CLOB_REST}/book"
            async with self.session.get(url, params={"token_id": token_id}, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    bids = data.get("bids", [])
                    asks = data.get("asks", [])

                    best_bid = float(bids[0]["price"]) if bids else 0.0
                    best_ask = float(asks[0]["price"]) if asks else 1.0

                    return best_bid, best_ask
        except Exception as e:
            logger.debug(f"Orderbook fetch error: {e}")

        return 0.0, 1.0

    async def update_prices(self) -> Dict[str, MarketPrices]:
        """Update all market prices."""
        tasks = []

        for asset, market in self.markets.items():
            tasks.append(self._update_market(market))

        await asyncio.gather(*tasks, return_exceptions=True)
        return self.markets

    async def _update_market(self, market: MarketPrices):
        """Update prices for a single market."""
        # Fetch both orderbooks in parallel
        up_task = self.fetch_orderbook(market.up_token_id)
        down_task = self.fetch_orderbook(market.down_token_id)

        (up_bid, up_ask), (down_bid, down_ask) = await asyncio.gather(up_task, down_task)

        market.up_best_bid = up_bid
        market.up_best_ask = up_ask
        market.down_best_bid = down_bid
        market.down_best_ask = down_ask
        market.up_mid = (up_bid + up_ask) / 2 if up_bid and up_ask else 0
        market.down_mid = (down_bid + down_ask) / 2 if down_bid and down_ask else 0
        market.last_update = datetime.now(timezone.utc)


# ============================================================================
# ARBITRAGE DETECTOR
# ============================================================================

class ArbDetector:
    """
    Detects arbitrage opportunities from price data.

    Strategy:
    1. If UP_ask + DOWN_ask < 0.99: BUY BOTH (guaranteed profit)
    2. If UP_bid + DOWN_bid > 1.01: SELL BOTH (if we hold shares)
    """

    def __init__(self, config: ArbConfig):
        self.config = config
        self.opportunities: List[ArbOpportunity] = []

    def scan(self, markets: Dict[str, MarketPrices]) -> List[ArbOpportunity]:
        """Scan markets for arbitrage opportunities."""
        self.opportunities = []

        for asset, market in markets.items():
            # Skip if prices not available
            if market.up_best_ask <= 0 or market.down_best_ask <= 0:
                continue

            # Check BUY BOTH opportunity
            cost = market.sum_asks
            if cost < self.config.max_sum_threshold:
                gross_edge = 1.0 - cost
                fees = 2 * self.config.taker_fee  # Buy both sides
                net_edge = gross_edge - fees

                if net_edge >= self.config.min_edge:
                    profit = net_edge * self.config.trade_size

                    opp = ArbOpportunity(
                        asset=asset,
                        condition_id=market.condition_id,
                        direction="BUY_BOTH",
                        up_price=market.up_best_ask,
                        down_price=market.down_best_ask,
                        total_cost=cost,
                        gross_edge=gross_edge,
                        net_edge=net_edge,
                        expected_profit=profit,
                    )
                    self.opportunities.append(opp)

            # Check SELL BOTH opportunity (if we have inventory)
            revenue = market.sum_bids
            if revenue > 1.0 + self.config.min_edge + 2 * self.config.taker_fee:
                gross_edge = revenue - 1.0
                fees = 2 * self.config.taker_fee
                net_edge = gross_edge - fees

                if net_edge >= self.config.min_edge:
                    profit = net_edge * self.config.trade_size

                    opp = ArbOpportunity(
                        asset=asset,
                        condition_id=market.condition_id,
                        direction="SELL_BOTH",
                        up_price=market.up_best_bid,
                        down_price=market.down_best_bid,
                        total_cost=revenue,
                        gross_edge=gross_edge,
                        net_edge=net_edge,
                        expected_profit=profit,
                    )
                    self.opportunities.append(opp)

        return self.opportunities


# ============================================================================
# EXECUTOR
# ============================================================================

class ArbExecutor:
    """
    Executes arbitrage trades.

    Places simultaneous orders for both UP and DOWN tokens.
    """

    def __init__(self, config: ArbConfig, mode: str = "paper"):
        self.config = config
        self.mode = mode  # "paper", "live", "scan"
        self.positions: List[ArbPosition] = []
        self.total_trades = 0
        self.total_profit = 0.0
        self.executor = None

        if mode == "live":
            from helpers.clob_executor import ClobExecutor, ExecutionMode
            self.executor = ClobExecutor(mode=ExecutionMode.LIVE)

    async def execute(self, opp: ArbOpportunity, market: MarketPrices) -> bool:
        """Execute an arbitrage opportunity."""
        if self.mode == "scan":
            self._log_opportunity(opp)
            return False

        if len(self.positions) >= self.config.max_positions:
            logger.warning(f"Max positions reached ({self.config.max_positions})")
            return False

        # Calculate shares to buy
        cost_per_share = opp.total_cost
        shares = self.config.trade_size / cost_per_share

        if self.mode == "paper":
            return await self._paper_execute(opp, market, shares)
        elif self.mode == "live":
            return await self._live_execute(opp, market, shares)

        return False

    async def _paper_execute(self, opp: ArbOpportunity, market: MarketPrices, shares: float) -> bool:
        """Simulate trade execution."""
        logger.info(f"\n{'='*60}")
        logger.info(f"[PAPER] ARBITRAGE DETECTED: {opp.asset}")
        logger.info(f"{'='*60}")
        logger.info(f"  Direction:    {opp.direction}")
        logger.info(f"  UP Price:     {opp.up_price:.4f}")
        logger.info(f"  DOWN Price:   {opp.down_price:.4f}")
        logger.info(f"  Total Cost:   {opp.total_cost:.4f} (< $1.00)")
        logger.info(f"  Gross Edge:   {opp.gross_edge:.2%}")
        logger.info(f"  Net Edge:     {opp.net_edge:.2%} (after 0.4% fees)")
        logger.info(f"  Trade Size:   ${self.config.trade_size:.2f}")
        logger.info(f"  Shares:       {shares:.2f}")
        logger.info(f"  Exp. Profit:  ${opp.expected_profit:.2f}")
        logger.info(f"{'='*60}\n")

        # Record position
        position = ArbPosition(
            asset=opp.asset,
            condition_id=opp.condition_id,
            up_token_id=market.up_token_id,
            down_token_id=market.down_token_id,
            up_shares=shares,
            down_shares=shares,
            entry_cost=self.config.trade_size,
            expected_payout=shares * 1.0,  # One side pays $1
        )
        self.positions.append(position)
        self.total_trades += 1
        self.total_profit += opp.expected_profit

        return True

    async def _live_execute(self, opp: ArbOpportunity, market: MarketPrices, shares: float) -> bool:
        """Execute real trades."""
        from helpers.clob_executor import OrderSide

        logger.info(f"\n{'='*60}")
        logger.info(f"[LIVE] EXECUTING ARBITRAGE: {opp.asset}")
        logger.info(f"{'='*60}")

        try:
            # Buy UP token
            up_cost = shares * opp.up_price
            up_order = self.executor.place_market_order(
                token_id=market.up_token_id,
                amount=up_cost,
                side=OrderSide.BUY,
                asset=f"{opp.asset} UP"
            )

            # Buy DOWN token
            down_cost = shares * opp.down_price
            down_order = self.executor.place_market_order(
                token_id=market.down_token_id,
                amount=down_cost,
                side=OrderSide.BUY,
                asset=f"{opp.asset} DOWN"
            )

            if up_order and down_order:
                logger.info(f"  UP Order:   {up_order.order_id[:16]}... ({up_order.status})")
                logger.info(f"  DOWN Order: {down_order.order_id[:16]}... ({down_order.status})")

                position = ArbPosition(
                    asset=opp.asset,
                    condition_id=opp.condition_id,
                    up_token_id=market.up_token_id,
                    down_token_id=market.down_token_id,
                    up_shares=up_order.filled_size,
                    down_shares=down_order.filled_size,
                    entry_cost=up_cost + down_cost,
                    expected_payout=min(up_order.filled_size, down_order.filled_size),
                )
                self.positions.append(position)
                self.total_trades += 1
                return True

        except Exception as e:
            logger.error(f"Execution failed: {e}")

        return False

    def _log_opportunity(self, opp: ArbOpportunity):
        """Log opportunity without executing."""
        logger.info(
            f"[SCAN] {opp.asset}: {opp.direction} | "
            f"UP={opp.up_price:.3f} DOWN={opp.down_price:.3f} | "
            f"Sum={opp.total_cost:.4f} | "
            f"Edge={opp.net_edge:.2%} | "
            f"Profit=${opp.expected_profit:.2f}"
        )


# ============================================================================
# MAIN BOT
# ============================================================================

class ArbitrageBot:
    """
    Main arbitrage bot orchestrator.

    Combines scanner, detector, and executor into a continuous loop.
    """

    MARKET_REFRESH_INTERVAL = 300  # Re-discover markets every 5 minutes

    def __init__(self, config: ArbConfig, mode: str = "paper"):
        self.config = config
        self.mode = mode

        self.scanner = PriceScanner(config)
        self.detector = ArbDetector(config)
        self.executor = ArbExecutor(config, mode)

        self._running = False
        self.scan_count = 0
        self.opportunities_found = 0
        self.start_time = None
        self.last_market_refresh = None

    async def run(self):
        """Run the bot."""
        logger.info("=" * 60)
        logger.info("ARBITRAGE BOT STARTING")
        logger.info("=" * 60)
        logger.info(f"Mode:           {self.mode.upper()}")
        logger.info(f"Min Edge:       {self.config.min_edge:.2%}")
        logger.info(f"Max Sum:        {self.config.max_sum_threshold}")
        logger.info(f"Trade Size:     ${self.config.trade_size:.2f}")
        logger.info(f"Scan Interval:  {self.config.rest_interval_s}s")
        logger.info(f"Assets:         {', '.join(self.config.assets)}")
        logger.info("=" * 60)

        await self.scanner.start()
        self._running = True
        self.start_time = datetime.now(timezone.utc)
        self.last_market_refresh = datetime.now(timezone.utc)

        try:
            while self._running:
                # Re-discover markets periodically (every 5 min)
                now = datetime.now(timezone.utc)
                since_refresh = (now - self.last_market_refresh).total_seconds()
                if since_refresh >= self.MARKET_REFRESH_INTERVAL:
                    logger.info(f"[*] Refreshing markets (every {self.MARKET_REFRESH_INTERVAL}s)...")
                    await self.scanner._discover_markets()
                    self.last_market_refresh = now

                await self._scan_cycle()
                await asyncio.sleep(self.config.rest_interval_s)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            await self.scanner.stop()
            self._print_summary()

    async def _scan_cycle(self):
        """Single scan cycle."""
        self.scan_count += 1

        # Update prices
        markets = await self.scanner.update_prices()

        # Detect opportunities
        opportunities = self.detector.scan(markets)

        # Log current state periodically
        if self.scan_count % 20 == 0:  # Every ~10 seconds
            self._log_status(markets)

        # Execute opportunities
        for opp in opportunities:
            self.opportunities_found += 1
            market = markets.get(opp.asset)
            if market:
                await self.executor.execute(opp, market)

    def _log_status(self, markets: Dict[str, MarketPrices]):
        """Log current market status."""
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        status_lines = [f"\n[Scan #{self.scan_count} | {runtime:.0f}s | Found: {self.opportunities_found}]"]

        for asset, m in markets.items():
            if m.up_best_ask > 0 and m.down_best_ask > 0:
                total = m.sum_asks
                edge = 1.0 - total
                marker = "<<< ARB!" if total < self.config.max_sum_threshold else ""
                status_lines.append(
                    f"  {asset}: UP={m.up_best_ask:.3f} DOWN={m.down_best_ask:.3f} "
                    f"Sum={total:.4f} Edge={edge:+.2%} {marker}"
                )

        logger.info("\n".join(status_lines))

    def _print_summary(self):
        """Print final summary."""
        runtime = (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0

        logger.info("\n" + "=" * 60)
        logger.info("ARBITRAGE BOT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Runtime:            {runtime:.0f} seconds")
        logger.info(f"Total Scans:        {self.scan_count}")
        logger.info(f"Opportunities:      {self.opportunities_found}")
        logger.info(f"Trades Executed:    {self.executor.total_trades}")
        logger.info(f"Total Profit:       ${self.executor.total_profit:.2f}")
        logger.info(f"Positions Open:     {len(self.executor.positions)}")
        logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="XRP Arbitrage Bot")
    parser.add_argument("--scan", action="store_true", help="Scan only, no trades")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--size", type=float, default=50.0, help="Trade size in USD")
    parser.add_argument("--edge", type=float, default=0.005, help="Minimum edge (0.005 = 0.5%)")
    parser.add_argument("--interval", type=float, default=0.5, help="Scan interval in seconds")
    args = parser.parse_args()

    # Determine mode
    if args.live:
        mode = "live"
    elif args.paper:
        mode = "paper"
    else:
        mode = "scan"

    # Build config
    config = ArbConfig(
        trade_size=args.size,
        min_edge=args.edge,
        rest_interval_s=args.interval,
    )

    # Run bot
    bot = ArbitrageBot(config, mode)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
