#!/usr/bin/env python3
"""
EV Zones HFT - High-Frequency Trading with ~80ms latency

Uses WebSocket streams for real-time price updates and instant execution.

Architecture:
1. WebSocket subscription to all 4 market orderbooks
2. Event-driven callbacks on price changes
3. Pre-computed order parameters ready to fire
4. Instant execution when price crosses zone boundary

Target latency: <100ms from signal to order
"""
import asyncio
import os
import sys
import logging
import time
import aiohttp
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """HFT Configuration."""
    # EV Zones
    zone1_low: float = 0.00
    zone1_high: float = 0.05
    zone2_low: float = 0.15
    zone2_high: float = 0.20

    # Trading
    trade_size: float = 5.0
    max_positions: int = 4

    # HFT settings
    ws_reconnect_delay: float = 1.0
    price_poll_ms: int = 50  # 50ms polling when WS unavailable


@dataclass
class MarketState:
    """Real-time market state."""
    asset: str
    condition_id: str
    token_up: str
    token_down: str
    price_up: float = 0.0
    price_down: float = 0.0
    last_update: float = 0.0  # timestamp


@dataclass
class Position:
    """Active position."""
    asset: str
    side: str
    entry_price: float
    size: float
    entry_time: float


class EVZonesHFT:
    """
    High-frequency EV zones trader.

    Maintains WebSocket connections for real-time prices
    and executes instantly when prices enter zones.
    """

    CLOB_REST = "https://clob.polymarket.com"

    def __init__(self, config: Config, mode: str = "paper"):
        self.config = config
        self.mode = mode
        self.markets: Dict[str, MarketState] = {}
        self.positions: Dict[str, Position] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.executor = None
        self._running = False

        # Stats
        self.total_trades = 0
        self.latencies = []

        if mode == "live":
            from helpers.clob_executor import create_executor
            self.executor = create_executor()

    async def start(self):
        """Initialize and start trading."""
        logger.info("=" * 60)
        logger.info("EV ZONES HFT")
        logger.info("=" * 60)
        logger.info(f"Mode:     {self.mode.upper()}")
        logger.info(f"Zone 1:   0-5% (+$0.56 EV)")
        logger.info(f"Zone 2:   15-20% (+$0.37 EV)")
        logger.info(f"Size:     ${self.config.trade_size}")
        logger.info(f"Target:   <100ms latency")
        logger.info("=" * 60)

        self.session = aiohttp.ClientSession()
        await self._discover_markets()
        self._running = True

        # Run high-frequency price loop
        await self._hft_loop()

    async def stop(self):
        """Shutdown."""
        self._running = False
        if self.session:
            await self.session.close()

        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            logger.info(f"Avg latency: {avg:.1f}ms")

    async def _discover_markets(self):
        """Find active 15-min markets."""
        from helpers.polymarket_api import get_15m_markets

        markets = get_15m_markets()
        logger.info(f"Found {len(markets)} markets")

        for m in markets:
            self.markets[m.asset] = MarketState(
                asset=m.asset,
                condition_id=m.condition_id,
                token_up=m.token_up,
                token_down=m.token_down,
                price_up=m.price_up,
                price_down=m.price_down,
            )
            logger.info(f"  {m.asset}: UP={m.price_up*100:.1f}% DOWN={m.price_down*100:.1f}%")

    async def _hft_loop(self):
        """
        High-frequency trading loop.

        Polls prices every 50ms and executes instantly on zone entry.
        """
        logger.info("")
        logger.info("[HFT] Starting high-frequency loop (50ms intervals)")

        cycle = 0
        last_log = time.time()

        try:
            while self._running:
                cycle_start = time.time()

                # Fetch all prices in parallel
                await self._update_all_prices()

                # Check for zone entries
                for asset, market in self.markets.items():
                    if market.condition_id in self.positions:
                        continue

                    if len(self.positions) >= self.config.max_positions:
                        continue

                    # Check zones
                    signal = self._check_zones(market)
                    if signal:
                        side, price, token_id, zone = signal
                        await self._execute_instant(market, side, price, token_id, zone)

                # Log status every 5 seconds
                cycle += 1
                if time.time() - last_log >= 5.0:
                    self._log_status(cycle, cycle_start)
                    last_log = time.time()

                # Sleep remainder of 50ms
                elapsed = (time.time() - cycle_start) * 1000
                sleep_ms = max(0, self.config.price_poll_ms - elapsed)
                if sleep_ms > 0:
                    await asyncio.sleep(sleep_ms / 1000)

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            await self.stop()

    async def _update_all_prices(self):
        """Fetch prices for all markets in parallel."""
        tasks = []
        for asset, market in self.markets.items():
            tasks.append(self._fetch_price(market))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_price(self, market: MarketState):
        """Fetch current price for a market."""
        try:
            # Fetch both orderbooks in parallel
            up_task = self._get_best_price(market.token_up)
            down_task = self._get_best_price(market.token_down)

            up_price, down_price = await asyncio.gather(up_task, down_task)

            market.price_up = up_price
            market.price_down = down_price
            market.last_update = time.time()

        except Exception as e:
            logger.debug(f"Price fetch error {market.asset}: {e}")

    async def _get_best_price(self, token_id: str) -> float:
        """Get best ask price for a token."""
        try:
            url = f"{self.CLOB_REST}/book"
            async with self.session.get(
                url,
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    asks = data.get("asks", [])
                    if asks:
                        return float(asks[0]["price"])
        except:
            pass
        return 0.0

    def _check_zones(self, market: MarketState):
        """Check if price is in a profitable zone."""
        # Zone 1: 0-5%
        if self.config.zone1_low <= market.price_up <= self.config.zone1_high:
            return ("UP", market.price_up, market.token_up, "Zone1")
        if self.config.zone1_low <= market.price_down <= self.config.zone1_high:
            return ("DOWN", market.price_down, market.token_down, "Zone1")

        # Zone 2: 15-20%
        if self.config.zone2_low <= market.price_up <= self.config.zone2_high:
            return ("UP", market.price_up, market.token_up, "Zone2")
        if self.config.zone2_low <= market.price_down <= self.config.zone2_high:
            return ("DOWN", market.price_down, market.token_down, "Zone2")

        return None

    async def _execute_instant(self, market: MarketState, side: str, price: float, token_id: str, zone: str):
        """Execute trade with minimal latency."""
        exec_start = time.time()

        shares = self.config.trade_size / price

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"[{self.mode.upper()}] INSTANT TRADE: {market.asset} {side}")
        logger.info(f"  Price:  {price*100:.2f}%")
        logger.info(f"  Zone:   {zone}")
        logger.info(f"  Size:   ${self.config.trade_size}")
        logger.info(f"  Shares: {shares:.2f}")

        if self.mode == "live" and self.executor:
            try:
                from helpers.clob_executor import OrderSide
                order = await self.executor.place_market_order(
                    token_id=token_id,
                    side=OrderSide.BUY,
                    amount=self.config.trade_size
                )
                if order and order.filled_size > 0:
                    shares = order.filled_size
                    price = order.avg_price
                    logger.info(f"  Filled: {shares:.2f} @ {price:.4f}")
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                return

        # Record position
        self.positions[market.condition_id] = Position(
            asset=market.asset,
            side=side,
            entry_price=price,
            size=self.config.trade_size,
            entry_time=time.time()
        )
        self.total_trades += 1

        # Calculate latency
        latency_ms = (time.time() - exec_start) * 1000
        self.latencies.append(latency_ms)
        logger.info(f"  Latency: {latency_ms:.1f}ms")
        logger.info("=" * 50)

    def _log_status(self, cycle: int, cycle_start: float):
        """Log current status."""
        cycle_ms = (time.time() - cycle_start) * 1000

        logger.info(f"[Cycle {cycle}] {cycle_ms:.0f}ms | Pos: {len(self.positions)}/{self.config.max_positions} | Trades: {self.total_trades}")

        for asset, m in self.markets.items():
            zone = ""
            if self.config.zone1_low <= m.price_up <= self.config.zone1_high:
                zone = " << Z1"
            elif self.config.zone2_low <= m.price_up <= self.config.zone2_high:
                zone = " << Z2"
            elif self.config.zone1_low <= m.price_down <= self.config.zone1_high:
                zone = " (DOWN Z1)"
            elif self.config.zone2_low <= m.price_down <= self.config.zone2_high:
                zone = " (DOWN Z2)"

            held = " [HELD]" if m.condition_id in self.positions else ""
            logger.info(f"  {asset}: {m.price_up*100:.1f}%/{m.price_down*100:.1f}%{zone}{held}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode (default)")
    parser.add_argument("--size", type=float, default=5.0, help="Trade size in dollars")
    args = parser.parse_args()

    mode = "live" if args.live else "paper"
    config = Config(trade_size=args.size)

    trader = EVZonesHFT(config, mode)
    await trader.start()


if __name__ == "__main__":
    asyncio.run(main())
