#!/usr/bin/env python3
"""
EV Zones Simulator - Test HFT strategy with simulated markets

Simulates 15-minute binary crypto markets with realistic price dynamics:
- Mean-reverting prices around 50%
- Occasional volatility spikes that push into EV zones
- Market resolution every 15 minutes
- Realistic bid/ask spreads

Usage:
    python ev_zones_simulator.py                    # Run simulation
    python ev_zones_simulator.py --speed 10         # 10x speed (1 min = 6 sec)
    python ev_zones_simulator.py --volatility 0.15  # Higher volatility
    python ev_zones_simulator.py --trades 20        # Stop after 20 trades
"""
import asyncio
import random
import time
import logging
import argparse
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class MarketOutcome(Enum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class SimulatedMarket:
    """Simulated 15-minute binary market."""
    asset: str
    condition_id: str
    token_up: str
    token_down: str

    # Current prices (probability of UP winning)
    price_up: float = 0.50
    price_down: float = 0.50

    # Market timing
    start_time: float = 0.0
    duration_seconds: float = 900.0  # 15 minutes

    # True outcome (determined at start, revealed at end)
    true_outcome: Optional[MarketOutcome] = None

    # Price dynamics
    volatility: float = 0.08
    mean_reversion: float = 0.1

    def __post_init__(self):
        self.start_time = time.time()
        # Randomly determine outcome at market creation
        self.true_outcome = random.choice([MarketOutcome.UP, MarketOutcome.DOWN])

        # Bias initial price slightly toward true outcome
        bias = random.uniform(0.45, 0.55)
        if self.true_outcome == MarketOutcome.UP:
            self.price_up = bias
        else:
            self.price_up = 1 - bias
        self.price_down = 1 - self.price_up

    def tick(self, dt: float):
        """Update prices with realistic dynamics."""
        # Time remaining affects volatility (increases near expiry)
        time_elapsed = time.time() - self.start_time
        time_remaining = max(0, self.duration_seconds - time_elapsed)
        time_factor = 1 + (1 - time_remaining / self.duration_seconds) * 0.5

        # Random walk with mean reversion toward true outcome
        if self.true_outcome == MarketOutcome.UP:
            drift = self.mean_reversion * (0.7 - self.price_up)  # Drift toward 70%
        else:
            drift = self.mean_reversion * (0.3 - self.price_up)  # Drift toward 30%

        # Add randomness
        noise = random.gauss(0, self.volatility * time_factor * dt)

        # Occasional large moves (simulates news/momentum)
        if random.random() < 0.02:  # 2% chance per tick
            noise += random.choice([-1, 1]) * random.uniform(0.05, 0.15)

        # Update price
        self.price_up = max(0.01, min(0.99, self.price_up + drift * dt + noise))
        self.price_down = 1 - self.price_up

    def time_remaining(self) -> float:
        """Seconds until market resolves."""
        return max(0, self.duration_seconds - (time.time() - self.start_time))

    def is_expired(self) -> bool:
        """Check if market has resolved."""
        return self.time_remaining() <= 0

    def resolve(self) -> MarketOutcome:
        """Resolve market and return outcome."""
        # Near expiry, snap to true outcome
        return self.true_outcome


@dataclass
class Position:
    """Active position."""
    asset: str
    condition_id: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    size: float
    shares: float
    entry_time: float


@dataclass
class TradeResult:
    """Result of a closed trade."""
    asset: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    shares: float
    pnl: float
    duration: float
    won: bool


@dataclass
class Config:
    """Simulator configuration."""
    # EV Zones
    zone1_low: float = 0.00
    zone1_high: float = 0.05
    zone2_low: float = 0.15
    zone2_high: float = 0.20

    # Trading
    trade_size: float = 5.0
    max_positions: int = 4

    # Simulation
    speed_multiplier: float = 1.0  # 1.0 = real-time, 10.0 = 10x speed
    tick_interval_ms: int = 50
    market_duration: float = 900.0  # 15 minutes
    volatility: float = 0.08


class EVZonesSimulator:
    """
    Simulated EV zones trader for testing.

    Creates fake markets with realistic price dynamics to test
    the entry/exit logic before going live.
    """

    ASSETS = ["BTC", "ETH", "SOL", "XRP"]

    def __init__(self, config: Config, max_trades: Optional[int] = None):
        self.config = config
        self.max_trades = max_trades
        self.markets: Dict[str, SimulatedMarket] = {}
        self.positions: Dict[str, Position] = {}
        self.results: List[TradeResult] = []
        self._running = False

        # Stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.latencies: List[float] = []

    async def run(self):
        """Run simulation."""
        logger.info("=" * 60)
        logger.info("EV ZONES SIMULATOR")
        logger.info("=" * 60)
        logger.info(f"Speed:      {self.config.speed_multiplier}x")
        logger.info(f"Volatility: {self.config.volatility}")
        logger.info(f"Zone 1:     0-5% (+$0.56 EV)")
        logger.info(f"Zone 2:     15-20% (+$0.37 EV)")
        logger.info(f"Size:       ${self.config.trade_size}")
        logger.info(f"Max trades: {self.max_trades or 'unlimited'}")
        logger.info("=" * 60)

        self._create_markets()
        self._running = True

        cycle = 0
        last_log = time.time()

        try:
            while self._running:
                cycle_start = time.time()

                # Update market prices
                dt = (self.config.tick_interval_ms / 1000) * self.config.speed_multiplier
                for market in self.markets.values():
                    market.tick(dt)

                # Check for expired markets and resolve positions
                await self._check_resolutions()

                # Refresh expired markets
                self._refresh_expired_markets()

                # Check for entry opportunities
                for asset, market in self.markets.items():
                    if market.condition_id in self.positions:
                        continue
                    if len(self.positions) >= self.config.max_positions:
                        continue

                    signal = self._check_zones(market)
                    if signal:
                        side, price, zone = signal
                        await self._execute_entry(market, side, price, zone)

                # Check if we've hit max trades
                if self.max_trades and self.total_trades >= self.max_trades:
                    logger.info(f"\nReached {self.max_trades} trades, stopping...")
                    break

                # Log status every 5 real seconds
                cycle += 1
                if time.time() - last_log >= 5.0:
                    self._log_status(cycle)
                    last_log = time.time()

                # Sleep
                elapsed = (time.time() - cycle_start) * 1000
                sleep_ms = max(0, self.config.tick_interval_ms - elapsed)
                if sleep_ms > 0:
                    await asyncio.sleep(sleep_ms / 1000)

        except KeyboardInterrupt:
            logger.info("\nStopping simulation...")

        finally:
            self._print_summary()

    def _create_markets(self):
        """Create initial simulated markets."""
        for i, asset in enumerate(self.ASSETS):
            market = SimulatedMarket(
                asset=asset,
                condition_id=f"sim_{asset}_{int(time.time())}",
                token_up=f"token_up_{asset}",
                token_down=f"token_down_{asset}",
                volatility=self.config.volatility,
                duration_seconds=self.config.market_duration / self.config.speed_multiplier,
            )
            self.markets[asset] = market
            logger.info(f"Created market: {asset} (outcome: {market.true_outcome.value})")

    def _refresh_expired_markets(self):
        """Replace expired markets with new ones."""
        for asset in self.ASSETS:
            market = self.markets[asset]
            if market.is_expired():
                # Create new market
                new_market = SimulatedMarket(
                    asset=asset,
                    condition_id=f"sim_{asset}_{int(time.time())}",
                    token_up=f"token_up_{asset}",
                    token_down=f"token_down_{asset}",
                    volatility=self.config.volatility,
                    duration_seconds=self.config.market_duration / self.config.speed_multiplier,
                )
                self.markets[asset] = new_market
                logger.info(f"New market: {asset} (outcome: {new_market.true_outcome.value})")

    async def _check_resolutions(self):
        """Check for and resolve expired positions."""
        to_remove = []

        for cid, pos in self.positions.items():
            # Find the market
            market = None
            for m in self.markets.values():
                if m.condition_id == cid:
                    market = m
                    break

            if market and market.is_expired():
                outcome = market.resolve()
                won = (pos.side == outcome.value)

                if won:
                    exit_price = 1.0
                    pnl = pos.shares * (1 - pos.entry_price)
                    self.wins += 1
                else:
                    exit_price = 0.0
                    pnl = -pos.size
                    self.losses += 1

                self.total_pnl += pnl

                result = TradeResult(
                    asset=pos.asset,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    size=pos.size,
                    shares=pos.shares,
                    pnl=pnl,
                    duration=time.time() - pos.entry_time,
                    won=won
                )
                self.results.append(result)

                logger.info("")
                logger.info("=" * 50)
                logger.info(f"[RESOLVED] {pos.asset} {pos.side} -> {outcome.value}")
                logger.info(f"  Entry:  {pos.entry_price*100:.1f}%")
                logger.info(f"  Result: {'WIN' if won else 'LOSS'}")
                logger.info(f"  PnL:    ${pnl:+.2f}")
                logger.info(f"  Total:  ${self.total_pnl:+.2f}")
                logger.info("=" * 50)

                to_remove.append(cid)

        for cid in to_remove:
            del self.positions[cid]

    def _check_zones(self, market: SimulatedMarket):
        """Check if price is in a profitable zone."""
        # Zone 1: 0-5%
        if self.config.zone1_low <= market.price_up <= self.config.zone1_high:
            return ("UP", market.price_up, "Zone1")
        if self.config.zone1_low <= market.price_down <= self.config.zone1_high:
            return ("DOWN", market.price_down, "Zone1")

        # Zone 2: 15-20%
        if self.config.zone2_low <= market.price_up <= self.config.zone2_high:
            return ("UP", market.price_up, "Zone2")
        if self.config.zone2_low <= market.price_down <= self.config.zone2_high:
            return ("DOWN", market.price_down, "Zone2")

        return None

    async def _execute_entry(self, market: SimulatedMarket, side: str, price: float, zone: str):
        """Execute simulated entry."""
        exec_start = time.time()

        shares = self.config.trade_size / price

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"[ENTRY] {market.asset} {side}")
        logger.info(f"  Price:    {price*100:.2f}%")
        logger.info(f"  Zone:     {zone}")
        logger.info(f"  Size:     ${self.config.trade_size}")
        logger.info(f"  Shares:   {shares:.2f}")
        logger.info(f"  Potential: ${shares * (1 - price):.2f}")
        logger.info(f"  True outcome: {market.true_outcome.value}")
        logger.info("=" * 50)

        self.positions[market.condition_id] = Position(
            asset=market.asset,
            condition_id=market.condition_id,
            side=side,
            entry_price=price,
            size=self.config.trade_size,
            shares=shares,
            entry_time=time.time()
        )
        self.total_trades += 1

        latency_ms = (time.time() - exec_start) * 1000
        self.latencies.append(latency_ms)

    def _log_status(self, cycle: int):
        """Log current status."""
        win_rate = (self.wins / (self.wins + self.losses) * 100) if (self.wins + self.losses) > 0 else 0

        logger.info("")
        logger.info(f"[Cycle {cycle}] Trades: {self.total_trades} | "
                   f"W/L: {self.wins}/{self.losses} ({win_rate:.0f}%) | "
                   f"PnL: ${self.total_pnl:+.2f} | "
                   f"Open: {len(self.positions)}")

        for asset, m in self.markets.items():
            zone = ""
            if self.config.zone1_low <= m.price_up <= self.config.zone1_high:
                zone = " << Z1 UP"
            elif self.config.zone2_low <= m.price_up <= self.config.zone2_high:
                zone = " << Z2 UP"
            elif self.config.zone1_low <= m.price_down <= self.config.zone1_high:
                zone = " << Z1 DOWN"
            elif self.config.zone2_low <= m.price_down <= self.config.zone2_high:
                zone = " << Z2 DOWN"

            held = ""
            if m.condition_id in self.positions:
                pos = self.positions[m.condition_id]
                held = f" [HOLD {pos.side}]"

            remaining = m.time_remaining()
            logger.info(f"  {asset}: {m.price_up*100:.1f}%/{m.price_down*100:.1f}% "
                       f"({remaining:.0f}s){zone}{held}")

    def _print_summary(self):
        """Print final summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("SIMULATION SUMMARY")
        logger.info("=" * 60)

        win_rate = (self.wins / (self.wins + self.losses) * 100) if (self.wins + self.losses) > 0 else 0

        logger.info(f"Total Trades:  {self.total_trades}")
        logger.info(f"Wins:          {self.wins}")
        logger.info(f"Losses:        {self.losses}")
        logger.info(f"Win Rate:      {win_rate:.1f}%")
        logger.info(f"Total PnL:     ${self.total_pnl:+.2f}")

        if self.total_trades > 0:
            avg_pnl = self.total_pnl / self.total_trades
            logger.info(f"Avg PnL/Trade: ${avg_pnl:+.2f}")

        if self.latencies:
            avg_lat = sum(self.latencies) / len(self.latencies)
            logger.info(f"Avg Latency:   {avg_lat:.1f}ms")

        # Zone analysis
        zone1_trades = [r for r in self.results if r.entry_price <= 0.05]
        zone2_trades = [r for r in self.results if 0.15 <= r.entry_price <= 0.20]

        if zone1_trades:
            z1_wins = sum(1 for r in zone1_trades if r.won)
            z1_pnl = sum(r.pnl for r in zone1_trades)
            logger.info(f"Zone 1 (0-5%): {len(zone1_trades)} trades, "
                       f"{z1_wins}/{len(zone1_trades)} wins, ${z1_pnl:+.2f}")

        if zone2_trades:
            z2_wins = sum(1 for r in zone2_trades if r.won)
            z2_pnl = sum(r.pnl for r in zone2_trades)
            logger.info(f"Zone 2 (15-20%): {len(zone2_trades)} trades, "
                       f"{z2_wins}/{len(zone2_trades)} wins, ${z2_pnl:+.2f}")

        logger.info("=" * 60)

        # Expected vs actual comparison
        if self.results:
            logger.info("")
            logger.info("EXPECTED VS ACTUAL (based on 513-trade analysis)")
            logger.info("-" * 40)
            logger.info(f"Expected Zone 1 win rate: 58.8%")
            logger.info(f"Expected Zone 2 win rate: 54.5%")
            logger.info(f"Actual overall win rate:  {win_rate:.1f}%")

            if zone1_trades:
                z1_actual = (z1_wins / len(zone1_trades) * 100)
                logger.info(f"Actual Zone 1 win rate:   {z1_actual:.1f}%")

            if zone2_trades:
                z2_actual = (z2_wins / len(zone2_trades) * 100)
                logger.info(f"Actual Zone 2 win rate:   {z2_actual:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="EV Zones Simulator")
    parser.add_argument("--speed", type=float, default=10.0,
                       help="Speed multiplier (default: 10x)")
    parser.add_argument("--volatility", type=float, default=0.12,
                       help="Price volatility (default: 0.12)")
    parser.add_argument("--trades", type=int, default=None,
                       help="Stop after N trades")
    parser.add_argument("--size", type=float, default=5.0,
                       help="Trade size in dollars")
    args = parser.parse_args()

    config = Config(
        speed_multiplier=args.speed,
        volatility=args.volatility,
        trade_size=args.size,
    )

    simulator = EVZonesSimulator(config, max_trades=args.trades)
    asyncio.run(simulator.run())


if __name__ == "__main__":
    main()
