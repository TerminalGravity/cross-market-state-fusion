#!/usr/bin/env python3
"""
EV Zones Trader - Simple strategy based on historical trade analysis

Strategy: Buy low-probability outcomes (0-5%) which have +$0.56 EV per share

Based on 513 trade analysis:
  - 0-5% entry zone: 58.8% win rate, +$0.56 EV
  - 15-20% entry zone: 54.5% win rate, +$0.37 EV

Usage:
    python ev_zones_trader.py --paper    # Paper trade
    python ev_zones_trader.py --live     # Live trading
"""
import asyncio
import os
import sys
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Trading configuration."""
    # EV Zone 1: 0-5% (best zone, +$0.56 EV, 58.8% win rate)
    zone1_low: float = 0.00
    zone1_high: float = 0.05

    # EV Zone 2: 15-20% (second best, +$0.37 EV, 54.5% win rate)
    zone2_low: float = 0.15
    zone2_high: float = 0.20

    # Trade settings
    trade_size: float = 5.0    # $5 per trade
    max_positions: int = 4     # Max concurrent positions

    # Risk management
    daily_loss_limit: float = 25.0  # Stop after losing $25

    # Scan settings
    scan_interval: float = 30.0  # Check every 30 seconds


@dataclass
class Position:
    """Active position."""
    asset: str
    token_id: str
    condition_id: str
    side: str  # "UP" or "DOWN"
    entry_price: float
    size: float
    shares: float
    entry_time: datetime


class EVZonesTrader:
    """
    Simple EV zones trader.

    Buys low-probability outcomes in the 0-5% zone where
    historical data shows +$0.56 EV per share.
    """

    def __init__(self, config: Config, mode: str = "paper"):
        self.config = config
        self.mode = mode
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.executor = None

        if mode == "live":
            from helpers.clob_executor import create_executor
            self.executor = create_executor()

    async def run(self):
        """Main trading loop."""
        logger.info("=" * 60)
        logger.info("EV ZONES TRADER")
        logger.info("=" * 60)
        logger.info(f"Mode:        {self.mode.upper()}")
        logger.info(f"Zone 1:      {self.config.zone1_low*100:.0f}%-{self.config.zone1_high*100:.0f}% (+$0.56 EV)")
        logger.info(f"Zone 2:      {self.config.zone2_low*100:.0f}%-{self.config.zone2_high*100:.0f}% (+$0.37 EV)")
        logger.info(f"Trade Size:  ${self.config.trade_size:.2f}")
        logger.info(f"Max Pos:     {self.config.max_positions}")
        logger.info(f"Daily Limit: ${self.config.daily_loss_limit:.2f}")
        logger.info("=" * 60)
        logger.info("Strategy: Buy when price is in positive EV zones")
        logger.info("=" * 60)

        try:
            while True:
                await self._scan_and_trade()
                await asyncio.sleep(self.config.scan_interval)
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
            self._print_summary()

    async def _scan_and_trade(self):
        """Scan markets and place trades."""
        from helpers.polymarket_api import get_15m_markets

        markets = get_15m_markets()

        if not markets:
            logger.warning("No active 15-min markets found")
            return

        # Check each market for entry opportunities
        for market in markets:
            # Skip if we already have a position in this market
            if market.condition_id in self.positions:
                continue

            # Skip if at max positions
            if len(self.positions) >= self.config.max_positions:
                continue

            # Check if daily loss limit hit
            if self.daily_pnl <= -self.config.daily_loss_limit:
                logger.warning(f"Daily loss limit hit: ${self.daily_pnl:.2f}")
                continue

            # Get current prices
            up_price = market.price_up
            down_price = market.price_down

            # Check for entry in profitable EV zones
            opportunity = None
            zone_name = ""

            # Zone 1: 0-5% (best zone, +$0.56 EV)
            if self.config.zone1_low <= up_price <= self.config.zone1_high:
                opportunity = ("UP", up_price, market.token_up)
                zone_name = "Zone 1 (0-5%, +$0.56 EV)"
            elif self.config.zone1_low <= down_price <= self.config.zone1_high:
                opportunity = ("DOWN", down_price, market.token_down)
                zone_name = "Zone 1 (0-5%, +$0.56 EV)"

            # Zone 2: 15-20% (second best, +$0.37 EV)
            elif self.config.zone2_low <= up_price <= self.config.zone2_high:
                opportunity = ("UP", up_price, market.token_up)
                zone_name = "Zone 2 (15-20%, +$0.37 EV)"
            elif self.config.zone2_low <= down_price <= self.config.zone2_high:
                opportunity = ("DOWN", down_price, market.token_down)
                zone_name = "Zone 2 (15-20%, +$0.37 EV)"

            if opportunity:
                side, price, token_id = opportunity
                await self._place_trade(market, side, price, token_id, zone_name)

        # Log status
        self._log_status(markets)

    async def _place_trade(self, market, side: str, price: float, token_id: str, zone_name: str = ""):
        """Place a trade."""
        shares = self.config.trade_size / price

        logger.info("")
        logger.info("=" * 50)
        logger.info(f"[{self.mode.upper()}] NEW TRADE: {market.asset} {side}")
        logger.info("=" * 50)
        logger.info(f"  Price:    {price*100:.1f}%")
        logger.info(f"  Size:     ${self.config.trade_size:.2f}")
        logger.info(f"  Shares:   {shares:.2f}")
        logger.info(f"  Potential Win: ${shares * (1 - price):.2f}")
        logger.info(f"  Potential Loss: ${self.config.trade_size:.2f}")
        logger.info(f"  EV Zone:  {zone_name}")
        logger.info("=" * 50)

        if self.mode == "live" and self.executor:
            try:
                from helpers.clob_executor import OrderSide
                order_side = OrderSide.BUY

                order = await self.executor.place_market_order(
                    token_id=token_id,
                    side=order_side,
                    amount=self.config.trade_size
                )

                if order and order.filled_size > 0:
                    logger.info(f"[LIVE] Order filled: {order.filled_size:.2f} shares @ {order.avg_price:.4f}")
                    shares = order.filled_size
                    price = order.avg_price
                else:
                    logger.warning("[LIVE] Order not filled")
                    return

            except Exception as e:
                logger.error(f"[LIVE] Order failed: {e}")
                return

        # Record position
        position = Position(
            asset=market.asset,
            token_id=token_id,
            condition_id=market.condition_id,
            side=side,
            entry_price=price,
            size=self.config.trade_size,
            shares=shares,
            entry_time=datetime.now(timezone.utc)
        )
        self.positions[market.condition_id] = position
        self.total_trades += 1

        # Send Discord alert
        await self._send_alert(
            f"**{self.mode.upper()} TRADE** | {market.asset} {side}\n"
            f"Entry: {price*100:.1f}% | Size: ${self.config.trade_size}\n"
            f"Shares: {shares:.2f} | EV Zone: 0-5%"
        )

    def _log_status(self, markets):
        """Log current status."""
        logger.info("")
        logger.info(f"[Status] Positions: {len(self.positions)}/{self.config.max_positions} | "
                   f"Trades: {self.total_trades} | PnL: ${self.daily_pnl:.2f}")

        for m in markets:
            in_zone = ""
            # Check Zone 1 (0-5%)
            if self.config.zone1_low <= m.price_up <= self.config.zone1_high:
                in_zone = " << UP IN ZONE 1 (BUY)"
            elif self.config.zone1_low <= m.price_down <= self.config.zone1_high:
                in_zone = " << DOWN IN ZONE 1 (BUY)"
            # Check Zone 2 (15-20%)
            elif self.config.zone2_low <= m.price_up <= self.config.zone2_high:
                in_zone = " << UP IN ZONE 2 (BUY)"
            elif self.config.zone2_low <= m.price_down <= self.config.zone2_high:
                in_zone = " << DOWN IN ZONE 2 (BUY)"

            pos_marker = ""
            if m.condition_id in self.positions:
                pos = self.positions[m.condition_id]
                pos_marker = f" [HOLDING {pos.side}]"

            logger.info(f"  {m.asset}: UP={m.price_up*100:.1f}% DOWN={m.price_down*100:.1f}%{in_zone}{pos_marker}")

    async def _send_alert(self, message: str):
        """Send Discord alert."""
        webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
        if not webhook_url:
            return

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(webhook_url, json={"content": message})
        except Exception as e:
            logger.debug(f"Discord alert failed: {e}")

    def _print_summary(self):
        """Print trading summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRADING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Trades:  {self.total_trades}")
        logger.info(f"Daily PnL:     ${self.daily_pnl:.2f}")
        logger.info(f"Open Positions: {len(self.positions)}")
        logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EV Zones Trader")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--live", action="store_true", help="Live trading mode")
    parser.add_argument("--size", type=float, default=5.0, help="Trade size in dollars")
    args = parser.parse_args()

    mode = "live" if args.live else "paper"

    # Use environment variables or defaults
    config = Config(
        trade_size=float(os.getenv("TRADE_SIZE", args.size)),
    )

    trader = EVZonesTrader(config, mode)
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
