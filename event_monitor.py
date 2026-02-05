#!/usr/bin/env python3
"""
Event Monitor - Law of Attraction Trading System

Monitors high-conviction event markets (NFL, politics, etc.) and only
trades when there's:
1. Genuine edge (price divergence from true probability)
2. Good liquidity (volume > threshold)
3. Market activity (recent trades happening)

Philosophy: Focus energy on ONE opportunity at a time, wait for alignment.
"""

import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, List, Dict
from decimal import Decimal

# Load environment
def load_env():
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"\''))

load_env()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class EventOpportunity:
    """A potential betting opportunity on an event."""
    event_name: str
    market_slug: str
    condition_id: str
    outcome: str  # e.g., "Jaguars", "Bears"
    current_price: float  # Polymarket price (0-1)
    estimated_true_prob: float  # Our estimate of true probability
    edge: float  # estimated_true_prob - current_price
    volume_24h: float
    min_volume_threshold: float
    conviction_level: str  # "HIGH", "MEDIUM", "LOW"
    reasoning: str
    game_time: Optional[datetime] = None


# High-conviction opportunities based on deep analysis
NFL_WILD_CARD_OPPORTUNITIES = [
    EventOpportunity(
        event_name="Bills @ Jaguars - Wild Card",
        market_slug="jaguars-vs-bills-wild-card",
        condition_id="",  # Will be fetched dynamically
        outcome="Jaguars",
        current_price=0.50,  # As of last check
        estimated_true_prob=0.55,  # Based on: 8-game streak, #1 rush D, home field, Allen 0-4 road playoffs
        edge=0.05,
        volume_24h=124840,
        min_volume_threshold=50000,
        conviction_level="HIGH",
        reasoning="""
        MOMENTUM: Jaguars on 8-game winning streak (hottest team in NFL)
        DEFENSE: #1 rush defense in NFL
        HOME FIELD: Playing at home in playoffs (~2.5 point advantage)
        JOSH ALLEN: 0-4 in road playoff games (statistical fact)
        MARKET: Bills are favorites by name/brand, not performance
        """,
        game_time=datetime(2026, 1, 11, 16, 0, tzinfo=timezone.utc)  # 11am ET
    ),
    EventOpportunity(
        event_name="Packers @ Bears - Wild Card",
        market_slug="packers-vs-bears-wild-card",
        condition_id="",
        outcome="Bears",
        current_price=0.48,
        estimated_true_prob=0.52,  # CBS model: Bears 25, Packers 23
        edge=0.04,
        volume_24h=236650,
        min_volume_threshold=50000,
        conviction_level="MEDIUM",
        reasoning="""
        HOME DOMINANCE: Bears 6-1 at home in last 7 games
        RECENT HISTORY: Beat Packers 22-16 OT on Dec 20 (3 weeks ago)
        PACKERS SLUMP: 4-game losing streak entering playoffs
        JORDAN LOVE: Returning from concussion (rust factor)
        MODEL: CBS advanced model predicts Bears 25, Packers 23
        """,
        game_time=datetime(2026, 1, 10, 23, 0, tzinfo=timezone.utc)  # 6pm ET
    ),
]


class EventMonitor:
    """
    Monitors event markets and identifies trading opportunities.

    Law of Attraction principles:
    - Focus on ONE high-conviction opportunity at a time
    - Wait for alignment (price + volume + timing)
    - Only act when conditions are optimal
    """

    def __init__(self):
        self.opportunities = NFL_WILD_CARD_OPPORTUNITIES
        self.min_edge = 0.03  # 3% minimum edge to consider
        self.min_volume = 50000  # $50k minimum 24h volume
        self.min_activity_trades = 10  # Minimum trades in last hour

        # Trading params
        self.trade_size = float(os.getenv("EVENT_TRADE_SIZE", "0.92"))  # All available capital
        self.dry_run = os.getenv("EVENT_DRY_RUN", "true").lower() == "true"

        # Discord alerts
        self.discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")

    async def check_market_activity(self, market_slug: str) -> Dict:
        """Check if market has recent trading activity."""
        # TODO: Implement via Polymarket API
        # For now, return mock data
        return {
            "trades_1h": 15,
            "volume_1h": 5000,
            "spread": 0.02,
            "liquidity_depth": 10000
        }

    async def get_current_price(self, market_slug: str, outcome: str) -> float:
        """Fetch current price from Polymarket."""
        # TODO: Implement via CLOB API
        # For now, return last known price
        for opp in self.opportunities:
            if opp.market_slug == market_slug:
                return opp.current_price
        return 0.0

    def calculate_position_size(self, edge: float, capital: float) -> float:
        """
        Kelly Criterion position sizing.

        f* = (p * b - q) / b
        where p = win prob, q = lose prob, b = win/loss ratio

        For binary markets: b = (1/price) - 1
        """
        # Use half-Kelly for safety
        kelly_fraction = edge * 0.5
        return min(capital * kelly_fraction, capital)  # Never bet more than capital

    async def evaluate_opportunity(self, opp: EventOpportunity) -> Dict:
        """
        Evaluate if an opportunity is ready to trade.

        Returns dict with:
        - ready: bool
        - reason: str
        - recommended_size: float
        """
        result = {
            "ready": False,
            "reason": "",
            "recommended_size": 0.0,
            "opportunity": opp
        }

        # Check edge
        if opp.edge < self.min_edge:
            result["reason"] = f"Edge {opp.edge:.1%} below minimum {self.min_edge:.1%}"
            return result

        # Check volume
        if opp.volume_24h < self.min_volume:
            result["reason"] = f"Volume ${opp.volume_24h:,.0f} below minimum ${self.min_volume:,.0f}"
            return result

        # Check market activity
        activity = await self.check_market_activity(opp.market_slug)
        if activity["trades_1h"] < self.min_activity_trades:
            result["reason"] = f"Only {activity['trades_1h']} trades in last hour (need {self.min_activity_trades}+)"
            return result

        # Check timing (don't trade too close to game)
        if opp.game_time:
            time_to_game = (opp.game_time - datetime.now(timezone.utc)).total_seconds() / 3600
            if time_to_game < 1:
                result["reason"] = f"Game starts in {time_to_game:.1f}h - too close to kickoff"
                return result
            if time_to_game > 72:
                result["reason"] = f"Game in {time_to_game:.0f}h - waiting for closer to game time"
                return result

        # All checks passed!
        result["ready"] = True
        result["reason"] = "All conditions met - alignment detected!"
        result["recommended_size"] = self.calculate_position_size(opp.edge, self.trade_size)

        return result

    async def send_alert(self, message: str, urgent: bool = False):
        """Send Discord alert."""
        if not self.discord_webhook:
            logger.info(f"[ALERT] {message}")
            return

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                emoji = "🚨" if urgent else "📊"
                await session.post(self.discord_webhook, json={
                    "content": f"{emoji} **Event Monitor**\n{message}"
                })
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")

    async def execute_trade(self, evaluation: Dict) -> bool:
        """Execute trade if conditions are met."""
        opp = evaluation["opportunity"]
        size = evaluation["recommended_size"]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would buy {opp.outcome} @ {opp.current_price:.2f} for ${size:.2f}")
            await self.send_alert(
                f"**DRY RUN TRADE**\n"
                f"Event: {opp.event_name}\n"
                f"Bet: {opp.outcome} @ {opp.current_price:.0%}\n"
                f"Size: ${size:.2f}\n"
                f"Edge: {opp.edge:.1%}\n"
                f"Conviction: {opp.conviction_level}"
            )
            return True

        # TODO: Implement actual trade execution
        # from helpers.clob_executor import ClobExecutor
        logger.info(f"[LIVE] Executing trade: {opp.outcome} @ {opp.current_price:.2f} for ${size:.2f}")
        return False

    async def run_scan(self):
        """Scan all opportunities and identify actionable ones."""
        logger.info("=" * 60)
        logger.info("EVENT MONITOR - Scanning for opportunities")
        logger.info("=" * 60)

        actionable = []

        for opp in self.opportunities:
            logger.info(f"\n[{opp.conviction_level}] {opp.event_name}")
            logger.info(f"  Outcome: {opp.outcome} @ {opp.current_price:.0%}")
            logger.info(f"  True Prob Est: {opp.estimated_true_prob:.0%}")
            logger.info(f"  Edge: {opp.edge:.1%}")
            logger.info(f"  Volume: ${opp.volume_24h:,.0f}")

            evaluation = await self.evaluate_opportunity(opp)

            if evaluation["ready"]:
                logger.info(f"  ✅ READY: {evaluation['reason']}")
                logger.info(f"  Recommended size: ${evaluation['recommended_size']:.2f}")
                actionable.append(evaluation)
            else:
                logger.info(f"  ⏳ WAITING: {evaluation['reason']}")

        return actionable

    async def monitor_loop(self, interval_minutes: int = 30):
        """Continuous monitoring loop."""
        logger.info(f"Starting event monitor (interval: {interval_minutes}min, dry_run: {self.dry_run})")

        await self.send_alert(
            f"Event Monitor started\n"
            f"Tracking {len(self.opportunities)} opportunities\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}"
        )

        while True:
            try:
                actionable = await self.run_scan()

                if actionable:
                    # Focus on highest conviction opportunity (Law of Attraction)
                    best = max(actionable, key=lambda x: x["opportunity"].edge)
                    opp = best["opportunity"]

                    logger.info(f"\n🎯 ALIGNMENT DETECTED: {opp.event_name}")
                    logger.info(f"   {opp.reasoning}")

                    await self.send_alert(
                        f"**ALIGNMENT DETECTED**\n"
                        f"Event: {opp.event_name}\n"
                        f"Bet: {opp.outcome}\n"
                        f"Edge: {opp.edge:.1%}\n"
                        f"Conviction: {opp.conviction_level}",
                        urgent=True
                    )

                    # Execute if not dry run
                    if not self.dry_run:
                        success = await self.execute_trade(best)
                        if success:
                            logger.info("Trade executed successfully!")
                            break  # Exit after trade (one focused bet)

                logger.info(f"\nNext scan in {interval_minutes} minutes...")
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(60)


async def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Event Monitor - Law of Attraction Trading")
    parser.add_argument("--scan", action="store_true", help="Run single scan and exit")
    parser.add_argument("--interval", type=int, default=30, help="Scan interval in minutes")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: dry run)")
    args = parser.parse_args()

    # Override dry_run if --live flag
    if args.live:
        os.environ["EVENT_DRY_RUN"] = "false"

    monitor = EventMonitor()

    if args.scan:
        actionable = await monitor.run_scan()
        if actionable:
            print(f"\n✅ Found {len(actionable)} actionable opportunities")
        else:
            print("\n⏳ No opportunities ready yet - waiting for alignment")
    else:
        await monitor.monitor_loop(args.interval)


if __name__ == "__main__":
    asyncio.run(main())
