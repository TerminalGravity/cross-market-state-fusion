#!/usr/bin/env python3
"""
Polymarket Universal Scanner

Scans ALL markets on Polymarket to find opportunities with:
1. Mispriced odds (vs external sources, models, or arbitrage)
2. High liquidity and activity
3. Clear resolution criteria
4. Favorable risk/reward

Uses multiple detection strategies:
- Arbitrage detection (UP + DOWN != 100%)
- Cross-reference with Vegas/prediction sites
- Volume anomalies (sudden interest)
- Price momentum (moving in one direction)
- LLM-based probability estimation

Philosophy: The scanner finds opportunities, YOU decide which align with your conviction.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import aiohttp

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
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class OpportunityType(Enum):
    ARBITRAGE = "arbitrage"  # UP + DOWN != 100%
    MISPRICING = "mispricing"  # Price differs from true probability
    MOMENTUM = "momentum"  # Price moving strongly one direction
    VOLUME_SPIKE = "volume_spike"  # Unusual trading activity
    VALUE_BET = "value_bet"  # Positive expected value


@dataclass
class MarketOpportunity:
    """A detected trading opportunity."""
    market_id: str
    market_title: str
    category: str
    opportunity_type: OpportunityType
    outcome: str
    current_price: float
    estimated_fair_price: float
    edge_percent: float
    volume_24h: float
    liquidity: float
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    url: str
    expires: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "market": self.market_title,
            "type": self.opportunity_type.value,
            "outcome": self.outcome,
            "price": f"{self.current_price:.0%}",
            "fair": f"{self.estimated_fair_price:.0%}",
            "edge": f"{self.edge_percent:.1%}",
            "volume": f"${self.volume_24h:,.0f}",
            "confidence": self.confidence,
            "url": self.url
        }


class PolymarketScanner:
    """
    Universal market scanner for Polymarket.

    Fetches markets via Gamma API and analyzes for opportunities.
    """

    GAMMA_API = "https://gamma-api.polymarket.com"
    CLOB_API = "https://clob.polymarket.com"

    CATEGORIES = [
        "sports", "politics", "crypto", "finance",
        "pop-culture", "science", "world"
    ]

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.min_volume = float(os.getenv("SCANNER_MIN_VOLUME", "10000"))
        self.min_edge = float(os.getenv("SCANNER_MIN_EDGE", "0.03"))
        self.opportunities: List[MarketOpportunity] = []

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def fetch_markets(self, category: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Fetch active markets from Gamma API."""
        params = {
            "limit": limit,
            "active": "true",
            "closed": "false",
        }
        if category:
            params["tag"] = category

        try:
            async with self.session.get(
                f"{self.GAMMA_API}/markets",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.warning(f"Gamma API returned {resp.status}")
                    return []
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []

    async def fetch_orderbook(self, token_id: str) -> Dict:
        """Fetch orderbook for a token."""
        try:
            async with self.session.get(
                f"{self.CLOB_API}/book",
                params={"token_id": token_id},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}
        except:
            return {}

    def detect_arbitrage(self, market: Dict) -> Optional[MarketOpportunity]:
        """
        Detect arbitrage opportunities where UP + DOWN != 100%.

        In binary markets, the sum of YES + NO should equal ~100%.
        Deviations indicate market inefficiency.
        """
        tokens = market.get("tokens", [])
        if len(tokens) != 2:
            return None

        try:
            price_yes = float(tokens[0].get("price", 0.5))
            price_no = float(tokens[1].get("price", 0.5))
            total = price_yes + price_no

            # Look for significant deviation from 1.0
            if total < 0.97:  # Can buy both for less than $1
                edge = 1.0 - total
                return MarketOpportunity(
                    market_id=market.get("id", ""),
                    market_title=market.get("question", "Unknown"),
                    category=market.get("category", "unknown"),
                    opportunity_type=OpportunityType.ARBITRAGE,
                    outcome="BUY BOTH",
                    current_price=total,
                    estimated_fair_price=1.0,
                    edge_percent=edge,
                    volume_24h=float(market.get("volume24hr", 0)),
                    liquidity=float(market.get("liquidity", 0)),
                    confidence="HIGH" if edge > 0.05 else "MEDIUM",
                    reasoning=f"Buy YES @ {price_yes:.0%} + NO @ {price_no:.0%} = {total:.0%} < 100%",
                    url=f"https://polymarket.com/event/{market.get('slug', '')}"
                )
            elif total > 1.03:  # Can sell both for more than $1
                edge = total - 1.0
                return MarketOpportunity(
                    market_id=market.get("id", ""),
                    market_title=market.get("question", "Unknown"),
                    category=market.get("category", "unknown"),
                    opportunity_type=OpportunityType.ARBITRAGE,
                    outcome="SELL BOTH",
                    current_price=total,
                    estimated_fair_price=1.0,
                    edge_percent=edge,
                    volume_24h=float(market.get("volume24hr", 0)),
                    liquidity=float(market.get("liquidity", 0)),
                    confidence="HIGH" if edge > 0.05 else "MEDIUM",
                    reasoning=f"Sell YES @ {price_yes:.0%} + NO @ {price_no:.0%} = {total:.0%} > 100%",
                    url=f"https://polymarket.com/event/{market.get('slug', '')}"
                )
        except:
            pass

        return None

    def detect_momentum(self, market: Dict) -> Optional[MarketOpportunity]:
        """
        Detect strong price momentum.

        Look for markets where price is moving strongly in one direction,
        suggesting informed trading or news.
        """
        # Would need historical price data - placeholder
        return None

    def detect_volume_spike(self, market: Dict) -> Optional[MarketOpportunity]:
        """
        Detect unusual volume spikes.

        High volume relative to liquidity suggests new information.
        """
        volume_24h = float(market.get("volume24hr", 0))
        liquidity = float(market.get("liquidity", 1))

        if liquidity > 0:
            volume_ratio = volume_24h / liquidity

            # Flag if 24h volume > 50% of liquidity
            if volume_ratio > 0.5 and volume_24h > self.min_volume:
                tokens = market.get("tokens", [])
                if tokens:
                    # Find the outcome that's moving
                    main_token = tokens[0]
                    price = float(main_token.get("price", 0.5))

                    return MarketOpportunity(
                        market_id=market.get("id", ""),
                        market_title=market.get("question", "Unknown"),
                        category=market.get("category", "unknown"),
                        opportunity_type=OpportunityType.VOLUME_SPIKE,
                        outcome=main_token.get("outcome", "YES"),
                        current_price=price,
                        estimated_fair_price=price,  # Unknown without analysis
                        edge_percent=0.0,
                        volume_24h=volume_24h,
                        liquidity=liquidity,
                        confidence="LOW",  # Needs investigation
                        reasoning=f"Volume ${volume_24h:,.0f} is {volume_ratio:.0%} of liquidity - unusual activity",
                        url=f"https://polymarket.com/event/{market.get('slug', '')}"
                    )
        return None

    async def scan_category(self, category: str) -> List[MarketOpportunity]:
        """Scan a single category for opportunities."""
        logger.info(f"Scanning {category}...")
        markets = await self.fetch_markets(category=category)
        opportunities = []

        for market in markets:
            volume = float(market.get("volume24hr", 0))
            if volume < self.min_volume:
                continue

            # Run all detectors
            arb = self.detect_arbitrage(market)
            if arb and arb.edge_percent >= self.min_edge:
                opportunities.append(arb)

            volume_spike = self.detect_volume_spike(market)
            if volume_spike:
                opportunities.append(volume_spike)

        return opportunities

    async def scan_all(self) -> List[MarketOpportunity]:
        """Scan all categories for opportunities."""
        logger.info("=" * 60)
        logger.info("POLYMARKET UNIVERSAL SCANNER")
        logger.info(f"Min Volume: ${self.min_volume:,.0f} | Min Edge: {self.min_edge:.0%}")
        logger.info("=" * 60)

        all_opportunities = []

        # Also scan without category filter to catch everything
        markets = await self.fetch_markets(limit=500)
        logger.info(f"Fetched {len(markets)} markets from Gamma API")

        for market in markets:
            volume = float(market.get("volume24hr", 0))
            if volume < self.min_volume:
                continue

            arb = self.detect_arbitrage(market)
            if arb and arb.edge_percent >= self.min_edge:
                all_opportunities.append(arb)

            volume_spike = self.detect_volume_spike(market)
            if volume_spike:
                all_opportunities.append(volume_spike)

        # Sort by edge
        all_opportunities.sort(key=lambda x: x.edge_percent, reverse=True)

        self.opportunities = all_opportunities
        return all_opportunities

    def display_opportunities(self):
        """Display found opportunities."""
        if not self.opportunities:
            logger.info("\n❌ No opportunities found matching criteria")
            return

        logger.info(f"\n✅ Found {len(self.opportunities)} opportunities:\n")

        for i, opp in enumerate(self.opportunities, 1):
            print(f"{i}. [{opp.opportunity_type.value.upper()}] {opp.market_title}")
            print(f"   Outcome: {opp.outcome} @ {opp.current_price:.0%}")
            print(f"   Edge: {opp.edge_percent:.1%} | Volume: ${opp.volume_24h:,.0f}")
            print(f"   Confidence: {opp.confidence}")
            print(f"   Reasoning: {opp.reasoning}")
            print(f"   URL: {opp.url}")
            print()

    def export_json(self, filepath: str = "scan_results.json"):
        """Export opportunities to JSON."""
        data = {
            "scan_time": datetime.now(timezone.utc).isoformat(),
            "count": len(self.opportunities),
            "opportunities": [opp.to_dict() for opp in self.opportunities]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported to {filepath}")


class NFLScanner:
    """
    Specialized scanner for NFL markets with external odds comparison.
    """

    # Known NFL Wild Card games with Vegas odds for comparison
    NFL_GAMES = {
        "packers-bears": {
            "home": "Bears",
            "away": "Packers",
            "vegas_home_ml": 0.48,  # -1.5 spread implies ~48%
            "notes": "Bears 6-1 at home, Packers 4-game losing streak"
        },
        "bills-jaguars": {
            "home": "Jaguars",
            "away": "Bills",
            "vegas_home_ml": 0.50,  # Pick'em essentially
            "notes": "Jaguars 8-game streak, Allen 0-4 road playoffs"
        },
        "rams-panthers": {
            "home": "Panthers",
            "away": "Rams",
            "vegas_home_ml": 0.17,  # +10.5 spread
            "notes": "Panthers beat Rams 31-28 in Nov"
        },
        "eagles-49ers": {
            "home": "Eagles",
            "away": "49ers",
            "vegas_home_ml": 0.68,  # -4.5 spread
            "notes": "Eagles strong at home"
        },
        "texans-steelers": {
            "home": "Steelers",
            "away": "Texans",
            "vegas_home_ml": 0.43,  # +3 spread
            "notes": "Steelers 4-1 ATS last 5"
        }
    }

    async def compare_with_vegas(self, polymarket_price: float, vegas_price: float,
                                  market_name: str, team: str) -> Optional[MarketOpportunity]:
        """Compare Polymarket price with Vegas implied probability."""
        edge = vegas_price - polymarket_price

        if abs(edge) >= 0.03:  # 3% edge minimum
            return MarketOpportunity(
                market_id=market_name,
                market_title=f"NFL: {market_name}",
                category="sports",
                opportunity_type=OpportunityType.MISPRICING,
                outcome=team,
                current_price=polymarket_price,
                estimated_fair_price=vegas_price,
                edge_percent=edge,
                volume_24h=0,  # Would need to fetch
                liquidity=0,
                confidence="MEDIUM" if abs(edge) < 0.05 else "HIGH",
                reasoning=f"Vegas implies {vegas_price:.0%} but Polymarket has {polymarket_price:.0%}",
                url=f"https://polymarket.com/sports/nfl/games"
            )
        return None


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Universal Scanner")
    parser.add_argument("--category", help="Scan specific category")
    parser.add_argument("--min-volume", type=float, default=10000, help="Min 24h volume")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Min edge percentage")
    parser.add_argument("--export", action="store_true", help="Export to JSON")
    args = parser.parse_args()

    os.environ["SCANNER_MIN_VOLUME"] = str(args.min_volume)
    os.environ["SCANNER_MIN_EDGE"] = str(args.min_edge)

    async with PolymarketScanner() as scanner:
        if args.category:
            opportunities = await scanner.scan_category(args.category)
            scanner.opportunities = opportunities
        else:
            await scanner.scan_all()

        scanner.display_opportunities()

        if args.export:
            scanner.export_json()


if __name__ == "__main__":
    asyncio.run(main())
