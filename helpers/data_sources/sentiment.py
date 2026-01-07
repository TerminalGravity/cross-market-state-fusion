"""
Sentiment data source for market psychology signals.

Provides:
- Fear & Greed Index (Alternative.me - free)
- Social volume (LunarCrush - freemium)
- Social sentiment score

Free APIs:
- Alternative.me Fear & Greed: https://alternative.me/crypto/fear-and-greed-index/
- LunarCrush: https://lunarcrush.com/developers (requires free API key)

Key insight: Extreme fear = buy opportunity, extreme greed = sell signal.
Social volume spikes often precede volatility (direction unclear).
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import deque

import aiohttp

from .base import DataSource, Features

logger = logging.getLogger(__name__)


class SentimentSource(DataSource):
    """
    Sentiment data source for market psychology.

    For 15-min trading, sentiment is a secondary signal because:
    1. Fear & Greed updates daily (too slow for intraday)
    2. Social volume has lag

    However, extreme values provide useful context:
    - Fear < 25: Consider being more aggressive on bullish signals
    - Greed > 75: Consider being more defensive

    Better for regime detection than entry timing.
    """

    def __init__(
        self,
        assets: list = None,
        lunarcrush_api_key: str = None,
        update_interval: float = 300.0,  # 5 min - sentiment changes slowly
        enabled: bool = True
    ):
        super().__init__(
            name="sentiment",
            assets=assets or ["BTC", "ETH", "SOL", "XRP"],
            update_interval=update_interval,
            enabled=enabled
        )

        self.lunarcrush_api_key = lunarcrush_api_key or os.getenv("LUNARCRUSH_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None

        # Cache Fear & Greed (same for all assets, updates daily)
        self._fear_greed_cache: tuple = (None, None)  # (value, timestamp)

        # Track sentiment history
        self._sentiment_history: Dict[str, deque] = {
            asset: deque(maxlen=24) for asset in self.assets
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_data(self, asset: str) -> Dict[str, float]:
        """Fetch sentiment metrics for an asset."""
        features = {}

        try:
            # Fear & Greed is global (same for all assets)
            fg_task = self._fetch_fear_greed()

            # Social metrics are per-asset
            social_task = self._fetch_social_metrics(asset)

            results = await asyncio.gather(
                fg_task, social_task,
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, dict):
                    features.update(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Sentiment/{asset} partial failure: {result}")

        except Exception as e:
            logger.warning(f"Sentiment/{asset} fetch error: {e}")

        return features

    async def _fetch_fear_greed(self) -> Dict[str, float]:
        """
        Fetch Crypto Fear & Greed Index.

        Scale: 0-100
        0-24: Extreme Fear
        25-49: Fear
        50-74: Greed
        75-100: Extreme Greed

        Contrarian signal: Buy fear, sell greed.
        """
        # Check cache (F&G updates once daily)
        if self._fear_greed_cache[0] is not None:
            cached_value, cached_time = self._fear_greed_cache
            if datetime.utcnow() - cached_time < timedelta(hours=1):
                return {Features.FEAR_GREED: cached_value / 100}  # Normalize to 0-1

        try:
            session = await self._get_session()

            url = "https://api.alternative.me/fng/"
            params = {"limit": 1}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if "data" not in data or not data["data"]:
                    return {}

                fg_data = data["data"][0]
                value = int(fg_data.get("value", 50))

                # Cache result
                self._fear_greed_cache = (value, datetime.utcnow())

                # Normalize to 0-1 range
                return {Features.FEAR_GREED: value / 100}

        except Exception as e:
            logger.debug(f"Fear & Greed fetch error: {e}")
            return {}

    async def _fetch_social_metrics(self, asset: str) -> Dict[str, float]:
        """
        Fetch social metrics from LunarCrush.

        Metrics:
        - Social volume: Total social posts mentioning the asset
        - Sentiment: Positive vs negative ratio
        - Galaxy score: LunarCrush's aggregate metric

        For 15-min trading:
        - Social volume spike = potential volatility
        - Sentiment extreme = potential reversal
        """
        if not self.lunarcrush_api_key:
            # Return defaults without API key
            return {
                Features.SOCIAL_VOLUME: 0.5,  # Neutral
                Features.SOCIAL_SENTIMENT: 0.5,  # Neutral
            }

        try:
            session = await self._get_session()

            # LunarCrush v2 API
            url = "https://lunarcrush.com/api3/coins"
            params = {
                "key": self.lunarcrush_api_key,
                "symbol": asset,
            }

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if "data" not in data or not data["data"]:
                    return {}

                coin_data = data["data"][0] if isinstance(data["data"], list) else data["data"]

                result = {}

                # Social volume (normalize by typical range)
                social_volume = coin_data.get("social_volume", 0)
                # Typical range varies wildly - use log scale
                if social_volume > 0:
                    import math
                    # Log scale, typical range 100-100000
                    normalized = (math.log10(social_volume + 1) - 2) / 3
                    result[Features.SOCIAL_VOLUME] = max(0, min(1, normalized))
                else:
                    result[Features.SOCIAL_VOLUME] = 0.0

                # Sentiment score (typically 0-5)
                sentiment = coin_data.get("sentiment", 2.5)
                result[Features.SOCIAL_SENTIMENT] = sentiment / 5.0

                return result

        except Exception as e:
            logger.debug(f"LunarCrush fetch error: {e}")
            return {}

    async def stop(self):
        """Clean up."""
        await super().stop()
        if self._session and not self._session.closed:
            await self._session.close()


class TwitterSentiment:
    """
    Twitter/X sentiment analysis (requires API access).

    For crypto, key accounts to monitor:
    - @whale_alert: Large transfers
    - @santaborgia: Technical analysis
    - Exchange official accounts

    This is a placeholder - implementation requires Twitter API v2 access.
    """

    def __init__(self, bearer_token: str = None):
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN", "")
        self._running = False

        # Track mention spikes
        self.mention_counts: Dict[str, deque] = {}

    async def start_monitoring(self, assets: list = None):
        """Start monitoring Twitter for crypto mentions."""
        if not self.bearer_token:
            logger.info("Twitter API not configured, skipping")
            return

        # Implementation would use Twitter Filtered Stream API
        # to monitor mentions of BTC, ETH, etc.
        pass

    def get_mention_spike(self, asset: str) -> float:
        """
        Get mention spike indicator.

        Returns value > 1 if current mentions exceed 2-hour average.
        """
        if asset not in self.mention_counts:
            return 1.0

        history = self.mention_counts[asset]
        if len(history) < 10:
            return 1.0

        recent = list(history)[-5:]
        baseline = list(history)[:-5]

        avg_recent = sum(recent) / len(recent)
        avg_baseline = sum(baseline) / len(baseline) if baseline else 1

        return avg_recent / max(1, avg_baseline)


class FearGreedClassifier:
    """
    Utility class for interpreting Fear & Greed Index.

    Use for regime-based position sizing or filtering.
    """

    @staticmethod
    def classify(value: float) -> str:
        """
        Classify F&G value (0-1 normalized).

        Returns: 'extreme_fear', 'fear', 'neutral', 'greed', 'extreme_greed'
        """
        if value < 0.25:
            return 'extreme_fear'
        elif value < 0.45:
            return 'fear'
        elif value < 0.55:
            return 'neutral'
        elif value < 0.75:
            return 'greed'
        else:
            return 'extreme_greed'

    @staticmethod
    def get_position_multiplier(value: float) -> float:
        """
        Get position size multiplier based on F&G.

        Contrarian approach:
        - Extreme fear: Increase long exposure (1.2x)
        - Extreme greed: Reduce exposure (0.8x)
        - Neutral: Normal (1.0x)
        """
        regime = FearGreedClassifier.classify(value)

        multipliers = {
            'extreme_fear': 1.2,
            'fear': 1.1,
            'neutral': 1.0,
            'greed': 0.9,
            'extreme_greed': 0.8,
        }

        return multipliers.get(regime, 1.0)

    @staticmethod
    def should_skip_trade(value: float, direction: str) -> bool:
        """
        Check if trade should be skipped based on F&G.

        Skip longs in extreme greed, skip shorts in extreme fear.
        """
        regime = FearGreedClassifier.classify(value)

        if direction == "long" and regime == 'extreme_greed':
            return True
        if direction == "short" and regime == 'extreme_fear':
            return True

        return False
