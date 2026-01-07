"""
Coinglass data source for aggregated CEX metrics.

Provides:
- Cross-exchange funding rates (Binance, OKX, Bybit, dYdX)
- Aggregated open interest and OI changes
- Long/short ratio across exchanges
- Liquidation data

Free API: https://www.coinglass.com/api (rate limited but sufficient for 15-min trading)
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

import aiohttp

from .base import DataSource, Features

logger = logging.getLogger(__name__)

# Coinglass API endpoints
COINGLASS_API = "https://open-api.coinglass.com/public/v2"

# Asset mapping
COINGLASS_SYMBOLS = {
    "BTC": "BTC",
    "ETH": "ETH",
    "SOL": "SOL",
    "XRP": "XRP",
}


class CoinglassSource(DataSource):
    """
    Coinglass aggregated CEX data source.

    Key signals for 15-min trading:
    - Funding rate divergence: When exchanges disagree, arbitrage corrects it
    - OI spikes: New positions opening often precede moves
    - Long/short ratio extremes: Crowded trades tend to reverse
    - Liquidation cascades: Forced selling creates momentum
    """

    def __init__(
        self,
        assets: list = None,
        api_key: str = None,
        update_interval: float = 30.0,  # Coinglass rate limits
        enabled: bool = True
    ):
        super().__init__(
            name="coinglass",
            assets=assets or ["BTC", "ETH", "SOL", "XRP"],
            update_interval=update_interval,
            enabled=enabled
        )

        self.api_key = api_key or os.getenv("COINGLASS_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None

        # Cache for rate limiting
        self._funding_cache: Dict[str, tuple] = {}  # asset -> (data, timestamp)
        self._oi_cache: Dict[str, tuple] = {}
        self._ls_cache: Dict[str, tuple] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["coinglassSecret"] = self.api_key
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def fetch_data(self, asset: str) -> Dict[str, float]:
        """Fetch all Coinglass metrics for an asset."""
        features = {}

        symbol = COINGLASS_SYMBOLS.get(asset)
        if not symbol:
            return features

        try:
            # Fetch in parallel
            funding_task = self._fetch_funding_rates(symbol)
            oi_task = self._fetch_open_interest(symbol)
            ls_task = self._fetch_long_short_ratio(symbol)
            liq_task = self._fetch_liquidations(symbol)

            results = await asyncio.gather(
                funding_task, oi_task, ls_task, liq_task,
                return_exceptions=True
            )

            # Merge results
            for result in results:
                if isinstance(result, dict):
                    features.update(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Coinglass/{asset} partial failure: {result}")

        except Exception as e:
            logger.warning(f"Coinglass/{asset} fetch error: {e}")

        return features

    async def _fetch_funding_rates(self, symbol: str) -> Dict[str, float]:
        """
        Fetch funding rates from multiple exchanges.

        Key insight: Funding rate DIVERGENCE between exchanges
        is more predictive than absolute funding rate.
        """
        # Check cache (funding updates every 8h, no need to hammer API)
        cache_key = f"funding_{symbol}"
        if cache_key in self._funding_cache:
            data, ts = self._funding_cache[cache_key]
            if datetime.utcnow() - ts < timedelta(minutes=5):
                return data

        try:
            session = await self._get_session()

            # Coinglass funding rate endpoint
            url = f"{COINGLASS_API}/funding"
            params = {"symbol": symbol}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if data.get("code") != "0" or not data.get("data"):
                    return {}

                # Parse exchange-specific funding
                rates = {}
                for item in data["data"]:
                    exchange = item.get("exchangeName", "").lower()
                    rate = float(item.get("rate", 0))

                    if "binance" in exchange:
                        rates[Features.FUNDING_RATE_BINANCE] = rate
                    elif "okx" in exchange:
                        rates[Features.FUNDING_RATE_OKX] = rate
                    elif "bybit" in exchange:
                        rates[Features.FUNDING_RATE_BYBIT] = rate

                # Calculate divergence (max spread)
                if rates:
                    values = list(rates.values())
                    rates[Features.FUNDING_DIVERGENCE] = max(values) - min(values)

                # Cache result
                self._funding_cache[cache_key] = (rates, datetime.utcnow())

                return rates

        except Exception as e:
            logger.debug(f"Coinglass funding error: {e}")
            return {}

    async def _fetch_open_interest(self, symbol: str) -> Dict[str, float]:
        """
        Fetch aggregated open interest.

        Key insight: OI changes show new money entering/exiting.
        Rising OI + rising price = new longs (bullish)
        Rising OI + falling price = new shorts (bearish)
        """
        try:
            session = await self._get_session()

            url = f"{COINGLASS_API}/open_interest"
            params = {"symbol": symbol}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if data.get("code") != "0" or not data.get("data"):
                    return {}

                oi_data = data["data"]

                result = {}

                # Total OI across exchanges
                if isinstance(oi_data, list) and oi_data:
                    total_oi = sum(float(x.get("openInterest", 0)) for x in oi_data)
                    result[Features.OI_TOTAL] = total_oi

                    # OI change (if history available)
                    if "h1OiChangePercent" in oi_data[0]:
                        result[Features.OI_CHANGE_1H] = float(oi_data[0]["h1OiChangePercent"])
                    if "h4OiChangePercent" in oi_data[0]:
                        result[Features.OI_CHANGE_4H] = float(oi_data[0]["h4OiChangePercent"])

                return result

        except Exception as e:
            logger.debug(f"Coinglass OI error: {e}")
            return {}

    async def _fetch_long_short_ratio(self, symbol: str) -> Dict[str, float]:
        """
        Fetch long/short position ratio.

        Key insight: Extreme ratios (>2.0 or <0.5) often precede reversals.
        Crowded trades get squeezed.
        """
        try:
            session = await self._get_session()

            url = f"{COINGLASS_API}/long_short"
            params = {"symbol": symbol}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if data.get("code") != "0" or not data.get("data"):
                    return {}

                ls_data = data["data"]

                result = {}

                # Aggregate ratio
                if isinstance(ls_data, list) and ls_data:
                    # Average across exchanges
                    ratios = [float(x.get("longShortRatio", 1.0)) for x in ls_data]
                    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
                    result[Features.LONG_SHORT_RATIO] = avg_ratio

                return result

        except Exception as e:
            logger.debug(f"Coinglass L/S error: {e}")
            return {}

    async def _fetch_liquidations(self, symbol: str) -> Dict[str, float]:
        """
        Fetch recent liquidation data.

        Key insight: Liquidation cascades create forced selling/buying,
        often overshooting fair value. Trade the reversion.
        """
        try:
            session = await self._get_session()

            url = f"{COINGLASS_API}/liquidation"
            params = {"symbol": symbol, "time_type": "h1"}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if data.get("code") != "0" or not data.get("data"):
                    return {}

                liq_data = data["data"]

                result = {}

                if isinstance(liq_data, dict):
                    long_liqs = float(liq_data.get("longLiquidationUsd", 0))
                    short_liqs = float(liq_data.get("shortLiquidationUsd", 0))
                    total = long_liqs + short_liqs

                    result[Features.LIQUIDATION_VOLUME_1H] = total

                    # Net pressure (positive = more longs liquidated = bearish)
                    if total > 0:
                        result[Features.LIQUIDATION_PRESSURE] = (long_liqs - short_liqs) / total
                    else:
                        result[Features.LIQUIDATION_PRESSURE] = 0.0

                return result

        except Exception as e:
            logger.debug(f"Coinglass liquidation error: {e}")
            return {}

    async def stop(self):
        """Clean up."""
        await super().stop()
        if self._session and not self._session.closed:
            await self._session.close()
