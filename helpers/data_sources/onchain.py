"""
On-chain data source for whale tracking and exchange flows.

Provides:
- Exchange inflows/outflows (CryptoQuant-style metrics)
- Whale wallet movements
- Large transfer alerts

Free APIs used:
- Blockchain.com (BTC)
- Etherscan (ETH - requires free API key)
- Whale Alert API (optional, for alerts)

Key insight: Large exchange deposits often precede selling pressure.
Whales move to exchanges to sell, move off exchanges to hold.
"""
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque

import aiohttp

from .base import DataSource, Features

logger = logging.getLogger(__name__)

# Known exchange addresses (simplified - production would use full lists)
# These are hot wallet addresses for major exchanges
EXCHANGE_ADDRESSES = {
    "BTC": {
        # Binance cold wallets
        "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo": "binance",
        "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97": "binance",
        # Coinbase
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j": "coinbase",
        # Kraken
        "3AfPT6h3vnQ1e2K6X5U4PW7d6J3cQk7vqw": "kraken",
    },
    "ETH": {
        # Binance
        "0x28c6c06298d514db089934071355e5743bf21d60": "binance",
        "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "binance",
        # Coinbase
        "0x71660c4005ba85c37ccec55d0c4493e66fe775d3": "coinbase",
        # Kraken
        "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "kraken",
    }
}

# Whale thresholds (in native units)
WHALE_THRESHOLDS = {
    "BTC": 100,    # 100 BTC = ~$6M
    "ETH": 1000,   # 1000 ETH = ~$3M
    "SOL": 50000,  # 50k SOL = ~$5M
    "XRP": 10000000,  # 10M XRP = ~$5M
}


class OnchainSource(DataSource):
    """
    On-chain data source for whale and exchange flow tracking.

    For 15-min trading, on-chain data has limited use because:
    1. Confirmations take time (BTC ~10min, ETH ~15sec)
    2. Most flow data has 30min+ lag

    However, sudden large exchange deposits CAN move prices within 15min
    as traders front-run expected selling.

    Key signals:
    - Large exchange inflows = selling pressure incoming
    - Large exchange outflows = accumulation (bullish)
    - Whale alert = potential volatility
    """

    def __init__(
        self,
        assets: list = None,
        etherscan_api_key: str = None,
        update_interval: float = 60.0,  # On-chain data is slower
        enabled: bool = True
    ):
        super().__init__(
            name="onchain",
            assets=assets or ["BTC", "ETH"],
            update_interval=update_interval,
            enabled=enabled
        )

        self.etherscan_api_key = etherscan_api_key or os.getenv("ETHERSCAN_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None

        # Track recent flows for trend
        self._flow_history: Dict[str, deque] = {
            asset: deque(maxlen=60) for asset in self.assets
        }

        # Whale alerts
        self.whale_alerts: deque = deque(maxlen=100)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_data(self, asset: str) -> Dict[str, float]:
        """Fetch on-chain metrics for an asset."""
        features = {}

        try:
            if asset == "BTC":
                features = await self._fetch_btc_flows()
            elif asset == "ETH":
                features = await self._fetch_eth_flows()
            else:
                # For other assets, use aggregated APIs
                features = await self._fetch_generic_flows(asset)

        except Exception as e:
            logger.warning(f"Onchain/{asset} fetch error: {e}")

        return features

    async def _fetch_btc_flows(self) -> Dict[str, float]:
        """
        Fetch BTC exchange flows.

        Uses blockchain.com API for recent blocks and transaction analysis.
        This is simplified - production would use dedicated analytics APIs.
        """
        try:
            session = await self._get_session()

            # Use blockchain.com stats API
            url = "https://api.blockchain.info/stats"

            async with session.get(url, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                # These are network-wide stats, not exchange-specific
                # In production, you'd use CryptoQuant or Glassnode
                result = {
                    # Placeholder - would calculate from actual exchange wallet monitoring
                    Features.EXCHANGE_INFLOW: 0.0,
                    Features.EXCHANGE_OUTFLOW: 0.0,
                    Features.EXCHANGE_NETFLOW: 0.0,
                }

                # Check for whale alerts in mempool (simplified)
                result[Features.WHALE_ALERT] = 0.0

                return result

        except Exception as e:
            logger.debug(f"BTC flow fetch error: {e}")
            return {}

    async def _fetch_eth_flows(self) -> Dict[str, float]:
        """
        Fetch ETH exchange flows using Etherscan.

        Monitors known exchange addresses for large deposits/withdrawals.
        """
        if not self.etherscan_api_key:
            return {}

        try:
            session = await self._get_session()

            # Get recent transactions to Binance hot wallet
            binance_addr = "0x28c6c06298d514db089934071355e5743bf21d60"

            url = "https://api.etherscan.io/api"
            params = {
                "module": "account",
                "action": "txlist",
                "address": binance_addr,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": 100,  # Last 100 transactions
                "sort": "desc",
                "apikey": self.etherscan_api_key
            }

            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if data.get("status") != "1":
                    return {}

                transactions = data.get("result", [])

                # Calculate inflows (deposits to exchange)
                inflow = 0.0
                outflow = 0.0
                whale_alert = 0.0

                one_hour_ago = datetime.utcnow() - timedelta(hours=1)

                for tx in transactions:
                    timestamp = datetime.fromtimestamp(int(tx.get("timeStamp", 0)))
                    if timestamp < one_hour_ago:
                        continue

                    value = int(tx.get("value", 0)) / 1e18  # Wei to ETH
                    to_addr = tx.get("to", "").lower()
                    from_addr = tx.get("from", "").lower()

                    # Check if it's a deposit (inflow) or withdrawal (outflow)
                    if to_addr == binance_addr.lower():
                        inflow += value
                        if value > WHALE_THRESHOLDS["ETH"]:
                            whale_alert = 1.0
                            self.whale_alerts.append({
                                "asset": "ETH",
                                "type": "exchange_deposit",
                                "amount": value,
                                "exchange": "binance",
                                "timestamp": timestamp
                            })
                    elif from_addr == binance_addr.lower():
                        outflow += value

                result = {
                    Features.EXCHANGE_INFLOW: inflow,
                    Features.EXCHANGE_OUTFLOW: outflow,
                    Features.EXCHANGE_NETFLOW: inflow - outflow,
                    Features.WHALE_ALERT: whale_alert,
                }

                return result

        except Exception as e:
            logger.debug(f"ETH flow fetch error: {e}")
            return {}

    async def _fetch_generic_flows(self, asset: str) -> Dict[str, float]:
        """
        Fetch exchange flows for other assets using aggregated APIs.

        This would use services like:
        - CryptoQuant (paid)
        - Glassnode (paid)
        - Santiment (freemium)
        """
        # Placeholder - return zeros for unsupported assets
        return {
            Features.EXCHANGE_INFLOW: 0.0,
            Features.EXCHANGE_OUTFLOW: 0.0,
            Features.EXCHANGE_NETFLOW: 0.0,
            Features.WHALE_ALERT: 0.0,
        }

    async def stop(self):
        """Clean up."""
        await super().stop()
        if self._session and not self._session.closed:
            await self._session.close()


class WhaleAlertWebSocket:
    """
    Whale Alert integration for real-time large transfer notifications.

    Requires Whale Alert API key (paid service).
    Free alternative: Monitor Twitter @whale_alert
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("WHALE_ALERT_API_KEY", "")
        self._running = False
        self.alerts: deque = deque(maxlen=100)

    async def start_polling(self, interval: float = 60.0):
        """Poll Whale Alert API for recent alerts."""
        if not self.api_key:
            logger.info("Whale Alert API key not set, skipping")
            return

        self._running = True

        while self._running:
            try:
                await self._fetch_alerts()
            except Exception as e:
                logger.warning(f"Whale Alert error: {e}")

            await asyncio.sleep(interval)

    async def _fetch_alerts(self):
        """Fetch recent whale alerts."""
        async with aiohttp.ClientSession() as session:
            url = "https://api.whale-alert.io/v1/transactions"
            params = {
                "api_key": self.api_key,
                "min_value": 5000000,  # $5M minimum
                "start": int((datetime.utcnow() - timedelta(minutes=15)).timestamp()),
            }

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return

                data = await resp.json()

                for tx in data.get("transactions", []):
                    self.alerts.append({
                        "blockchain": tx.get("blockchain"),
                        "symbol": tx.get("symbol"),
                        "amount": tx.get("amount"),
                        "amount_usd": tx.get("amount_usd"),
                        "from_owner_type": tx.get("from", {}).get("owner_type"),
                        "to_owner_type": tx.get("to", {}).get("owner_type"),
                        "timestamp": datetime.fromtimestamp(tx.get("timestamp", 0))
                    })

    def stop(self):
        """Stop polling."""
        self._running = False
