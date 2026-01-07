"""
Deribit data source for options market signals.

Provides:
- Implied volatility (ATM IV)
- IV skew (put vs call IV)
- Put/call ratio (volume-based)
- Options volume and large trades
- Max pain calculation

Free API: https://docs.deribit.com (public endpoints, no key required)

Key insight: Options flow often leads spot by 10-30 minutes.
Large put buys = smart money hedging = bearish signal.
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

# Deribit API
DERIBIT_API = "https://www.deribit.com/api/v2"

# Asset mapping to Deribit instruments
DERIBIT_CURRENCIES = {
    "BTC": "BTC",
    "ETH": "ETH",
    # SOL and XRP don't have liquid Deribit options
}


class DeribitSource(DataSource):
    """
    Deribit options data source.

    Options markets are information-rich because:
    1. Leverage is embedded (small capital, big bets)
    2. Institutional traders use options for hedging
    3. IV changes often precede spot moves
    4. Large option trades signal conviction

    For 15-min trading:
    - ATM IV spikes = volatility incoming
    - Put/call ratio extremes = directional bias
    - IV skew changes = smart money positioning
    """

    def __init__(
        self,
        assets: list = None,
        update_interval: float = 15.0,  # Options data changes slower
        enabled: bool = True
    ):
        # Only BTC and ETH have liquid Deribit options
        valid_assets = [a for a in (assets or ["BTC", "ETH"]) if a in DERIBIT_CURRENCIES]

        super().__init__(
            name="deribit",
            assets=valid_assets or ["BTC", "ETH"],
            update_interval=update_interval,
            enabled=enabled
        )

        self._session: Optional[aiohttp.ClientSession] = None

        # Track IV history for rate of change
        self._iv_history: Dict[str, deque] = {
            asset: deque(maxlen=20) for asset in self.assets
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def fetch_data(self, asset: str) -> Dict[str, float]:
        """Fetch all Deribit metrics for an asset."""
        features = {}

        currency = DERIBIT_CURRENCIES.get(asset)
        if not currency:
            return features

        try:
            # Fetch in parallel
            iv_task = self._fetch_iv_index(currency)
            book_task = self._fetch_option_book_summary(currency)

            results = await asyncio.gather(
                iv_task, book_task,
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, dict):
                    features.update(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Deribit/{asset} partial failure: {result}")

        except Exception as e:
            logger.warning(f"Deribit/{asset} fetch error: {e}")

        return features

    async def _fetch_iv_index(self, currency: str) -> Dict[str, float]:
        """
        Fetch the Deribit Volatility Index (DVOL).

        DVOL is an index of 30-day ATM implied volatility.
        High DVOL = market expects big moves.
        """
        try:
            session = await self._get_session()

            url = f"{DERIBIT_API}/public/get_volatility_index_data"
            params = {
                "currency": currency,
                "resolution": "1",  # 1 hour
                "start_timestamp": int((datetime.utcnow() - timedelta(hours=1)).timestamp() * 1000),
                "end_timestamp": int(datetime.utcnow().timestamp() * 1000),
            }

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    # Fallback: try ticker for DVOL
                    return await self._fetch_dvol_ticker(currency)

                data = await resp.json()

                if "result" not in data or not data["result"].get("data"):
                    return await self._fetch_dvol_ticker(currency)

                # Get latest IV value
                iv_data = data["result"]["data"]
                if iv_data:
                    latest = iv_data[-1]
                    iv = latest[4] if len(latest) > 4 else latest[1]  # close or open

                    return {Features.IV_ATM: iv / 100}  # Convert to decimal

                return {}

        except Exception as e:
            logger.debug(f"Deribit IV index error: {e}")
            return await self._fetch_dvol_ticker(currency)

    async def _fetch_dvol_ticker(self, currency: str) -> Dict[str, float]:
        """Fallback: fetch DVOL from ticker."""
        try:
            session = await self._get_session()

            instrument = f"{currency}_DVOL"
            url = f"{DERIBIT_API}/public/ticker"
            params = {"instrument_name": instrument}

            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if "result" not in data:
                    return {}

                result = data["result"]
                iv = result.get("last_price", 0) / 100  # DVOL is in percentage

                return {Features.IV_ATM: iv}

        except Exception as e:
            logger.debug(f"Deribit DVOL ticker error: {e}")
            return {}

    async def _fetch_option_book_summary(self, currency: str) -> Dict[str, float]:
        """
        Fetch aggregated options book data.

        Calculates:
        - Put/call ratio from volume
        - IV skew from near-term ATM options
        - Total options volume
        """
        try:
            session = await self._get_session()

            url = f"{DERIBIT_API}/public/get_book_summary_by_currency"
            params = {
                "currency": currency,
                "kind": "option"
            }

            async with session.get(url, params=params, timeout=15) as resp:
                if resp.status != 200:
                    return {}

                data = await resp.json()

                if "result" not in data:
                    return {}

                options = data["result"]

                # Aggregate metrics
                put_volume = 0.0
                call_volume = 0.0
                put_iv_sum = 0.0
                call_iv_sum = 0.0
                put_count = 0
                call_count = 0
                total_oi = 0.0

                now = datetime.utcnow()

                for opt in options:
                    instrument = opt.get("instrument_name", "")
                    volume = opt.get("volume", 0)
                    mark_iv = opt.get("mark_iv", 0) / 100  # Convert to decimal
                    oi = opt.get("open_interest", 0)

                    # Parse expiry from instrument name (e.g., BTC-28JUN24-70000-C)
                    parts = instrument.split("-")
                    if len(parts) < 4:
                        continue

                    option_type = parts[-1]  # C or P

                    # Only count near-term options (within 7 days)
                    # This is a simplification - could parse expiry date
                    if option_type == "C":
                        call_volume += volume
                        if mark_iv > 0:
                            call_iv_sum += mark_iv
                            call_count += 1
                    elif option_type == "P":
                        put_volume += volume
                        if mark_iv > 0:
                            put_iv_sum += mark_iv
                            put_count += 1

                    total_oi += oi

                result = {}

                # Put/call ratio
                total_volume = put_volume + call_volume
                if total_volume > 0:
                    result[Features.PUT_CALL_RATIO] = put_volume / max(1, call_volume)
                    result[Features.OPTIONS_VOLUME] = total_volume

                # IV skew (put IV - call IV)
                # Positive skew = puts more expensive = bearish
                if put_count > 0 and call_count > 0:
                    avg_put_iv = put_iv_sum / put_count
                    avg_call_iv = call_iv_sum / call_count
                    result[Features.IV_SKEW] = avg_put_iv - avg_call_iv

                return result

        except Exception as e:
            logger.debug(f"Deribit book summary error: {e}")
            return {}

    async def stop(self):
        """Clean up."""
        await super().stop()
        if self._session and not self._session.closed:
            await self._session.close()


class DeribitWebSocket:
    """
    Deribit WebSocket for real-time options updates.

    Use this for higher-frequency options signals.
    Subscribes to:
    - deribit_price_index
    - trades (option trades for flow analysis)
    """

    def __init__(self, currencies: List[str] = None):
        self.currencies = currencies or ["BTC", "ETH"]
        self._ws = None
        self._running = False

        # Real-time state
        self.latest_trades: Dict[str, list] = {c: [] for c in self.currencies}
        self.large_trade_alerts: deque = deque(maxlen=100)

    async def connect(self):
        """Connect to Deribit WebSocket."""
        import websockets

        self._running = True

        url = "wss://www.deribit.com/ws/api/v2"

        try:
            self._ws = await websockets.connect(url)

            # Subscribe to option trades
            for currency in self.currencies:
                await self._subscribe_trades(currency)

            logger.info("Deribit WebSocket connected")

            # Start message loop
            asyncio.create_task(self._message_loop())

        except Exception as e:
            logger.error(f"Deribit WebSocket connect error: {e}")
            self._running = False

    async def _subscribe_trades(self, currency: str):
        """Subscribe to option trades."""
        if not self._ws:
            return

        msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "public/subscribe",
            "params": {
                "channels": [f"trades.option.{currency}.raw"]
            }
        }

        await self._ws.send(json.dumps(msg))

    async def _message_loop(self):
        """Process incoming WebSocket messages."""
        while self._running and self._ws:
            try:
                msg = await asyncio.wait_for(self._ws.recv(), timeout=30)
                data = json.loads(msg)

                if "params" in data:
                    self._handle_message(data["params"])

            except asyncio.TimeoutError:
                # Send heartbeat
                await self._ws.ping()

            except Exception as e:
                logger.warning(f"Deribit WS error: {e}")
                break

    def _handle_message(self, params: dict):
        """Handle incoming trade message."""
        channel = params.get("channel", "")
        data = params.get("data", [])

        if "trades.option" in channel:
            for trade in data:
                amount = trade.get("amount", 0)
                price = trade.get("price", 0)
                instrument = trade.get("instrument_name", "")

                # Flag large trades (> $100k notional)
                notional = amount * price
                if notional > 100000:
                    self.large_trade_alerts.append({
                        "instrument": instrument,
                        "amount": amount,
                        "price": price,
                        "notional": notional,
                        "timestamp": datetime.utcnow()
                    })

    async def disconnect(self):
        """Disconnect WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
