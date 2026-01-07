#!/usr/bin/env python3
"""
Polymarket Data Client - Authenticated access to CLOB and Data APIs.

Provides methods to fetch:
- User trading history (authenticated)
- Order status and details
- Current positions
- Redeemable positions
- Market information

Uses py-clob-client for L2 authentication.
"""
import os
import asyncio
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# API endpoints
CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# py-clob-client imports (optional - for authenticated methods)
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.exceptions import PolyApiException
    HAS_CLOB_CLIENT = True
except ImportError:
    HAS_CLOB_CLIENT = False
    logger.warning("py-clob-client not installed - authenticated methods unavailable")


@dataclass
class TradeRecord:
    """Represents a trade from the CLOB API."""
    trade_id: str
    market: str
    asset_id: str
    side: str
    size: float
    price: float
    fee_rate_bps: int
    status: str
    match_time: str
    outcome: str
    transaction_hash: Optional[str] = None
    maker_address: Optional[str] = None
    type: str = "TAKER"  # TAKER or MAKER


class PolymarketDataClient:
    """
    Authenticated client for Polymarket APIs.

    Handles L2 authentication via py-clob-client and provides
    async methods for fetching user data.
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        funder_address: Optional[str] = None,
        signature_type: int = 0
    ):
        """
        Initialize the data client.

        Args:
            private_key: Wallet private key (or from POLYMARKET_PRIVATE_KEY env)
            funder_address: Funder address (or from POLYMARKET_FUNDER_ADDRESS env)
            signature_type: 0=EOA, 1=Email/Magic, 2=Browser
        """
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self.funder_address = funder_address or os.getenv("POLYMARKET_FUNDER_ADDRESS")
        self.signature_type = signature_type
        self.clob_client: Optional[ClobClient] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the authenticated CLOB client.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        if not HAS_CLOB_CLIENT:
            logger.error("py-clob-client not installed")
            return False

        if not self.private_key or not self.funder_address:
            logger.error("Missing POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS")
            return False

        try:
            # Initialize in thread (py-clob-client is sync)
            self.clob_client = await asyncio.to_thread(
                self._create_client
            )
            self._initialized = True
            logger.info(f"PolymarketDataClient initialized for {self.funder_address[:10]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize CLOB client: {e}")
            return False

    def _create_client(self) -> ClobClient:
        """Create and authenticate CLOB client (sync)."""
        client = ClobClient(
            host=CLOB_API,
            key=self.private_key,
            chain_id=137,  # Polygon
            signature_type=self.signature_type,
            funder=self.funder_address
        )
        # Derive API credentials (cached by library)
        client.set_api_creds(client.create_or_derive_api_creds())
        return client

    async def get_user_trades(
        self,
        limit: int = 50,
        before: Optional[str] = None,
        after: Optional[str] = None,
        market: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get authenticated user's trade history.

        Args:
            limit: Maximum trades to return
            before: Unix timestamp for trades before this time
            after: Unix timestamp for trades after this time
            market: Filter by condition ID

        Returns:
            List of trade records
        """
        if not self._initialized:
            await self.initialize()

        if not self.clob_client:
            return []

        try:
            # Build params dict
            params = {}
            if before:
                params["before"] = before
            if after:
                params["after"] = after
            if market:
                params["market"] = market

            # Call CLOB API in thread
            trades = await asyncio.to_thread(
                lambda: self.clob_client.get_trades(**params) if params else self.clob_client.get_trades()
            )

            # Limit results
            if trades and len(trades) > limit:
                trades = trades[:limit]

            return trades or []

        except PolyApiException as e:
            logger.error(f"CLOB API error getting trades: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to get trades: {e}")
            return []

    async def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific order.

        Args:
            order_id: The order ID to look up

        Returns:
            Order details dict or None
        """
        if not self._initialized:
            await self.initialize()

        if not self.clob_client:
            return None

        try:
            order = await asyncio.to_thread(
                self.clob_client.get_order,
                order_id
            )
            return order

        except PolyApiException as e:
            logger.error(f"CLOB API error getting order {order_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get order: {e}")
            return None

    async def get_user_positions(self) -> List[Dict[str, Any]]:
        """
        Get user's current open positions from Data API.

        Returns:
            List of position records
        """
        if not self.funder_address:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{DATA_API}/positions"
                params = {
                    "user": self.funder_address,
                    "limit": 100
                }

                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Data API error: {resp.status}")
                        return []

                    data = await resp.json()
                    return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_redeemable_positions(self) -> List[Dict[str, Any]]:
        """
        Get positions that can be redeemed (resolved markets).

        Returns:
            List of redeemable position records
        """
        if not self.funder_address:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{DATA_API}/positions"
                params = {
                    "user": self.funder_address,
                    "redeemable": "true",
                    "limit": 100
                }

                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Data API error: {resp.status}")
                        return []

                    data = await resp.json()
                    return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Failed to get redeemable positions: {e}")
            return []

    async def get_market_info(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Get market information from CLOB API.

        Args:
            condition_id: Market condition ID

        Returns:
            Market details dict or None
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{CLOB_API}/markets/{condition_id}"

                async with session.get(url) as resp:
                    if resp.status != 200:
                        return None

                    return await resp.json()

        except Exception as e:
            logger.error(f"Failed to get market info: {e}")
            return None

    async def get_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get user's activity history from Data API.

        Args:
            limit: Maximum records to return

        Returns:
            List of activity records
        """
        if not self.funder_address:
            return []

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{DATA_API}/activity"
                params = {
                    "user": self.funder_address,
                    "limit": limit
                }

                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        logger.error(f"Data API activity error: {resp.status}")
                        return []

                    data = await resp.json()
                    return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Failed to get activity: {e}")
            return []

    async def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """
        Get last trade price for a token.

        Args:
            token_id: The token ID

        Returns:
            Last trade price or None
        """
        if not self._initialized:
            await self.initialize()

        if not self.clob_client:
            return None

        try:
            price = await asyncio.to_thread(
                self.clob_client.get_last_trade_price,
                token_id
            )
            return float(price) if price else None

        except Exception as e:
            logger.error(f"Failed to get last trade price: {e}")
            return None

    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        return self._initialized and self.clob_client is not None


# Convenience function
def create_polymarket_client() -> PolymarketDataClient:
    """
    Create a Polymarket data client using environment variables.

    Environment variables:
        POLYMARKET_PRIVATE_KEY: Wallet private key
        POLYMARKET_FUNDER_ADDRESS: Funder address
        POLYMARKET_SIGNATURE_TYPE: 0=EOA, 1=Email/Magic, 2=Browser
    """
    sig_type = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0"))
    return PolymarketDataClient(signature_type=sig_type)


if __name__ == "__main__":
    import asyncio

    async def test():
        client = create_polymarket_client()

        if not await client.initialize():
            print("Failed to initialize client")
            return

        print("Client initialized!")

        # Test getting trades
        trades = await client.get_user_trades(limit=5)
        print(f"\nFound {len(trades)} trades:")
        for t in trades[:3]:
            print(f"  - {t.get('side')} @ {t.get('price')} | {t.get('status')}")

        # Test getting positions
        positions = await client.get_user_positions()
        print(f"\nFound {len(positions)} positions")

    asyncio.run(test())
