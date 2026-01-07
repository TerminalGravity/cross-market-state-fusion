"""
Base class for all data sources.

Provides a standardized interface for pluggable data providers.
Each source can be enabled/disabled independently for A/B testing.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class DataSourceState:
    """
    State container for a single data source.

    Each source populates its own state which gets merged
    into the expanded MarketState by the aggregator.
    """
    source_name: str
    asset: str

    # Health tracking
    last_update: Optional[datetime] = None
    update_count: int = 0
    error_count: int = 0
    is_healthy: bool = False

    # Generic feature storage (source-specific)
    features: Dict[str, float] = field(default_factory=dict)

    # Raw data for debugging
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def get_feature(self, name: str, default: float = 0.0) -> float:
        """Get feature value with default."""
        return self.features.get(name, default)

    def set_feature(self, name: str, value: float):
        """Set feature value."""
        self.features[name] = value

    def mark_updated(self):
        """Mark this state as freshly updated."""
        self.last_update = datetime.utcnow()
        self.update_count += 1
        self.is_healthy = True

    def mark_error(self):
        """Mark an error occurred."""
        self.error_count += 1
        # Degrade health after 3 consecutive errors
        if self.error_count > 3:
            self.is_healthy = False

    @property
    def age_seconds(self) -> float:
        """Seconds since last update."""
        if not self.last_update:
            return float('inf')
        return (datetime.utcnow() - self.last_update).total_seconds()

    @property
    def is_stale(self) -> bool:
        """Data older than 60 seconds is stale."""
        return self.age_seconds > 60


class DataSource(ABC):
    """
    Abstract base class for all data sources.

    Implement this to add new alpha signal sources.
    Each source runs independently and can have different
    update frequencies based on data availability.
    """

    def __init__(
        self,
        name: str,
        assets: List[str],
        update_interval: float = 5.0,
        enabled: bool = True
    ):
        self.name = name
        self.assets = assets
        self.update_interval = update_interval
        self.enabled = enabled

        # State per asset
        self.states: Dict[str, DataSourceState] = {
            asset: DataSourceState(source_name=name, asset=asset)
            for asset in assets
        }

        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @abstractmethod
    async def fetch_data(self, asset: str) -> Dict[str, float]:
        """
        Fetch latest data for an asset.

        Returns dict of feature_name -> value.
        Must be implemented by each source.
        """
        pass

    async def start(self):
        """Start the data source update loop."""
        if not self.enabled:
            logger.info(f"{self.name}: Disabled, not starting")
            return

        self._running = True
        self._task = asyncio.create_task(self._update_loop())
        logger.info(f"{self.name}: Started (interval={self.update_interval}s)")

    async def stop(self):
        """Stop the data source."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"{self.name}: Stopped")

    async def _update_loop(self):
        """Main update loop."""
        while self._running:
            try:
                # Fetch for all assets in parallel
                tasks = [self._update_asset(asset) for asset in self.assets]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"{self.name}: Update loop error: {e}")

            await asyncio.sleep(self.update_interval)

    async def _update_asset(self, asset: str):
        """Update data for a single asset."""
        try:
            features = await self.fetch_data(asset)

            state = self.states[asset]
            state.features.update(features)
            state.mark_updated()
            state.error_count = 0  # Reset on success

        except Exception as e:
            logger.warning(f"{self.name}/{asset}: Fetch error: {e}")
            self.states[asset].mark_error()

    def get_state(self, asset: str) -> DataSourceState:
        """Get state for an asset."""
        return self.states.get(asset, DataSourceState(self.name, asset))

    def get_features(self, asset: str) -> Dict[str, float]:
        """Get all features for an asset."""
        return self.states[asset].features if asset in self.states else {}

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health status for all assets."""
        return {
            "source": self.name,
            "enabled": self.enabled,
            "running": self._running,
            "assets": {
                asset: {
                    "healthy": state.is_healthy,
                    "stale": state.is_stale,
                    "age_sec": round(state.age_seconds, 1),
                    "updates": state.update_count,
                    "errors": state.error_count,
                }
                for asset, state in self.states.items()
            }
        }


# Feature name constants for consistency
class Features:
    """Standard feature names across all sources."""

    # Funding (Coinglass)
    FUNDING_RATE_BINANCE = "funding_rate_binance"
    FUNDING_RATE_OKX = "funding_rate_okx"
    FUNDING_RATE_BYBIT = "funding_rate_bybit"
    FUNDING_DIVERGENCE = "funding_divergence"  # Max spread across exchanges

    # Open Interest (Coinglass)
    OI_TOTAL = "oi_total"
    OI_CHANGE_1H = "oi_change_1h"
    OI_CHANGE_4H = "oi_change_4h"
    LONG_SHORT_RATIO = "long_short_ratio"

    # Liquidations (Coinglass)
    LIQUIDATION_PRESSURE = "liquidation_pressure"  # Net long vs short liqs
    LIQUIDATION_VOLUME_1H = "liquidation_volume_1h"

    # Options (Deribit)
    IV_ATM = "iv_atm"  # At-the-money implied vol
    IV_SKEW = "iv_skew"  # Put vs call IV difference
    PUT_CALL_RATIO = "put_call_ratio"
    OPTIONS_VOLUME = "options_volume"
    MAX_PAIN = "max_pain"  # Options max pain price

    # On-chain
    EXCHANGE_INFLOW = "exchange_inflow"
    EXCHANGE_OUTFLOW = "exchange_outflow"
    EXCHANGE_NETFLOW = "exchange_netflow"
    WHALE_ALERT = "whale_alert"  # Large transfers flag

    # Sentiment
    FEAR_GREED = "fear_greed"
    SOCIAL_VOLUME = "social_volume"
    SOCIAL_SENTIMENT = "social_sentiment"  # Positive vs negative
