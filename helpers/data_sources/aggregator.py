"""
Data Aggregator - Combines all data sources into unified market state.

This is the central hub that:
1. Manages all data source lifecycles
2. Merges features from all sources
3. Handles missing data gracefully
4. Provides health monitoring
5. Enables A/B testing by toggling sources
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base import DataSource, DataSourceState, Features
from .coinglass import CoinglassSource
from .deribit import DeribitSource
from .onchain import OnchainSource
from .sentiment import SentimentSource

logger = logging.getLogger(__name__)


@dataclass
class AggregatedState:
    """
    Combined state from all data sources for a single asset.

    Provides normalized features ready for ML model consumption.
    """
    asset: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Source states
    sources: Dict[str, DataSourceState] = field(default_factory=dict)

    # Merged features (all sources combined)
    features: Dict[str, float] = field(default_factory=dict)

    # Health
    healthy_sources: int = 0
    total_sources: int = 0

    def get_feature(self, name: str, default: float = 0.0) -> float:
        """Get a feature value with default."""
        return self.features.get(name, default)

    def to_feature_vector(self) -> List[float]:
        """
        Convert to ordered feature vector for ML.

        Returns 24 features in consistent order:
        - 6 Funding/OI features (Coinglass)
        - 4 Options features (Deribit)
        - 4 On-chain features
        - 3 Sentiment features
        - 7 Reserved/padding

        All normalized to [-1, 1] or [0, 1] range.
        """
        def clamp(x, lo=-1.0, hi=1.0):
            return max(lo, min(hi, x))

        return [
            # === COINGLASS (6) ===
            # Funding rates (typically -0.1% to 0.1%, scale by 100)
            clamp(self.get_feature(Features.FUNDING_RATE_BINANCE, 0) * 100),
            clamp(self.get_feature(Features.FUNDING_RATE_OKX, 0) * 100),
            clamp(self.get_feature(Features.FUNDING_DIVERGENCE, 0) * 200),  # Divergence is smaller
            # Long/short ratio (typically 0.5-2.0, center at 1.0)
            clamp((self.get_feature(Features.LONG_SHORT_RATIO, 1.0) - 1.0) * 2),
            # OI change (typically -10% to +10%)
            clamp(self.get_feature(Features.OI_CHANGE_1H, 0) * 10),
            # Liquidation pressure (already -1 to 1)
            clamp(self.get_feature(Features.LIQUIDATION_PRESSURE, 0)),

            # === DERIBIT OPTIONS (4) ===
            # ATM IV (typically 0.3-1.5, center at 0.6)
            clamp((self.get_feature(Features.IV_ATM, 0.6) - 0.6) * 2),
            # IV skew (typically -0.1 to 0.1)
            clamp(self.get_feature(Features.IV_SKEW, 0) * 10),
            # Put/call ratio (typically 0.5-2.0, center at 1.0)
            clamp((self.get_feature(Features.PUT_CALL_RATIO, 1.0) - 1.0)),
            # Options volume (log-normalized, 0-1)
            clamp(self.get_feature(Features.OPTIONS_VOLUME, 0) / 10000, 0, 1),

            # === ON-CHAIN (4) ===
            # Exchange netflow (positive = bearish, scale by typical daily flow)
            clamp(self.get_feature(Features.EXCHANGE_NETFLOW, 0) / 1000),
            # Inflow/outflow ratio
            self._safe_ratio(
                self.get_feature(Features.EXCHANGE_INFLOW, 0),
                self.get_feature(Features.EXCHANGE_OUTFLOW, 0)
            ),
            # Whale alert (0 or 1)
            self.get_feature(Features.WHALE_ALERT, 0),
            # Reserved
            0.0,

            # === SENTIMENT (3) ===
            # Fear & Greed (0-1, center at 0.5)
            (self.get_feature(Features.FEAR_GREED, 0.5) - 0.5) * 2,
            # Social volume (0-1)
            self.get_feature(Features.SOCIAL_VOLUME, 0.5),
            # Social sentiment (0-1, center at 0.5)
            (self.get_feature(Features.SOCIAL_SENTIMENT, 0.5) - 0.5) * 2,

            # === RESERVED (7) ===
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]

    def _safe_ratio(self, a: float, b: float) -> float:
        """Calculate ratio with safe division."""
        if b == 0:
            return 0.0 if a == 0 else 1.0
        ratio = a / b
        # Center at 1.0 and clamp
        return max(-1.0, min(1.0, (ratio - 1.0)))

    @property
    def health_score(self) -> float:
        """0-1 health score based on source availability."""
        if self.total_sources == 0:
            return 0.0
        return self.healthy_sources / self.total_sources


class DataAggregator:
    """
    Central aggregator for all data sources.

    Usage:
        aggregator = DataAggregator(assets=["BTC", "ETH", "SOL", "XRP"])
        await aggregator.start()

        # Get combined state
        state = aggregator.get_state("BTC")
        features = state.to_feature_vector()

        await aggregator.stop()
    """

    def __init__(
        self,
        assets: List[str] = None,
        enabled_sources: List[str] = None,
        coinglass_api_key: str = None,
        lunarcrush_api_key: str = None,
        etherscan_api_key: str = None,
    ):
        self.assets = assets or ["BTC", "ETH", "SOL", "XRP"]

        # Determine which sources to enable
        all_sources = {"coinglass", "deribit", "onchain", "sentiment"}
        self.enabled_sources = set(enabled_sources or all_sources)

        # Initialize sources
        self.sources: Dict[str, DataSource] = {}

        if "coinglass" in self.enabled_sources:
            self.sources["coinglass"] = CoinglassSource(
                assets=self.assets,
                api_key=coinglass_api_key,
                enabled=True
            )

        if "deribit" in self.enabled_sources:
            # Deribit only supports BTC/ETH
            deribit_assets = [a for a in self.assets if a in ["BTC", "ETH"]]
            if deribit_assets:
                self.sources["deribit"] = DeribitSource(
                    assets=deribit_assets,
                    enabled=True
                )

        if "onchain" in self.enabled_sources:
            self.sources["onchain"] = OnchainSource(
                assets=self.assets,
                etherscan_api_key=etherscan_api_key,
                enabled=True
            )

        if "sentiment" in self.enabled_sources:
            self.sources["sentiment"] = SentimentSource(
                assets=self.assets,
                lunarcrush_api_key=lunarcrush_api_key,
                enabled=True
            )

        # Aggregated state cache
        self._states: Dict[str, AggregatedState] = {
            asset: AggregatedState(asset=asset)
            for asset in self.assets
        }

        self._running = False

    async def start(self):
        """Start all data sources."""
        self._running = True

        # Start each source
        for name, source in self.sources.items():
            try:
                await source.start()
                logger.info(f"Started data source: {name}")
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")

        # Start aggregation loop
        asyncio.create_task(self._aggregation_loop())

        logger.info(f"DataAggregator started with {len(self.sources)} sources")

    async def stop(self):
        """Stop all data sources."""
        self._running = False

        for name, source in self.sources.items():
            try:
                await source.stop()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        logger.info("DataAggregator stopped")

    async def _aggregation_loop(self):
        """Periodically aggregate data from all sources."""
        while self._running:
            try:
                for asset in self.assets:
                    self._aggregate_asset(asset)
            except Exception as e:
                logger.error(f"Aggregation error: {e}")

            await asyncio.sleep(1.0)  # Aggregate every second

    def _aggregate_asset(self, asset: str):
        """Aggregate features for a single asset."""
        state = AggregatedState(asset=asset)

        healthy = 0
        total = 0

        for name, source in self.sources.items():
            if asset not in source.assets:
                continue

            total += 1
            source_state = source.get_state(asset)
            state.sources[name] = source_state

            if source_state.is_healthy and not source_state.is_stale:
                healthy += 1
                # Merge features
                state.features.update(source_state.features)

        state.healthy_sources = healthy
        state.total_sources = total

        self._states[asset] = state

    def get_state(self, asset: str) -> AggregatedState:
        """Get aggregated state for an asset."""
        return self._states.get(asset, AggregatedState(asset=asset))

    def get_features(self, asset: str) -> Dict[str, float]:
        """Get all features for an asset."""
        return self.get_state(asset).features

    def get_feature_vector(self, asset: str) -> List[float]:
        """Get ML-ready feature vector for an asset."""
        return self.get_state(asset).to_feature_vector()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health status of all sources."""
        return {
            "running": self._running,
            "sources": {
                name: source.get_health_summary()
                for name, source in self.sources.items()
            },
            "assets": {
                asset: {
                    "health_score": state.health_score,
                    "healthy_sources": state.healthy_sources,
                    "total_sources": state.total_sources,
                    "feature_count": len(state.features),
                }
                for asset, state in self._states.items()
            }
        }

    def toggle_source(self, name: str, enabled: bool):
        """Enable/disable a source for A/B testing."""
        if name in self.sources:
            self.sources[name].enabled = enabled
            logger.info(f"Source {name} {'enabled' if enabled else 'disabled'}")


# Convenience function for quick setup
async def create_aggregator(
    assets: List[str] = None,
    enable_all: bool = True,
) -> DataAggregator:
    """
    Create and start a DataAggregator with default config.

    Args:
        assets: List of assets to track
        enable_all: Enable all available sources

    Returns:
        Started DataAggregator instance
    """
    aggregator = DataAggregator(
        assets=assets or ["BTC", "ETH", "SOL", "XRP"],
        enabled_sources=None if enable_all else ["coinglass", "sentiment"],
    )

    await aggregator.start()
    return aggregator
