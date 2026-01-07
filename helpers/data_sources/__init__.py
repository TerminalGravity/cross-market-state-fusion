"""
Multi-source data framework for expanded alpha signals.

This module provides pluggable data providers for:
- Coinglass: Aggregated CEX funding rates, OI, liquidations
- Deribit: Options IV, put/call ratio, large trades
- On-chain: Exchange flows, whale movements
- Sentiment: Fear & Greed index, social volume

Each provider is independent and can be enabled/disabled for A/B testing.
"""

from .base import DataSource, DataSourceState
from .coinglass import CoinglassSource
from .deribit import DeribitSource
from .onchain import OnchainSource
from .sentiment import SentimentSource
from .aggregator import DataAggregator

__all__ = [
    'DataSource',
    'DataSourceState',
    'CoinglassSource',
    'DeribitSource',
    'OnchainSource',
    'SentimentSource',
    'DataAggregator',
]
