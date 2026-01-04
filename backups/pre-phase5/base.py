#!/usr/bin/env python3
"""
Base classes for trading strategies.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Action(Enum):
    HOLD = 0
    BUY = 1   # Buy UP token
    SELL = 2  # Sell UP token

    @property
    def is_buy(self) -> bool:
        return self == Action.BUY

    @property
    def is_sell(self) -> bool:
        return self == Action.SELL

    @property
    def size_multiplier(self) -> float:
        """Fixed 50% sizing for all trades."""
        return 0.5 if self in (Action.BUY, Action.SELL) else 0.0


@dataclass
class MarketState:
    """Rich market state for 15-min trading decisions."""
    # Core
    asset: str
    prob: float  # Current UP probability
    time_remaining: float  # Fraction of 15 min left (0-1)

    # Orderbook - CRITICAL for 15-min
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    order_book_imbalance_l1: float = 0.0  # Top of book imbalance
    order_book_imbalance_l5: float = 0.0  # Depth imbalance (top 5 levels)

    # Price data
    binance_price: float = 0.0
    binance_change: float = 0.0  # % change since market open

    # History (last N observations)
    prob_history: List[float] = field(default_factory=list)

    # Position
    has_position: bool = False
    position_side: Optional[str] = None  # "UP" or "DOWN"
    position_pnl: float = 0.0  # Unrealized P&L

    # === 15-MIN FOCUSED FEATURES ===
    # Ultra-short momentum (most relevant for 15-min)
    returns_1m: float = 0.0
    returns_5m: float = 0.0
    returns_10m: float = 0.0  # Middle timeframe

    # Order flow - THIS IS THE EDGE
    trade_flow_imbalance: float = 0.0  # [-1, 1] buy vs sell pressure
    cvd: float = 0.0  # Cumulative volume delta
    cvd_acceleration: float = 0.0  # Is CVD speeding up?
    prev_cvd: float = 0.0  # For acceleration calc

    # Microstructure
    trade_intensity: float = 0.0  # Trades per second (rolling)
    large_trade_flag: float = 0.0  # Big order just hit? (0 or 1)
    trade_count: int = 0  # For intensity calc
    last_trade_time: float = 0.0

    # Volatility (short-term)
    realized_vol_5m: float = 0.0
    vol_expansion: float = 0.0  # Current vol vs recent average

    # Regime context (only slow features worth keeping)
    vol_regime: float = 0.0  # High/low vol environment
    trend_regime: float = 0.0  # Trending or ranging

    def to_features(self) -> np.ndarray:
        """Convert to feature vector for ML models. Returns 18 features optimized for 15-min."""
        velocity = self._velocity(3)  # Shorter window
        vol_5m = self._volatility(30)  # ~5 min of ticks

        # Spread as percentage
        spread_pct = self.spread / max(0.01, self.prob) if self.prob > 0 else 0.0

        # Time remaining features (non-linear urgency)
        time_remaining_sq = self.time_remaining ** 2

        return np.array([
            # Ultra-short momentum (3)
            self.returns_1m * 100,
            self.returns_5m * 100,
            self.returns_10m * 100,

            # Order flow - THE EDGE (4)
            self.order_book_imbalance_l1,
            self.order_book_imbalance_l5,
            self.trade_flow_imbalance,
            self.cvd_acceleration,

            # Microstructure (3)
            spread_pct * 100,
            self.trade_intensity,
            self.large_trade_flag,

            # Volatility (2)
            vol_5m,
            self.vol_expansion,

            # Position (4)
            float(self.has_position),
            1.0 if self.position_side == "UP" else (-1.0 if self.position_side == "DOWN" else 0.0),
            self.position_pnl,
            self.time_remaining,  # CRITICAL

            # Regime (2)
            self.vol_regime,
            self.trend_regime,
        ], dtype=np.float32)

    def _velocity(self, window: int = 5) -> float:
        """Prob change over last N ticks."""
        if len(self.prob_history) < window:
            return 0.0
        return self.prob - self.prob_history[-window]

    def _volatility(self, window: int = 10) -> float:
        """Rolling std of prob."""
        if len(self.prob_history) < window:
            return 0.0
        recent = self.prob_history[-window:]
        return float(np.std(recent))

    def _momentum(self, window: int = 20) -> float:
        """Longer-term trend."""
        if len(self.prob_history) < window:
            return 0.0
        return self.prob - self.prob_history[-window]

    @property
    def near_expiry(self) -> bool:
        return self.time_remaining < 0.133  # < 2 min

    @property
    def very_near_expiry(self) -> bool:
        return self.time_remaining < 0.033  # < 30 sec


class Strategy(ABC):
    """Base class for all strategies."""

    def __init__(self, name: str):
        self.name = name
        self.training = False

    @abstractmethod
    def act(self, state: MarketState) -> Action:
        """Select action given current state."""
        pass

    def reset(self):
        """Reset any internal state (called between episodes/markets)."""
        pass

    def train(self):
        """Set to training mode."""
        self.training = True

    def eval(self):
        """Set to evaluation mode."""
        self.training = False

    def save(self, path: str):
        """Save strategy parameters."""
        pass

    def load(self, path: str):
        """Load strategy parameters."""
        pass
