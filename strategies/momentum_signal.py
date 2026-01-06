#!/usr/bin/env python3
"""
High-Win-Rate Momentum Signal Generator (HiFi v3)

Uses information lag between Binance futures (fast) and Polymarket (slow)
to generate high-confidence directional signals.

Key differences from RL approach:
1. Trades BOTH directions (UP and DOWN tokens)
2. Only trades on strong momentum confirmation
3. Uses multi-timeframe momentum confluence
4. Stricter entry criteria for higher win rate

Target: 80%+ win rate, +$10 avg profit per trade
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Trading signal direction."""
    NONE = 0
    UP = 1      # Buy UP token (bullish)
    DOWN = 2    # Buy DOWN token (bearish)


@dataclass
class MomentumSignal:
    """High-confidence trading signal."""
    direction: SignalDirection
    confidence: float  # 0-1 signal strength
    momentum_1m: float
    momentum_5m: float
    momentum_10m: float
    entry_price: float  # Recommended entry (prob)
    target_pnl_pct: float  # Expected profit %
    reason: str


class MomentumSignalGenerator:
    """
    Generate high-confidence directional signals from Binance futures data.

    Strategy:
    - Monitor Binance 1m, 5m, 10m returns
    - When ALL timeframes align strongly, generate signal
    - Buy UP when bullish, buy DOWN when bearish
    - Only trade when confidence > threshold
    """

    # === SIGNAL THRESHOLDS (Aggressive for profit) ===
    # Lowered thresholds to catch more moves while maintaining edge

    # Minimum returns required for signal (all must align)
    MIN_RETURN_1M = 0.0003   # 0.03% minimum 1-min move (lowered from 0.08%)
    MIN_RETURN_5M = 0.0006   # 0.06% minimum 5-min move (lowered from 0.15%)
    MIN_RETURN_10M = 0.0008  # 0.08% minimum 10-min move (lowered from 0.20%)

    # Strong signal thresholds (higher confidence)
    STRONG_RETURN_1M = 0.0008  # 0.08%
    STRONG_RETURN_5M = 0.0015  # 0.15%
    STRONG_RETURN_10M = 0.002  # 0.20%

    # Very strong signal (near-certain)
    VERY_STRONG_RETURN_1M = 0.0015  # 0.15%
    VERY_STRONG_RETURN_5M = 0.003   # 0.30%

    # Orderbook imbalance thresholds (relaxed)
    MIN_OB_IMBALANCE = 0.05  # 5% imbalance minimum (was 10%)
    STRONG_OB_IMBALANCE = 0.15  # 15% = strong (was 25%)

    # Entry price bounds (widened for more opportunities)
    MAX_UP_ENTRY = 0.72     # Buy UP up to 72% (was 62%)
    MIN_UP_ENTRY = 0.25     # Buy UP down to 25% (was 30%)
    MAX_DOWN_ENTRY = 0.75   # Buy DOWN up to 75% (was 70%)
    MIN_DOWN_ENTRY = 0.28   # Buy DOWN down to 28% (was 38%)

    # Time constraints (widened)
    MIN_TIME_REMAINING = 0.15   # 15% = ~2.25 min for 15-min market (was 25%)
    MAX_TIME_REMAINING = 0.95   # 95% = start earlier (was 90%)

    # Minimum confidence to trade (lowered)
    MIN_CONFIDENCE = 0.45
    LIVE_MIN_CONFIDENCE = 0.55  # Lower for more action (was 0.70)

    def __init__(self):
        self.last_signal_time = {}  # Prevent rapid-fire signals
        self.signal_cooldown_sec = 30  # Min 30 sec between signals per asset

    def generate_signal(
        self,
        asset: str,
        returns_1m: float,
        returns_5m: float,
        returns_10m: float,
        prob_up: float,
        ob_imbalance_l1: float,
        ob_imbalance_l5: float,
        time_remaining: float,
        spread: float,
        cvd_acceleration: float = 0.0,
        trade_flow_imbalance: float = 0.0,
    ) -> Optional[MomentumSignal]:
        """
        Generate trading signal from market data.

        Args:
            asset: Asset name (BTC, ETH, etc.)
            returns_1m: Binance 1-minute return
            returns_5m: Binance 5-minute return
            returns_10m: Binance 10-minute return
            prob_up: Current UP token probability
            ob_imbalance_l1: Orderbook L1 imbalance [-1, 1]
            ob_imbalance_l5: Orderbook L5 imbalance [-1, 1]
            time_remaining: Fraction of market time remaining [0, 1]
            spread: Bid-ask spread
            cvd_acceleration: CVD momentum (optional)
            trade_flow_imbalance: Trade flow direction (optional)

        Returns:
            MomentumSignal if conditions met, None otherwise
        """
        # === PRE-CHECKS ===

        # Time filter - avoid extremes
        if time_remaining < self.MIN_TIME_REMAINING:
            return None
        if time_remaining > self.MAX_TIME_REMAINING:
            return None

        # Spread filter - avoid illiquid markets
        if spread > 0.06:  # 6% max spread
            return None

        # === DIRECTION DETECTION ===
        # Check if momentum is bullish or bearish

        bullish_1m = returns_1m > self.MIN_RETURN_1M
        bullish_5m = returns_5m > self.MIN_RETURN_5M
        bullish_10m = returns_10m > self.MIN_RETURN_10M

        bearish_1m = returns_1m < -self.MIN_RETURN_1M
        bearish_5m = returns_5m < -self.MIN_RETURN_5M
        bearish_10m = returns_10m < -self.MIN_RETURN_10M

        # === BULLISH SIGNAL (Buy UP) ===
        if bullish_1m and bullish_5m and bullish_10m:
            # Price filter for UP entry
            if prob_up > self.MAX_UP_ENTRY:
                return None  # Price too high, limited upside
            if prob_up < self.MIN_UP_ENTRY:
                return None  # Price too low, might be a trap

            # Calculate confidence
            confidence = self._calc_bullish_confidence(
                returns_1m, returns_5m, returns_10m,
                ob_imbalance_l1, ob_imbalance_l5,
                cvd_acceleration, trade_flow_imbalance
            )

            if confidence < self.MIN_CONFIDENCE:
                return None

            # Expected profit: (1.0 - entry_price) / entry_price
            target_pnl = (1.0 - prob_up) / prob_up

            return MomentumSignal(
                direction=SignalDirection.UP,
                confidence=confidence,
                momentum_1m=returns_1m,
                momentum_5m=returns_5m,
                momentum_10m=returns_10m,
                entry_price=prob_up,
                target_pnl_pct=target_pnl,
                reason=f"Bullish confluence: 1m={returns_1m*100:.2f}% 5m={returns_5m*100:.2f}% 10m={returns_10m*100:.2f}%"
            )

        # === BEARISH SIGNAL (Buy DOWN) ===
        if bearish_1m and bearish_5m and bearish_10m:
            prob_down = 1.0 - prob_up

            # Price filter for DOWN entry
            if prob_down > self.MAX_DOWN_ENTRY:
                return None  # DOWN too expensive
            if prob_down < self.MIN_DOWN_ENTRY:
                return None  # DOWN too cheap (UP too high)

            # Calculate confidence (same logic, reversed signs)
            confidence = self._calc_bearish_confidence(
                returns_1m, returns_5m, returns_10m,
                ob_imbalance_l1, ob_imbalance_l5,
                cvd_acceleration, trade_flow_imbalance
            )

            if confidence < self.MIN_CONFIDENCE:
                return None

            # Expected profit for DOWN token
            target_pnl = (1.0 - prob_down) / prob_down

            return MomentumSignal(
                direction=SignalDirection.DOWN,
                confidence=confidence,
                momentum_1m=returns_1m,
                momentum_5m=returns_5m,
                momentum_10m=returns_10m,
                entry_price=prob_down,
                target_pnl_pct=target_pnl,
                reason=f"Bearish confluence: 1m={returns_1m*100:.2f}% 5m={returns_5m*100:.2f}% 10m={returns_10m*100:.2f}%"
            )

        return None

    def _calc_bullish_confidence(
        self,
        ret_1m: float, ret_5m: float, ret_10m: float,
        ob_l1: float, ob_l5: float,
        cvd_accel: float, flow: float
    ) -> float:
        """Calculate bullish signal confidence [0, 1]."""
        score = 0.0
        max_score = 0.0

        # Momentum strength (50% weight)
        max_score += 0.50
        if ret_1m >= self.VERY_STRONG_RETURN_1M:
            score += 0.20
        elif ret_1m >= self.STRONG_RETURN_1M:
            score += 0.15
        else:
            score += 0.10

        if ret_5m >= self.VERY_STRONG_RETURN_5M:
            score += 0.20
        elif ret_5m >= self.STRONG_RETURN_5M:
            score += 0.15
        else:
            score += 0.10

        if ret_10m >= self.STRONG_RETURN_10M:
            score += 0.10
        else:
            score += 0.05

        # Orderbook confirmation (25% weight)
        max_score += 0.25
        if ob_l1 >= self.STRONG_OB_IMBALANCE:
            score += 0.15
        elif ob_l1 >= self.MIN_OB_IMBALANCE:
            score += 0.10
        elif ob_l1 >= 0:
            score += 0.05
        # Negative imbalance = no points

        if ob_l5 >= self.MIN_OB_IMBALANCE:
            score += 0.10
        elif ob_l5 >= 0:
            score += 0.05

        # Flow confirmation (25% weight)
        max_score += 0.25
        if cvd_accel > 0.1:
            score += 0.10
        elif cvd_accel > 0:
            score += 0.05

        if flow > 0.2:
            score += 0.15
        elif flow > 0:
            score += 0.08
        elif flow > -0.1:
            score += 0.02

        return min(1.0, score / max_score)

    def _calc_bearish_confidence(
        self,
        ret_1m: float, ret_5m: float, ret_10m: float,
        ob_l1: float, ob_l5: float,
        cvd_accel: float, flow: float
    ) -> float:
        """Calculate bearish signal confidence [0, 1]."""
        # Mirror of bullish with signs reversed
        score = 0.0
        max_score = 0.0

        # Momentum strength (50% weight) - use abs values
        max_score += 0.50
        if abs(ret_1m) >= self.VERY_STRONG_RETURN_1M:
            score += 0.20
        elif abs(ret_1m) >= self.STRONG_RETURN_1M:
            score += 0.15
        else:
            score += 0.10

        if abs(ret_5m) >= self.VERY_STRONG_RETURN_5M:
            score += 0.20
        elif abs(ret_5m) >= self.STRONG_RETURN_5M:
            score += 0.15
        else:
            score += 0.10

        if abs(ret_10m) >= self.STRONG_RETURN_10M:
            score += 0.10
        else:
            score += 0.05

        # Orderbook confirmation (25% weight) - reversed
        max_score += 0.25
        if ob_l1 <= -self.STRONG_OB_IMBALANCE:
            score += 0.15
        elif ob_l1 <= -self.MIN_OB_IMBALANCE:
            score += 0.10
        elif ob_l1 <= 0:
            score += 0.05

        if ob_l5 <= -self.MIN_OB_IMBALANCE:
            score += 0.10
        elif ob_l5 <= 0:
            score += 0.05

        # Flow confirmation (25% weight) - reversed
        max_score += 0.25
        if cvd_accel < -0.1:
            score += 0.10
        elif cvd_accel < 0:
            score += 0.05

        if flow < -0.2:
            score += 0.15
        elif flow < 0:
            score += 0.08
        elif flow < 0.1:
            score += 0.02

        return min(1.0, score / max_score)


# === QUICK TEST ===
if __name__ == "__main__":
    gen = MomentumSignalGenerator()

    # Test bullish signal
    signal = gen.generate_signal(
        asset="BTC",
        returns_1m=0.002,   # 0.2% up
        returns_5m=0.004,   # 0.4% up
        returns_10m=0.003,  # 0.3% up
        prob_up=0.48,       # 48% UP price (good entry)
        ob_imbalance_l1=0.15,
        ob_imbalance_l5=0.10,
        time_remaining=0.60,
        spread=0.02,
    )

    if signal:
        print(f"✓ Bullish signal: {signal.direction.name}")
        print(f"  Confidence: {signal.confidence*100:.0f}%")
        print(f"  Target PnL: {signal.target_pnl_pct*100:.0f}%")
        print(f"  Reason: {signal.reason}")
    else:
        print("No bullish signal")

    # Test bearish signal
    signal = gen.generate_signal(
        asset="ETH",
        returns_1m=-0.002,   # 0.2% down
        returns_5m=-0.004,   # 0.4% down
        returns_10m=-0.003,  # 0.3% down
        prob_up=0.55,        # 55% UP = 45% DOWN (good DOWN entry)
        ob_imbalance_l1=-0.20,
        ob_imbalance_l5=-0.15,
        time_remaining=0.50,
        spread=0.02,
    )

    if signal:
        print(f"\n✓ Bearish signal: {signal.direction.name}")
        print(f"  Confidence: {signal.confidence*100:.0f}%")
        print(f"  Entry (DOWN): {signal.entry_price*100:.0f}%")
        print(f"  Target PnL: {signal.target_pnl_pct*100:.0f}%")
        print(f"  Reason: {signal.reason}")
    else:
        print("\nNo bearish signal")
