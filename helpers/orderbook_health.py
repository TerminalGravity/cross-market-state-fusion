#!/usr/bin/env python3
"""
Orderbook Health Monitor

Monitors the freshness of orderbook data from Polymarket CLOB WebSocket.
Prevents trading with stale data that could cause positions to expire unmanaged.

Critical for preventing the failure mode where:
- Bot opens positions with fresh data
- Orderbook feed goes stale
- Bot can't exit positions before market expiry
- Positions held to expiration causing losses

Health States:
- HEALTHY: Data fresh (<30s old)
- DEGRADED: Data stale (30-60s old) - stop opening positions
- CRITICAL: Data stale (>60s old) - emergency exit all positions
"""
import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime, timezone
from enum import Enum


logger = logging.getLogger(__name__)


class HealthState(Enum):
    """Orderbook health states."""
    HEALTHY = "healthy"       # <30s: Normal operations
    DEGRADED = "degraded"     # 30-60s: Stop new positions
    CRITICAL = "critical"     # >60s: Emergency exit all positions
    UNKNOWN = "unknown"       # No data received yet


class OrderbookHealthMonitor:
    """
    Monitors orderbook data freshness and provides health status.

    Architecture:
    - Tracks last update timestamp per market
    - Background check loop (every 5 seconds)
    - Exposes health state for decision making
    - Logs health transitions for debugging

    Usage:
        monitor = OrderbookHealthMonitor()
        await monitor.start()

        # In orderbook callback:
        monitor.mark_update("BTC")

        # Before trading decisions:
        if monitor.get_state("BTC") == HealthState.CRITICAL:
            await emergency_exit("BTC")
    """

    def __init__(
        self,
        healthy_threshold_seconds: int = 30,
        critical_threshold_seconds: int = 60,
        check_interval_seconds: int = 5
    ):
        """
        Initialize health monitor.

        Args:
            healthy_threshold_seconds: Max age for HEALTHY state
            critical_threshold_seconds: Max age before CRITICAL state
            check_interval_seconds: How often to check staleness
        """
        self.healthy_threshold = healthy_threshold_seconds
        self.critical_threshold = critical_threshold_seconds
        self.check_interval = check_interval_seconds

        # Track last update time per market
        self.last_update: Dict[str, datetime] = {}

        # Track health state per market
        self.health_state: Dict[str, HealthState] = {}

        # Control background task
        self.running = False
        self._check_task: Optional[asyncio.Task] = None

        logger.info(
            f"OrderbookHealthMonitor initialized "
            f"(healthy<{healthy_threshold_seconds}s, "
            f"critical>{critical_threshold_seconds}s)"
        )

    def mark_update(self, asset: str) -> None:
        """
        Mark that orderbook data was received for an asset.

        Call this from orderbook WebSocket callback on every update.

        Args:
            asset: Asset name (BTC, ETH, SOL, XRP)
        """
        now = datetime.now(timezone.utc)
        self.last_update[asset] = now

        # Optimistically set to HEALTHY on update
        old_state = self.health_state.get(asset, HealthState.UNKNOWN)
        self.health_state[asset] = HealthState.HEALTHY

        # Log state transition
        if old_state != HealthState.HEALTHY:
            logger.info(f"[Health] {asset}: {old_state.value} → HEALTHY")

    def get_state(self, asset: str) -> HealthState:
        """
        Get current health state for an asset.

        Returns:
            HealthState enum (HEALTHY, DEGRADED, CRITICAL, UNKNOWN)
        """
        return self.health_state.get(asset, HealthState.UNKNOWN)

    def get_staleness_seconds(self, asset: str) -> Optional[float]:
        """
        Get how many seconds since last update.

        Returns:
            Seconds since last update, or None if no data received
        """
        if asset not in self.last_update:
            return None

        now = datetime.now(timezone.utc)
        delta = now - self.last_update[asset]
        return delta.total_seconds()

    def is_healthy(self, asset: str) -> bool:
        """Check if asset orderbook is healthy (fresh data)."""
        return self.get_state(asset) == HealthState.HEALTHY

    def is_degraded(self, asset: str) -> bool:
        """Check if asset orderbook is degraded (stale but not critical)."""
        return self.get_state(asset) == HealthState.DEGRADED

    def is_critical(self, asset: str) -> bool:
        """Check if asset orderbook is critical (very stale, emergency exit needed)."""
        return self.get_state(asset) == HealthState.CRITICAL

    def get_all_critical(self) -> list[str]:
        """Get list of assets in CRITICAL state (need emergency exit)."""
        return [
            asset for asset, state in self.health_state.items()
            if state == HealthState.CRITICAL
        ]

    async def start(self) -> None:
        """Start background health check loop."""
        if self.running:
            logger.warning("[Health] Monitor already running")
            return

        self.running = True
        self._check_task = asyncio.create_task(self._check_loop())
        logger.info(f"[Health] Monitor started (check every {self.check_interval}s)")

    async def stop(self) -> None:
        """Stop background health check loop."""
        self.running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("[Health] Monitor stopped")

    async def _check_loop(self) -> None:
        """Background loop that checks staleness every N seconds."""
        while self.running:
            try:
                await asyncio.sleep(self.check_interval)
                self._check_all_assets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Health] Check loop error: {e}", exc_info=True)

    def _check_all_assets(self) -> None:
        """Check staleness for all tracked assets and update states."""
        now = datetime.now(timezone.utc)

        for asset, last_update in self.last_update.items():
            age_seconds = (now - last_update).total_seconds()
            old_state = self.health_state[asset]
            new_state = self._determine_state(age_seconds)

            # Update state
            self.health_state[asset] = new_state

            # Log state transitions
            if old_state != new_state:
                logger.warning(
                    f"[Health] {asset}: {old_state.value} → {new_state.value} "
                    f"(stale for {age_seconds:.1f}s)"
                )

    def _determine_state(self, age_seconds: float) -> HealthState:
        """
        Determine health state based on data age.

        Args:
            age_seconds: How old the data is

        Returns:
            HealthState based on thresholds
        """
        if age_seconds < self.healthy_threshold:
            return HealthState.HEALTHY
        elif age_seconds < self.critical_threshold:
            return HealthState.DEGRADED
        else:
            return HealthState.CRITICAL

    def get_summary(self) -> Dict[str, any]:
        """
        Get health summary for all assets.

        Returns:
            Dict with per-asset health info
        """
        summary = {}
        for asset in self.last_update.keys():
            summary[asset] = {
                "state": self.get_state(asset).value,
                "staleness_seconds": self.get_staleness_seconds(asset),
                "last_update": self.last_update[asset].isoformat()
            }
        return summary
