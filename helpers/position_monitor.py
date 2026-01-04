#!/usr/bin/env python3
"""
Position Lifecycle Monitor

Prevents positions from expiring unmanaged by enforcing timeouts.

Critical safeguard against the failure mode where:
- Bot opens position with fresh data
- Orderbook goes stale or strategy doesn't signal exit
- Position held to market expiration
- Result: Uncontrolled wins/losses based on market resolution

This monitor ensures ALL positions exit before market expiry.

Timeout Rules:
1. T-2min: Force exit (gives 2min buffer for execution)
2. Max 12min: Force exit (for 15-min markets, 80% duration)
3. Position age tracked for all trades

Architecture:
- Background check loop (every 30 seconds)
- Queries database for open positions
- Calculates time remaining until expiry
- Signals force-exit when timeout reached
"""
import asyncio
import logging
from typing import List, Dict, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class PositionTimeout:
    """Details of a position that needs force-exit."""
    trade_id: str
    asset: str
    condition_id: str
    side: str
    entry_time: datetime
    market_expiry: datetime
    age_seconds: float
    time_to_expiry_seconds: float
    timeout_reason: str  # 'expiry_approaching' or 'max_duration'


class PositionMonitor:
    """
    Monitors open positions and enforces timeout rules.

    Prevents the critical failure where positions expire unmanaged.

    Usage:
        monitor = PositionMonitor(
            db=database,
            session_id=session_id,
            timeout_callback=force_exit_position
        )
        await monitor.start()

        # Monitor runs in background, calls timeout_callback
        # when positions need to be force-closed
    """

    def __init__(
        self,
        db,
        session_id: str,
        timeout_callback: Callable,
        expiry_buffer_seconds: int = 120,  # Exit 2min before expiry
        max_duration_seconds: int = 720,   # 12min max (80% of 15min)
        check_interval_seconds: int = 30
    ):
        """
        Initialize position monitor.

        Args:
            db: Database instance
            session_id: Current trading session ID
            timeout_callback: Async function(trade_id, reason) to call on timeout
            expiry_buffer_seconds: How many seconds before expiry to force-exit
            max_duration_seconds: Max position duration regardless of expiry
            check_interval_seconds: How often to check positions
        """
        self.db = db
        self.session_id = session_id
        self.timeout_callback = timeout_callback
        self.expiry_buffer = expiry_buffer_seconds
        self.max_duration = max_duration_seconds
        self.check_interval = check_interval_seconds

        # Control background task
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # Stats
        self.timeouts_triggered = 0
        self.expiry_timeouts = 0
        self.duration_timeouts = 0

        logger.info(
            f"PositionMonitor initialized "
            f"(expiry_buffer={expiry_buffer_seconds}s, "
            f"max_duration={max_duration_seconds}s)"
        )

    async def start(self) -> None:
        """Start background position monitoring loop."""
        if self.running:
            logger.warning("[PositionMonitor] Already running")
            return

        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"[PositionMonitor] Started "
            f"(check every {self.check_interval}s)"
        )

    async def stop(self) -> None:
        """Stop background monitoring loop."""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"[PositionMonitor] Stopped "
            f"(triggered {self.timeouts_triggered} timeouts: "
            f"{self.expiry_timeouts} expiry, {self.duration_timeouts} duration)"
        )

    async def _monitor_loop(self) -> None:
        """Background loop that checks positions every N seconds."""
        while self.running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[PositionMonitor] Check loop error: {e}",
                    exc_info=True
                )

    async def _check_positions(self) -> None:
        """Check all open positions for timeouts."""
        try:
            # Get open positions from database
            open_trades = await self.db.get_open_trades(self.session_id)

            if not open_trades:
                return  # No positions to monitor

            now = datetime.now(timezone.utc)

            # Check each position
            for trade in open_trades:
                timeout = self._check_trade_timeout(trade, now)

                if timeout:
                    await self._handle_timeout(timeout)

        except Exception as e:
            logger.error(
                f"[PositionMonitor] Error checking positions: {e}",
                exc_info=True
            )

    def _check_trade_timeout(
        self,
        trade,
        now: datetime
    ) -> Optional[PositionTimeout]:
        """
        Check if a trade needs force-exit.

        Args:
            trade: Database trade record
            now: Current time

        Returns:
            PositionTimeout if timeout triggered, None otherwise
        """
        # Calculate position age
        entry_time = trade["entry_time"]
        age = (now - entry_time).total_seconds()

        # Check max duration timeout
        if age >= self.max_duration:
            return PositionTimeout(
                trade_id=str(trade["id"]),
                asset=trade["asset"],
                condition_id=trade["condition_id"],
                side=trade["side"],
                entry_time=entry_time,
                market_expiry=trade.get("market_expiry_time"),
                age_seconds=age,
                time_to_expiry_seconds=0,  # Unknown
                timeout_reason="max_duration"
            )

        # Check expiry timeout (if market_expiry_time is set)
        market_expiry = trade.get("market_expiry_time")
        if market_expiry:
            time_to_expiry = (market_expiry - now).total_seconds()

            if time_to_expiry <= self.expiry_buffer:
                return PositionTimeout(
                    trade_id=str(trade["id"]),
                    asset=trade["asset"],
                    condition_id=trade["condition_id"],
                    side=trade["side"],
                    entry_time=entry_time,
                    market_expiry=market_expiry,
                    age_seconds=age,
                    time_to_expiry_seconds=time_to_expiry,
                    timeout_reason="expiry_approaching"
                )

        return None

    async def _handle_timeout(self, timeout: PositionTimeout) -> None:
        """
        Handle a position timeout by calling the callback.

        Args:
            timeout: Timeout details
        """
        # Log timeout
        if timeout.timeout_reason == "expiry_approaching":
            logger.warning(
                f"[PositionMonitor] TIMEOUT: {timeout.asset} "
                f"(T-{timeout.time_to_expiry_seconds:.0f}s to expiry, "
                f"age={timeout.age_seconds:.0f}s) - force exit"
            )
            self.expiry_timeouts += 1
        else:
            logger.warning(
                f"[PositionMonitor] TIMEOUT: {timeout.asset} "
                f"(age={timeout.age_seconds:.0f}s exceeds {self.max_duration}s) "
                f"- force exit"
            )
            self.duration_timeouts += 1

        self.timeouts_triggered += 1

        # Call the timeout callback
        try:
            await self.timeout_callback(
                trade_id=timeout.trade_id,
                asset=timeout.asset,
                condition_id=timeout.condition_id,
                side=timeout.side,
                reason=timeout.timeout_reason,
                age_seconds=timeout.age_seconds
            )
        except Exception as e:
            logger.error(
                f"[PositionMonitor] Timeout callback failed for "
                f"{timeout.asset} trade {timeout.trade_id}: {e}",
                exc_info=True
            )

    async def get_position_ages(self) -> List[Dict]:
        """
        Get current age of all open positions.

        Returns:
            List of dicts with asset, age_seconds, time_to_expiry_seconds
        """
        try:
            open_trades = await self.db.get_open_trades(self.session_id)
            now = datetime.now(timezone.utc)

            positions = []
            for trade in open_trades:
                age = (now - trade["entry_time"]).total_seconds()

                time_to_expiry = None
                if trade.get("market_expiry_time"):
                    time_to_expiry = (
                        trade["market_expiry_time"] - now
                    ).total_seconds()

                positions.append({
                    "trade_id": str(trade["id"]),
                    "asset": trade["asset"],
                    "age_seconds": age,
                    "time_to_expiry_seconds": time_to_expiry,
                    "entry_time": trade["entry_time"].isoformat()
                })

            return positions

        except Exception as e:
            logger.error(
                f"[PositionMonitor] Error getting position ages: {e}",
                exc_info=True
            )
            return []

    def get_stats(self) -> Dict:
        """Get monitor statistics."""
        return {
            "running": self.running,
            "total_timeouts": self.timeouts_triggered,
            "expiry_timeouts": self.expiry_timeouts,
            "duration_timeouts": self.duration_timeouts,
            "expiry_buffer_seconds": self.expiry_buffer,
            "max_duration_seconds": self.max_duration
        }
