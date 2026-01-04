#!/usr/bin/env python3
"""
Emergency Exit Executor

Force-closes positions when normal exit mechanisms fail.

Scenarios requiring emergency exit:
1. Position timeout (T-2min before market expiry)
2. Orderbook health critical (>60s stale data)
3. Strategy stuck (position open too long)

This is the last-resort safeguard to prevent unmanaged position expiry.

Retry Strategy:
- Attempt 1: Immediate
- Attempt 2: +5s delay
- Attempt 3: +10s delay
- If all fail: Alert to Discord, log critical event

Uses market orders if limit orders fail (prioritizes exit over price).
"""
import asyncio
import logging
from typing import Optional, Dict
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


class EmergencyExitExecutor:
    """
    Handles forced position exits with retry logic.

    Ensures positions don't expire unmanaged even when normal
    exit mechanisms fail.

    Usage:
        executor = EmergencyExitExecutor(
            clob_executor=clob_executor,
            db=database,
            discord=discord_notifier
        )

        # Force-close position
        success = await executor.force_exit(
            trade_id="...",
            asset="BTC",
            condition_id="...",
            side="UP",
            reason="timeout"
        )
    """

    def __init__(
        self,
        clob_executor,
        db,
        discord=None,
        max_retries: int = 3,
        retry_delays: list[int] = None
    ):
        """
        Initialize emergency exit executor.

        Args:
            clob_executor: CLOBExecutor instance for placing orders
            db: Database instance
            discord: Optional Discord notifier
            max_retries: Maximum retry attempts
            retry_delays: Delays between retries (seconds)
        """
        self.clob = clob_executor
        self.db = db
        self.discord = discord
        self.max_retries = max_retries
        self.retry_delays = retry_delays or [0, 5, 10]  # 0s, 5s, 10s

        # Stats
        self.total_exits = 0
        self.successful_exits = 0
        self.failed_exits = 0

        logger.info(
            f"EmergencyExitExecutor initialized "
            f"(max_retries={max_retries}, "
            f"delays={self.retry_delays})"
        )

    async def force_exit(
        self,
        trade_id: str,
        asset: str,
        condition_id: str,
        side: str,
        reason: str,
        age_seconds: float,
        current_price: Optional[float] = None
    ) -> bool:
        """
        Force-exit a position with retries.

        Args:
            trade_id: Database trade ID
            asset: Asset name (BTC, ETH, etc.)
            condition_id: Polymarket condition ID
            side: Trade side (UP/DOWN)
            reason: Why forcing exit (timeout, health_check, max_duration)
            age_seconds: How long position has been open
            current_price: Optional current market price

        Returns:
            True if exit successful, False if all retries failed
        """
        self.total_exits += 1

        logger.warning(
            f"[EmergencyExit] Force-exiting {asset} {side} "
            f"(reason={reason}, age={age_seconds:.0f}s, "
            f"trade={trade_id[:8]}...)"
        )

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                # Delay before retry (skip for first attempt)
                if attempt > 0:
                    delay = self.retry_delays[min(attempt, len(self.retry_delays) - 1)]
                    logger.info(
                        f"[EmergencyExit] Retry {attempt + 1}/{self.max_retries} "
                        f"after {delay}s delay"
                    )
                    await asyncio.sleep(delay)

                # Attempt exit
                success = await self._attempt_exit(
                    trade_id=trade_id,
                    asset=asset,
                    condition_id=condition_id,
                    side=side,
                    reason=reason,
                    attempt=attempt + 1
                )

                if success:
                    self.successful_exits += 1
                    logger.info(
                        f"[EmergencyExit] ✅ {asset} exited successfully "
                        f"(attempt {attempt + 1})"
                    )
                    return True

            except Exception as e:
                logger.error(
                    f"[EmergencyExit] Attempt {attempt + 1} failed: {e}",
                    exc_info=True
                )

        # All retries exhausted
        self.failed_exits += 1
        await self._handle_exit_failure(
            trade_id=trade_id,
            asset=asset,
            side=side,
            reason=reason,
            age_seconds=age_seconds
        )
        return False

    async def _attempt_exit(
        self,
        trade_id: str,
        asset: str,
        condition_id: str,
        side: str,
        reason: str,
        attempt: int
    ) -> bool:
        """
        Single exit attempt.

        Args:
            trade_id: Database trade ID
            asset: Asset name
            condition_id: Polymarket condition ID
            side: Trade side
            reason: Exit reason
            attempt: Attempt number

        Returns:
            True if successful
        """
        # Get trade details from database
        trade = await self.db.fetchrow(
            "SELECT * FROM trades WHERE id = $1",
            trade_id
        )

        if not trade:
            logger.error(f"[EmergencyExit] Trade {trade_id} not found in DB")
            return False

        if trade["exit_time"]:
            logger.warning(
                f"[EmergencyExit] Trade {trade_id} already closed "
                f"(exit_time={trade['exit_time']})"
            )
            return True  # Already closed

        # Determine opposite action (SELL if bought UP, BUY if bought DOWN)
        exit_side = "SELL" if side == "UP" else "BUY"

        # Get current price from CLOB executor
        # (This should query current best bid/ask)
        try:
            if exit_side == "SELL":
                # Use best bid price (selling)
                current_price = await self.clob.get_best_bid(
                    asset, condition_id, side
                )
            else:
                # Use best ask price (buying)
                current_price = await self.clob.get_best_ask(
                    asset, condition_id, side
                )

            if not current_price:
                logger.warning(
                    f"[EmergencyExit] No price available for {asset}, "
                    f"using fallback"
                )
                current_price = 0.50  # Fallback to 50% if no data

        except Exception as e:
            logger.error(
                f"[EmergencyExit] Error getting current price: {e}",
                exc_info=True
            )
            current_price = 0.50  # Fallback

        # Place exit order via CLOB
        try:
            order_id = await self.clob.place_order(
                asset=asset,
                condition_id=condition_id,
                side=exit_side,
                price=current_price,
                size_dollars=trade["size_dollars"],
                force_exit=True  # Flag to indicate emergency exit
            )

            if not order_id:
                logger.error(
                    f"[EmergencyExit] Order placement returned None "
                    f"for {asset} {exit_side}"
                )
                return False

            # Update database with forced exit
            await self._record_forced_exit(
                trade_id=trade_id,
                exit_price=current_price,
                reason=reason
            )

            return True

        except Exception as e:
            logger.error(
                f"[EmergencyExit] Order placement failed: {e}",
                exc_info=True
            )
            return False

    async def _record_forced_exit(
        self,
        trade_id: str,
        exit_price: float,
        reason: str
    ) -> None:
        """
        Record forced exit in database.

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            reason: Why force-closed
        """
        try:
            await self.db.execute("""
                UPDATE trades
                SET force_closed = TRUE,
                    force_close_reason = $2,
                    exit_price = $3,
                    exit_time = NOW(),
                    exit_reason = 'forced_exit'
                WHERE id = $1
            """, trade_id, reason, exit_price)

            logger.info(
                f"[EmergencyExit] Recorded forced exit "
                f"(trade={trade_id[:8]}..., reason={reason})"
            )

        except Exception as e:
            logger.error(
                f"[EmergencyExit] Failed to record exit in DB: {e}",
                exc_info=True
            )

    async def _handle_exit_failure(
        self,
        trade_id: str,
        asset: str,
        side: str,
        reason: str,
        age_seconds: float
    ) -> None:
        """
        Handle case where all exit attempts failed.

        Args:
            trade_id: Trade ID
            asset: Asset name
            side: Trade side
            reason: Exit reason
            age_seconds: Position age
        """
        logger.critical(
            f"[EmergencyExit] ❌ FAILED TO EXIT {asset} {side} "
            f"after {self.max_retries} attempts "
            f"(reason={reason}, age={age_seconds:.0f}s, "
            f"trade={trade_id[:8]}...)"
        )

        # Alert to Discord if available
        if self.discord:
            try:
                await self.discord.send_critical_alert(
                    title="🚨 Emergency Exit Failed",
                    message=(
                        f"Failed to force-exit position after "
                        f"{self.max_retries} attempts\n\n"
                        f"**Asset**: {asset} {side}\n"
                        f"**Reason**: {reason}\n"
                        f"**Age**: {age_seconds:.0f}s\n"
                        f"**Trade ID**: {trade_id[:8]}...\n\n"
                        f"**Action Required**: Manual intervention needed!"
                    )
                )
            except Exception as e:
                logger.error(
                    f"[EmergencyExit] Failed to send Discord alert: {e}",
                    exc_info=True
                )

        # Record health event in database
        try:
            await self.db.execute("""
                INSERT INTO health_events (
                    event_type, asset, action_taken, affected_trades
                ) VALUES (
                    'critical', $1, 'exit_failed', ARRAY[$2]::UUID[]
                )
            """, asset, trade_id)
        except Exception as e:
            logger.error(
                f"[EmergencyExit] Failed to log health event: {e}",
                exc_info=True
            )

    def get_stats(self) -> Dict:
        """Get exit statistics."""
        success_rate = (
            self.successful_exits / self.total_exits * 100
            if self.total_exits > 0
            else 0
        )

        return {
            "total_exits": self.total_exits,
            "successful_exits": self.successful_exits,
            "failed_exits": self.failed_exits,
            "success_rate": f"{success_rate:.1f}%"
        }
