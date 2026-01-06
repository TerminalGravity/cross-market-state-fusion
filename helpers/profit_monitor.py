#!/usr/bin/env python3
"""
Profit monitor - background task that checks profit and triggers transfers.

Runs every 60 seconds and checks:
1. Threshold trigger: profit >= PROFIT_TRANSFER_THRESHOLD
2. Time trigger: time_since_last_transfer >= MAX_INTERVAL_HOURS

When triggered, transfers available profit to cold wallet.
"""
import asyncio
import os
import logging
from typing import Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


class ProfitMonitor:
    """
    Background task: Monitor profit and trigger transfers.

    Usage:
        monitor = ProfitMonitor(session_id, db, balance_tracker, transfer_executor, initial_balance)
        asyncio.create_task(monitor.start())
    """

    def __init__(
        self,
        session_id: str,
        db,  # Database instance
        balance_tracker,  # WalletBalanceTracker instance
        transfer_executor,  # ProfitTransferExecutor instance
        initial_balance: float
    ):
        """
        Initialize profit monitor.

        Args:
            session_id: Trading session ID
            db: Database instance
            balance_tracker: WalletBalanceTracker for querying balance
            transfer_executor: ProfitTransferExecutor for executing transfers
            initial_balance: Starting balance (working capital to preserve)
        """
        self.session_id = session_id
        self.db = db
        self.balance_tracker = balance_tracker
        self.transfer_executor = transfer_executor
        self.initial_balance = initial_balance

        # Config from environment
        self.threshold = float(os.getenv("PROFIT_TRANSFER_THRESHOLD", "100"))
        self.max_interval_hours = float(
            os.getenv("PROFIT_TRANSFER_MAX_INTERVAL_HOURS", "24")
        )
        self.check_interval = 60  # Check every 60 seconds

        # State
        self.last_transfer_time: Optional[datetime] = None
        self.running = False
        self.total_transferred = 0.0
        self.transfer_count = 0

        logger.info(
            f"ProfitMonitor initialized: "
            f"initial_balance=${initial_balance:.2f}, "
            f"threshold=${self.threshold:.2f}, "
            f"max_interval={self.max_interval_hours}h"
        )

    async def start(self) -> None:
        """
        Start monitoring loop (runs every 60s).

        This is the main background task - should be launched with asyncio.create_task().
        """
        self.running = True
        logger.info("Profit monitor started")

        while self.running:
            try:
                await self._check_and_transfer()
            except Exception as e:
                logger.error(f"Profit monitor error: {e}", exc_info=True)

            # Sleep for check interval
            await asyncio.sleep(self.check_interval)

        logger.info("Profit monitor stopped")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self.running = False

    async def _check_and_transfer(self) -> None:
        """
        Check conditions and initiate transfer if triggered.

        This is called every 60 seconds by the main loop.
        """
        try:
            # Query current balance from blockchain
            current_balance = await self.balance_tracker.get_balance()

            # Query open positions value (locked capital)
            open_trades = await self.db.get_open_trades(self.session_id)
            open_positions_value = sum(
                float(trade["size_dollars"])
                for trade in open_trades
                if trade.get("size_dollars")
            )

            # Calculate available profit
            # Profit = current_balance - initial_balance - locked_capital
            available_profit = (
                current_balance - self.initial_balance - open_positions_value
            )

            # Check threshold trigger
            threshold_triggered = available_profit >= self.threshold

            # Check time trigger
            time_triggered = False
            hours_since_last = None

            if self.last_transfer_time:
                time_delta = datetime.now(timezone.utc) - self.last_transfer_time
                hours_since_last = time_delta.total_seconds() / 3600
                time_triggered = hours_since_last >= self.max_interval_hours

            # Log status (every 10th check = every 10 minutes)
            if not hasattr(self, '_check_count'):
                self._check_count = 0
            self._check_count += 1

            if self._check_count % 10 == 0:
                logger.info(
                    f"Profit check: current=${current_balance:.2f}, "
                    f"initial=${self.initial_balance:.2f}, "
                    f"locked=${open_positions_value:.2f}, "
                    f"available=${available_profit:.2f}, "
                    f"hours_since_last={f'{hours_since_last:.1f}' if hours_since_last is not None else 'N/A'}"
                )

            # Trigger transfer?
            if threshold_triggered or time_triggered:
                trigger_reason = "threshold" if threshold_triggered else "time_interval"

                logger.info(
                    f"Transfer triggered ({trigger_reason}): "
                    f"available_profit=${available_profit:.2f}, "
                    f"threshold=${self.threshold:.2f}, "
                    f"hours_since_last={f'{hours_since_last:.1f}' if hours_since_last is not None else 'N/A'}h"
                )

                # Sanity check: don't transfer if available profit is negative or tiny
                if available_profit < 1.0:
                    logger.warning(
                        f"Transfer skipped: available profit too low (${available_profit:.2f})"
                    )
                    return

                # Sanity check: don't transfer if it would deplete working capital
                post_transfer_balance = current_balance - available_profit
                if post_transfer_balance < self.initial_balance:
                    logger.warning(
                        f"Transfer skipped: would deplete working capital "
                        f"(post_transfer=${post_transfer_balance:.2f}, "
                        f"initial=${self.initial_balance:.2f})"
                    )
                    return

                # Execute transfer
                try:
                    tx_hash = await self.transfer_executor.transfer(
                        amount_dollars=available_profit,
                        session_id=self.session_id,
                        trigger_reason=trigger_reason
                    )

                    if tx_hash:
                        self.last_transfer_time = datetime.now(timezone.utc)
                        self.total_transferred += available_profit
                        self.transfer_count += 1

                        logger.info(
                            f"✅ Transfer initiated: ${available_profit:.2f} "
                            f"(tx: {tx_hash}, total: ${self.total_transferred:.2f})"
                        )
                    elif not self.transfer_executor.dry_run:
                        # Transfer failed (but not in dry-run mode)
                        logger.error("Transfer failed - see logs above")

                except ValueError as e:
                    # Pre-flight check failed
                    logger.warning(f"Transfer skipped: {e}")
                except Exception as e:
                    # Unexpected error
                    logger.error(f"Transfer failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"Error in check_and_transfer: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """
        Get monitor statistics.

        Returns:
            Dict with transfer stats
        """
        return {
            "total_transferred": self.total_transferred,
            "transfer_count": self.transfer_count,
            "last_transfer_time": self.last_transfer_time.isoformat() if self.last_transfer_time else None,
            "initial_balance": self.initial_balance,
            "threshold": self.threshold,
            "max_interval_hours": self.max_interval_hours,
        }
