#!/usr/bin/env python3
"""
Loss Tracker for daily and consecutive loss limits.

Tracks trading P&L and detects limit breaches that should
trigger the kill switch. Persists state to PostgreSQL.

Safety limits:
- Daily loss limit: Maximum cumulative loss per day
- Consecutive loss limit: Maximum losing trades in a row

Usage:
    tracker = LossTracker(
        db=database,
        daily_loss_limit=100.0,
        consecutive_loss_limit=3
    )

    # Record trade result
    await tracker.record_pnl(pnl=-25.0)

    # Check limits
    exceeded, reason = await tracker.check_all_limits()
    if exceeded:
        await kill_switch.activate(reason)
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any

from db.connection import Database

logger = logging.getLogger(__name__)


class LossTracker:
    """
    Tracks trading losses and detects limit breaches.

    Uses PostgreSQL for persistence across worker restarts.
    Automatically resets daily metrics at midnight UTC.
    """

    def __init__(
        self,
        db: Database,
        session_id: Optional[str] = None,
        daily_loss_limit: float = 100.0,
        consecutive_loss_limit: int = 3
    ):
        """
        Initialize loss tracker.

        Args:
            db: Database connection
            session_id: Optional trading session ID
            daily_loss_limit: Maximum daily loss in dollars (positive number)
            consecutive_loss_limit: Maximum consecutive losing trades
        """
        self.db = db
        self.session_id = session_id
        self.daily_loss_limit = daily_loss_limit
        self.consecutive_loss_limit = consecutive_loss_limit

        # In-memory state (synced from DB on init)
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._last_trade_date: Optional[datetime] = None
        self._initialized = False

    async def ensure_table(self) -> None:
        """
        Ensure loss_tracking table exists.

        Creates table and safety_events table if they don't exist.
        """
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS loss_tracking (
                id SERIAL PRIMARY KEY,
                session_id UUID,
                trading_date DATE NOT NULL DEFAULT CURRENT_DATE,
                daily_pnl DECIMAL(12, 4) DEFAULT 0,
                consecutive_losses INTEGER DEFAULT 0,
                total_trades_today INTEGER DEFAULT 0,
                wins_today INTEGER DEFAULT 0,
                losses_today INTEGER DEFAULT 0,
                largest_loss_today DECIMAL(12, 4) DEFAULT 0,
                last_trade_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(trading_date, session_id)
            )
        """)

        # Safety events table for audit trail
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                id SERIAL PRIMARY KEY,
                event_type TEXT NOT NULL,
                session_id UUID,
                reason TEXT,
                details JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

    async def initialize(self) -> None:
        """
        Initialize tracker from database state.

        Loads today's metrics or creates new record.
        """
        await self.ensure_table()

        today = datetime.now(timezone.utc).date()

        # Get or create today's record
        row = await self.db.fetchrow("""
            SELECT * FROM loss_tracking
            WHERE trading_date = $1 AND (session_id = $2 OR session_id IS NULL)
            ORDER BY session_id NULLS LAST
            LIMIT 1
        """, today, self.session_id)

        if row:
            self._daily_pnl = float(row["daily_pnl"])
            self._consecutive_losses = row["consecutive_losses"]
            self._last_trade_date = today
            logger.info(
                f"[LossTracker] Loaded state: daily_pnl=${self._daily_pnl:.2f}, "
                f"consecutive_losses={self._consecutive_losses}"
            )
        else:
            # Create today's record
            await self.db.execute("""
                INSERT INTO loss_tracking (session_id, trading_date, daily_pnl, consecutive_losses)
                VALUES ($1, $2, 0, 0)
            """, self.session_id, today)
            self._daily_pnl = 0.0
            self._consecutive_losses = 0
            self._last_trade_date = today
            logger.info("[LossTracker] Created new daily record")

        self._initialized = True

    async def record_pnl(self, pnl: float) -> None:
        """
        Record trade result and update metrics.

        Args:
            pnl: Trade P&L in dollars (negative for loss)
        """
        if not self._initialized:
            await self.initialize()

        today = datetime.now(timezone.utc).date()

        # Check if new day (reset daily metrics)
        if self._last_trade_date and self._last_trade_date < today:
            await self._reset_daily_metrics()
            self._last_trade_date = today

        # Update metrics
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0  # Reset on win

        # Persist to database
        await self.db.execute("""
            UPDATE loss_tracking
            SET daily_pnl = $1,
                consecutive_losses = $2,
                total_trades_today = total_trades_today + 1,
                wins_today = wins_today + CASE WHEN $3 > 0 THEN 1 ELSE 0 END,
                losses_today = losses_today + CASE WHEN $3 < 0 THEN 1 ELSE 0 END,
                largest_loss_today = CASE
                    WHEN $3 < 0 AND $3 < largest_loss_today THEN $3
                    ELSE largest_loss_today
                END,
                last_trade_at = NOW(),
                updated_at = NOW()
            WHERE trading_date = $4 AND (session_id = $5 OR session_id IS NULL)
        """, self._daily_pnl, self._consecutive_losses, pnl, today, self.session_id)

        logger.info(
            f"[LossTracker] Recorded PnL: ${pnl:+.2f} | "
            f"Daily: ${self._daily_pnl:.2f} | "
            f"Consecutive losses: {self._consecutive_losses}"
        )

    async def check_all_limits(self) -> Tuple[bool, str]:
        """
        Check if any limits are exceeded.

        Returns:
            Tuple of (exceeded, reason) where exceeded is True if
            any limit is breached
        """
        if not self._initialized:
            await self.initialize()

        # Check daily loss limit
        if self._daily_pnl <= -self.daily_loss_limit:
            reason = (
                f"Daily loss limit exceeded: "
                f"${self._daily_pnl:.2f} <= ${-self.daily_loss_limit:.2f}"
            )
            logger.warning(f"[LossTracker] {reason}")
            return True, reason

        # Check consecutive loss limit
        if self._consecutive_losses >= self.consecutive_loss_limit:
            reason = (
                f"Consecutive loss limit hit: "
                f"{self._consecutive_losses} >= {self.consecutive_loss_limit}"
            )
            logger.warning(f"[LossTracker] {reason}")
            return True, reason

        return False, ""

    async def get_remaining_daily_loss_allowance(self) -> float:
        """
        Get remaining daily loss allowance.

        Returns:
            Dollars remaining before daily limit (0 if exceeded)
        """
        if not self._initialized:
            await self.initialize()

        remaining = self.daily_loss_limit + self._daily_pnl
        return max(0.0, remaining)

    async def get_state(self) -> Dict[str, Any]:
        """
        Get current loss tracking state.

        Returns:
            Dict with all metrics
        """
        if not self._initialized:
            await self.initialize()

        return {
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": self.daily_loss_limit,
            "remaining_allowance": await self.get_remaining_daily_loss_allowance(),
            "consecutive_losses": self._consecutive_losses,
            "consecutive_loss_limit": self.consecutive_loss_limit,
            "last_trade_date": self._last_trade_date.isoformat() if self._last_trade_date else None,
        }

    async def _reset_daily_metrics(self) -> None:
        """Reset daily metrics (called at start of new day)."""
        today = datetime.now(timezone.utc).date()

        logger.info("[LossTracker] New day - resetting daily metrics")

        # Create new record for today
        await self.db.execute("""
            INSERT INTO loss_tracking (session_id, trading_date, daily_pnl, consecutive_losses)
            VALUES ($1, $2, 0, $3)
            ON CONFLICT (trading_date, session_id) DO NOTHING
        """, self.session_id, today, self._consecutive_losses)

        self._daily_pnl = 0.0
        # Note: consecutive_losses carries over across days

        # Log safety event
        await self._log_safety_event("daily_reset", "New trading day started")

    async def reset_daily_metrics(self) -> None:
        """Manual reset of daily metrics."""
        await self._reset_daily_metrics()
        logger.info("[LossTracker] Manual daily reset complete")

    async def _log_safety_event(
        self,
        event_type: str,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log safety event to database."""
        try:
            await self.db.execute("""
                INSERT INTO safety_events (
                    event_type, session_id, reason, details
                ) VALUES ($1, $2, $3, $4)
            """, event_type, self.session_id, reason, details or {})
        except Exception as e:
            logger.debug(f"[LossTracker] Could not log safety event: {e}")
