#!/usr/bin/env python3
"""
Position Limiter for max positions and position size enforcement.

Prevents overexposure by limiting:
- Maximum concurrent positions
- Maximum position size in dollars

Usage:
    limiter = PositionLimiter(
        db=database,
        max_positions=4,
        max_position_size=100.0
    )

    # Check before opening position
    can_open, reason = await limiter.can_open_position(
        symbol="BTC",
        size_usd=50.0
    )

    if can_open:
        # Execute trade
        await limiter.update_position_count()
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List

from db.connection import Database

logger = logging.getLogger(__name__)


class PositionLimiter:
    """
    Enforces position limits for risk management.

    Tracks open positions and validates new position requests
    against configured limits.
    """

    def __init__(
        self,
        db: Database,
        session_id: Optional[str] = None,
        max_positions: int = 4,
        max_position_size: float = 100.0,
        max_total_exposure: float = 400.0
    ):
        """
        Initialize position limiter.

        Args:
            db: Database connection
            session_id: Optional trading session ID
            max_positions: Maximum concurrent positions
            max_position_size: Maximum single position size (dollars)
            max_total_exposure: Maximum total exposure across positions (dollars)
        """
        self.db = db
        self.session_id = session_id
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        self.max_total_exposure = max_total_exposure

        # Cached state
        self._open_positions: int = 0
        self._total_exposure: float = 0.0
        self._last_sync: Optional[datetime] = None

    async def ensure_table(self) -> None:
        """
        Ensure position_limits table exists.

        Stores current position state for persistence.
        """
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS position_limits (
                id SERIAL PRIMARY KEY,
                session_id UUID,
                open_positions INTEGER DEFAULT 0,
                total_exposure DECIMAL(12, 4) DEFAULT 0,
                max_positions INTEGER DEFAULT 4,
                max_position_size DECIMAL(12, 4) DEFAULT 100,
                max_total_exposure DECIMAL(12, 4) DEFAULT 400,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(session_id)
            )
        """)

    async def initialize(self) -> None:
        """Initialize limiter and sync from database."""
        await self.ensure_table()
        await self.sync_from_db()

    async def sync_from_db(self) -> None:
        """Sync position count from trades table."""
        # Count actual open positions from trades table
        row = await self.db.fetchrow("""
            SELECT
                COUNT(*) as open_positions,
                COALESCE(SUM(size_dollars), 0) as total_exposure
            FROM trades
            WHERE session_id = $1
              AND exit_time IS NULL
        """, self.session_id)

        if row:
            self._open_positions = row["open_positions"]
            self._total_exposure = float(row["total_exposure"])
        else:
            self._open_positions = 0
            self._total_exposure = 0.0

        self._last_sync = datetime.now(timezone.utc)

        # Update position_limits table
        await self.db.execute("""
            INSERT INTO position_limits (
                session_id, open_positions, total_exposure,
                max_positions, max_position_size, max_total_exposure
            ) VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (session_id) DO UPDATE SET
                open_positions = $2,
                total_exposure = $3,
                updated_at = NOW()
        """,
            self.session_id, self._open_positions, self._total_exposure,
            self.max_positions, self.max_position_size, self.max_total_exposure
        )

        logger.debug(
            f"[PositionLimiter] Synced: {self._open_positions} positions, "
            f"${self._total_exposure:.2f} exposure"
        )

    async def can_open_position(
        self,
        symbol: str,
        size_usd: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Trading symbol (e.g., "BTC")
            size_usd: Position size in dollars

        Returns:
            Tuple of (can_open, reason)
        """
        # Refresh from DB if stale (> 10 seconds)
        if self._last_sync is None or \
           (datetime.now(timezone.utc) - self._last_sync).total_seconds() > 10:
            await self.sync_from_db()

        # Check position count limit
        if self._open_positions >= self.max_positions:
            reason = (
                f"Maximum positions reached: "
                f"{self._open_positions}/{self.max_positions}"
            )
            logger.warning(f"[PositionLimiter] {reason}")
            return False, reason

        # Check single position size limit
        if size_usd > self.max_position_size:
            reason = (
                f"Position size exceeds limit: "
                f"${size_usd:.2f} > ${self.max_position_size:.2f}"
            )
            logger.warning(f"[PositionLimiter] {reason}")
            return False, reason

        # Check total exposure limit
        new_total = self._total_exposure + size_usd
        if new_total > self.max_total_exposure:
            reason = (
                f"Total exposure would exceed limit: "
                f"${new_total:.2f} > ${self.max_total_exposure:.2f}"
            )
            logger.warning(f"[PositionLimiter] {reason}")
            return False, reason

        # Check if already have position in this symbol
        existing = await self.db.fetchrow("""
            SELECT id FROM trades
            WHERE session_id = $1
              AND asset = $2
              AND exit_time IS NULL
            LIMIT 1
        """, self.session_id, symbol)

        if existing:
            reason = f"Already have open position in {symbol}"
            logger.debug(f"[PositionLimiter] {reason}")
            return False, reason

        return True, ""

    async def update_position_count(self) -> int:
        """
        Update position count from database.

        Call after opening or closing positions.

        Returns:
            Current open position count
        """
        await self.sync_from_db()
        return self._open_positions

    async def get_available_position_slots(self) -> int:
        """
        Get number of available position slots.

        Returns:
            Number of positions that can still be opened
        """
        if self._last_sync is None:
            await self.sync_from_db()
        return max(0, self.max_positions - self._open_positions)

    async def get_remaining_exposure_capacity(self) -> float:
        """
        Get remaining exposure capacity.

        Returns:
            Dollars remaining before hitting exposure limit
        """
        if self._last_sync is None:
            await self.sync_from_db()
        return max(0.0, self.max_total_exposure - self._total_exposure)

    async def get_open_positions_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all open positions.

        Returns:
            List of position dicts with asset, side, size, entry_time
        """
        rows = await self.db.fetch("""
            SELECT asset, side, size_dollars, entry_price, entry_time
            FROM trades
            WHERE session_id = $1
              AND exit_time IS NULL
            ORDER BY entry_time
        """, self.session_id)

        return [
            {
                "asset": r["asset"],
                "side": r["side"],
                "size_usd": float(r["size_dollars"]),
                "entry_price": float(r["entry_price"]),
                "entry_time": r["entry_time"].isoformat() if r["entry_time"] else None,
            }
            for r in rows
        ]

    async def get_state(self) -> Dict[str, Any]:
        """
        Get current position limiter state.

        Returns:
            Dict with all metrics
        """
        if self._last_sync is None:
            await self.sync_from_db()

        return {
            "open_positions": self._open_positions,
            "max_positions": self.max_positions,
            "available_slots": self.max_positions - self._open_positions,
            "total_exposure": self._total_exposure,
            "max_total_exposure": self.max_total_exposure,
            "remaining_capacity": self.max_total_exposure - self._total_exposure,
            "max_position_size": self.max_position_size,
        }
