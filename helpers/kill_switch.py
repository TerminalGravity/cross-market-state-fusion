#!/usr/bin/env python3
"""
Kill Switch for emergency trading halt.

Provides a database-backed mechanism to immediately halt all trading
when safety limits are breached. Persists across worker restarts.

Activation triggers:
- Daily loss limit exceeded
- Consecutive loss limit hit
- Manual emergency halt

Usage:
    kill_switch = KillSwitch(db=database)

    # Check before trading
    if await kill_switch.is_active():
        return  # Skip trading

    # Activate on limit breach
    await kill_switch.activate(
        reason="daily_loss_limit",
        details={"loss": -150.0, "limit": -100.0}
    )

    # Manual deactivation (after review)
    await kill_switch.deactivate(reason="manual_reset")
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from db.connection import Database

logger = logging.getLogger(__name__)


class KillSwitch:
    """
    Emergency trading halt with database persistence.

    Stores activation state in PostgreSQL so it survives
    worker restarts and can be monitored externally.
    """

    def __init__(
        self,
        db: Database,
        session_id: Optional[str] = None
    ):
        """
        Initialize kill switch.

        Args:
            db: Database connection
            session_id: Optional trading session ID
        """
        self.db = db
        self.session_id = session_id
        self._cached_state: Optional[bool] = None
        self._last_check: Optional[datetime] = None

    async def ensure_table(self) -> None:
        """
        Ensure kill_switch_state table exists.

        Creates the table if it doesn't exist (idempotent).
        """
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS kill_switch_state (
                id SERIAL PRIMARY KEY,
                is_active BOOLEAN NOT NULL DEFAULT FALSE,
                activated_at TIMESTAMPTZ,
                deactivated_at TIMESTAMPTZ,
                activation_reason TEXT,
                deactivation_reason TEXT,
                details JSONB DEFAULT '{}',
                session_id UUID,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # Ensure at least one row exists
        row = await self.db.fetchrow(
            "SELECT id FROM kill_switch_state ORDER BY id LIMIT 1"
        )
        if not row:
            await self.db.execute(
                "INSERT INTO kill_switch_state (is_active) VALUES (FALSE)"
            )
            logger.info("[KillSwitch] Created initial state record")

    async def is_active(self, use_cache: bool = True) -> bool:
        """
        Check if kill switch is active.

        Args:
            use_cache: Use cached value if available (5s TTL)

        Returns:
            True if trading should be halted
        """
        now = datetime.now(timezone.utc)

        # Use cache if fresh (< 5 seconds old)
        if use_cache and self._cached_state is not None and self._last_check:
            age = (now - self._last_check).total_seconds()
            if age < 5.0:
                return self._cached_state

        # Query database
        row = await self.db.fetchrow("""
            SELECT is_active FROM kill_switch_state
            ORDER BY id LIMIT 1
        """)

        is_active = row["is_active"] if row else False

        # Update cache
        self._cached_state = is_active
        self._last_check = now

        return is_active

    async def activate(
        self,
        reason: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Activate kill switch to halt all trading.

        Args:
            reason: Why the kill switch was activated
            details: Additional context (e.g., loss amounts)
        """
        await self.db.execute("""
            UPDATE kill_switch_state
            SET is_active = TRUE,
                activated_at = NOW(),
                activation_reason = $1,
                details = $2,
                session_id = $3,
                updated_at = NOW()
            WHERE id = (SELECT id FROM kill_switch_state ORDER BY id LIMIT 1)
        """, reason, details or {}, self.session_id)

        # Update cache immediately
        self._cached_state = True
        self._last_check = datetime.now(timezone.utc)

        logger.critical(
            f"[KillSwitch] 🛑 ACTIVATED: {reason} | "
            f"Details: {details}"
        )

        # Log safety event
        await self._log_safety_event("kill_switch_activated", reason, details)

    async def deactivate(self, reason: str = "manual_reset") -> None:
        """
        Deactivate kill switch to resume trading.

        Args:
            reason: Why the kill switch was deactivated
        """
        await self.db.execute("""
            UPDATE kill_switch_state
            SET is_active = FALSE,
                deactivated_at = NOW(),
                deactivation_reason = $1,
                updated_at = NOW()
            WHERE id = (SELECT id FROM kill_switch_state ORDER BY id LIMIT 1)
        """, reason)

        # Update cache immediately
        self._cached_state = False
        self._last_check = datetime.now(timezone.utc)

        logger.warning(f"[KillSwitch] ✓ Deactivated: {reason}")

        # Log safety event
        await self._log_safety_event("kill_switch_deactivated", reason)

    async def get_state(self) -> Dict[str, Any]:
        """
        Get full kill switch state.

        Returns:
            Dict with activation state and history
        """
        row = await self.db.fetchrow("""
            SELECT * FROM kill_switch_state
            ORDER BY id LIMIT 1
        """)

        if not row:
            return {"is_active": False, "error": "No state found"}

        return {
            "is_active": row["is_active"],
            "activated_at": row["activated_at"].isoformat() if row["activated_at"] else None,
            "deactivated_at": row["deactivated_at"].isoformat() if row["deactivated_at"] else None,
            "activation_reason": row["activation_reason"],
            "deactivation_reason": row["deactivation_reason"],
            "details": row["details"],
        }

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
            # Table might not exist yet - log but don't fail
            logger.debug(f"[KillSwitch] Could not log safety event: {e}")
