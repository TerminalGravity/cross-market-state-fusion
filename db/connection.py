#!/usr/bin/env python3
"""
Database connection management for Railway PostgreSQL.

Provides async connection pooling with automatic reconnection
and health checking.
"""
import os
import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import json

import asyncpg

logger = logging.getLogger(__name__)


class Database:
    """
    Async PostgreSQL connection pool manager.

    Usage:
        db = Database()
        await db.connect()

        # Use context manager for connections
        async with db.acquire() as conn:
            await conn.fetch("SELECT * FROM sessions")

        await db.close()
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database manager.

        Args:
            database_url: PostgreSQL connection URL.
                          Defaults to DATABASE_URL env var.
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not provided")

        self.pool: Optional[asyncpg.Pool] = None
        self._connected = False

    async def connect(
        self,
        min_size: int = 2,
        max_size: int = 10,
        command_timeout: int = 60
    ) -> None:
        """
        Create connection pool.

        Args:
            min_size: Minimum pool connections
            max_size: Maximum pool connections
            command_timeout: Query timeout in seconds
        """
        if self._connected:
            return

        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=command_timeout,
            # Handle JSON serialization
            init=self._init_connection
        )
        self._connected = True
        logger.info(f"Database pool connected (min={min_size}, max={max_size})")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        """Initialize connection with JSON codec."""
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
        await conn.set_type_codec(
            'json',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        if not self._connected:
            raise RuntimeError("Database not connected. Call connect() first.")
        async with self.pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        """Execute a query without returning results."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch multiple rows."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    # =========================================================================
    # Session Operations
    # =========================================================================

    async def create_session(
        self,
        mode: str = "paper",
        trade_size: float = 50.0,
        model_version: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> str:
        """
        Create a new trading session.

        Returns:
            Session UUID as string
        """
        row = await self.fetchrow("""
            INSERT INTO sessions (mode, trade_size, model_version, config)
            VALUES ($1, $2, $3, $4)
            RETURNING id
        """, mode, trade_size, model_version, config or {})
        return str(row["id"])

    async def get_active_session(self) -> Optional[asyncpg.Record]:
        """Get the currently running session."""
        return await self.fetchrow("""
            SELECT * FROM sessions
            WHERE status = 'running'
            ORDER BY started_at DESC
            LIMIT 1
        """)

    async def update_session_checkpoint(
        self,
        session_id: str,
        checkpoint_data: Dict,
        total_pnl: float,
        trade_count: int,
        win_count: int
    ) -> None:
        """Update session checkpoint for crash recovery."""
        await self.execute("""
            UPDATE sessions
            SET last_checkpoint = NOW(),
                checkpoint_data = $2,
                total_pnl = $3,
                trade_count = $4,
                win_count = $5
            WHERE id = $1
        """, session_id, checkpoint_data, total_pnl, trade_count, win_count)

    async def end_session(self, session_id: str, status: str = "stopped") -> None:
        """Mark session as ended."""
        await self.execute("""
            UPDATE sessions
            SET status = $2, ended_at = NOW()
            WHERE id = $1
        """, session_id, status)

    # =========================================================================
    # Trade Operations
    # =========================================================================

    async def record_trade_open(
        self,
        session_id: str,
        condition_id: str,
        asset: str,
        entry_price: float,
        entry_binance_price: float,
        side: str,
        size_dollars: float,
        time_remaining: float,
        action_probs: Dict[str, float],
        market_state: Dict[str, Any],
        order_id: Optional[str] = None,
        execution_type: str = "paper",
        fill_status: Optional[str] = None,
        clob_response: Optional[Dict] = None
    ) -> str:
        """
        Record a new trade entry.

        Args:
            session_id: Session UUID
            condition_id: Polymarket condition ID
            asset: Asset name (BTC, ETH, etc.)
            entry_price: Entry price
            entry_binance_price: Binance price at entry
            side: Trade side (UP/DOWN)
            size_dollars: Position size in dollars
            time_remaining: Time remaining in market
            action_probs: Model action probabilities
            market_state: Full market state features
            order_id: Polymarket CLOB order ID (for live trades)
            execution_type: 'paper' or 'live'
            fill_status: Order fill status (matched, rejected, etc.)
            clob_response: Full CLOB API response for audit

        Returns:
            Trade UUID as string
        """
        shares = size_dollars / entry_price if entry_price > 0 else 0
        row = await self.fetchrow("""
            INSERT INTO trades (
                session_id, condition_id, asset, entry_time, entry_price,
                entry_binance_price, side, size_dollars, shares,
                time_remaining_at_entry, action_probs, market_state,
                order_id, execution_type, fill_status, clob_response
            ) VALUES ($1, $2, $3, NOW(), $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            RETURNING id
        """,
            session_id, condition_id, asset, entry_price,
            entry_binance_price, side, size_dollars, shares,
            time_remaining, action_probs, market_state,
            order_id, execution_type, fill_status, clob_response
        )
        return str(row["id"])

    async def record_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        exit_binance_price: float,
        exit_reason: str,
        pnl: float,
        duration_seconds: int
    ) -> None:
        """Record trade exit."""
        await self.execute("""
            UPDATE trades
            SET exit_time = NOW(),
                exit_price = $2,
                exit_binance_price = $3,
                exit_reason = $4,
                pnl = $5,
                duration_seconds = $6
            WHERE id = $1
        """, trade_id, exit_price, exit_binance_price, exit_reason, pnl, duration_seconds)

    async def get_open_trades(self, session_id: str) -> List[asyncpg.Record]:
        """Get trades without exit (open positions)."""
        return await self.fetch("""
            SELECT * FROM trades
            WHERE session_id = $1 AND exit_time IS NULL
            ORDER BY entry_time
        """, session_id)

    async def get_recent_trades(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[asyncpg.Record]:
        """Get recent completed trades."""
        return await self.fetch("""
            SELECT * FROM trades
            WHERE session_id = $1 AND exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT $2
        """, session_id, limit)

    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated session statistics."""
        row = await self.fetchrow("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MIN(pnl) as worst_trade,
                MAX(pnl) as best_trade,
                AVG(duration_seconds) as avg_duration
            FROM trades
            WHERE session_id = $1 AND pnl IS NOT NULL
        """, session_id)

        if not row or row["total_trades"] == 0:
            return {
                "total_trades": 0,
                "wins": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "worst_trade": 0.0,
                "best_trade": 0.0,
                "avg_duration": 0.0
            }

        return {
            "total_trades": row["total_trades"],
            "wins": row["wins"] or 0,
            "win_rate": (row["wins"] or 0) / row["total_trades"],
            "total_pnl": float(row["total_pnl"] or 0),
            "avg_pnl": float(row["avg_pnl"] or 0),
            "worst_trade": float(row["worst_trade"] or 0),
            "best_trade": float(row["best_trade"] or 0),
            "avg_duration": float(row["avg_duration"] or 0)
        }

    # =========================================================================
    # Metrics Operations
    # =========================================================================

    async def record_metrics(
        self,
        session_id: str,
        cumulative_pnl: float,
        trades_today: int,
        win_rate_today: float,
        open_positions: int = 0,
        total_exposure: float = 0,
        active_markets: int = 0,
        markets_data: Optional[Dict] = None
    ) -> None:
        """Record a metrics snapshot."""
        await self.execute("""
            INSERT INTO metrics (
                session_id, cumulative_pnl, trades_today, win_rate_today,
                open_position_count, total_exposure, active_markets, markets_data
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
            session_id, cumulative_pnl, trades_today, win_rate_today,
            open_positions, total_exposure, active_markets, markets_data or {}
        )

    async def get_metrics_history(
        self,
        session_id: str,
        hours: int = 24
    ) -> List[asyncpg.Record]:
        """Get metrics history for time-series charts."""
        return await self.fetch("""
            SELECT recorded_at, cumulative_pnl, trades_today, win_rate_today
            FROM metrics
            WHERE session_id = $1
              AND recorded_at > NOW() - ($2 || ' hours')::INTERVAL
            ORDER BY recorded_at
        """, session_id, str(hours))

    # =========================================================================
    # Transfer Operations (Profit Transfer System)
    # =========================================================================

    async def record_transfer(
        self,
        session_id: str,
        amount: float,
        tx_hash: Optional[str] = None,
        status: str = "pending",
        trigger_reason: str = "threshold"
    ) -> int:
        """
        Record a profit transfer attempt.

        Args:
            session_id: Trading session ID
            amount: Transfer amount in dollars
            tx_hash: Optional transaction hash (set after sending)
            status: Transfer status ('pending', 'sent', 'confirmed', 'failed', 'timeout')
            trigger_reason: Why transfer triggered ('threshold', 'time_interval', 'manual')

        Returns:
            Transfer ID (integer)
        """
        row = await self.fetchrow("""
            INSERT INTO transfers (session_id, amount, tx_hash, status, trigger_reason)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, session_id, amount, tx_hash, status, trigger_reason)
        return int(row["id"])

    async def update_transfer_status(
        self,
        transfer_id: int,
        status: str,
        tx_hash: Optional[str] = None,
        gas_used: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update transfer status after confirmation or failure.

        Args:
            transfer_id: Transfer ID from record_transfer()
            status: New status ('sent', 'confirmed', 'failed', 'timeout')
            tx_hash: Optional transaction hash (if newly available)
            gas_used: Gas used for confirmed transactions
            error_message: Error details for failed transactions
        """
        await self.execute("""
            UPDATE transfers
            SET status = $2,
                tx_hash = COALESCE($3, tx_hash),
                gas_used = COALESCE($4, gas_used),
                error_message = COALESCE($5, error_message),
                confirmed_at = CASE WHEN $2 = 'confirmed' THEN NOW() ELSE confirmed_at END,
                retry_count = retry_count + 1
            WHERE id = $1
        """, transfer_id, status, tx_hash, gas_used, error_message)

    async def get_transfer_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[asyncpg.Record]:
        """
        Get recent transfers for a session.

        Args:
            session_id: Trading session ID
            limit: Maximum number of transfers to return

        Returns:
            List of transfer records
        """
        return await self.fetch("""
            SELECT * FROM transfers
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, session_id, limit)

    async def get_session_by_mode(
        self,
        mode: str = "paper",
        status: str = "running"
    ) -> Optional[asyncpg.Record]:
        """
        Get session by mode and status (for crash recovery).

        Args:
            mode: 'paper' or 'live'
            status: Session status (default: 'running')

        Returns:
            Session record or None
        """
        return await self.fetchrow("""
            SELECT * FROM sessions
            WHERE mode = $1 AND status = $2
            ORDER BY started_at DESC
            LIMIT 1
        """, mode, status)

    # =========================================================================
    # Dual-Mode Comparison Operations
    # =========================================================================

    async def get_dual_mode_summary(self) -> Optional[asyncpg.Record]:
        """Get current dual-mode session summary (paper + live)."""
        return await self.fetchrow("""
            SELECT * FROM dual_mode_sessions
            WHERE status = 'running'
            LIMIT 1
        """)

    async def get_execution_comparison(
        self,
        hours: int = 24
    ) -> List[asyncpg.Record]:
        """Compare execution quality between paper and live modes."""
        return await self.fetch("""
            SELECT * FROM execution_quality
        """)

    async def get_session_by_mode(
        self,
        mode: str,
        status: str = 'running'
    ) -> Optional[asyncpg.Record]:
        """Get active session for specific mode (paper or live)."""
        return await self.fetchrow("""
            SELECT * FROM sessions
            WHERE mode = $1 AND status = $2
            ORDER BY started_at DESC
            LIMIT 1
        """, mode, status)

    async def get_hourly_performance(self) -> List[asyncpg.Record]:
        """Get hourly performance comparison between modes."""
        return await self.fetch("""
            SELECT * FROM hourly_performance_comparison
            ORDER BY hour DESC
            LIMIT 24
        """)

    async def get_trade_outcomes_comparison(self) -> List[asyncpg.Record]:
        """Get trade outcome statistics by mode."""
        return await self.fetch("""
            SELECT * FROM trade_outcomes_by_mode
        """)

    # =========================================================================
    # Alert Operations
    # =========================================================================

    async def record_alert(
        self,
        session_id: Optional[str],
        alert_type: str,
        payload: Dict,
        status: str = "sent",
        discord_message_id: Optional[str] = None
    ) -> int:
        """
        Record a sent alert.

        Returns:
            Alert ID
        """
        row = await self.fetchrow("""
            INSERT INTO alerts (session_id, alert_type, payload, status, discord_message_id)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, session_id, alert_type, payload, status, discord_message_id)
        return row["id"]

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check database health.

        Returns:
            Health status dict
        """
        try:
            result = await self.fetchval("SELECT 1")
            session = await self.get_active_session()

            return {
                "status": "healthy",
                "connected": True,
                "session_active": session is not None,
                "last_checkpoint": session["last_checkpoint"].isoformat() if session and session["last_checkpoint"] else None
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }


# Global database instance for convenience
_db: Optional[Database] = None


def get_database() -> Database:
    """Get or create global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


async def init_database() -> Database:
    """Initialize and connect global database instance."""
    db = get_database()
    await db.connect()
    return db


# Alias for backward compatibility
DatabaseConnection = Database
