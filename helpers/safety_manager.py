#!/usr/bin/env python3
"""
Safety Manager - Orchestrates all risk controls.

Central coordination of:
- KillSwitch: Emergency trading halt
- LossTracker: Daily and consecutive loss limits
- PositionLimiter: Max positions and exposure limits

Usage:
    safety = SafetyManager(
        db=database,
        session_id=session_id,
        daily_loss_limit=100.0,
        consecutive_loss_limit=3,
        max_positions=4,
        max_position_size=100.0
    )

    await safety.initialize()

    # Validate before every trade
    can_trade, reason = await safety.validate_trade(
        symbol="BTC",
        size_usd=50.0
    )

    if not can_trade:
        logger.warning(f"Trade blocked: {reason}")
        return

    # After trade closes, record P&L
    await safety.record_trade_result(pnl=-15.0)
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

from db.connection import Database
from helpers.kill_switch import KillSwitch
from helpers.loss_tracker import LossTracker
from helpers.position_limiter import PositionLimiter

logger = logging.getLogger(__name__)


class SafetyManager:
    """
    Orchestrates all safety mechanisms.

    Provides a single interface for:
    - Pre-trade validation (can we trade?)
    - Post-trade recording (update metrics)
    - Emergency controls (halt trading)
    - Status monitoring (dashboard data)
    """

    def __init__(
        self,
        db: Database,
        session_id: Optional[str] = None,
        daily_loss_limit: float = 100.0,
        consecutive_loss_limit: int = 3,
        max_positions: int = 4,
        max_position_size: float = 100.0,
        max_total_exposure: float = 400.0
    ):
        """
        Initialize safety manager.

        Args:
            db: Database connection
            session_id: Trading session ID
            daily_loss_limit: Maximum daily loss (dollars)
            consecutive_loss_limit: Maximum consecutive losses
            max_positions: Maximum concurrent positions
            max_position_size: Maximum single position size (dollars)
            max_total_exposure: Maximum total exposure (dollars)
        """
        self.db = db
        self.session_id = session_id

        # Store config for status reporting
        self._config = {
            "daily_loss_limit": daily_loss_limit,
            "consecutive_loss_limit": consecutive_loss_limit,
            "max_positions": max_positions,
            "max_position_size": max_position_size,
            "max_total_exposure": max_total_exposure,
        }

        # Initialize components
        self.kill_switch = KillSwitch(db=db, session_id=session_id)

        self.loss_tracker = LossTracker(
            db=db,
            session_id=session_id,
            daily_loss_limit=daily_loss_limit,
            consecutive_loss_limit=consecutive_loss_limit
        )

        self.position_limiter = PositionLimiter(
            db=db,
            session_id=session_id,
            max_positions=max_positions,
            max_position_size=max_position_size,
            max_total_exposure=max_total_exposure
        )

        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize all safety components.

        Creates database tables and loads existing state.
        """
        logger.info("[SafetyManager] Initializing...")

        # Initialize components (creates tables if needed)
        await self.kill_switch.ensure_table()
        await self.loss_tracker.initialize()
        await self.position_limiter.initialize()

        self._initialized = True

        # Log initial state
        state = await self.get_status()
        logger.info(
            f"[SafetyManager] Ready | "
            f"Kill switch: {'ACTIVE' if state['kill_switch_active'] else 'off'} | "
            f"Daily P&L: ${state['daily_pnl']:.2f}/{state['daily_loss_limit']:.0f} | "
            f"Consecutive losses: {state['consecutive_losses']}/{state['consecutive_loss_limit']} | "
            f"Positions: {state['open_positions']}/{state['max_positions']}"
        )

    async def validate_trade(
        self,
        symbol: str,
        size_usd: float
    ) -> Tuple[bool, str]:
        """
        Validate if a trade can be executed.

        Checks all safety conditions in order:
        1. Kill switch not active
        2. Loss limits not exceeded
        3. Position limits not exceeded

        Args:
            symbol: Trading symbol (e.g., "BTC")
            size_usd: Position size in dollars

        Returns:
            Tuple of (is_valid, reason)
        """
        if not self._initialized:
            await self.initialize()

        # 1. Check kill switch
        if await self.kill_switch.is_active():
            return False, "Kill switch is active"

        # 2. Check loss limits
        limit_exceeded, limit_reason = await self.loss_tracker.check_all_limits()
        if limit_exceeded:
            # Auto-activate kill switch
            await self.kill_switch.activate(
                reason=limit_reason,
                details={"auto_activated": True, "source": "loss_tracker"}
            )
            return False, limit_reason

        # 3. Check position limits
        can_open, position_reason = await self.position_limiter.can_open_position(
            symbol=symbol,
            size_usd=size_usd
        )
        if not can_open:
            return False, position_reason

        return True, ""

    async def record_trade_result(self, pnl: float) -> None:
        """
        Record trade result and update metrics.

        Updates loss tracker and checks if limits now exceeded.

        Args:
            pnl: Trade P&L in dollars
        """
        if not self._initialized:
            await self.initialize()

        # Record P&L
        await self.loss_tracker.record_pnl(pnl)

        # Update position count
        await self.position_limiter.update_position_count()

        # Check if limits exceeded after trade
        limit_exceeded, reason = await self.loss_tracker.check_all_limits()
        if limit_exceeded:
            await self.kill_switch.activate(
                reason=reason,
                details={
                    "triggered_by_trade_result": True,
                    "pnl": pnl,
                    "source": "post_trade_check"
                }
            )

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive safety status.

        Returns all safety metrics for monitoring/dashboard.

        Returns:
            Dict with all safety metrics
        """
        if not self._initialized:
            await self.initialize()

        kill_state = await self.kill_switch.get_state()
        loss_state = await self.loss_tracker.get_state()
        position_state = await self.position_limiter.get_state()

        return {
            # Kill switch
            "kill_switch_active": kill_state["is_active"],
            "kill_switch_reason": kill_state.get("activation_reason"),
            "kill_switch_activated_at": kill_state.get("activated_at"),

            # Loss tracking
            "daily_pnl": loss_state["daily_pnl"],
            "daily_loss_limit": loss_state["daily_loss_limit"],
            "remaining_loss_allowance": loss_state["remaining_allowance"],
            "consecutive_losses": loss_state["consecutive_losses"],
            "consecutive_loss_limit": loss_state["consecutive_loss_limit"],

            # Position limits
            "open_positions": position_state["open_positions"],
            "max_positions": position_state["max_positions"],
            "available_slots": position_state["available_slots"],
            "total_exposure": position_state["total_exposure"],
            "max_total_exposure": position_state["max_total_exposure"],
            "remaining_capacity": position_state["remaining_capacity"],
            "max_position_size": position_state["max_position_size"],

            # Meta
            "initialized": self._initialized,
            "config": self._config,
        }

    async def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at start of new trading day)."""
        await self.loss_tracker.reset_daily_metrics()
        logger.info("[SafetyManager] Daily metrics reset")

    async def emergency_halt(self, reason: str) -> None:
        """
        Emergency halt all trading.

        Args:
            reason: Reason for halt
        """
        await self.kill_switch.activate(
            reason=f"EMERGENCY HALT: {reason}",
            details={"emergency": True, "source": "manual"}
        )
        logger.critical(f"[SafetyManager] EMERGENCY HALT: {reason}")

    async def resume_trading(self, reason: str = "manual_resume") -> None:
        """
        Resume trading after halt.

        Should only be called after reviewing the halt reason.

        Args:
            reason: Why trading is being resumed
        """
        await self.kill_switch.deactivate(reason=reason)
        logger.warning(f"[SafetyManager] Trading resumed: {reason}")

    async def is_trading_allowed(self) -> bool:
        """
        Quick check if trading is currently allowed.

        Returns:
            True if trading is allowed
        """
        return not await self.kill_switch.is_active()

    async def get_safety_summary_for_discord(self) -> str:
        """
        Get formatted safety summary for Discord alerts.

        Returns:
            Formatted string for Discord message
        """
        status = await self.get_status()

        lines = [
            f"**Safety Status**",
            f"Kill Switch: {'🛑 ACTIVE' if status['kill_switch_active'] else '✅ Off'}",
            f"Daily P&L: ${status['daily_pnl']:.2f} / ${status['daily_loss_limit']:.0f}",
            f"Consecutive Losses: {status['consecutive_losses']} / {status['consecutive_loss_limit']}",
            f"Positions: {status['open_positions']} / {status['max_positions']}",
            f"Exposure: ${status['total_exposure']:.2f} / ${status['max_total_exposure']:.0f}",
        ]

        if status['kill_switch_active']:
            lines.append(f"\n⚠️ Reason: {status['kill_switch_reason']}")

        return "\n".join(lines)
