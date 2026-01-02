#!/usr/bin/env python3
"""
Discord webhook integration for trade alerts.

Sends formatted embeds for trade events, daily summaries,
errors, and recovery notifications.
"""
import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import aiohttp

logger = logging.getLogger(__name__)

# Rate limiting
RATE_LIMIT_DELAY = 0.5  # Minimum seconds between messages


class DiscordWebhook:
    """
    Discord webhook client with rate limiting.

    Usage:
        webhook = DiscordWebhook()
        await webhook.send_trade_open(asset="BTC", side="UP", entry_price=0.52, ...)
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize Discord webhook.

        Args:
            webhook_url: Discord webhook URL.
                         Defaults to DISCORD_WEBHOOK_URL env var.
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self._last_send = 0.0
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def enabled(self) -> bool:
        """Check if webhook is configured."""
        return bool(self.webhook_url)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _send(self, embed: Dict[str, Any]) -> bool:
        """
        Send embed to Discord.

        Args:
            embed: Discord embed object

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        # Rate limiting
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_send
        if elapsed < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - elapsed)

        try:
            session = await self._get_session()
            async with session.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                self._last_send = asyncio.get_event_loop().time()

                if resp.status == 429:  # Rate limited
                    retry_after = float(resp.headers.get("Retry-After", 1))
                    logger.warning(f"Discord rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._send(embed)

                return resp.status in (200, 204)

        except Exception as e:
            logger.error(f"Discord send failed: {e}")
            return False

    async def send_trade_open(
        self,
        asset: str,
        side: str,
        entry_price: float,
        size: float,
        confidence: float,
        session_pnl: float = 0.0
    ) -> bool:
        """
        Send trade open notification.

        Args:
            asset: Trading asset (BTC, ETH, etc.)
            side: Position side (UP or DOWN)
            entry_price: Entry probability
            size: Position size in dollars
            confidence: Model confidence
            session_pnl: Current session PnL

        Returns:
            True if sent successfully
        """
        color = 0x00ff00 if side == "UP" else 0xff0000  # Green for UP, red for DOWN

        embed = {
            "title": f"Trade Opened: {asset} {side}",
            "color": color,
            "fields": [
                {"name": "Entry", "value": f"{entry_price*100:.1f}%", "inline": True},
                {"name": "Size", "value": f"${size:.2f}", "inline": True},
                {"name": "Confidence", "value": f"{confidence*100:.0f}%", "inline": True},
                {"name": "Session PnL", "value": f"${session_pnl:.2f}", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)

    async def send_trade_close(
        self,
        asset: str,
        side: str,
        pnl: float,
        duration: int,
        entry_price: float,
        exit_price: float,
        session_pnl: float,
        session_trades: int,
        session_win_rate: float
    ) -> bool:
        """
        Send trade close notification.

        Args:
            asset: Trading asset
            side: Position side
            pnl: Trade PnL in dollars
            duration: Trade duration in seconds
            entry_price: Entry probability
            exit_price: Exit probability
            session_pnl: Total session PnL
            session_trades: Total session trades
            session_win_rate: Session win rate

        Returns:
            True if sent successfully
        """
        color = 0x00ff00 if pnl >= 0 else 0xff0000
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        embed = {
            "title": f"Trade Closed: {asset} {side} | {pnl_str}",
            "color": color,
            "fields": [
                {"name": "PnL", "value": pnl_str, "inline": True},
                {"name": "Duration", "value": f"{duration}s", "inline": True},
                {"name": "Entry → Exit", "value": f"{entry_price*100:.1f}% → {exit_price*100:.1f}%", "inline": True},
                {"name": "Session Total", "value": f"${session_pnl:.2f}", "inline": True},
                {"name": "Trades", "value": str(session_trades), "inline": True},
                {"name": "Win Rate", "value": f"{session_win_rate*100:.0f}%", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)

    async def send_daily_summary(
        self,
        pnl: float,
        trades: int,
        win_rate: float,
        best_trade: float,
        worst_trade: float,
        exposure_pct: float = 0.0
    ) -> bool:
        """
        Send daily summary.

        Args:
            pnl: Day's PnL
            trades: Number of trades
            win_rate: Win rate (0-1)
            best_trade: Best trade PnL
            worst_trade: Worst trade PnL
            exposure_pct: Average exposure percentage

        Returns:
            True if sent successfully
        """
        color = 0x00ff00 if pnl >= 0 else 0xff0000
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

        embed = {
            "title": f"Daily Summary | {pnl_str}",
            "color": color,
            "fields": [
                {"name": "PnL", "value": pnl_str, "inline": True},
                {"name": "Trades", "value": str(trades), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate*100:.0f}%", "inline": True},
                {"name": "Best Trade", "value": f"${best_trade:.2f}", "inline": True},
                {"name": "Worst Trade", "value": f"${worst_trade:.2f}", "inline": True},
                {"name": "Exposure", "value": f"{exposure_pct:.0f}%", "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)

    async def send_error(self, message: str, details: Optional[str] = None) -> bool:
        """
        Send error notification.

        Args:
            message: Error message
            details: Optional error details

        Returns:
            True if sent successfully
        """
        embed = {
            "title": "Trading Error",
            "color": 0xff0000,
            "description": message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        if details:
            embed["fields"] = [{"name": "Details", "value": details[:1000]}]

        return await self._send(embed)

    async def send_recovery(
        self,
        session_id: str,
        recovered_pnl: float,
        recovered_trades: int,
        open_positions: int = 0
    ) -> bool:
        """
        Send session recovery notification.

        Args:
            session_id: Recovered session ID
            recovered_pnl: PnL at recovery
            recovered_trades: Trade count at recovery
            open_positions: Number of open positions recovered

        Returns:
            True if sent successfully
        """
        embed = {
            "title": "Session Recovered",
            "color": 0xffff00,  # Yellow
            "description": f"Recovered session `{session_id[:8]}` after restart",
            "fields": [
                {"name": "Recovered PnL", "value": f"${recovered_pnl:.2f}", "inline": True},
                {"name": "Trades", "value": str(recovered_trades), "inline": True},
                {"name": "Open Positions", "value": str(open_positions), "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)

    async def send_startup(
        self,
        mode: str,
        trade_size: float,
        model_version: str
    ) -> bool:
        """
        Send startup notification.

        Args:
            mode: Trading mode (paper/live)
            trade_size: Trade size in dollars
            model_version: Model version string

        Returns:
            True if sent successfully
        """
        color = 0x3498db if mode == "paper" else 0x9b59b6

        embed = {
            "title": f"Trading Started ({mode.upper()})",
            "color": color,
            "fields": [
                {"name": "Mode", "value": mode.upper(), "inline": True},
                {"name": "Trade Size", "value": f"${trade_size:.2f}", "inline": True},
                {"name": "Model", "value": model_version, "inline": True},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)

    async def send_milestone(self, pnl: float, milestone: float) -> bool:
        """
        Send PnL milestone notification.

        Args:
            pnl: Current PnL
            milestone: Milestone crossed ($100, $500, $1000, etc.)

        Returns:
            True if sent successfully
        """
        embed = {
            "title": f"Milestone Reached: ${milestone:.0f}!",
            "color": 0xf1c40f,  # Gold
            "description": f"Current PnL: ${pnl:.2f}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)


# Global webhook instance
_webhook: Optional[DiscordWebhook] = None


def get_webhook() -> DiscordWebhook:
    """Get or create global webhook instance."""
    global _webhook
    if _webhook is None:
        _webhook = DiscordWebhook()
    return _webhook
