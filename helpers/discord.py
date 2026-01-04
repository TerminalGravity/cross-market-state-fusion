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
        # Dynamic color based on side
        color = 0x2ecc71 if side == "UP" else 0xe74c3c  # Brighter green/red

        # Emoji indicators
        side_emoji = "📈" if side == "UP" else "📉"
        conf_emoji = "🔥" if confidence >= 0.15 else "⚡" if confidence >= 0.10 else "🎯"

        # Confidence bar visualization
        conf_bars = "█" * int(confidence * 50) + "░" * (10 - int(confidence * 50))

        # Session PnL indicator
        pnl_emoji = "💰" if session_pnl > 0 else "💸" if session_pnl < 0 else "💵"

        embed = {
            "title": f"{side_emoji} {asset} {side} Opened",
            "color": color,
            "description": f"**{conf_emoji} Confidence: {confidence*100:.1f}%**\n`{conf_bars}`",
            "fields": [
                {"name": "📍 Entry Price", "value": f"**{entry_price*100:.1f}%**", "inline": True},
                {"name": "💵 Position Size", "value": f"**${size:.2f}**", "inline": True},
                {"name": f"{pnl_emoji} Session P&L", "value": f"**${session_pnl:+.2f}**", "inline": True},
            ],
            "footer": {"text": f"🤖 RL Model • Live Trading"},
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
        # Dynamic color based on PnL magnitude
        if pnl >= 2.0:
            color = 0x27ae60  # Bright green for big wins
        elif pnl >= 0:
            color = 0x2ecc71  # Green for small wins
        elif pnl >= -2.0:
            color = 0xe67e22  # Orange for small losses
        else:
            color = 0xe74c3c  # Red for big losses

        # Win/loss emoji
        result_emoji = "✅" if pnl >= 0 else "❌"

        # Calculate ROI percentage
        size = abs(pnl / ((exit_price - entry_price) if exit_price != entry_price else 0.01))
        roi_pct = (pnl / size * 100) if size > 0 else 0

        # Format PnL with sign
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"${pnl:.2f}"

        # Format duration (convert seconds to readable format)
        if duration < 60:
            duration_str = f"{duration}s"
        elif duration < 3600:
            duration_str = f"{duration // 60}m {duration % 60}s"
        else:
            duration_str = f"{duration // 3600}h {(duration % 3600) // 60}m"

        # Win rate bar visualization
        win_bars = "█" * int(session_win_rate * 10) + "░" * (10 - int(session_win_rate * 10))

        # Session PnL emoji
        session_emoji = "💰" if session_pnl > 0 else "💸" if session_pnl < 0 else "💵"

        embed = {
            "title": f"{result_emoji} {asset} {side} Closed | {pnl_str}",
            "color": color,
            "description": f"**ROI: {roi_pct:+.1f}%** • Duration: {duration_str}",
            "fields": [
                {"name": "💵 Trade P&L", "value": f"**{pnl_str}**", "inline": True},
                {"name": "📊 Entry → Exit", "value": f"{entry_price*100:.1f}% → {exit_price*100:.1f}%", "inline": True},
                {"name": f"{session_emoji} Session P&L", "value": f"**${session_pnl:+.2f}**", "inline": True},
                {"name": "📈 Trades", "value": f"**{session_trades}**", "inline": True},
                {"name": "🎯 Win Rate", "value": f"**{session_win_rate*100:.0f}%**\n`{win_bars}`", "inline": True},
                {"name": "⏱️ Hold Time", "value": f"**{duration_str}**", "inline": True},
            ],
            "footer": {"text": f"🤖 RL Model • {'🟢 Profitable' if pnl >= 0 else '🔴 Loss'}"},
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
        # Dynamic color based on daily performance
        if pnl >= 50:
            color = 0xf1c40f  # Gold for great days
        elif pnl >= 10:
            color = 0x2ecc71  # Green for good days
        elif pnl >= 0:
            color = 0x3498db  # Blue for small wins
        elif pnl >= -10:
            color = 0xe67e22  # Orange for small losses
        else:
            color = 0xe74c3c  # Red for bad days

        # Format PnL with sign
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"${pnl:.2f}"

        # Daily result emoji
        if pnl >= 50:
            result_emoji = "🏆"
        elif pnl >= 10:
            result_emoji = "💰"
        elif pnl >= 0:
            result_emoji = "✅"
        else:
            result_emoji = "📉"

        # Win rate bar visualization
        win_bars = "█" * int(win_rate * 10) + "░" * (10 - int(win_rate * 10))

        # Win/loss count
        wins = int(trades * win_rate)
        losses = trades - wins

        embed = {
            "title": f"{result_emoji} Daily Summary | {pnl_str}",
            "color": color,
            "description": f"**{wins}W - {losses}L** ({win_rate*100:.0f}% win rate)\n`{win_bars}`",
            "fields": [
                {"name": "💵 Total P&L", "value": f"**{pnl_str}**", "inline": True},
                {"name": "📊 Total Trades", "value": f"**{trades}**", "inline": True},
                {"name": "📈 Avg Exposure", "value": f"**{exposure_pct:.0f}%**", "inline": True},
                {"name": "🏆 Best Trade", "value": f"**+${best_trade:.2f}**", "inline": True},
                {"name": "💸 Worst Trade", "value": f"**${worst_trade:.2f}**", "inline": True},
                {"name": "📊 P&L Range", "value": f"**${abs(best_trade - worst_trade):.2f}**", "inline": True},
            ],
            "footer": {"text": f"🤖 RL Model • End of Day Report"},
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
        # Color based on recovered PnL
        if recovered_pnl > 0:
            color = 0x2ecc71  # Green for profitable recovery
        elif recovered_pnl < 0:
            color = 0xe67e22  # Orange for loss recovery
        else:
            color = 0x3498db  # Blue for neutral

        # PnL emoji
        pnl_emoji = "💰" if recovered_pnl > 0 else "💸" if recovered_pnl < 0 else "💵"

        # Format PnL with sign
        pnl_str = f"+${recovered_pnl:.2f}" if recovered_pnl >= 0 else f"${recovered_pnl:.2f}"

        embed = {
            "title": "🔄 Session Recovered",
            "color": color,
            "description": f"Successfully recovered session after restart\n**Session ID:** `{session_id[:16]}...`",
            "fields": [
                {"name": f"{pnl_emoji} Recovered P&L", "value": f"**{pnl_str}**", "inline": True},
                {"name": "📊 Trades", "value": f"**{recovered_trades}**", "inline": True},
                {"name": "📈 Open Positions", "value": f"**{open_positions}**", "inline": True},
            ],
            "footer": {"text": "🤖 RL Model • Resuming trading operations"},
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
        # Color and emoji based on mode
        if mode == "live":
            color = 0x9b59b6  # Purple for live
            mode_emoji = "🚀"
            mode_text = "LIVE TRADING"
        else:
            color = 0x3498db  # Blue for paper
            mode_emoji = "📝"
            mode_text = "PAPER TRADING"

        embed = {
            "title": f"{mode_emoji} Trading Bot Started",
            "color": color,
            "description": f"**Mode: {mode_text}**\nBot is now actively monitoring markets",
            "fields": [
                {"name": "💵 Trade Size", "value": f"**${trade_size:.2f}**", "inline": True},
                {"name": "🤖 Model", "value": f"**{model_version}**", "inline": True},
                {"name": "📊 Markets", "value": "**BTC, ETH, SOL, XRP**", "inline": True},
            ],
            "footer": {"text": f"🤖 RL Model • {mode_text}"},
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
        # Celebratory emoji based on milestone size
        if milestone >= 1000:
            emoji = "🎊💰🎊"
        elif milestone >= 500:
            emoji = "🏆💎"
        elif milestone >= 100:
            emoji = "🎉💰"
        else:
            emoji = "✨"

        # Progress bar to next milestone
        next_milestone = milestone * 2
        progress_pct = min((pnl - milestone) / (next_milestone - milestone), 1.0)
        progress_bars = "█" * int(progress_pct * 10) + "░" * (10 - int(progress_pct * 10))

        embed = {
            "title": f"{emoji} Milestone Reached: ${milestone:.0f}!",
            "color": 0xf1c40f,  # Gold
            "description": f"**Current P&L: ${pnl:.2f}**\n\nProgress to ${next_milestone:.0f}:\n`{progress_bars}` {progress_pct*100:.0f}%",
            "fields": [
                {"name": "🎯 Milestone", "value": f"**${milestone:.0f}**", "inline": True},
                {"name": "💰 Current P&L", "value": f"**${pnl:.2f}**", "inline": True},
                {"name": "📈 Next Goal", "value": f"**${next_milestone:.0f}**", "inline": True},
            ],
            "footer": {"text": "🤖 RL Model • Keep up the great work!"},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return await self._send(embed)

    async def send_profit_transfer(
        self,
        amount: float,
        tx_hash: str,
        gas_used: int
    ) -> bool:
        """
        Send profit transfer success notification.

        Args:
            amount: Transfer amount in dollars
            tx_hash: Polygon transaction hash
            gas_used: Gas units consumed

        Returns:
            True if sent successfully
        """
        # Polygon block explorer link
        explorer_url = f"https://polygonscan.com/tx/{tx_hash}"

        # Calculate approximate gas cost (assuming ~30 gwei and 0.50 MATIC/USD)
        gas_cost_matic = (gas_used * 30e9) / 1e18  # Convert to MATIC
        gas_cost_usd = gas_cost_matic * 0.50  # Rough MATIC price

        embed = {
            "title": "✅ Profit Transfer Complete",
            "color": 0x2ecc71,  # Green
            "description": f"Successfully transferred ${amount:.2f} USDC to cold wallet",
            "fields": [
                {"name": "Amount", "value": f"${amount:.2f} USDC", "inline": True},
                {"name": "Gas Used", "value": f"{gas_used:,} units", "inline": True},
                {"name": "Gas Cost", "value": f"~${gas_cost_usd:.3f}", "inline": True},
                {
                    "name": "Transaction",
                    "value": f"[View on PolygonScan]({explorer_url})",
                    "inline": False
                },
            ],
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
