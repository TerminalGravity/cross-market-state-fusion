#!/usr/bin/env python3
"""
Error Monitoring System for Polymarket Trading Bot.

Monitors log output for critical errors and sends Discord alerts:
- Connection failures (Cloudflare 403s, WebSocket disconnects)
- Balance/allowance errors
- Order rejections
- Health degradation

Rate-limited to prevent Discord spam.
"""
import os
import re
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for alert types."""
    pattern: str
    title: str
    color: int
    emoji: str
    cooldown_seconds: int = 300  # 5 minutes between alerts of same type
    severity: str = "warning"  # info, warning, error, critical


# Alert configurations for different error patterns
ALERT_CONFIGS: Dict[str, AlertConfig] = {
    # Connection issues
    "cloudflare_403": AlertConfig(
        pattern=r"(403|cloudflare|cf-ray|blocked|access denied)",
        title="Cloudflare Block Detected",
        color=0xff6b6b,  # Red
        emoji="🚫",
        cooldown_seconds=60,  # Alert every 1 minute if persistent
        severity="critical"
    ),
    "connection_failed": AlertConfig(
        pattern=r"(connection.*failed|connect.*error|connection.*refused|connection.*reset|timed?\s*out)",
        title="Connection Failed",
        color=0xffa500,  # Orange
        emoji="🔌",
        cooldown_seconds=120,
        severity="error"
    ),
    "websocket_disconnect": AlertConfig(
        pattern=r"(websocket.*disconnect|ws.*closed|websocket.*error|wss.*failed)",
        title="WebSocket Disconnected",
        color=0xffa500,  # Orange
        emoji="📡",
        cooldown_seconds=120,
        severity="warning"
    ),

    # Trading issues
    "balance_error": AlertConfig(
        pattern=r"(not enough balance|insufficient.*balance|allowance|balance too low)",
        title="Balance/Allowance Error",
        color=0xffcc00,  # Yellow
        emoji="💰",
        cooldown_seconds=300,  # 5 minutes - this is expected sometimes
        severity="warning"
    ),
    "order_rejected": AlertConfig(
        pattern=r"(order.*rejected|order.*failed|trade.*rejected|execution.*failed)",
        title="Order Rejected",
        color=0xffa500,  # Orange
        emoji="❌",
        cooldown_seconds=60,
        severity="error"
    ),

    # System health
    "system_unhealthy": AlertConfig(
        pattern=r"(system.*unhealthy|health.*check.*failed|stale.*data|markets.*stale)",
        title="System Health Degraded",
        color=0xff0000,  # Bright red
        emoji="🚨",
        cooldown_seconds=300,
        severity="critical"
    ),
    "emergency_close": AlertConfig(
        pattern=r"(emergency.*close|force.*close|safety.*triggered)",
        title="Emergency Action Triggered",
        color=0xff0000,  # Bright red
        emoji="🆘",
        cooldown_seconds=60,
        severity="critical"
    ),
}


class ErrorMonitor:
    """
    Monitors logs for errors and sends Discord alerts.

    Features:
    - Pattern-based error detection
    - Rate limiting per error type
    - Connection health tracking
    - Error aggregation (batch similar errors)
    """

    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initialize error monitor.

        Args:
            webhook_url: Discord webhook URL (defaults to DISCORD_WEBHOOK_URL env var)
        """
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self._last_alert_times: Dict[str, datetime] = {}
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False

        # Connection health tracking
        self._connection_status = {
            "polymarket_clob": True,
            "binance_wss": True,
            "okx_wss": True,
        }
        self._last_health_check = datetime.now(timezone.utc)

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

    def _should_alert(self, alert_type: str, config: AlertConfig) -> bool:
        """Check if we should send an alert (rate limiting)."""
        now = datetime.now(timezone.utc)
        last_alert = self._last_alert_times.get(alert_type)

        if last_alert is None:
            return True

        elapsed = (now - last_alert).total_seconds()
        return elapsed >= config.cooldown_seconds

    async def _send_alert(
        self,
        config: AlertConfig,
        message: str,
        details: Optional[str] = None
    ) -> bool:
        """Send a Discord alert."""
        if not self.enabled:
            return False

        try:
            session = await self._get_session()

            embed = {
                "title": f"{config.emoji} {config.title}",
                "color": config.color,
                "description": message[:2000],  # Discord limit
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {
                    "text": f"Fly.io Toronto (yyz) • {config.severity.upper()}"
                }
            }

            if details:
                embed["fields"] = [{
                    "name": "Details",
                    "value": details[:1000],
                    "inline": False
                }]

            async with session.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 429:  # Rate limited by Discord
                    retry_after = float(resp.headers.get("Retry-After", 5))
                    logger.warning(f"Discord rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._send_alert(config, message, details)

                return resp.status in (200, 204)

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    async def check_message(self, message: str, level: str = "ERROR") -> bool:
        """
        Check a log message for error patterns and send alerts.

        Args:
            message: Log message to check
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

        Returns:
            True if alert was sent
        """
        message_lower = message.lower()

        for alert_type, config in ALERT_CONFIGS.items():
            # Skip if log level doesn't match severity
            if config.severity == "critical" and level not in ("ERROR", "CRITICAL"):
                continue
            if config.severity == "error" and level not in ("WARNING", "ERROR", "CRITICAL"):
                continue

            # Check pattern match
            if re.search(config.pattern, message_lower, re.IGNORECASE):
                self._error_counts[alert_type] += 1

                if self._should_alert(alert_type, config):
                    self._last_alert_times[alert_type] = datetime.now(timezone.utc)

                    # Include error count in message
                    count = self._error_counts[alert_type]
                    count_str = f" (occurrence #{count})" if count > 1 else ""

                    await self._send_alert(
                        config,
                        f"{message[:500]}{count_str}",
                        f"Error type: {alert_type}\nTotal occurrences: {count}"
                    )
                    return True

        return False

    async def alert_connection_status(
        self,
        service: str,
        connected: bool,
        error: Optional[str] = None
    ) -> None:
        """
        Alert on connection status changes.

        Args:
            service: Service name (polymarket_clob, binance_wss, etc.)
            connected: Whether service is now connected
            error: Optional error message
        """
        was_connected = self._connection_status.get(service, True)
        self._connection_status[service] = connected

        # Alert on disconnection
        if was_connected and not connected:
            await self._send_alert(
                AlertConfig(
                    pattern="",
                    title=f"{service} Disconnected",
                    color=0xff6b6b,
                    emoji="🔴",
                    cooldown_seconds=60,
                    severity="error"
                ),
                f"Lost connection to {service}",
                error
            )

        # Alert on reconnection
        elif not was_connected and connected:
            await self._send_alert(
                AlertConfig(
                    pattern="",
                    title=f"{service} Reconnected",
                    color=0x2ecc71,
                    emoji="🟢",
                    cooldown_seconds=60,
                    severity="info"
                ),
                f"Successfully reconnected to {service}",
                None
            )

    async def send_health_summary(self) -> None:
        """Send periodic health summary."""
        if not self.enabled:
            return

        now = datetime.now(timezone.utc)

        # Connection status
        status_lines = []
        for service, connected in self._connection_status.items():
            emoji = "🟢" if connected else "🔴"
            status_lines.append(f"{emoji} {service}: {'Connected' if connected else 'DISCONNECTED'}")

        # Error counts (last hour)
        error_lines = []
        for alert_type, count in sorted(self._error_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                error_lines.append(f"• {alert_type}: {count}")

        embed = {
            "title": "📊 System Health Report",
            "color": 0x3498db,  # Blue
            "timestamp": now.isoformat(),
            "fields": [
                {
                    "name": "Connection Status",
                    "value": "\n".join(status_lines) or "No services tracked",
                    "inline": False
                },
                {
                    "name": "Error Counts (Session)",
                    "value": "\n".join(error_lines[:10]) or "No errors",
                    "inline": False
                }
            ],
            "footer": {
                "text": "Fly.io Toronto (yyz) • Health Check"
            }
        }

        try:
            session = await self._get_session()
            async with session.post(
                self.webhook_url,
                json={"embeds": [embed]},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status not in (200, 204):
                    logger.warning(f"Health summary failed: status {resp.status}")
        except Exception as e:
            logger.error(f"Failed to send health summary: {e}")

    def get_error_stats(self) -> Dict[str, int]:
        """Get current error counts by type."""
        return dict(self._error_counts)

    def reset_counts(self) -> None:
        """Reset error counts (call at session start or daily)."""
        self._error_counts.clear()


class LogHandler(logging.Handler):
    """
    Logging handler that forwards errors to ErrorMonitor.

    Install this handler to automatically capture and alert on log errors.
    """

    def __init__(self, monitor: ErrorMonitor):
        super().__init__(level=logging.WARNING)
        self.monitor = monitor
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def emit(self, record: logging.LogRecord):
        """Process a log record."""
        try:
            # Get or create event loop
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, skip async processing
                return

            message = self.format(record)
            level = record.levelname

            # Schedule alert check (non-blocking)
            asyncio.create_task(self.monitor.check_message(message, level))

        except Exception:
            self.handleError(record)


# Global monitor instance
_monitor: Optional[ErrorMonitor] = None


def get_error_monitor() -> ErrorMonitor:
    """Get or create the global error monitor."""
    global _monitor
    if _monitor is None:
        _monitor = ErrorMonitor()
    return _monitor


def install_log_handler() -> None:
    """
    Install the error monitoring log handler.

    Call this early in application startup to enable automatic error alerting.
    """
    monitor = get_error_monitor()
    handler = LogHandler(monitor)
    handler.setFormatter(logging.Formatter('%(message)s'))

    # Add to root logger
    logging.getLogger().addHandler(handler)
    logger.info("[ERROR-MONITOR] Log handler installed")


async def test_alerts():
    """Test alert functionality."""
    monitor = get_error_monitor()

    print(f"Webhook configured: {monitor.enabled}")

    if monitor.enabled:
        # Test connection alert
        await monitor.alert_connection_status("test_service", False, "Test disconnection")
        await asyncio.sleep(1)
        await monitor.alert_connection_status("test_service", True)

        # Test error pattern
        await monitor.check_message("not enough balance / allowance for order", "ERROR")

        # Test health summary
        await monitor.send_health_summary()

        print("Test alerts sent!")
    else:
        print("No webhook configured - set DISCORD_WEBHOOK_URL")

    await monitor.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_alerts())
