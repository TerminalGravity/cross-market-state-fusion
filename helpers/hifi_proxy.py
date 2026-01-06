"""
HiFi Proxy Manager - Zero Cloudflare blocks with residential SOCKS5 proxies.

This module provides:
1. Centralized proxy configuration for all components
2. Connection health monitoring
3. Auto-retry with exponential backoff
4. Pre-configured httpx clients with SOCKS5 support
5. aiohttp connectors for WebSocket connections

Designed for Decodo (Smartproxy) residential SOCKS5 proxies via Canada.
"""
import os
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# ============================================================================
# PROXY CONFIGURATION
# ============================================================================

# Decodo (Smartproxy) SOCKS5 residential proxy
# Format: socks5://user:pass@gate.decodo.com:PORT
# For Canada geo-targeting, use: gate.decodo.com with country=ca in username
# Example: socks5://user-country-ca:pass@gate.decodo.com:7000
PROXY_SOCKS5_URL = os.environ.get("RESIDENTIAL_SOCKS5_URL", "")
PROXY_HTTP_URL = os.environ.get("RESIDENTIAL_PROXY_URL", "")

# Connection timeouts (tuned for reliability)
CONNECT_TIMEOUT = 15.0  # Time to establish connection through proxy
REQUEST_TIMEOUT = 30.0  # Total request timeout
WEBSOCKET_TIMEOUT = 60.0  # WebSocket operation timeout

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # Exponential backoff: 2^n seconds
RETRY_JITTER = 0.5  # Random jitter +-50%

# Health monitoring
HEALTH_CHECK_INTERVAL = 60.0  # Check proxy health every 60s
MAX_CONSECUTIVE_FAILURES = 5  # Alert after this many failures


@dataclass
class ProxyHealth:
    """Track proxy connection health."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    last_error: Optional[str] = None
    cloudflare_blocks: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def is_healthy(self) -> bool:
        return self.consecutive_failures < MAX_CONSECUTIVE_FAILURES


class HiFiProxyManager:
    """
    Centralized proxy manager for HiFi trading system.

    Provides:
    - Pre-configured httpx client with SOCKS5 support
    - aiohttp connector for WebSocket connections
    - Health monitoring and auto-recovery
    - Retry logic with exponential backoff
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern for consistent proxy state."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.health = ProxyHealth()
        self._httpx_client = None
        self._aiohttp_connector = None

        # Log proxy configuration
        if PROXY_SOCKS5_URL:
            # Mask password in logs
            masked = self._mask_url(PROXY_SOCKS5_URL)
            logger.info(f"[HIFI-PROXY] SOCKS5 configured: {masked}")
        else:
            logger.warning("[HIFI-PROXY] No SOCKS5 proxy configured - using direct connections")

        if PROXY_HTTP_URL:
            masked = self._mask_url(PROXY_HTTP_URL)
            logger.info(f"[HIFI-PROXY] HTTP proxy configured: {masked}")

    def _mask_url(self, url: str) -> str:
        """Mask password in proxy URL for logging."""
        if "@" in url:
            prefix, host = url.rsplit("@", 1)
            if ":" in prefix:
                parts = prefix.rsplit(":", 1)
                if len(parts) == 2:
                    return f"{parts[0]}:****@{host}"
        return url

    def get_httpx_client(self, force_new: bool = False):
        """
        Get a configured httpx client with SOCKS5 proxy support.

        This client is used for:
        - py-clob-client API calls
        - REST API polling
        - Balance checks

        Returns:
            httpx.Client configured with SOCKS5 proxy if available
        """
        import httpx

        if self._httpx_client is not None and not force_new:
            return self._httpx_client

        # Close existing client
        if self._httpx_client is not None:
            try:
                self._httpx_client.close()
            except:
                pass

        # Browser-like headers to avoid bot detection
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        }

        timeout = httpx.Timeout(
            REQUEST_TIMEOUT,
            connect=CONNECT_TIMEOUT
        )

        if PROXY_SOCKS5_URL:
            self._httpx_client = httpx.Client(
                http2=True,
                proxy=PROXY_SOCKS5_URL,
                timeout=timeout,
                headers=headers,
                follow_redirects=True
            )
            logger.info("[HIFI-PROXY] Created httpx client with SOCKS5 proxy")
        else:
            self._httpx_client = httpx.Client(
                http2=True,
                timeout=timeout,
                headers=headers,
                follow_redirects=True
            )
            logger.info("[HIFI-PROXY] Created httpx client (direct, no proxy)")

        return self._httpx_client

    def get_async_httpx_client(self, force_new: bool = False):
        """
        Get an async httpx client with SOCKS5 proxy support.

        Returns:
            httpx.AsyncClient configured with SOCKS5 proxy if available
        """
        import httpx

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }

        timeout = httpx.Timeout(
            REQUEST_TIMEOUT,
            connect=CONNECT_TIMEOUT
        )

        if PROXY_SOCKS5_URL:
            return httpx.AsyncClient(
                http2=True,
                proxy=PROXY_SOCKS5_URL,
                timeout=timeout,
                headers=headers,
                follow_redirects=True
            )
        else:
            return httpx.AsyncClient(
                http2=True,
                timeout=timeout,
                headers=headers,
                follow_redirects=True
            )

    def get_aiohttp_connector(self):
        """
        Get an aiohttp connector for WebSocket connections through SOCKS5.

        This is used for:
        - Polymarket orderbook WebSocket

        Returns:
            ProxyConnector if aiohttp-socks available and proxy configured,
            None otherwise (will use direct connection)
        """
        if not PROXY_SOCKS5_URL:
            return None

        try:
            from aiohttp_socks import ProxyConnector
            return ProxyConnector.from_url(PROXY_SOCKS5_URL)
        except ImportError:
            logger.warning("[HIFI-PROXY] aiohttp-socks not installed, WebSocket will use direct connection")
            return None
        except Exception as e:
            logger.error(f"[HIFI-PROXY] Failed to create aiohttp connector: {e}")
            return None

    def get_websocket_proxy(self):
        """
        Get proxy configuration for websockets library (python-socks).

        This is used for:
        - Binance WebSocket streams

        Returns:
            Proxy object if python-socks available and proxy configured,
            None otherwise
        """
        if not PROXY_SOCKS5_URL:
            return None

        try:
            from python_socks.async_.asyncio.v2 import Proxy
            return Proxy.from_url(PROXY_SOCKS5_URL)
        except ImportError:
            logger.warning("[HIFI-PROXY] python-socks not installed, WebSocket will use direct connection")
            return None
        except Exception as e:
            logger.error(f"[HIFI-PROXY] Failed to create websocket proxy: {e}")
            return None

    def record_success(self):
        """Record a successful proxy request."""
        self.health.total_requests += 1
        self.health.successful_requests += 1
        self.health.consecutive_failures = 0
        self.health.last_success = datetime.now(timezone.utc)

    def record_failure(self, error: str, is_cloudflare: bool = False):
        """Record a failed proxy request."""
        self.health.total_requests += 1
        self.health.failed_requests += 1
        self.health.consecutive_failures += 1
        self.health.last_failure = datetime.now(timezone.utc)
        self.health.last_error = error

        if is_cloudflare:
            self.health.cloudflare_blocks += 1

        # Alert on consecutive failures
        if self.health.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            logger.error(
                f"[HIFI-PROXY] ALERT: {self.health.consecutive_failures} consecutive failures! "
                f"Last error: {error}"
            )

    def get_health_status(self) -> Dict[str, Any]:
        """Get current proxy health status."""
        return {
            "is_healthy": self.health.is_healthy,
            "success_rate": f"{self.health.success_rate*100:.1f}%",
            "total_requests": self.health.total_requests,
            "consecutive_failures": self.health.consecutive_failures,
            "cloudflare_blocks": self.health.cloudflare_blocks,
            "last_success": self.health.last_success.isoformat() if self.health.last_success else None,
            "last_error": self.health.last_error,
            "proxy_configured": bool(PROXY_SOCKS5_URL)
        }

    def patch_clob_client(self):
        """
        Monkey-patch py-clob-client to use our SOCKS5 proxy.

        Call this AFTER importing py_clob_client.
        """
        if not PROXY_SOCKS5_URL:
            logger.warning("[HIFI-PROXY] No proxy configured, py-clob-client will use direct connection")
            return False

        try:
            from py_clob_client.http_helpers import helpers as clob_http_helpers

            # Replace the module-level _http_client with our proxied version
            clob_http_helpers._http_client = self.get_httpx_client(force_new=True)
            logger.info("[HIFI-PROXY] Successfully patched py-clob-client with SOCKS5 proxy")
            return True
        except ImportError:
            logger.warning("[HIFI-PROXY] py-clob-client not installed")
            return False
        except Exception as e:
            logger.error(f"[HIFI-PROXY] Failed to patch py-clob-client: {e}")
            return False

    def close(self):
        """Clean up resources."""
        if self._httpx_client is not None:
            try:
                self._httpx_client.close()
            except:
                pass
            self._httpx_client = None


def with_retry(max_retries: int = MAX_RETRIES, backoff_base: float = RETRY_BACKOFF_BASE):
    """
    Decorator for retrying operations with exponential backoff.

    Usage:
        @with_retry(max_retries=3)
        async def fetch_data():
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import random
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = backoff_base ** attempt
                        jitter = delay * RETRY_JITTER * (2 * random.random() - 1)
                        wait_time = delay + jitter

                        logger.warning(
                            f"[HIFI-PROXY] Retry {attempt + 1}/{max_retries} in {wait_time:.1f}s: {e}"
                        )
                        import asyncio
                        await asyncio.sleep(wait_time)

            raise last_error
        return wrapper
    return decorator


def get_proxy_manager() -> HiFiProxyManager:
    """Get the singleton proxy manager instance."""
    return HiFiProxyManager()


# Initialize on import for convenience
_proxy_manager = None

def init_proxy():
    """Initialize the proxy manager singleton."""
    global _proxy_manager
    if _proxy_manager is None:
        _proxy_manager = HiFiProxyManager()
    return _proxy_manager


# ============================================================================
# DECODO/SMARTPROXY CONFIGURATION HELPER
# ============================================================================

def get_decodo_socks5_url(
    username: str,
    password: str,
    country: str = "ca",  # Canada by default
    session: Optional[str] = None,
    port: int = 7000
) -> str:
    """
    Generate a Decodo (Smartproxy) SOCKS5 residential proxy URL.

    Args:
        username: Decodo username
        password: Decodo password
        country: Country code (ca=Canada, us=USA, etc.)
        session: Optional session ID for sticky IP (same IP for duration)
        port: SOCKS5 port (default 7000 for residential)

    Returns:
        SOCKS5 proxy URL in format socks5://user:pass@host:port

    Example:
        url = get_decodo_socks5_url("myuser", "mypass", country="ca")
        # Returns: socks5://myuser-country-ca:mypass@gate.decodo.com:7000
    """
    # Build username with geo-targeting
    user_parts = [username]

    if country:
        user_parts.append(f"country-{country.lower()}")

    if session:
        user_parts.append(f"session-{session}")

    final_username = "-".join(user_parts)

    return f"socks5://{final_username}:{password}@gate.decodo.com:{port}"


if __name__ == "__main__":
    # Test proxy configuration
    import asyncio

    logging.basicConfig(level=logging.INFO)

    pm = get_proxy_manager()
    print(f"Proxy configured: {bool(PROXY_SOCKS5_URL)}")
    print(f"Health: {pm.get_health_status()}")

    # Test httpx client
    client = pm.get_httpx_client()
    print(f"httpx client ready: {client is not None}")

    # Example Decodo URL generation
    print("\nExample Decodo URL for Canada:")
    print(get_decodo_socks5_url("testuser", "testpass", country="ca"))
