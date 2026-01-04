"""
Polymarket CLOB WebSocket helpers for orderbook streaming.

Uses:
- websockets library with python-socks for SOCKS5 proxy support (more reliable than aiohttp)
- Falls back to REST polling if WSS fails after MAX_WSS_FAILURES attempts
"""
import asyncio
import json
import logging
import os
import aiohttp  # For REST fallback
import websockets
from python_socks.async_.asyncio.v2 import Proxy
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any

# Residential proxy for bypassing Cloudflare datacenter IP blocks
# SOCKS5 proxy is more reliable for WebSocket than HTTP proxy
# SmartProxy: HTTP=10001, SOCKS5=10000
PROXY_HTTP_URL = os.environ.get("RESIDENTIAL_PROXY_URL", "")  # HTTP proxy (fallback)
PROXY_SOCKS5_URL = os.environ.get("RESIDENTIAL_SOCKS5_URL", "")  # SOCKS5 proxy (preferred for WSS)
_try_direct_raw = os.environ.get("ORDERBOOK_TRY_DIRECT", "true")
TRY_DIRECT_FIRST = _try_direct_raw.lower() == "true"

logger = logging.getLogger(__name__)

CLOB_WSS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
CLOB_REST = "https://clob.polymarket.com/book"

# After N consecutive WebSocket failures, switch to REST polling
# Set to 1 for immediate REST fallback since SOCKS5 consistently times out on Fly.io
MAX_WSS_FAILURES = 1
REST_POLL_INTERVAL = 1.5  # Poll every 1.5 seconds for faster data

# Browser-like headers to bypass Cloudflare bot detection
REST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",  # Removed 'br' (Brotli) - requires Brotli package
    "Connection": "keep-alive",
    "Origin": "https://polymarket.com",
    "Referer": "https://polymarket.com/",
}


@dataclass
class OrderbookState:
    """Orderbook state for a market."""
    condition_id: str
    token_id: str
    side: str  # "UP" or "DOWN"
    bids: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    asks: List[tuple] = field(default_factory=list)
    last_update: Optional[datetime] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


class OrderbookStreamer:
    """Stream orderbook data from Polymarket CLOB."""

    def __init__(self):
        self.orderbooks: Dict[str, OrderbookState] = {}
        self.running = False
        self.callbacks: List[Callable] = []
        self._subscriptions: List[tuple] = []  # [(condition_id, token_id, side), ...]
        self._pending_subs: List[str] = []  # New token IDs to subscribe
        self._ws: Optional[Any] = None
        self._force_reconnect = False  # Flag to trigger reconnection
        self._use_rest_fallback = False  # Switch to REST polling after WSS failures
        self._wss_failure_count = 0

    def subscribe(self, condition_id: str, token_up: str, token_down: str):
        """Subscribe to orderbook for a market."""
        # Check if already subscribed
        existing_tokens = {t for _, t, _ in self._subscriptions}
        added = []

        if token_up not in existing_tokens:
            self._subscriptions.append((condition_id, token_up, "UP"))
            self._pending_subs.append(token_up)
            added.append("UP")

        if token_down not in existing_tokens:
            self._subscriptions.append((condition_id, token_down, "DOWN"))
            self._pending_subs.append(token_down)
            added.append("DOWN")

        if added:
            logger.info(f"[OB] Queued {condition_id[:8]}... ({', '.join(added)}) - pending: {len(self._pending_subs)}")

        # Initialize orderbook states
        self.orderbooks[f"{condition_id}_UP"] = OrderbookState(
            condition_id=condition_id,
            token_id=token_up,
            side="UP"
        )
        self.orderbooks[f"{condition_id}_DOWN"] = OrderbookState(
            condition_id=condition_id,
            token_id=token_down,
            side="DOWN"
        )

    def clear_stale(self, active_condition_ids: set):
        """Remove orderbooks for expired markets and trigger reconnection."""
        stale_keys = [k for k in self.orderbooks.keys()
                      if k.rsplit('_', 1)[0] not in active_condition_ids]

        had_stale = len(stale_keys) > 0
        for k in stale_keys:
            del self.orderbooks[k]

        # Also clean up subscriptions list
        old_sub_count = len(self._subscriptions)
        self._subscriptions = [(cid, tid, side) for cid, tid, side in self._subscriptions
                               if cid in active_condition_ids]

        # If we removed subscriptions, force reconnect to cleanly re-subscribe
        # This fixes the issue where WSS gets stuck on stale token IDs
        if had_stale or len(self._subscriptions) < old_sub_count:
            logger.info(f"[OB] Cleared {len(stale_keys)} stale orderbooks, triggering reconnect")
            self._force_reconnect = True

    def on_update(self, callback: Callable):
        """Register a callback for orderbook updates."""
        self.callbacks.append(callback)

    def get_orderbook(self, condition_id: str, side: str) -> Optional[OrderbookState]:
        """Get orderbook state for a market side."""
        return self.orderbooks.get(f"{condition_id}_{side}")

    async def _poll_rest(self, client, token_id: str, use_proxy: bool = False) -> Optional[dict]:
        """Poll orderbook via REST API using httpx (same as py-clob-client uses for auth).

        Uses asyncio.wait_for() to guarantee timeout since httpx async timeouts
        can be bypassed by DNS resolution or h2 library internals.
        """
        import time as _time
        import httpx

        start = _time.time()
        url = f"{CLOB_REST}?token_id={token_id}"
        REQUEST_TIMEOUT = 10.0  # Hard timeout for entire request

        try:
            # Wrap in asyncio.wait_for to guarantee timeout - httpx timeouts
            # can be bypassed during DNS/SSL phases in async context
            resp = await asyncio.wait_for(client.get(url), timeout=REQUEST_TIMEOUT)
            elapsed = _time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                if "bids" in data and "asks" in data:
                    if elapsed > 2.0:
                        logger.debug(f"[OB] REST OK for {token_id[:8]}... ({elapsed:.1f}s)")
                    return data
                logger.warning(f"[OB] REST response missing bids/asks for {token_id[:8]}...")
            elif resp.status_code == 403:
                is_cloudflare = "cloudflare" in resp.text.lower()
                logger.warning(f"[OB] REST 403 for {token_id[:8]}... cloudflare={is_cloudflare} ({elapsed:.1f}s)")
            else:
                logger.warning(f"[OB] REST {resp.status_code} for {token_id[:8]}... ({elapsed:.1f}s): {resp.text[:100]}")
            return None

        except asyncio.TimeoutError:
            elapsed = _time.time() - start
            logger.info(f"[OB] REST hard timeout for {token_id[:15]}... ({elapsed:.1f}s)")
        except httpx.TimeoutException as e:
            elapsed = _time.time() - start
            logger.info(f"[OB] REST httpx timeout for {token_id[:15]}... ({elapsed:.1f}s): {type(e).__name__}")
        except httpx.HTTPError as e:
            elapsed = _time.time() - start
            logger.warning(f"[OB] REST httpx error for {token_id[:8]}... ({elapsed:.1f}s): {type(e).__name__}: {e}")
        except Exception as e:
            elapsed = _time.time() - start
            logger.warning(f"[OB] REST poll error for {token_id[:8]}... ({elapsed:.1f}s): {type(e).__name__}: {e}")
        return None

    def _sync_fetch_orderbook(self, sync_client, token_id: str) -> Optional[dict]:
        """Fetch orderbook using sync httpx client (proven to work in 0.2s from Fly.io)."""
        import time as _time

        url = f"{CLOB_REST}?token_id={token_id}"
        start = _time.time()

        try:
            resp = sync_client.get(url)
            elapsed = _time.time() - start

            if resp.status_code == 200:
                data = resp.json()
                if "bids" in data and "asks" in data:
                    if elapsed > 2.0:
                        logger.debug(f"[OB] REST OK for {token_id[:8]}... ({elapsed:.1f}s)")
                    return data
                logger.warning(f"[OB] REST missing bids/asks for {token_id[:8]}...")
            elif resp.status_code == 403:
                logger.warning(f"[OB] REST 403 for {token_id[:8]}... ({elapsed:.1f}s)")
            else:
                logger.warning(f"[OB] REST {resp.status_code} for {token_id[:8]}...: {resp.text[:100]}")
        except Exception as e:
            elapsed = _time.time() - start
            logger.warning(f"[OB] REST sync error for {token_id[:8]}... ({elapsed:.1f}s): {type(e).__name__}")
        return None

    async def _rest_polling_loop(self):
        """Fallback polling loop using REST API with sync httpx in ThreadPoolExecutor.

        Uses synchronous httpx.Client which is proven to work in 0.2s from Fly.io,
        run in ThreadPoolExecutor to avoid blocking the event loop.
        """
        import httpx
        from concurrent.futures import ThreadPoolExecutor

        logger.info("[OB] ✓ Starting REST polling fallback mode (sync httpx in thread)")

        consecutive_failures = 0
        max_consecutive_failures = 10
        last_log_time = datetime.now(timezone.utc)

        # Create sync client with aligned timeouts and proper connection limits.
        # IMPORTANT: httpx timeout must be LESS than asyncio.wait_for timeout to avoid
        # race condition where asyncio times out but httpx keeps running in thread.
        # asyncio timeout is 60s warmup / 35s normal, so httpx should be ~25s max.
        #
        # Pool settings: Allow enough connections for all concurrent requests.
        # With HTTP/2, connections are multiplexed but we still need enough pool slots.
        sync_client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=15.0, write=5.0, pool=30.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=30),
            http2=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

        # ThreadPoolExecutor to run sync requests without blocking event loop.
        # Must be >= orderbook count to prevent queue backup that causes timeouts.
        # With 4 markets × 2 sides × ~3 time windows = ~24 orderbooks max.
        executor = ThreadPoolExecutor(max_workers=32)
        loop = asyncio.get_event_loop()

        try:
            iteration = 0
            first_success_time = None  # Track when connection pool warms up
            while self.running and self._use_rest_fallback:
                iteration += 1
                success_count = 0
                total_count = len(self.orderbooks)

                # Adaptive timeout: keep at 60s always
                # Polymarket API can be slow (50-60s for batch requests)
                # httpx timeout is 10s connect + 15s read, but pool/thread waits add more
                request_timeout = 60.0

                # Log iteration status
                iteration_start = datetime.now(timezone.utc)
                if iteration == 1:
                    logger.info(f"[OB] REST polling: iteration 1 ({total_count} orderbooks in parallel, timeout={request_timeout}s)")
                    sample_tokens = [ob.token_id[:20] for ob in list(self.orderbooks.values())[:2]]
                    logger.info(f"[OB] Sample tokens: {sample_tokens}")
                elif iteration == 2:
                    logger.info(f"[OB] REST polling: iteration 2 (timeout={request_timeout}s)")
                elif iteration == 3:
                    logger.info(f"[OB] REST polling: iteration 3 (timeout={request_timeout}s)")

                # Create parallel fetch tasks for ALL orderbooks at once
                orderbook_items = list(self.orderbooks.items())

                async def fetch_with_timeout(key, ob):
                    """Fetch single orderbook with timeout wrapper."""
                    try:
                        data = await asyncio.wait_for(
                            loop.run_in_executor(
                                executor,
                                self._sync_fetch_orderbook,
                                sync_client,
                                ob.token_id
                            ),
                            timeout=request_timeout
                        )
                        return (key, ob, data, None)
                    except asyncio.TimeoutError:
                        return (key, ob, None, "timeout")
                    except Exception as e:
                        return (key, ob, None, str(e))

                # Run ALL requests in parallel using asyncio.gather
                tasks = [fetch_with_timeout(key, ob) for key, ob in orderbook_items]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Debug: log result types
                exception_count = sum(1 for r in results if isinstance(r, Exception))
                tuple_count = sum(1 for r in results if isinstance(r, tuple))
                other_count = len(results) - exception_count - tuple_count
                if iteration <= 5 or success_count == 0:
                    logger.info(f"[OB] DEBUG: gather returned {len(results)} results - {tuple_count} tuples, {exception_count} exceptions, {other_count} other")

                # Process all results
                data_none_count = 0
                error_count = 0
                for result in results:
                    if isinstance(result, Exception):
                        consecutive_failures += 1
                        continue

                    key, ob, data, error = result

                    if error:
                        error_count += 1
                        # Log all errors during first 5 iterations for debugging
                        if iteration <= 5 or success_count == 0:
                            logger.info(f"[OB] DEBUG error: {ob.token_id[:8]}... => {error[:60]}")
                        elif error == "timeout":
                            logger.debug(f"[OB] REST timeout for {ob.token_id[:15]}...")
                        else:
                            logger.warning(f"[OB] REST error for {ob.token_id[:8]}...: {error[:80]}")
                        consecutive_failures += 1
                        continue

                    if not data:
                        data_none_count += 1
                        consecutive_failures += 1
                        continue

                    if data:
                        try:
                            # Parse REST response
                            bids = data.get("bids", [])
                            asks = data.get("asks", [])

                            parsed_bids = [(float(b["price"]), float(b["size"])) for b in bids]
                            parsed_asks = [(float(a["price"]), float(a["size"])) for a in asks]

                            ob.bids = sorted(parsed_bids, key=lambda x: x[0], reverse=True)[:10]
                            ob.asks = sorted(parsed_asks, key=lambda x: x[0])[:10]
                            ob.last_update = datetime.now(timezone.utc)

                            success_count += 1
                            consecutive_failures = 0

                            # Log first success to confirm warmup
                            if first_success_time is None:
                                first_success_time = datetime.now(timezone.utc)
                                logger.info(f"[OB] ✓ First REST success! Connection pool warmed up")

                            # Call callbacks
                            for cb in self.callbacks:
                                try:
                                    cb(ob)
                                except Exception as e:
                                    logger.debug(f"[OB] Callback error: {e}")
                        except (KeyError, ValueError, TypeError) as e:
                            logger.error(f"[OB] Parse error for {ob.token_id[:8]}...: {e}")
                            consecutive_failures += 1
                    else:
                        consecutive_failures += 1

                # Log iteration completion with timing
                iteration_elapsed = (datetime.now(timezone.utc) - iteration_start).total_seconds()
                if iteration <= 5 or success_count == 0:
                    logger.info(f"[OB] DEBUG: errors={error_count}, data_none={data_none_count}, success={success_count}")
                    logger.info(f"[OB] REST iteration {iteration}: {success_count}/{total_count} in {iteration_elapsed:.1f}s")
                elif (datetime.now(timezone.utc) - last_log_time).total_seconds() > 30:
                    logger.info(f"[OB] REST poll: {success_count}/{total_count} successful ({iteration_elapsed:.1f}s)")
                    last_log_time = datetime.now(timezone.utc)

                # Warn if consistently failing
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"[OB] REST polling failing ({consecutive_failures} consecutive failures)")
                    consecutive_failures = 0

                await asyncio.sleep(REST_POLL_INTERVAL)

        finally:
            # Clean up sync client and executor
            sync_client.close()
            executor.shutdown(wait=False)
            logger.info("[OB] REST polling loop ended")

    async def _connect_with_socks5_proxy(self, proxy_url: str) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to WebSocket through SOCKS5 proxy using python-socks.

        SOCKS5 proxies work better than HTTP proxies for WebSocket because they
        operate at the TCP level rather than HTTP level, avoiding issues with
        WebSocket upgrade handshake.

        Note: python-socks returns an AsyncioSocketStream which is not compatible
        with websockets library directly. We extract the underlying socket via
        the stream's writer transport and use socket.fromfd() to create a real
        socket object that websockets can use.
        """
        import socket
        import ssl

        try:
            proxy = Proxy.from_url(proxy_url)
            stream = await asyncio.wait_for(
                proxy.connect(
                    dest_host="ws-subscriptions-clob.polymarket.com",
                    dest_port=443,
                    timeout=25
                ),
                timeout=30
            )
            logger.info("[OB] ✓ SOCKS5 tunnel established")

            # Extract underlying socket from asyncio stream
            # python-socks returns AsyncioSocketStream with _writer -> transport -> socket
            writer = stream._writer
            transport_socket = writer.get_extra_info('socket')
            fd = transport_socket.fileno()

            # Create a real socket.socket from the file descriptor
            # This is required because websockets expects a standard socket, not AsyncioSocketStream
            raw_sock = socket.fromfd(fd, socket.AF_INET, socket.SOCK_STREAM)

            # Connect websockets over the proxied socket with SSL
            ssl_context = ssl.create_default_context()
            ws = await asyncio.wait_for(
                websockets.connect(
                    CLOB_WSS,
                    sock=raw_sock,
                    ssl=ssl_context,
                    server_hostname="ws-subscriptions-clob.polymarket.com",
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=10,
                ),
                timeout=30
            )
            return ws
        except asyncio.TimeoutError:
            logger.error(f"[OB] SOCKS5 proxy connection failed: ProxyTimeoutError: Proxy connection timed out: 30")
            return None
        except Exception as e:
            logger.error(f"[OB] SOCKS5 proxy connection failed: {type(e).__name__}: {e}")
            return None

    async def _connect_direct(self) -> Optional[websockets.WebSocketClientProtocol]:
        """Connect to WebSocket directly (no proxy)."""
        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    CLOB_WSS,
                    ping_interval=30,
                    ping_timeout=60,
                    close_timeout=10,
                ),
                timeout=15
            )
            return ws
        except asyncio.TimeoutError:
            logger.error(f"[OB] Direct WSS connection timeout (15s)")
            return None
        except Exception as e:
            logger.error(f"[OB] Direct WSS connection failed: {type(e).__name__}: {e}")
            return None

    async def stream(self):
        """Start streaming orderbooks using websockets library with SOCKS5 proxy support.

        Connection strategy:
        1. Try direct connection first (works if Cloudflare isn't blocking)
        2. Try SOCKS5 proxy if direct fails (more reliable than HTTP proxy for WSS)
        3. Try HTTP proxy via aiohttp as last resort
        4. Fall back to REST polling after MAX_WSS_FAILURES total failures
        """
        self.running = True
        # Use TRY_DIRECT_FIRST to decide initial connection strategy
        # If ORDERBOOK_TRY_DIRECT=false, skip direct attempts and go straight to proxy
        self._use_proxy = not TRY_DIRECT_FIRST
        self._direct_failed_count = 0
        logger.info(f"[OB] Orderbook streamer starting... (TRY_DIRECT_FIRST={TRY_DIRECT_FIRST}, use_proxy={self._use_proxy})")

        while self.running:
            # Wait for subscriptions if none exist yet
            if not self._subscriptions and not self._pending_subs:
                await asyncio.sleep(0.5)
                continue

            ws = None
            connection_type = "unknown"

            try:
                # Strategy: Direct → SOCKS5 → REST fallback
                if not self._use_proxy:
                    logger.info(f"[OB] Connecting (direct) with {len(self._subscriptions)} subs, {len(self._pending_subs)} pending")
                    logger.info("[OB] Connecting directly (no proxy)...")
                    ws = await self._connect_direct()
                    if ws:
                        connection_type = "direct"
                    else:
                        self._direct_failed_count += 1
                        self._wss_failure_count += 1
                        if self._direct_failed_count >= 2 and PROXY_SOCKS5_URL:
                            logger.warning(f"[OB] Direct failed {self._direct_failed_count}x, trying SOCKS5 proxy...")
                            self._use_proxy = True

                if self._use_proxy and not ws and PROXY_SOCKS5_URL:
                    logger.info(f"[OB] Connecting (via SOCKS5 proxy) with {len(self._subscriptions)} subs")
                    ws = await self._connect_with_socks5_proxy(PROXY_SOCKS5_URL)
                    if ws:
                        connection_type = "SOCKS5 proxy"
                    else:
                        self._wss_failure_count += 1

                # Check if we should fall back to REST
                if self._wss_failure_count >= MAX_WSS_FAILURES:
                    logger.warning(f"[OB] ✓ WSS failed {self._wss_failure_count}x, activating REST polling fallback")
                    self._use_rest_fallback = True
                    await self._rest_polling_loop()
                    return  # Exit WSS stream loop

                if not ws:
                    await asyncio.sleep(1)
                    continue

                # Successfully connected
                self._direct_failed_count = 0
                logger.info(f"✓ Connected to Polymarket CLOB WSS ({connection_type})")

                # Collect all token IDs for initial subscription
                token_ids = [token_id for _, token_id, _ in self._subscriptions]

                # Also include any pending subs
                if self._pending_subs:
                    token_ids.extend(self._pending_subs)
                    self._pending_subs.clear()

                if token_ids:
                    # Send single subscription with all assets
                    sub_msg = {
                        "assets_ids": token_ids,
                        "type": "market"
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"Subscribed to {len(token_ids)} orderbooks")

                self._ws = ws

                # Listen for updates
                while self.running:
                    try:
                        # Check for forced reconnection (markets changed)
                        if self._force_reconnect:
                            logger.info("[OB] Force reconnect triggered, closing connection...")
                            self._force_reconnect = False
                            break  # Exit inner loop to reconnect

                        # Check for pending subscriptions FIRST (new markets added dynamically)
                        if self._pending_subs:
                            new_tokens = self._pending_subs.copy()
                            self._pending_subs.clear()
                            sub_msg = {
                                "assets_ids": new_tokens,
                                "type": "market"
                            }
                            await ws.send(json.dumps(sub_msg))
                            logger.info(f"[OB] Sent subscription for {len(new_tokens)} new tokens")

                        # Receive with short timeout to check pending subs frequently
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                            data = json.loads(msg)

                            # Handle different message types
                            if isinstance(data, list):
                                # Initial snapshot is an array
                                for item in data:
                                    if isinstance(item, dict):
                                        self._handle_book_update(item)
                            elif isinstance(data, dict):
                                # Check for orderbook update (has bids/asks)
                                if "bids" in data or "asks" in data:
                                    self._handle_book_update(data)
                                # Check for price_changes
                                elif "price_changes" in data:
                                    self._handle_price_change(data)

                        except asyncio.TimeoutError:
                            pass

                    except websockets.ConnectionClosed as e:
                        logger.warning(f"[OB] WebSocket closed: {e}")
                        break
                    except Exception as e:
                        logger.error(f"[OB] Message handling error: {type(e).__name__}: {e}")
                        break

                self._ws = None

            except Exception as e:
                logger.error(f"CLOB WSS error: {type(e).__name__}: {e}")
                self._wss_failure_count += 1
                if not self._use_proxy:
                    self._direct_failed_count += 1
                    if self._direct_failed_count >= 2 and PROXY_SOCKS5_URL:
                        logger.warning(f"[OB] Direct failed {self._direct_failed_count}x, trying SOCKS5 proxy...")
                        self._use_proxy = True
                # After MAX_WSS_FAILURES total failures, switch to REST polling
                if self._wss_failure_count >= MAX_WSS_FAILURES:
                    logger.warning(f"[OB] ✓ WSS failed {self._wss_failure_count}x, activating REST polling fallback")
                    self._use_rest_fallback = True
                    await self._rest_polling_loop()
                    return  # Exit WSS stream loop
                await asyncio.sleep(1)
            finally:
                # Clean up WebSocket connection
                if ws:
                    try:
                        await ws.close()
                    except:
                        pass

    def _handle_book_update(self, data: dict):
        """Handle orderbook update message."""
        asset_id = data.get("asset_id")
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        # Find matching orderbook
        for key, ob in self.orderbooks.items():
            if ob.token_id == asset_id:
                # Parse and sort: bids descending, asks ascending
                parsed_bids = [(float(b["price"]), float(b["size"])) for b in bids]
                parsed_asks = [(float(a["price"]), float(a["size"])) for a in asks]

                # Sort bids high to low, asks low to high
                ob.bids = sorted(parsed_bids, key=lambda x: x[0], reverse=True)[:10]
                ob.asks = sorted(parsed_asks, key=lambda x: x[0])[:10]
                ob.last_update = datetime.now(timezone.utc)

                # Call callbacks
                for cb in self.callbacks:
                    try:
                        cb(ob)
                    except:
                        pass
                break

    def _handle_price_change(self, data: dict):
        """Handle price change message (simpler update)."""
        changes = data.get("price_changes", [])
        for change in changes:
            asset_id = change.get("asset_id")
            price = change.get("price")

            # Find matching orderbook and update mid estimate
            for key, ob in self.orderbooks.items():
                if ob.token_id == asset_id:
                    ob.last_update = datetime.now(timezone.utc)
                    break

    def reconnect(self):
        """Force a reconnection to pick up new subscriptions cleanly."""
        logger.info("[OB] Manual reconnect requested")
        self._force_reconnect = True

    def stop(self):
        """Stop streaming."""
        self.running = False


if __name__ == "__main__":
    # Test with a real market
    from polymarket_api import get_active_markets

    print("Testing Orderbook WSS...")

    async def test():
        markets = get_active_markets()

        if not markets:
            print("No active markets!")
            return

        m = markets[0]
        print(f"\nSubscribing to: {m.question[:50]}...")

        streamer = OrderbookStreamer()
        streamer.subscribe(m.condition_id, m.token_up, m.token_down)

        def on_update(ob: OrderbookState):
            print(f"  {ob.side}: bid={ob.best_bid:.3f} ask={ob.best_ask:.3f} spread={ob.spread:.3f}")

        streamer.on_update(on_update)

        # Run for 15 seconds
        task = asyncio.create_task(streamer.stream())
        await asyncio.sleep(15)
        streamer.stop()
        task.cancel()

    asyncio.run(test())
