#!/usr/bin/env python3
"""
Railway Worker: 24/7 Paper Trading Service

Runs the RL inference loop, manages WebSocket streams,
persists to PostgreSQL, and sends Discord alerts.

Usage:
    # Paper trading (default)
    python railway_worker.py

    # Live trading (requires POLYMARKET_PRIVATE_KEY)
    TRADING_MODE=live python railway_worker.py

Environment Variables:
    DATABASE_URL: PostgreSQL connection URL
    TRADING_MODE: 'paper' or 'live' (default: paper)
    TRADE_SIZE: Trade size in dollars (default: 50)
    DISCORD_WEBHOOK_URL: Discord webhook for alerts
    POLYMARKET_PRIVATE_KEY: Required for live trading
    POLYMARKET_FUNDER_ADDRESS: Required for live trading
"""
import asyncio
import os
import signal
import sys
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure residential proxy for Polymarket CLOB API (bypasses Cloudflare)
# Only apply to CLOB requests - we'll handle this in clob_executor
PROXY_URL = os.getenv("RESIDENTIAL_PROXY_URL")
if PROXY_URL:
    # Set for requests library (used by py-clob-client)
    os.environ["HTTP_PROXY"] = PROXY_URL
    os.environ["HTTPS_PROXY"] = PROXY_URL
    print(f"[PROXY] Configured residential proxy: {PROXY_URL.split('@')[-1] if '@' in PROXY_URL else 'configured'}")

import json

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.connection import Database
from helpers.discord import DiscordWebhook
from helpers.polymarket_api import get_15m_markets, Market
from helpers.orderbook_wss import OrderbookStreamer, OrderbookState
from helpers.binance_wss import BinanceStreamer
# Use OKX for futures data - datacenter-friendly (Binance blocks cloud IPs)
from helpers.okx_futures import OKXFuturesStreamer as FuturesStreamer
from helpers.clob_executor import create_executor, OrderSide
from strategies.base import MarketState, Action

# Conditional import: Use NumPy on Linux, MLX on macOS
try:
    import mlx.core as mx
    from strategies.rl_mlx import RLStrategy
    USE_MLX = True
except ImportError:
    from strategies.rl_numpy import NumpyRLStrategy as RLStrategy
    USE_MLX = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("worker")


@dataclass
class Position:
    """Active trading position."""
    asset: str
    side: str = ""
    size: float = 0.0
    entry_price: float = 0.0
    entry_binance_price: float = 0.0
    token_id: str = ""
    entry_time: Optional[datetime] = None
    condition_id: str = ""
    trade_id: Optional[str] = None  # Database trade ID
    action_probs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "asset": self.asset,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "entry_binance_price": self.entry_binance_price,
            "token_id": self.token_id,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "condition_id": self.condition_id,
            "trade_id": self.trade_id,
            "action_probs": self.action_probs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create from dict."""
        entry_time = None
        if data.get("entry_time"):
            entry_time = datetime.fromisoformat(data["entry_time"])
        return cls(
            asset=data["asset"],
            side=data.get("side", ""),
            size=data.get("size", 0.0),
            entry_price=data.get("entry_price", 0.0),
            entry_binance_price=data.get("entry_binance_price", 0.0),
            token_id=data.get("token_id", ""),
            entry_time=entry_time,
            condition_id=data.get("condition_id", ""),
            trade_id=data.get("trade_id"),
            action_probs=data.get("action_probs", {}),
        )


class TradingWorker:
    """
    Main trading worker for Railway deployment.

    Manages:
    - Database connection and persistence
    - WebSocket data streams
    - RL inference and trade execution
    - Discord alerts
    - Crash recovery via checkpoints
    """

    def __init__(self):
        # Configuration from environment
        self.trade_size = float(os.getenv("TRADE_SIZE", "50"))
        self.dual_mode = os.getenv("DUAL_MODE", "false").lower() == "true"
        self.live_auto_start = os.getenv("TRADING_MODE", "paper") == "live"
        self.model_path = os.getenv("MODEL_PATH", "rl_model")

        # Components
        self.db = Database()
        self.discord = DiscordWebhook()

        # Sessions (dual mode: separate sessions for paper and live)
        self.paper_session_id: Optional[str] = None
        self.live_session_id: Optional[str] = None
        self.live_enabled = self.live_auto_start

        # Strategy (NumPy or MLX depending on platform)
        logger.info(f"Loading strategy (MLX={USE_MLX})")
        self.strategy = RLStrategy(self.model_path) if not USE_MLX else None

        # Profit transfer components (LIVE mode only)
        self.profit_transfer_enabled = os.getenv("COLD_WALLET_ADDRESS") is not None
        self.balance_tracker: Optional[Any] = None
        self.transfer_executor: Optional[Any] = None
        self.profit_monitor: Optional[Any] = None
        self.initial_balance: float = 0.0

        # Executors (dual mode: both paper and live)
        self.paper_executor = create_executor(live=False)
        self.live_executor = None
        if self.dual_mode or self.live_auto_start:
            try:
                self.live_executor = create_executor(live=True)
                logger.info("✓ Live executor initialized")
            except Exception as e:
                logger.error(f"✗ Live executor failed to initialize: {e}")
                self.live_enabled = False

        # Data streams (shared between executors)
        self.assets = ["BTC", "ETH", "SOL", "XRP"]
        self.orderbook_streamer = OrderbookStreamer()
        self.price_streamer = BinanceStreamer(self.assets)
        self.futures_streamer = FuturesStreamer(self.assets)

        # State (shared market data)
        self.markets: Dict[str, Market] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.open_prices: Dict[str, float] = {}

        # Positions (separate per executor)
        self.paper_positions: Dict[str, Position] = {}
        self.live_positions: Dict[str, Position] = {}

        # Stats (separate per mode)
        self.paper_total_pnl = 0.0
        self.paper_trade_count = 0
        self.paper_win_count = 0

        self.live_total_pnl = 0.0
        self.live_trade_count = 0
        self.live_win_count = 0

        # Live failure tracking
        self.live_failure_count = 0
        self.live_max_failures = 3

        # Shared milestones
        self.milestones_hit = set()

        # === SAFETY SYSTEMS ===
        # Orderbook health tracking (last successful update per market)
        self.orderbook_last_update: Dict[str, datetime] = {}
        self.orderbook_stale_threshold = 60  # seconds without update = stale
        self.system_healthy = True

        # Position timeout settings
        self.position_timeout_minutes = 2  # Force close 2 min before expiry
        self.position_max_age_seconds = 840  # 14 minutes max (for 15-min markets)

        # Control
        self.running = False
        self.shutdown_event = asyncio.Event()

        logger.info(f"TradingWorker initialized (dual_mode={self.dual_mode}, live_enabled={self.live_enabled})")

    async def initialize(self) -> None:
        """Initialize all components."""
        # Load MLX strategy (if on macOS)
        if USE_MLX:
            self.strategy = RLStrategy()
            self.strategy.load(self.model_path)
            self.strategy.training = False

        # Connect to database
        await self.db.connect()
        logger.info("Database connected")

        # Try to recover from crashed sessions
        recovered = await self._recover_session()

        # Create paper session if not recovered
        if not self.paper_session_id:
            self.paper_session_id = await self.db.create_session(
                mode="paper",
                trade_size=self.trade_size,
                model_version=f"rl_model_{'mlx' if USE_MLX else 'numpy'}",
                config={
                    "assets": self.assets,
                    "use_mlx": USE_MLX,
                    "dual_mode": self.dual_mode,
                }
            )
            logger.info(f"[PAPER] Created session: {self.paper_session_id}")

        # Create live session if enabled and not recovered
        if self.live_enabled and self.live_executor and not self.live_session_id:
            self.live_session_id = await self.db.create_session(
                mode="live",
                trade_size=self.trade_size,
                model_version=f"rl_model_{'mlx' if USE_MLX else 'numpy'}",
                config={
                    "assets": self.assets,
                    "use_mlx": USE_MLX,
                    "dual_mode": self.dual_mode,
                }
            )
            logger.info(f"[LIVE] Created session: {self.live_session_id}")

        # Send startup notification (if not recovered)
        if not recovered:
            if self.dual_mode:
                await self.discord.send_startup(
                    mode="dual (paper + live)",
                    trade_size=self.trade_size,
                    model_version=f"rl_model ({'MLX' if USE_MLX else 'NumPy'})"
                )
            else:
                await self.discord.send_startup(
                    mode="live" if self.live_enabled else "paper",
                    trade_size=self.trade_size,
                    model_version=f"rl_model ({'MLX' if USE_MLX else 'NumPy'})"
                )

        # Initialize profit transfer system (LIVE mode only)
        if self.profit_transfer_enabled and self.live_enabled and self.live_session_id:
            try:
                from helpers.wallet_balance import WalletBalanceTracker
                from helpers.profit_transfer import ProfitTransferExecutor
                from helpers.profit_monitor import ProfitMonitor

                rpc_url = os.getenv("POLYGON_RPC_URL")
                if not rpc_url:
                    logger.warning("POLYGON_RPC_URL not set - profit transfer disabled")
                    self.profit_transfer_enabled = False
                else:
                    cold_wallet = os.getenv("COLD_WALLET_ADDRESS")
                    hot_wallet = os.getenv("POLYMARKET_FUNDER_ADDRESS")
                    private_key = os.getenv("POLYMARKET_PRIVATE_KEY")

                    # Initialize balance tracker
                    self.balance_tracker = WalletBalanceTracker(rpc_url, hot_wallet)
                    self.initial_balance = await self.balance_tracker.get_balance()

                    # Initialize transfer executor
                    self.transfer_executor = ProfitTransferExecutor(
                        rpc_url, private_key, hot_wallet, cold_wallet,
                        self.db, self.discord
                    )

                    # Initialize profit monitor
                    self.profit_monitor = ProfitMonitor(
                        self.live_session_id, self.db, self.balance_tracker,
                        self.transfer_executor, self.initial_balance
                    )

                    logger.info(
                        f"✓ Profit transfer enabled (initial balance: ${self.initial_balance:.2f})"
                    )
            except Exception as e:
                logger.error(f"✗ Profit transfer initialization failed: {e}")
                self.profit_transfer_enabled = False

    async def _recover_session(self) -> bool:
        """Attempt to recover from crashed sessions (dual-mode aware)."""
        recovered_any = False

        # Try to recover paper session
        paper_session = await self.db.get_session_by_mode(mode="paper", status="running")
        if paper_session and paper_session.get("checkpoint_data"):
            logger.info(f"[PAPER] Recovering session {paper_session['id']}")
            self.paper_session_id = str(paper_session["id"])
            self.paper_total_pnl = float(paper_session["total_pnl"] or 0)
            self.paper_trade_count = paper_session["trade_count"] or 0
            self.paper_win_count = paper_session["win_count"] or 0

            # Restore paper positions
            checkpoint = paper_session["checkpoint_data"]
            if "positions" in checkpoint:
                for cid, pos_data in checkpoint["positions"].items():
                    if pos_data.get("size", 0) > 0:
                        self.paper_positions[cid] = Position.from_dict(pos_data)

            recovered_any = True

        # Try to recover live session (if enabled)
        if self.live_enabled:
            live_session = await self.db.get_session_by_mode(mode="live", status="running")
            if live_session and live_session.get("checkpoint_data"):
                logger.info(f"[LIVE] Recovering session {live_session['id']}")
                self.live_session_id = str(live_session["id"])
                self.live_total_pnl = float(live_session["total_pnl"] or 0)
                self.live_trade_count = live_session["trade_count"] or 0
                self.live_win_count = live_session["win_count"] or 0

                # Restore live positions
                checkpoint = live_session["checkpoint_data"]
                if "positions" in checkpoint:
                    for cid, pos_data in checkpoint["positions"].items():
                        if pos_data.get("size", 0) > 0:
                            self.live_positions[cid] = Position.from_dict(pos_data)

                recovered_any = True

        # Send recovery notification
        if recovered_any:
            if self.dual_mode:
                await self.discord.send_recovery(
                    session_id=f"paper:{self.paper_session_id}, live:{self.live_session_id}",
                    recovered_pnl=self.paper_total_pnl + self.live_total_pnl,
                    recovered_trades=self.paper_trade_count + self.live_trade_count,
                    open_positions=len(self.paper_positions) + len(self.live_positions)
                )
            else:
                session_id = self.paper_session_id or self.live_session_id
                total_pnl = self.paper_total_pnl or self.live_total_pnl
                trade_count = self.paper_trade_count or self.live_trade_count
                positions = self.paper_positions if self.paper_session_id else self.live_positions
                await self.discord.send_recovery(
                    session_id=session_id,
                    recovered_pnl=total_pnl,
                    recovered_trades=trade_count,
                    open_positions=len(positions)
                )

        return recovered_any

    async def checkpoint(self) -> None:
        """Save current state for crash recovery (dual-mode aware)."""
        # Checkpoint paper session
        if self.paper_session_id:
            paper_checkpoint = {
                "positions": {
                    cid: pos.to_dict()
                    for cid, pos in self.paper_positions.items()
                    if pos.size > 0
                },
                "open_prices": self.open_prices,
            }
            await self.db.update_session_checkpoint(
                session_id=self.paper_session_id,
                checkpoint_data=paper_checkpoint,
                total_pnl=self.paper_total_pnl,
                trade_count=self.paper_trade_count,
                win_count=self.paper_win_count
            )

        # Checkpoint live session (if enabled)
        if self.live_session_id and self.live_enabled:
            live_checkpoint = {
                "positions": {
                    cid: pos.to_dict()
                    for cid, pos in self.live_positions.items()
                    if pos.size > 0
                },
                "open_prices": self.open_prices,
            }
            await self.db.update_session_checkpoint(
                session_id=self.live_session_id,
                checkpoint_data=live_checkpoint,
                total_pnl=self.live_total_pnl,
                trade_count=self.live_trade_count,
                win_count=self.live_win_count
            )

    async def refresh_markets(self) -> None:
        """Refresh active Polymarket markets."""
        markets = get_15m_markets(assets=self.assets)

        if not markets:
            logger.warning("No active markets found")
            return

        self.markets = {m.condition_id: m for m in markets}

        for m in markets:
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            if m.condition_id not in self.paper_positions:
                self.paper_positions[m.condition_id] = Position(asset=m.asset)
            if m.condition_id not in self.live_positions:
                self.live_positions[m.condition_id] = Position(asset=m.asset)

            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0 and m.condition_id not in self.open_prices:
                self.open_prices[m.condition_id] = current_price

        logger.info(f"Found {len(markets)} active markets: {', '.join(m.asset for m in markets)}")

    def update_market_state(
        self,
        cid: str,
        market: Market,
        ob_up: OrderbookState
    ) -> MarketState:
        """Update market state from data streams."""
        if cid not in self.market_states:
            self.market_states[cid] = MarketState(
                asset=market.asset,
                prob=market.price_up,
                time_remaining=1.0
            )

        state = self.market_states[cid]
        now = datetime.now(timezone.utc)

        # Orderbook data
        if ob_up and ob_up.mid_price:
            state.prob = ob_up.mid_price
            state.prob_history.append(ob_up.mid_price)
            if len(state.prob_history) > 100:
                state.prob_history = state.prob_history[-100:]
            state.best_bid = ob_up.best_bid or 0.0
            state.best_ask = ob_up.best_ask or 0.0
            state.spread = ob_up.spread or 0.0

            if ob_up.bids and ob_up.asks:
                bid_vol_l1 = ob_up.bids[0][1] if ob_up.bids else 0
                ask_vol_l1 = ob_up.asks[0][1] if ob_up.asks else 0
                total_l1 = bid_vol_l1 + ask_vol_l1
                state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0

            bid_vol_l5 = sum(s for _, s in ob_up.bids[:5])
            ask_vol_l5 = sum(s for _, s in ob_up.asks[:5])
            total_l5 = bid_vol_l5 + ask_vol_l5
            state.order_book_imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0

        # Binance spot
        binance_price = self.price_streamer.get_price(market.asset)
        if binance_price > 0:
            state.binance_price = binance_price
            open_price = self.open_prices.get(cid, binance_price)
            if open_price > 0:
                state.binance_change = (binance_price - open_price) / open_price

        # Binance futures
        futures = self.futures_streamer.get_state(market.asset)
        if futures:
            old_cvd = state.cvd
            state.cvd = futures.cvd
            state.cvd_acceleration = (futures.cvd - old_cvd) / 1e6 if old_cvd != 0 else 0.0
            state.trade_flow_imbalance = futures.trade_flow_imbalance
            state.returns_1m = futures.returns_1m
            state.returns_5m = futures.returns_5m
            state.returns_10m = futures.returns_10m
            state.trade_intensity = futures.trade_intensity
            state.large_trade_flag = futures.large_trade_flag
            state.realized_vol_5m = futures.realized_vol_1h / 3.5 if futures.realized_vol_1h > 0 else 0.0
            state.vol_expansion = futures.vol_ratio - 1.0
            state.vol_regime = 1.0 if futures.realized_vol_1h > 0.01 else 0.0
            state.trend_regime = 1.0 if abs(futures.returns_1h) > 0.005 else 0.0

        state.time_remaining = max(0, (market.end_time - now).total_seconds() / 900)

        # Position state (use paper position for state calculation)
        pos = self.paper_positions.get(cid)
        if pos and pos.size > 0:
            state.has_position = True
            state.position_side = pos.side
            shares = pos.size / pos.entry_price if pos.entry_price > 0 else 0
            if pos.side == "UP":
                state.position_pnl = (state.prob - pos.entry_price) * shares
            else:
                state.position_pnl = ((1 - state.prob) - pos.entry_price) * shares
        else:
            state.has_position = False
            state.position_side = None
            state.position_pnl = 0.0

        return state

    def get_rl_action(self, state: MarketState) -> tuple:
        """Get action from RL strategy."""
        if USE_MLX:
            import mlx.core as mx
            features = state.to_features()
            features_mx = mx.array(features.reshape(1, -1))
            probs = self.strategy.actor(features_mx)
            probs_np = np.array(probs[0])
        else:
            probs_np = self.strategy.get_action_probs(state)

        action_idx = int(np.argmax(probs_np))
        action = Action(action_idx)
        confidence = np.max(probs_np) - np.partition(probs_np, -2)[-2]

        return action, probs_np, confidence

    async def execute_action_dual_mode(
        self,
        cid: str,
        market: Market,
        action: Action,
        state: MarketState,
        probs: np.ndarray,
        confidence: float
    ) -> None:
        """Execute action on both paper and live modes in parallel."""
        tasks = []

        # Always execute paper
        tasks.append(
            self._execute_action_single(
                cid, market, action, state, probs, confidence,
                mode='paper',
                executor=self.paper_executor,
                positions=self.paper_positions,
                session_id=self.paper_session_id
            )
        )

        # Execute live if enabled
        if self.live_enabled and self.live_executor and self.live_session_id:
            tasks.append(
                self._execute_action_single(
                    cid, market, action, state, probs, confidence,
                    mode='live',
                    executor=self.live_executor,
                    positions=self.live_positions,
                    session_id=self.live_session_id
                )
            )

        # Run in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for live failures
        if len(results) > 1 and isinstance(results[1], Exception):
            logger.error(f"[LIVE] Execution failed: {results[1]}")
            self.live_failure_count += 1

            if self.live_failure_count >= self.live_max_failures:
                logger.error(f"[LIVE] Disabled after {self.live_max_failures} failures")
                self.live_enabled = False
                await self.discord.send_error(
                    message=f"⚠️ Live trading disabled after {self.live_max_failures} failures",
                    details=f"Error: {results[1]}\n\nPaper mode continues running."
                )

    async def _execute_action_single(
        self,
        cid: str,
        market: Market,
        action: Action,
        state: MarketState,
        probs: np.ndarray,
        confidence: float,
        mode: str,
        executor,
        positions: Dict[str, Position],
        session_id: str
    ) -> None:
        """Execute trading action for a single mode (paper or live)."""
        pos = positions.get(cid)

        if action == Action.HOLD:
            return

        if action == Action.BUY and (not pos or pos.size == 0):
            # Open position
            amount = self.trade_size * 0.5
            binance_price = self.price_streamer.get_price(market.asset)

            # *** CRITICAL: Actually execute the order on Polymarket ***
            order = executor.place_market_order(
                token_id=market.token_up,
                amount=amount,
                side=OrderSide.BUY,
                asset=market.asset
            )

            # For live mode, verify order was successful
            if mode == 'live' and (not order or order.status != "matched"):
                logger.warning(
                    f"[LIVE] Order failed for {market.asset}: "
                    f"{order.status if order else 'no response'}"
                )
                return  # Don't record failed trades

            # Record to database with order tracking
            trade_id = await self.db.record_trade_open(
                session_id=session_id,
                condition_id=cid,
                asset=market.asset,
                entry_price=state.prob,
                entry_binance_price=binance_price,
                side="UP",
                size_dollars=amount,
                time_remaining=state.time_remaining,
                action_probs={"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])},
                market_state={"prob": state.prob, "spread": state.spread, "imbalance_l1": state.order_book_imbalance_l1},
                order_id=order.order_id if order else None,
                execution_type=order.execution_type if order else "paper",
                fill_status=order.status if order else None,
                clob_response=order.clob_response if order else None
            )

            # Update local state
            positions[cid] = Position(
                asset=market.asset,
                side="UP",
                size=amount,
                entry_price=state.prob,
                entry_binance_price=binance_price,
                token_id=market.token_up,
                entry_time=datetime.now(timezone.utc),
                condition_id=cid,
                trade_id=trade_id,
                action_probs={"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])}
            )

            # Update mode-specific stats
            if mode == 'paper':
                self.paper_trade_count += 1
                session_pnl = self.paper_total_pnl
            else:
                self.live_trade_count += 1
                session_pnl = self.live_total_pnl

            logger.info(
                f"[{mode.upper()}] BUY {market.asset} UP @ {state.prob*100:.1f}% | "
                f"${amount:.2f} | conf={confidence*100:.0f}%"
            )

            # Discord alert (only for live or paper if not dual mode)
            if mode == 'live' or not self.dual_mode:
                await self.discord.send_trade_open(
                    asset=market.asset,
                    side="UP",
                    entry_price=state.prob,
                    size=amount,
                    confidence=confidence,
                    session_pnl=session_pnl
                )

        elif action == Action.SELL and pos and pos.size > 0:
            # Close position
            if pos.side == "UP":
                pnl = (state.prob - pos.entry_price) * (pos.size / pos.entry_price)
            else:
                pnl = ((1 - state.prob) - pos.entry_price) * (pos.size / pos.entry_price)

            duration = int((datetime.now(timezone.utc) - pos.entry_time).total_seconds()) if pos.entry_time else 0
            binance_price = self.price_streamer.get_price(market.asset)

            # *** CRITICAL: Actually execute the sell order on Polymarket ***
            # Calculate shares to sell (amount / entry_price gives shares)
            shares_to_sell = pos.size / pos.entry_price if pos.entry_price > 0 else 0
            if shares_to_sell > 0:
                order = executor.place_market_order(
                    token_id=pos.token_id,
                    amount=shares_to_sell,  # For SELL, amount is shares
                    side=OrderSide.SELL,
                    asset=market.asset
                )

                # For live mode, verify order was successful
                if mode == 'live' and (not order or order.status != "matched"):
                    logger.warning(
                        f"[LIVE] Sell order failed for {market.asset}: "
                        f"{order.status if order else 'no response'}"
                    )
                    return  # Don't record failed trades

            # Record to database
            if pos.trade_id:
                await self.db.record_trade_close(
                    trade_id=pos.trade_id,
                    exit_price=state.prob,
                    exit_binance_price=binance_price,
                    exit_reason="signal",
                    pnl=pnl,
                    duration_seconds=duration
                )

            # Update mode-specific stats
            if mode == 'paper':
                self.paper_total_pnl += pnl
                if pnl > 0:
                    self.paper_win_count += 1
                session_pnl = self.paper_total_pnl
                trade_count = self.paper_trade_count
                win_count = self.paper_win_count
            else:
                self.live_total_pnl += pnl
                if pnl > 0:
                    self.live_win_count += 1
                session_pnl = self.live_total_pnl
                trade_count = self.live_trade_count
                win_count = self.live_win_count

            win_rate = win_count / trade_count if trade_count > 0 else 0

            logger.info(
                f"[{mode.upper()}] SELL {market.asset} @ {state.prob*100:.1f}% | "
                f"PnL: {'+'if pnl >= 0 else ''}{pnl:.2f} | {duration}s | "
                f"Total: ${session_pnl:.2f}"
            )

            # Discord alert (only for live or paper if not dual mode)
            if mode == 'live' or not self.dual_mode:
                await self.discord.send_trade_close(
                    asset=market.asset,
                    side=pos.side,
                    pnl=pnl,
                    duration=duration,
                    entry_price=pos.entry_price,
                    exit_price=state.prob,
                    session_pnl=session_pnl,
                    session_trades=trade_count,
                    session_win_rate=win_rate
                )

            # Check milestones (shared across modes)
            if mode == 'paper':
                await self._check_milestones()

            # Clear position
            positions[cid] = Position(asset=market.asset)

    async def _check_milestones(self) -> None:
        """Check and alert on PnL milestones (paper mode only - NO combining)."""
        milestones = [100, 250, 500, 1000, 2500, 5000, 10000]

        for m in milestones:
            if self.paper_total_pnl >= m and m not in self.milestones_hit:
                self.milestones_hit.add(m)
                await self.discord.send_milestone(self.paper_total_pnl, m)

    # === SAFETY SYSTEM METHODS ===

    async def _position_safety_loop(self) -> None:
        """
        Background task: Check positions for timeout/expiry every 10 seconds.

        CRITICAL SAFETY: This prevents positions from being held until market
        expiration when orderbook data fails or system becomes unhealthy.
        """
        logger.info("[SAFETY] Position safety loop started")
        while self.running:
            await asyncio.sleep(10)
            try:
                now = datetime.now(timezone.utc)

                # Count positions checked
                live_checked = 0
                live_closed = 0
                paper_checked = 0
                paper_closed = 0

                # Check all live positions first (MOST IMPORTANT)
                for cid, pos in list(self.live_positions.items()):
                    if pos.size <= 0:
                        continue

                    live_checked += 1
                    market = self.markets.get(cid)
                    if not market:
                        logger.warning(f"[SAFETY] LIVE {pos.asset}: Market not found (cid={cid[:8]}...)")
                        continue

                    # Check if position needs emergency close
                    should_close, reason = self._should_emergency_close(pos, market, now)
                    if should_close:
                        logger.critical(f"[SAFETY] 🚨 TIMEOUT TRIGGERED: LIVE {pos.asset} - {reason}")
                        live_closed += 1
                        await self._emergency_close_position(
                            cid, market, pos,
                            mode='live',
                            executor=self.live_executor,
                            positions=self.live_positions,
                            session_id=self.live_session_id,
                            reason=reason
                        )
                    else:
                        # Log healthy state periodically (every minute)
                        if int(now.timestamp()) % 60 < 10:
                            time_to_expiry = (market.end_time - now).total_seconds()
                            age = (now - pos.entry_time).total_seconds() if pos.entry_time else 0
                            logger.debug(
                                f"[SAFETY] LIVE {pos.asset} healthy: "
                                f"T-{time_to_expiry/60:.1f}min to expiry, age={age:.0f}s"
                            )

                # Check paper positions
                for cid, pos in list(self.paper_positions.items()):
                    if pos.size <= 0:
                        continue

                    paper_checked += 1
                    market = self.markets.get(cid)
                    if not market:
                        logger.warning(f"[SAFETY] PAPER {pos.asset}: Market not found (cid={cid[:8]}...)")
                        continue

                    should_close, reason = self._should_emergency_close(pos, market, now)
                    if should_close:
                        logger.warning(f"[SAFETY] TIMEOUT: PAPER {pos.asset} - {reason}")
                        paper_closed += 1
                        await self._emergency_close_position(
                            cid, market, pos,
                            mode='paper',
                            executor=self.paper_executor,
                            positions=self.paper_positions,
                            session_id=self.paper_session_id,
                            reason=reason
                        )

                # Check orderbook health
                await self._check_orderbook_health()

                # Log summary every minute
                if live_checked > 0 or paper_checked > 0:
                    if int(now.timestamp()) % 60 < 10:
                        logger.info(
                            f"[SAFETY] Checked {live_checked} LIVE + {paper_checked} PAPER positions | "
                            f"Closed: {live_closed} LIVE + {paper_closed} PAPER"
                        )

            except Exception as e:
                logger.error(f"[SAFETY] Position safety check error: {e}", exc_info=True)

    def _should_emergency_close(
        self,
        pos: Position,
        market: Market,
        now: datetime
    ) -> tuple[bool, str]:
        """
        Determine if a position should be emergency closed.

        Returns: (should_close, reason)
        """
        # 1. Market expiry timeout (2 minutes before end)
        time_to_expiry = (market.end_time - now).total_seconds()
        if time_to_expiry < self.position_timeout_minutes * 60:
            return True, f"market_expiry (expires in {time_to_expiry:.0f}s)"

        # 2. Position age timeout (max 14 minutes for 15-min markets)
        if pos.entry_time:
            position_age = (now - pos.entry_time).total_seconds()
            if position_age > self.position_max_age_seconds:
                return True, f"max_age ({position_age:.0f}s > {self.position_max_age_seconds}s)"

        # 3. System unhealthy (orderbook data stale for this market)
        cid = pos.condition_id
        if cid and cid in self.orderbook_last_update:
            last_update = self.orderbook_last_update[cid]
            staleness = (now - last_update).total_seconds()
            if staleness > self.orderbook_stale_threshold:
                return True, f"stale_data ({staleness:.0f}s without update)"

        return False, ""

    async def _emergency_close_position(
        self,
        cid: str,
        market: Market,
        pos: Position,
        mode: str,
        executor,
        positions: Dict[str, Position],
        session_id: str,
        reason: str
    ) -> None:
        """
        Emergency close a position at market price.

        Called by safety system when position must be closed immediately.
        """
        try:
            # Get current price (try orderbook first, fallback to entry price)
            ob_up = self.orderbook_streamer.get_orderbook(cid, "UP")
            exit_price = ob_up.mid_price if ob_up and ob_up.mid_price else pos.entry_price

            # Calculate PnL
            if pos.side == "UP":
                pnl = (exit_price - pos.entry_price) * (pos.size / pos.entry_price)
            else:
                pnl = ((1 - exit_price) - pos.entry_price) * (pos.size / pos.entry_price)

            duration = int((datetime.now(timezone.utc) - pos.entry_time).total_seconds()) if pos.entry_time else 0
            binance_price = self.price_streamer.get_price(market.asset)

            # Execute the sell order
            shares_to_sell = pos.size / pos.entry_price if pos.entry_price > 0 else 0
            if shares_to_sell > 0 and executor:
                try:
                    order = executor.place_market_order(
                        token_id=pos.token_id,
                        amount=shares_to_sell,
                        side=OrderSide.SELL,
                        asset=market.asset
                    )
                    logger.info(f"[{mode.upper()}] Emergency sell executed: {order.status if order else 'submitted'}")
                except Exception as e:
                    logger.error(f"[{mode.upper()}] Emergency sell order failed: {e}")
                    # Continue anyway to update state - position expiring is worse than tracking error

            # Record to database
            if pos.trade_id:
                await self.db.record_trade_close(
                    trade_id=pos.trade_id,
                    exit_price=exit_price,
                    exit_binance_price=binance_price,
                    exit_reason=f"emergency:{reason}",
                    pnl=pnl,
                    duration_seconds=duration
                )

            # Update stats
            if mode == 'paper':
                self.paper_total_pnl += pnl
                if pnl > 0:
                    self.paper_win_count += 1
            else:
                self.live_total_pnl += pnl
                if pnl > 0:
                    self.live_win_count += 1

            logger.warning(
                f"[{mode.upper()}] EMERGENCY CLOSE {market.asset} | reason={reason} | "
                f"PnL: {'+'if pnl >= 0 else ''}{pnl:.2f}"
            )

            # Discord alert for emergency close
            await self.discord.send_error(
                message=f"⚠️ Emergency Position Close ({mode.upper()})",
                details=(
                    f"**Asset:** {market.asset}\n"
                    f"**Reason:** {reason}\n"
                    f"**Entry:** {pos.entry_price*100:.1f}%\n"
                    f"**Exit:** {exit_price*100:.1f}%\n"
                    f"**PnL:** ${pnl:+.2f}\n"
                    f"**Duration:** {duration}s"
                )
            )

            # Clear position
            positions[cid] = Position(asset=market.asset)

        except Exception as e:
            logger.error(f"[SAFETY] Emergency close failed for {market.asset}: {e}")
            # Last resort: clear local state anyway to prevent double-close attempts
            positions[cid] = Position(asset=market.asset)

    async def _check_orderbook_health(self) -> None:
        """
        Check orderbook health across all markets.
        Alert if system becomes unhealthy.
        """
        now = datetime.now(timezone.utc)
        stale_markets = []

        for cid in self.markets:
            if cid in self.orderbook_last_update:
                staleness = (now - self.orderbook_last_update[cid]).total_seconds()
                if staleness > self.orderbook_stale_threshold:
                    stale_markets.append(cid)

        # Update system health
        was_healthy = self.system_healthy
        self.system_healthy = len(stale_markets) < len(self.markets) / 2  # <50% stale = healthy

        # Alert on health state change
        if was_healthy and not self.system_healthy:
            logger.error(f"[SAFETY] System UNHEALTHY: {len(stale_markets)}/{len(self.markets)} markets stale")
            await self.discord.send_error(
                message="🚨 Trading System Unhealthy",
                details=(
                    f"**Stale Markets:** {len(stale_markets)}/{len(self.markets)}\n"
                    f"**Threshold:** {self.orderbook_stale_threshold}s\n"
                    f"**Action:** Emergency closing positions and pausing new trades"
                )
            )
        elif not was_healthy and self.system_healthy:
            logger.info(f"[SAFETY] System recovered: orderbook data restored")
            await self.discord.send_startup(
                mode="RECOVERED",
                trade_size=self.trade_size,
                model_version="System health restored"
            )

    def update_orderbook_health(self, cid: str) -> None:
        """Mark orderbook as recently updated (called when we get valid data)."""
        self.orderbook_last_update[cid] = datetime.now(timezone.utc)

    async def emergency_close_all(self, reason: str = "manual") -> None:
        """
        Emergency close ALL open positions.
        Called when system needs immediate shutdown or critical failure.
        """
        logger.warning(f"[SAFETY] EMERGENCY CLOSE ALL: {reason}")

        # Close live positions first (real money)
        if self.live_executor and self.live_session_id:
            for cid, pos in list(self.live_positions.items()):
                if pos.size > 0:
                    market = self.markets.get(cid)
                    if market:
                        await self._emergency_close_position(
                            cid, market, pos,
                            mode='live',
                            executor=self.live_executor,
                            positions=self.live_positions,
                            session_id=self.live_session_id,
                            reason=reason
                        )

        # Then paper positions
        if self.paper_session_id:
            for cid, pos in list(self.paper_positions.items()):
                if pos.size > 0:
                    market = self.markets.get(cid)
                    if market:
                        await self._emergency_close_position(
                            cid, market, pos,
                            mode='paper',
                            executor=self.paper_executor,
                            positions=self.paper_positions,
                            session_id=self.paper_session_id,
                            reason=reason
                        )

    async def _checkpoint_loop(self) -> None:
        """Background task: checkpoint every 30 seconds."""
        while self.running:
            await asyncio.sleep(30)
            try:
                await self.checkpoint()
            except Exception as e:
                logger.error(f"Checkpoint failed: {e}")

    async def _metrics_loop(self) -> None:
        """Background task: record metrics every minute (separate for paper and live)."""
        while self.running:
            await asyncio.sleep(60)
            try:
                markets_data = {
                    a: {"prob": self.market_states.get(cid, MarketState(a, 0.5, 0)).prob}
                    for cid, m in self.markets.items()
                    for a in [m.asset]
                }

                # Record paper metrics
                if self.paper_session_id:
                    paper_open_pos = len([p for p in self.paper_positions.values() if p.size > 0])
                    paper_exposure = sum(p.size for p in self.paper_positions.values() if p.size > 0)
                    await self.db.record_metrics(
                        session_id=self.paper_session_id,
                        cumulative_pnl=self.paper_total_pnl,
                        trades_today=self.paper_trade_count,
                        win_rate_today=self.paper_win_count / max(1, self.paper_trade_count),
                        open_positions=paper_open_pos,
                        total_exposure=paper_exposure,
                        active_markets=len(self.markets),
                        markets_data=markets_data
                    )

                # Record live metrics (if enabled)
                if self.live_session_id and self.live_enabled:
                    live_open_pos = len([p for p in self.live_positions.values() if p.size > 0])
                    live_exposure = sum(p.size for p in self.live_positions.values() if p.size > 0)
                    await self.db.record_metrics(
                        session_id=self.live_session_id,
                        cumulative_pnl=self.live_total_pnl,
                        trades_today=self.live_trade_count,
                        win_rate_today=self.live_win_count / max(1, self.live_trade_count),
                        open_positions=live_open_pos,
                        total_exposure=live_exposure,
                        active_markets=len(self.markets),
                        markets_data=markets_data
                    )
            except Exception as e:
                logger.error(f"Metrics recording failed: {e}")

    async def _daily_summary_loop(self) -> None:
        """Background task: send daily summary at midnight UTC."""
        while self.running:
            # Calculate time until midnight UTC
            now = datetime.now(timezone.utc)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            seconds_until_midnight = (tomorrow - now).total_seconds()

            await asyncio.sleep(seconds_until_midnight)

            try:
                # Send paper summary
                if self.paper_session_id:
                    paper_stats = await self.db.get_session_stats(self.paper_session_id)
                    await self.discord.send_daily_summary(
                        pnl=paper_stats["total_pnl"],
                        trades=paper_stats["total_trades"],
                        win_rate=paper_stats["win_rate"],
                        best_trade=paper_stats["best_trade"],
                        worst_trade=paper_stats["worst_trade"],
                        exposure_pct=0,
                        mode="PAPER"
                    )

                # Send live summary (if enabled)
                if self.live_session_id and self.live_enabled:
                    live_stats = await self.db.get_session_stats(self.live_session_id)
                    await self.discord.send_daily_summary(
                        pnl=live_stats["total_pnl"],
                        trades=live_stats["total_trades"],
                        win_rate=live_stats["win_rate"],
                        best_trade=live_stats["best_trade"],
                        worst_trade=live_stats["worst_trade"],
                        exposure_pct=0,
                        mode="LIVE"
                    )
            except Exception as e:
                logger.error(f"Daily summary failed: {e}")

    async def run(self) -> None:
        """Main trading loop."""
        self.running = True

        try:
            await self.initialize()
            await self.refresh_markets()

            # Start data streams
            stream_tasks = [
                asyncio.create_task(self.orderbook_streamer.stream()),
                asyncio.create_task(self.price_streamer.stream()),
                asyncio.create_task(self.futures_streamer.stream()),
                asyncio.create_task(self._checkpoint_loop()),
                asyncio.create_task(self._metrics_loop()),
                asyncio.create_task(self._daily_summary_loop()),
                asyncio.create_task(self._position_safety_loop()),  # CRITICAL: Position safety monitor
            ]

            # Add profit monitor if enabled
            if self.profit_transfer_enabled and self.profit_monitor:
                stream_tasks.append(asyncio.create_task(self.profit_monitor.start()))

            last_refresh = datetime.now(timezone.utc)
            last_status = datetime.now(timezone.utc)

            while self.running and not self.shutdown_event.is_set():
                now = datetime.now(timezone.utc)

                # Refresh markets every 30s
                if (now - last_refresh).total_seconds() > 30:
                    await self.refresh_markets()
                    last_refresh = now

                # Log status every 60s (separate paper and live - NO combining)
                if (now - last_status).total_seconds() > 60:
                    paper_wr = self.paper_win_count / max(1, self.paper_trade_count)
                    if self.dual_mode or self.live_enabled:
                        live_wr = self.live_win_count / max(1, self.live_trade_count)
                        logger.info(
                            f"Status [DUAL]: Paper PnL=${self.paper_total_pnl:.2f} (trades={self.paper_trade_count}, win={paper_wr*100:.0f}%) | "
                            f"Live PnL=${self.live_total_pnl:.2f} (trades={self.live_trade_count}, win={live_wr*100:.0f}%)"
                        )
                    else:
                        logger.info(f"Status: PnL=${self.paper_total_pnl:.2f} | Trades={self.paper_trade_count} | Win={paper_wr*100:.0f}%")
                    last_status = now

                # Process each market
                for cid, market in list(self.markets.items()):
                    try:
                        ob_up = self.orderbook_streamer.get_orderbook(cid, "UP")
                        if not ob_up or not ob_up.mid_price:
                            continue

                        # SAFETY: Mark orderbook as healthy for this market
                        self.update_orderbook_health(cid)

                        state = self.update_market_state(cid, market, ob_up)

                        # Skip if market nearly expired
                        if state.time_remaining < 0.05:
                            continue

                        # SAFETY: Skip new trades if system unhealthy (but still allow closes)
                        if not self.system_healthy:
                            continue

                        action, probs, confidence = self.get_rl_action(state)

                        # Execute on both modes if dual-mode enabled
                        if self.dual_mode or self.live_enabled:
                            await self.execute_action_dual_mode(cid, market, action, state, probs, confidence)
                        else:
                            await self._execute_action_single(
                                cid, market, action, state, probs, confidence,
                                mode='paper',
                                executor=self.paper_executor,
                                positions=self.paper_positions,
                                session_id=self.paper_session_id
                            )

                    except Exception as e:
                        logger.error(f"Error processing {market.asset}: {e}")

                await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await self.discord.send_error(str(e))

        finally:
            await self.shutdown(stream_tasks if 'stream_tasks' in locals() else [])

    async def shutdown(self, tasks: list) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self.running = False

        # SAFETY: Emergency close all open positions before shutdown
        try:
            await self.emergency_close_all(reason="shutdown")
        except Exception as e:
            logger.error(f"Emergency close on shutdown failed: {e}")

        # Stop streams
        self.orderbook_streamer.stop()
        self.price_streamer.stop()
        self.futures_streamer.stop()

        # Stop profit monitor
        if self.profit_monitor:
            self.profit_monitor.stop()

        # Cancel background tasks
        for task in tasks:
            task.cancel()

        # Final checkpoint
        try:
            await self.checkpoint()
        except Exception as e:
            logger.error(f"Final checkpoint failed: {e}")

        # Mark sessions as stopped (both paper and live)
        try:
            if self.paper_session_id:
                await self.db.end_session(self.paper_session_id, "stopped")
                logger.info("[PAPER] Session ended")
            if self.live_session_id:
                await self.db.end_session(self.live_session_id, "stopped")
                logger.info("[LIVE] Session ended")
        except Exception as e:
            logger.error(f"Failed to end sessions: {e}")

        # Close connections
        await self.db.close()
        await self.discord.close()

        logger.info("Shutdown complete")


def main():
    """Entry point."""
    worker = TradingWorker()

    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        logger.info("Received shutdown signal")
        worker.shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        loop.run_until_complete(worker.run())
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
