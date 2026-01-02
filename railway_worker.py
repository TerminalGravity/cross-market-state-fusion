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
import json

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.connection import Database
from helpers.discord import DiscordWebhook
from helpers.polymarket_api import get_15m_markets, Market
from helpers.orderbook_wss import OrderbookStreamer, OrderbookState
from helpers.binance_wss import BinanceStreamer
from helpers.binance_futures import FuturesStreamer
from helpers.clob_executor import create_executor
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
        self.live_mode = os.getenv("TRADING_MODE", "paper") == "live"
        self.model_path = os.getenv("MODEL_PATH", "rl_model")

        # Components
        self.db = Database()
        self.discord = DiscordWebhook()
        self.session_id: Optional[str] = None

        # Strategy (NumPy or MLX depending on platform)
        logger.info(f"Loading strategy (MLX={USE_MLX})")
        self.strategy = RLStrategy(self.model_path) if not USE_MLX else None

        # Executor (paper or live)
        self.executor = create_executor(live=self.live_mode)

        # Data streams
        self.assets = ["BTC", "ETH", "SOL", "XRP"]
        self.orderbook_streamer = OrderbookStreamer()
        self.price_streamer = BinanceStreamer(self.assets)
        self.futures_streamer = FuturesStreamer(self.assets)

        # State
        self.markets: Dict[str, Market] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.positions: Dict[str, Position] = {}
        self.open_prices: Dict[str, float] = {}

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.milestones_hit = set()

        # Control
        self.running = False
        self.shutdown_event = asyncio.Event()

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

        # Try to recover from crashed session
        recovered = await self._recover_session()

        if not recovered:
            # Create new session
            self.session_id = await self.db.create_session(
                mode="live" if self.live_mode else "paper",
                trade_size=self.trade_size,
                model_version=f"rl_model_{'mlx' if USE_MLX else 'numpy'}",
                config={
                    "assets": self.assets,
                    "use_mlx": USE_MLX,
                }
            )
            logger.info(f"Created new session: {self.session_id}")

            # Send startup notification
            await self.discord.send_startup(
                mode="live" if self.live_mode else "paper",
                trade_size=self.trade_size,
                model_version=f"rl_model ({'MLX' if USE_MLX else 'NumPy'})"
            )

    async def _recover_session(self) -> bool:
        """Attempt to recover from crashed session."""
        session = await self.db.get_active_session()

        if not session or not session.get("checkpoint_data"):
            return False

        logger.info(f"Recovering session {session['id']}")
        self.session_id = str(session["id"])
        self.total_pnl = float(session["total_pnl"] or 0)
        self.trade_count = session["trade_count"] or 0
        self.win_count = session["win_count"] or 0

        # Restore positions
        checkpoint = session["checkpoint_data"]
        if "positions" in checkpoint:
            for cid, pos_data in checkpoint["positions"].items():
                if pos_data.get("size", 0) > 0:
                    self.positions[cid] = Position.from_dict(pos_data)

        # Send recovery notification
        await self.discord.send_recovery(
            session_id=self.session_id,
            recovered_pnl=self.total_pnl,
            recovered_trades=self.trade_count,
            open_positions=len([p for p in self.positions.values() if p.size > 0])
        )

        return True

    async def checkpoint(self) -> None:
        """Save current state for crash recovery."""
        if not self.session_id:
            return

        checkpoint_data = {
            "positions": {
                cid: pos.to_dict()
                for cid, pos in self.positions.items()
                if pos.size > 0
            },
            "open_prices": self.open_prices,
        }

        await self.db.update_session_checkpoint(
            session_id=self.session_id,
            checkpoint_data=checkpoint_data,
            total_pnl=self.total_pnl,
            trade_count=self.trade_count,
            win_count=self.win_count
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

            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

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

        # Position state
        pos = self.positions.get(cid)
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

    async def execute_action(
        self,
        cid: str,
        market: Market,
        action: Action,
        state: MarketState,
        probs: np.ndarray,
        confidence: float
    ) -> None:
        """Execute trading action."""
        pos = self.positions.get(cid)

        if action == Action.HOLD:
            return

        if action == Action.BUY and (not pos or pos.size == 0):
            # Open position
            amount = self.trade_size * 0.5
            binance_price = self.price_streamer.get_price(market.asset)

            # Record to database
            trade_id = await self.db.record_trade_open(
                session_id=self.session_id,
                condition_id=cid,
                asset=market.asset,
                entry_price=state.prob,
                entry_binance_price=binance_price,
                side="UP",
                size_dollars=amount,
                time_remaining=state.time_remaining,
                action_probs={"hold": float(probs[0]), "buy": float(probs[1]), "sell": float(probs[2])},
                market_state={"prob": state.prob, "spread": state.spread, "imbalance_l1": state.order_book_imbalance_l1}
            )

            # Update local state
            self.positions[cid] = Position(
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

            self.trade_count += 1

            logger.info(
                f"BUY {market.asset} UP @ {state.prob*100:.1f}% | "
                f"${amount:.2f} | conf={confidence*100:.0f}%"
            )

            # Discord alert
            await self.discord.send_trade_open(
                asset=market.asset,
                side="UP",
                entry_price=state.prob,
                size=amount,
                confidence=confidence,
                session_pnl=self.total_pnl
            )

        elif action == Action.SELL and pos and pos.size > 0:
            # Close position
            if pos.side == "UP":
                pnl = (state.prob - pos.entry_price) * (pos.size / pos.entry_price)
            else:
                pnl = ((1 - state.prob) - pos.entry_price) * (pos.size / pos.entry_price)

            duration = int((datetime.now(timezone.utc) - pos.entry_time).total_seconds()) if pos.entry_time else 0
            binance_price = self.price_streamer.get_price(market.asset)

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

            # Update stats
            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1

            win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0

            logger.info(
                f"SELL {market.asset} @ {state.prob*100:.1f}% | "
                f"PnL: {'+'if pnl >= 0 else ''}{pnl:.2f} | {duration}s | "
                f"Total: ${self.total_pnl:.2f}"
            )

            # Discord alert
            await self.discord.send_trade_close(
                asset=market.asset,
                side=pos.side,
                pnl=pnl,
                duration=duration,
                entry_price=pos.entry_price,
                exit_price=state.prob,
                session_pnl=self.total_pnl,
                session_trades=self.trade_count,
                session_win_rate=win_rate
            )

            # Check milestones
            await self._check_milestones()

            # Clear position
            self.positions[cid] = Position(asset=market.asset)

    async def _check_milestones(self) -> None:
        """Check and alert on PnL milestones."""
        milestones = [100, 250, 500, 1000, 2500, 5000, 10000]

        for m in milestones:
            if self.total_pnl >= m and m not in self.milestones_hit:
                self.milestones_hit.add(m)
                await self.discord.send_milestone(self.total_pnl, m)

    async def _checkpoint_loop(self) -> None:
        """Background task: checkpoint every 30 seconds."""
        while self.running:
            await asyncio.sleep(30)
            try:
                await self.checkpoint()
            except Exception as e:
                logger.error(f"Checkpoint failed: {e}")

    async def _metrics_loop(self) -> None:
        """Background task: record metrics every minute."""
        while self.running:
            await asyncio.sleep(60)
            try:
                open_positions = len([p for p in self.positions.values() if p.size > 0])
                total_exposure = sum(p.size for p in self.positions.values() if p.size > 0)

                await self.db.record_metrics(
                    session_id=self.session_id,
                    cumulative_pnl=self.total_pnl,
                    trades_today=self.trade_count,
                    win_rate_today=self.win_count / max(1, self.trade_count),
                    open_positions=open_positions,
                    total_exposure=total_exposure,
                    active_markets=len(self.markets),
                    markets_data={
                        a: {"prob": self.market_states.get(cid, MarketState(a, 0.5, 0)).prob}
                        for cid, m in self.markets.items()
                        for a in [m.asset]
                    }
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
                stats = await self.db.get_session_stats(self.session_id)
                await self.discord.send_daily_summary(
                    pnl=stats["total_pnl"],
                    trades=stats["total_trades"],
                    win_rate=stats["win_rate"],
                    best_trade=stats["best_trade"],
                    worst_trade=stats["worst_trade"],
                    exposure_pct=0  # TODO: Calculate
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
            ]

            last_refresh = datetime.now(timezone.utc)
            last_status = datetime.now(timezone.utc)

            while self.running and not self.shutdown_event.is_set():
                now = datetime.now(timezone.utc)

                # Refresh markets every 30s
                if (now - last_refresh).total_seconds() > 30:
                    await self.refresh_markets()
                    last_refresh = now

                # Log status every 60s
                if (now - last_status).total_seconds() > 60:
                    win_rate = self.win_count / max(1, self.trade_count)
                    logger.info(f"Status: PnL=${self.total_pnl:.2f} | Trades={self.trade_count} | Win={win_rate*100:.0f}%")
                    last_status = now

                # Process each market
                for cid, market in list(self.markets.items()):
                    try:
                        ob_up = self.orderbook_streamer.get_orderbook(cid, "UP")
                        if not ob_up or not ob_up.mid_price:
                            continue

                        state = self.update_market_state(cid, market, ob_up)

                        # Skip if market nearly expired
                        if state.time_remaining < 0.05:
                            continue

                        action, probs, confidence = self.get_rl_action(state)
                        await self.execute_action(cid, market, action, state, probs, confidence)

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

        # Stop streams
        self.orderbook_streamer.stop()
        self.price_streamer.stop()
        self.futures_streamer.stop()

        # Cancel background tasks
        for task in tasks:
            task.cancel()

        # Final checkpoint
        try:
            await self.checkpoint()
        except Exception as e:
            logger.error(f"Final checkpoint failed: {e}")

        # Mark session as stopped
        if self.session_id:
            try:
                await self.db.end_session(self.session_id, "stopped")
            except Exception as e:
                logger.error(f"Failed to end session: {e}")

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
