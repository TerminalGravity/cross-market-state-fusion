#!/usr/bin/env python3
"""
Background paper trading with RL agent.
Logs all trades and PnL to a file for monitoring.
"""
import asyncio
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.polymarket_api import get_15m_markets, Market
from helpers.orderbook_wss import OrderbookStreamer, OrderbookState
from helpers.binance_wss import BinanceStreamer
from helpers.binance_futures import FuturesStreamer
from helpers.clob_executor import ClobExecutor, ExecutionMode, OrderSide, create_executor
from strategies.base import MarketState, Action
from strategies.rl_mlx import RLStrategy


@dataclass
class Position:
    asset: str
    side: str = ""
    size: float = 0.0
    entry_price: float = 0.0
    token_id: str = ""
    entry_time: Optional[datetime] = None


class PaperTrader:
    def __init__(self, trade_size: float = 50.0):
        self.trade_size = trade_size
        self.executor = create_executor(live=False)

        # Load RL strategy
        self.strategy = RLStrategy()
        self.strategy.load("rl_model")
        self.strategy.training = False

        # Data sources
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
        self.running = False

        # Logging
        self.log_file = open("paper_trading.log", "a")
        self.log(f"\n{'='*60}")
        self.log(f"Paper Trading Session Started")
        self.log(f"Trade Size: ${trade_size}")
        self.log(f"{'='*60}")

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def update_market_state(self, cid: str, market: Market, ob_up: OrderbookState) -> MarketState:
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
        import mlx.core as mx

        features = state.to_features()
        features_mx = mx.array(features.reshape(1, -1))

        probs = self.strategy.actor(features_mx)
        probs_np = np.array(probs[0])

        action_idx = int(np.argmax(probs_np))
        action = Action(action_idx)
        confidence = np.max(probs_np) - np.partition(probs_np, -2)[-2]

        return action, probs_np, confidence

    def execute_action(self, cid: str, market: Market, action: Action, state: MarketState, probs: np.ndarray, confidence: float):
        pos = self.positions.get(cid)

        if action == Action.HOLD:
            return

        if action == Action.BUY and (not pos or pos.size == 0):
            amount = self.trade_size * 0.5
            self.positions[cid] = Position(
                asset=market.asset,
                side="UP",
                size=amount,
                entry_price=state.prob,
                token_id=market.token_up,
                entry_time=datetime.now(timezone.utc)
            )
            self.trade_count += 1
            self.log(f"📈 BUY {market.asset} UP @ {state.prob*100:.1f}% | ${amount:.2f} | conf={confidence*100:.0f}%")
            self.log(f"   Probs: HOLD={probs[0]*100:.1f}% BUY={probs[1]*100:.1f}% SELL={probs[2]*100:.1f}%")

        elif action == Action.SELL and pos and pos.size > 0:
            # Close position
            if pos.side == "UP":
                pnl = (state.prob - pos.entry_price) * (pos.size / pos.entry_price)
            else:
                pnl = ((1 - state.prob) - pos.entry_price) * (pos.size / pos.entry_price)

            self.total_pnl += pnl
            if pnl > 0:
                self.win_count += 1

            duration = (datetime.now(timezone.utc) - pos.entry_time).total_seconds() if pos.entry_time else 0
            win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0

            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            self.log(f"📉 SELL {market.asset} @ {state.prob*100:.1f}% | PnL: {pnl_str} | {duration:.0f}s")
            self.log(f"   Total PnL: ${self.total_pnl:.2f} | Trades: {self.trade_count} | Win: {win_rate:.0f}%")

            self.positions[cid] = Position(asset=market.asset)

    async def refresh_markets(self):
        markets = get_15m_markets(assets=self.assets)

        if not markets:
            return

        self.markets = {m.condition_id: m for m in markets}

        for m in markets:
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)
            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0 and m.condition_id not in self.open_prices:
                self.open_prices[m.condition_id] = current_price

        self.log(f"Found {len(markets)} active markets: {', '.join(m.asset for m in markets)}")

    async def run(self):
        self.running = True

        await self.refresh_markets()

        ob_task = asyncio.create_task(self.orderbook_streamer.stream())
        price_task = asyncio.create_task(self.price_streamer.stream())
        futures_task = asyncio.create_task(self.futures_streamer.stream())

        try:
            last_refresh = datetime.now(timezone.utc)
            last_status = datetime.now(timezone.utc)

            while self.running:
                now = datetime.now(timezone.utc)

                # Refresh markets every 30s
                if (now - last_refresh).total_seconds() > 30:
                    await self.refresh_markets()
                    last_refresh = now

                # Status update every 60s
                if (now - last_status).total_seconds() > 60:
                    win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
                    self.log(f"📊 Status: PnL=${self.total_pnl:.2f} | Trades={self.trade_count} | Win={win_rate:.0f}%")
                    last_status = now

                # Process each market
                for cid, market in self.markets.items():
                    ob_up = self.orderbook_streamer.get_orderbook(cid, "UP")
                    if not ob_up or not ob_up.mid_price:
                        continue

                    state = self.update_market_state(cid, market, ob_up)

                    # Skip if market nearly expired
                    if state.time_remaining < 0.05:
                        continue

                    action, probs, confidence = self.get_rl_action(state)
                    self.execute_action(cid, market, action, state, probs, confidence)

                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            self.log("Shutting down...")
        finally:
            self.running = False
            self.orderbook_streamer.stop()
            self.price_streamer.stop()
            self.futures_streamer.stop()
            ob_task.cancel()
            price_task.cancel()
            futures_task.cancel()

            win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
            self.log(f"\n{'='*60}")
            self.log(f"Session Ended")
            self.log(f"Final PnL: ${self.total_pnl:.2f}")
            self.log(f"Total Trades: {self.trade_count}")
            self.log(f"Win Rate: {win_rate:.0f}%")
            self.log(f"{'='*60}")
            self.log_file.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float, default=50.0, help="Trade size in $")
    args = parser.parse_args()

    trader = PaperTrader(trade_size=args.size)
    asyncio.run(trader.run())
