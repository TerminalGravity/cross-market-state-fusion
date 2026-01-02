#!/usr/bin/env python3
"""
Terminal UI for Polymarket 15m trading with RL integration.

Displays live orderbook, RL agent signals, and trading metrics
in a rich terminal interface.

Usage:
    python terminal_ui.py                    # Paper trading, rule-based signals
    python terminal_ui.py --rl               # Paper trading, RL agent decisions
    python terminal_ui.py --rl --live        # Live trading with RL agent
    python terminal_ui.py --rl --load model  # Load specific RL model
    python terminal_ui.py --asset BTC        # Focus on single asset
"""
import asyncio
import argparse
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers.polymarket_api import get_15m_markets, Market
from helpers.orderbook_wss import OrderbookStreamer, OrderbookState
from helpers.binance_wss import BinanceStreamer
from helpers.binance_futures import FuturesStreamer
from helpers.clob_executor import ClobExecutor, ExecutionMode, OrderSide, create_executor
from strategies.base import MarketState, Action


# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"


@dataclass
class SignalConfig:
    """Trading signal configuration for rule-based mode."""
    entry_low: float = 0.10
    entry_high: float = 0.25
    take_profit: float = 0.30
    stop_loss_offset: float = 0.05


@dataclass
class Position:
    """Track a live position."""
    asset: str
    side: str = ""  # "UP" or "DOWN"
    size: float = 0.0
    entry_price: float = 0.0
    token_id: str = ""


@dataclass
class RLSignalState:
    """RL agent signal state."""
    action: Action = Action.HOLD
    action_probs: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    confidence: float = 0.0
    value_estimate: float = 0.0


class TerminalUI:
    """
    Rich terminal UI for Polymarket trading with RL integration.

    Displays:
    - Market info (asset, time remaining)
    - UP/DOWN probabilities
    - RL agent action probabilities and decision
    - Full orderbook depth (L10)
    - Volume and liquidity metrics
    """

    def __init__(
        self,
        executor: ClobExecutor,
        rl_strategy=None,
        signal_config: SignalConfig = None,
        assets: List[str] = None,
        trade_size: float = 50.0,
        refresh_rate: float = 0.5,
        auto_trade: bool = False,
    ):
        self.executor = executor
        self.rl_strategy = rl_strategy
        self.config = signal_config or SignalConfig()
        self.assets = assets or ["BTC", "ETH", "SOL", "XRP"]
        self.trade_size = trade_size
        self.refresh_rate = refresh_rate
        self.auto_trade = auto_trade

        # Data sources
        self.orderbook_streamer = OrderbookStreamer()
        self.price_streamer = BinanceStreamer(self.assets)
        self.futures_streamer = FuturesStreamer(self.assets)

        # State
        self.markets: Dict[str, Market] = {}
        self.market_states: Dict[str, MarketState] = {}
        self.positions: Dict[str, Position] = {}
        self.rl_signals: Dict[str, RLSignalState] = {}
        self.open_prices: Dict[str, float] = {}

        self.current_asset_idx = 0
        self.running = False
        self.last_ws_update: Optional[datetime] = None

        # Stats
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

    def clear_screen(self):
        print("\033[2J\033[H", end="")

    def draw_box(self, title: str, width: int = 60) -> str:
        c = Colors
        border = "─" * (width - 2)
        return f"{c.CYAN}┌{border}┐{c.RESET}\n{c.CYAN}│{c.RESET} {c.BOLD}{c.CYAN}{title}{c.RESET}"

    def format_prob(self, prob: float, is_up: bool = True) -> str:
        c = Colors
        pct = prob * 100
        color = c.GREEN if is_up else c.RED
        arrow = "▲" if is_up else "▼"
        return f"{color}{arrow} {pct:.1f}%{c.RESET}"

    def update_market_state(self, cid: str, market: Market, ob_up: OrderbookState):
        """Update MarketState with all live data sources."""
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

            # L1 imbalance
            if ob_up.bids and ob_up.asks:
                bid_vol_l1 = ob_up.bids[0][1] if ob_up.bids else 0
                ask_vol_l1 = ob_up.asks[0][1] if ob_up.asks else 0
                total_l1 = bid_vol_l1 + ask_vol_l1
                state.order_book_imbalance_l1 = (bid_vol_l1 - ask_vol_l1) / total_l1 if total_l1 > 0 else 0.0

            # L5 imbalance
            bid_vol_l5 = sum(s for _, s in ob_up.bids[:5])
            ask_vol_l5 = sum(s for _, s in ob_up.asks[:5])
            total_l5 = bid_vol_l5 + ask_vol_l5
            state.order_book_imbalance_l5 = (bid_vol_l5 - ask_vol_l5) / total_l5 if total_l5 > 0 else 0.0

        # Binance spot price
        binance_price = self.price_streamer.get_price(market.asset)
        if binance_price > 0:
            state.binance_price = binance_price
            open_price = self.open_prices.get(cid, binance_price)
            if open_price > 0:
                state.binance_change = (binance_price - open_price) / open_price

        # Binance futures data
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

        # Time remaining
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
                current_down_price = 1 - state.prob
                state.position_pnl = (current_down_price - pos.entry_price) * shares
        else:
            state.has_position = False
            state.position_side = None
            state.position_pnl = 0.0

        return state

    def get_rl_signal(self, cid: str, state: MarketState) -> RLSignalState:
        """Get RL agent's action for the current state."""
        if not self.rl_strategy:
            return RLSignalState()

        import mlx.core as mx

        features = state.to_features()
        features_mx = mx.array(features.reshape(1, -1))

        # Get action probabilities and value
        probs = self.rl_strategy.actor(features_mx)
        value = self.rl_strategy.critic(features_mx)

        # Force MLX to compute the lazy arrays
        probs_computed = mx.array(probs)
        value_computed = mx.array(value)

        probs_np = np.array(probs_computed[0])
        value_np = float(np.array(value_computed[0, 0]))

        # Get action (greedy in inference mode)
        action_idx = int(np.argmax(probs_np))
        action = Action(action_idx)

        # Confidence = max prob - second max
        sorted_probs = np.sort(probs_np)[::-1]
        confidence = sorted_probs[0] - sorted_probs[1]

        return RLSignalState(
            action=action,
            action_probs=probs_np,
            confidence=confidence,
            value_estimate=value_np
        )

    def execute_rl_action(self, cid: str, market: Market, signal: RLSignalState, state: MarketState):
        """Execute the RL agent's action."""
        if not self.auto_trade:
            return

        action = signal.action
        pos = self.positions.get(cid)

        if action == Action.HOLD:
            return

        if action == Action.BUY and (not pos or pos.size == 0):
            # Open UP position
            amount = self.trade_size * 0.5  # 50% sizing
            order = self.executor.place_market_order(
                token_id=market.token_up,
                amount=amount,
                side=OrderSide.BUY,
                asset=market.asset
            )
            if order and order.status in ("matched", "OPEN"):
                self.positions[cid] = Position(
                    asset=market.asset,
                    side="UP",
                    size=amount,
                    entry_price=state.prob,
                    token_id=market.token_up
                )
                self.trade_count += 1

        elif action == Action.SELL and pos and pos.size > 0:
            # Close position
            order = self.executor.place_market_order(
                token_id=pos.token_id,
                amount=pos.size,
                side=OrderSide.SELL,
                asset=market.asset
            )
            if order:
                # Calculate PnL
                if pos.side == "UP":
                    pnl = (state.prob - pos.entry_price) * (pos.size / pos.entry_price)
                else:
                    pnl = ((1 - state.prob) - pos.entry_price) * (pos.size / pos.entry_price)

                self.total_pnl += pnl
                if pnl > 0:
                    self.win_count += 1

                # Clear position
                self.positions[cid] = Position(asset=market.asset)

    def render_market(self, market: Market, ob_up: OrderbookState, ob_down: OrderbookState, state: MarketState) -> str:
        c = Colors
        lines = []

        # Time remaining
        now = datetime.now(timezone.utc)
        remaining = market.end_time - now
        mins = int(remaining.total_seconds() // 60)
        secs = int(remaining.total_seconds() % 60)
        time_color = c.GREEN if remaining.total_seconds() > 60 else c.YELLOW if remaining.total_seconds() > 30 else c.RED

        # Header
        lines.append(self.draw_box(f"{market.asset} Up/Down 15m", 60))
        lines.append("")

        # WebSocket status
        ws_status = f"{c.GREEN}●{c.RESET}" if self.last_ws_update else f"{c.RED}●{c.RESET}"
        ws_time = self.last_ws_update.strftime("%I:%M:%S %p") if self.last_ws_update else "Connecting..."
        lines.append(f"{ws_status} {c.DIM}WS{c.RESET} Last: {ws_time}")
        lines.append("")

        # Market info
        lines.append(f"{c.BOLD}{market.question}{c.RESET}")
        lines.append(f"Time: {time_color}{mins:02d}:{secs:02d}{c.RESET}")
        lines.append("")

        # Probabilities
        up_prob = ob_up.mid_price or market.price_up
        down_prob = ob_down.mid_price or market.price_down
        lines.append(f"{self.format_prob(up_prob, True)}    {self.format_prob(down_prob, False)}")
        lines.append("")

        # Signal Module (RL or rule-based)
        cid = market.condition_id
        if self.rl_strategy:
            signal = self.rl_signals.get(cid, RLSignalState())
            lines.append(self._render_rl_signal(signal, state))
        else:
            lines.append(self._render_rule_signal(up_prob))
        lines.append("")

        # Orderbooks
        lines.append(self._render_orderbooks(ob_up, ob_down))

        # Stats
        up_liq = sum(s for _, s in ob_up.bids[:5]) + sum(s for _, s in ob_up.asks[:5])
        down_liq = sum(s for _, s in ob_down.bids[:5]) + sum(s for _, s in ob_down.asks[:5])
        total_liq = up_liq + down_liq

        lines.append("")
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0
        pnl_color = c.GREEN if self.total_pnl >= 0 else c.RED
        lines.append(f"{c.DIM}Vol: ${self.executor.total_volume:.0f} | Liq: ${total_liq:.0f} | "
                    f"PnL: {pnl_color}${self.total_pnl:.2f}{c.RESET} | "
                    f"Trades: {self.trade_count} ({win_rate:.0f}% win){c.RESET}")

        return "\n".join(lines)

    def _render_rl_signal(self, signal: RLSignalState, state: MarketState) -> str:
        """Render RL agent signal module."""
        c = Colors
        lines = []

        lines.append(f"    {c.BOLD}🤖 RL Agent (PPO){c.RESET}")

        # Action probabilities bar
        probs = signal.action_probs
        hold_bar = "█" * int(probs[0] * 20)
        buy_bar = "█" * int(probs[1] * 20)
        sell_bar = "█" * int(probs[2] * 20)

        lines.append(f"    {c.DIM}HOLD:{c.RESET} {c.YELLOW}{hold_bar:<20}{c.RESET} {probs[0]*100:5.1f}%")
        lines.append(f"    {c.DIM}BUY: {c.RESET} {c.GREEN}{buy_bar:<20}{c.RESET} {probs[1]*100:5.1f}%")
        lines.append(f"    {c.DIM}SELL:{c.RESET} {c.RED}{sell_bar:<20}{c.RESET} {probs[2]*100:5.1f}%")
        lines.append("")

        # Current action
        if signal.action == Action.HOLD:
            action_str = f"{c.YELLOW}⏸ HOLD{c.RESET}"
        elif signal.action == Action.BUY:
            action_str = f"{c.GREEN}📈 BUY UP{c.RESET}"
        else:
            action_str = f"{c.RED}📉 SELL (BUY DOWN){c.RESET}"

        confidence_pct = signal.confidence * 100
        conf_color = c.GREEN if confidence_pct > 30 else c.YELLOW if confidence_pct > 15 else c.DIM

        lines.append(f"    Action: {action_str} | "
                    f"Confidence: {conf_color}{confidence_pct:.0f}%{c.RESET} | "
                    f"Value: {signal.value_estimate:+.2f}")

        # Position info
        if state.has_position:
            pnl_color = c.GREEN if state.position_pnl >= 0 else c.RED
            pnl_sign = "+" if state.position_pnl >= 0 else ""
            lines.append(f"    {c.CYAN}Position: {state.position_side} | "
                        f"PnL: {pnl_color}{pnl_sign}${state.position_pnl:.2f}{c.RESET}")

        return "\n".join(lines)

    def _render_rule_signal(self, current_prob: float) -> str:
        """Render rule-based signal module."""
        c = Colors
        cfg = self.config
        lines = []

        lines.append(f"    {c.BOLD}📊 Signal Module{c.RESET}")
        lines.append(f"    Entry: {cfg.entry_low*100:.0f}%-{cfg.entry_high*100:.0f}% | "
                    f"TP: ≥{cfg.take_profit*100:.0f}% | SL: Entry-{cfg.stop_loss_offset*100:.0f}%")
        lines.append("")

        if current_prob < cfg.entry_low:
            status = f"{c.YELLOW}⏳ Below entry zone (<{cfg.entry_low*100:.0f}%){c.RESET}"
        elif current_prob > cfg.entry_high:
            status = f"{c.YELLOW}⏳ Above entry zone (>{cfg.entry_high*100:.0f}%){c.RESET}"
        else:
            status = f"{c.GREEN}✓ In entry zone ({cfg.entry_low*100:.0f}%-{cfg.entry_high*100:.0f}%){c.RESET}"

        lines.append(f"    {status}")

        return "\n".join(lines)

    def _render_orderbooks(self, ob_up: OrderbookState, ob_down: OrderbookState) -> str:
        c = Colors
        lines = []

        # Headers
        lines.append(f"{c.GREEN}{c.BOLD}UP Order Book{c.RESET}                    "
                    f"{c.RED}{c.BOLD}DOWN Order Book{c.RESET}")

        up_bid = ob_up.best_bid or 0
        up_ask = ob_up.best_ask or 0
        down_bid = ob_down.best_bid or 0
        down_ask = ob_down.best_ask or 0

        lines.append(f"Bid: {up_bid*100:.1f}% | Ask: {up_ask*100:.1f}%      "
                    f"Bid: {down_bid*100:.1f}% | Ask: {down_ask*100:.1f}%")
        lines.append("")

        lines.append(f"{c.DIM}BIDS ({len(ob_up.bids)})    ASKS ({len(ob_up.asks)})        "
                    f"BIDS ({len(ob_down.bids)})    ASKS ({len(ob_down.asks)}){c.RESET}")

        max_levels = 10
        for i in range(max_levels):
            up_bid_str = ""
            up_ask_str = ""
            if i < len(ob_up.bids):
                p, s = ob_up.bids[i]
                up_bid_str = f"{c.GREEN}{p*100:.1f}% @ {s:.0f}{c.RESET}"
            if i < len(ob_up.asks):
                p, s = ob_up.asks[i]
                up_ask_str = f"{c.RED}{p*100:.1f}% @ {s:.0f}{c.RESET}"

            down_bid_str = ""
            down_ask_str = ""
            if i < len(ob_down.bids):
                p, s = ob_down.bids[i]
                down_bid_str = f"{c.GREEN}{p*100:.1f}% @ {s:.0f}{c.RESET}"
            if i < len(ob_down.asks):
                p, s = ob_down.asks[i]
                down_ask_str = f"{c.RED}{p*100:.1f}% @ {s:.0f}{c.RESET}"

            lines.append(f"{up_bid_str:30} {up_ask_str:30}  {down_bid_str:30} {down_ask_str:30}")

        return "\n".join(lines)

    def on_orderbook_update(self, ob: OrderbookState):
        self.last_ws_update = datetime.now(timezone.utc)

    async def refresh_markets(self):
        print(f"{Colors.DIM}Searching for active markets...{Colors.RESET}")

        markets = get_15m_markets(assets=self.assets)

        if not markets:
            print(f"{Colors.YELLOW}No active 15m markets found. Waiting...{Colors.RESET}")
            return

        self.markets = {m.condition_id: m for m in markets}

        for m in markets:
            self.orderbook_streamer.subscribe(m.condition_id, m.token_up, m.token_down)

            if m.condition_id not in self.positions:
                self.positions[m.condition_id] = Position(asset=m.asset)

            # Record open price
            current_price = self.price_streamer.get_price(m.asset)
            if current_price > 0 and m.condition_id not in self.open_prices:
                self.open_prices[m.condition_id] = current_price

        print(f"{Colors.GREEN}Found {len(markets)} active markets{Colors.RESET}")

    async def run(self):
        self.running = True
        self.orderbook_streamer.on_update(self.on_orderbook_update)

        await self.refresh_markets()

        ob_task = asyncio.create_task(self.orderbook_streamer.stream())
        price_task = asyncio.create_task(self.price_streamer.stream())
        futures_task = asyncio.create_task(self.futures_streamer.stream())

        try:
            last_refresh = datetime.now(timezone.utc)

            while self.running:
                now = datetime.now(timezone.utc)
                if (now - last_refresh).total_seconds() > 30:
                    await self.refresh_markets()
                    last_refresh = now

                self.clear_screen()

                if not self.markets:
                    print(f"\n{Colors.YELLOW}Waiting for markets...{Colors.RESET}")
                    await asyncio.sleep(1)
                    continue

                market_ids = list(self.markets.keys())
                if self.current_asset_idx >= len(market_ids):
                    self.current_asset_idx = 0

                cid = market_ids[self.current_asset_idx]
                market = self.markets[cid]

                ob_up = self.orderbook_streamer.get_orderbook(cid, "UP")
                ob_down = self.orderbook_streamer.get_orderbook(cid, "DOWN")

                if ob_up and ob_down:
                    # Update full market state
                    state = self.update_market_state(cid, market, ob_up)

                    # Get RL signal if enabled
                    if self.rl_strategy:
                        signal = self.get_rl_signal(cid, state)
                        self.rl_signals[cid] = signal

                        # Execute if auto-trade enabled
                        self.execute_rl_action(cid, market, signal, state)

                    # Render
                    output = self.render_market(market, ob_up, ob_down, state)
                    print(output)
                else:
                    print(f"\n{Colors.YELLOW}Waiting for orderbook data...{Colors.RESET}")

                # Navigation hint
                mode_str = "RL" if self.rl_strategy else "RULES"
                exec_str = self.executor.mode.value.upper()
                auto_str = " | AUTO-TRADE" if self.auto_trade else ""
                print(f"\n{Colors.DIM}[←/→] Switch asset | [q] Quit | "
                      f"Signal: {mode_str} | Exec: {exec_str}{auto_str}{Colors.RESET}")

                await asyncio.sleep(self.refresh_rate)

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Shutting down...{Colors.RESET}")
        finally:
            self.running = False
            self.orderbook_streamer.stop()
            self.price_streamer.stop()
            self.futures_streamer.stop()
            ob_task.cancel()
            price_task.cancel()
            futures_task.cancel()


def main():
    parser = argparse.ArgumentParser(description="Polymarket Terminal UI with RL")
    parser.add_argument("--live", action="store_true", help="Enable live trading mode")
    parser.add_argument("--rl", action="store_true", help="Use RL agent for signals")
    parser.add_argument("--load", type=str, default="rl_model", help="RL model to load (default: rl_model)")
    parser.add_argument("--auto", action="store_true", help="Auto-execute RL agent trades")
    parser.add_argument("--asset", type=str, help="Focus on single asset (BTC, ETH, SOL, XRP)")
    parser.add_argument("--size", type=float, default=50.0, help="Trade size in $ (default: 50)")
    parser.add_argument("--entry-low", type=float, default=0.10, help="Entry zone low (default: 0.10)")
    parser.add_argument("--entry-high", type=float, default=0.25, help="Entry zone high (default: 0.25)")
    parser.add_argument("--tp", type=float, default=0.30, help="Take profit threshold (default: 0.30)")
    parser.add_argument("--sl", type=float, default=0.05, help="Stop loss offset (default: 0.05)")

    args = parser.parse_args()

    # Load RL strategy if requested
    rl_strategy = None
    if args.rl:
        try:
            from strategies.rl_mlx import RLStrategy
            rl_strategy = RLStrategy()
            rl_strategy.load(args.load)
            rl_strategy.training = False  # Set to inference mode
            print(f"{Colors.GREEN}Loaded RL model: {args.load}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}Failed to load RL model: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}Falling back to rule-based signals{Colors.RESET}")
            rl_strategy = None

    # Create executor
    executor = create_executor(live=args.live)

    # Signal config (for rule-based mode)
    signal_config = SignalConfig(
        entry_low=args.entry_low,
        entry_high=args.entry_high,
        take_profit=args.tp,
        stop_loss_offset=args.sl
    )

    assets = [args.asset] if args.asset else ["BTC", "ETH", "SOL", "XRP"]

    # Create and run UI
    ui = TerminalUI(
        executor=executor,
        rl_strategy=rl_strategy,
        signal_config=signal_config,
        assets=assets,
        trade_size=args.size,
        auto_trade=args.auto
    )

    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Polymarket 15m Terminal UI{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"Execution: {Colors.GREEN if args.live else Colors.YELLOW}"
          f"{executor.mode.value.upper()}{Colors.RESET}")
    print(f"Signals: {Colors.CYAN if args.rl else Colors.YELLOW}"
          f"{'RL Agent (PPO)' if args.rl else 'Rule-based'}{Colors.RESET}")
    if args.auto:
        print(f"Auto-trade: {Colors.GREEN}ENABLED{Colors.RESET} (${args.size:.0f}/trade)")
    print(f"Assets: {', '.join(assets)}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

    asyncio.run(ui.run())


if __name__ == "__main__":
    main()
