# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPO (Proximal Policy Optimization) agent that paper trades Polymarket's 15-minute binary crypto markets by exploiting information lag between fast markets (Binance futures) and slow markets (Polymarket). Trades 4 concurrent markets (BTC, ETH, SOL, XRP) with a single shared policy network.

**Status**: Paper trading only - live market data, simulated trades.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Training (starts live paper trading with learning enabled)
python run.py rl --train --size 50

# Training with real-time dashboard
python run.py rl --train --size 50 --dashboard --port 5050

# Inference (load trained model, no learning)
python run.py rl --load rl_model --size 100

# Baseline strategies
python run.py momentum
python run.py mean_revert
python run.py random

# Standalone dashboard
python dashboard.py --port 5001

# Trade analysis
python analyze_trades.py logs/trades_YYYYMMDD_HHMMSS.csv
```

## Architecture

### Data Flow
```
Binance Futures WSS ──┐
                      ├──→ MarketState (18-dim) ──→ Strategy.act() ──→ Action
Polymarket CLOB WSS ──┘
```

### Core Modules

- **run.py**: `TradingEngine` - orchestrates market discovery, data streaming, action execution, PnL calculation, RL experience collection
- **strategies/base.py**: `MarketState` (18 features), `Action` enum (HOLD/BUY/SELL), `Strategy` ABC
- **strategies/rl_mlx.py**: `RLStrategy` - PPO actor-critic (MLX), experience buffer, training loop
- **helpers/binance_futures.py**: `FuturesStreamer` - multi-timeframe returns, CVD, order flow, liquidations
- **helpers/orderbook_wss.py**: `OrderbookStreamer` - Polymarket CLOB bid/ask, imbalance metrics
- **helpers/polymarket_api.py**: Market discovery via Gamma API (finds active 15-min markets)
- **helpers/training_logger.py**: CSV logging for trades, PPO updates, episodes

### State Space (18 dimensions)
| Category | Features |
|----------|----------|
| Momentum | returns_1m, returns_5m, returns_10m (Binance, scaled 100x) |
| Order Flow | ob_imbalance_l1, ob_imbalance_l5, trade_flow, cvd_acceleration |
| Microstructure | spread_pct, trade_intensity, large_trade_flag |
| Volatility | vol_5m, vol_expansion |
| Position | has_position, position_side, position_pnl, time_remaining |
| Regime | vol_regime, trend_regime |

### Neural Network
```
Actor:  18 → 128 (tanh) → 128 (tanh) → 3 (softmax)
Critic: 18 → 128 (tanh) → 128 (tanh) → 1
```

### PPO Hyperparameters
- Learning rates: actor=1e-4, critic=3e-4
- Buffer: 512 experiences → triggers update
- Batch: 64, Epochs: 10
- GAE: gamma=0.99, lambda=0.95
- Clip epsilon: 0.2, Entropy coef: 0.10

## Key Design Decisions

1. **Share-based PnL reward**: `(exit - entry) × (dollars / entry)` matches actual binary market economics. 4.5x better ROI than probability-based rewards.

2. **Sparse rewards only**: Reward only on position close. Intermediate shaping rewards (Phase 1) caused policy collapse via reward hacking.

3. **High entropy coefficient (0.10)**: Prevents premature policy collapse. Values ≤0.05 caused entropy to drop to 0.36.

4. **Multi-asset single policy**: Same network trades all 4 assets, learning generalizable patterns vs overfitting.

5. **MLX on Apple Silicon**: Real-time PPO updates without cloud GPU.

## Model Files

- `rl_model.safetensors` / `rl_model_stats.npz`: Trained actor/critic weights and reward normalization stats
- `logs/trades_*.csv`: Trade records (entry/exit prices, PnL, duration)
- `logs/updates_*.csv`: PPO metrics per update (losses, entropy, KL)

## Debugging

Watch for entropy collapse (healthy ~1.0, collapsed <0.5). If entropy drops:
- Increase entropy_coef in strategies/rl_mlx.py
- Check reward signal isn't gameable
- Verify buffer/trade win rate aren't diverging (indicates reward hacking)
