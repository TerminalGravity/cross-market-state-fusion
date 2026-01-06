# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPO (Proximal Policy Optimization) agent that paper trades Polymarket's 15-minute binary crypto markets by exploiting information lag between fast markets (Binance futures) and slow markets (Polymarket). Trades 4 concurrent markets (BTC, ETH, SOL, XRP) with a single shared policy network.

**Status**: Paper trading by default. Live execution available via py-clob-client.

**⚠️ INFRASTRUCTURE**: Fly.io ONLY. Railway is permanently deprecated - do not use Railway for any deployments.

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

# Terminal UI (paper trading)
python terminal_ui.py

# Terminal UI (live trading - requires .env)
python terminal_ui.py --live

# Terminal UI with custom signals
python terminal_ui.py --entry-low 0.10 --entry-high 0.25 --tp 0.30 --sl 0.05

# Single asset focus
python terminal_ui.py --asset BTC

# Fly.io worker with dual-mode (paper + live)
python fly_worker.py

# Profit transfer dry-run (test without sending transactions)
PROFIT_TRANSFER_DRY_RUN=true python fly_worker.py

# Fly.io deployment commands
fly deploy                                     # Deploy worker
fly logs -a cross-market-state-fusion          # View logs
fly secrets list -a cross-market-state-fusion  # List secrets
fly secrets set KEY=value -a cross-market-state-fusion  # Set secret
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
- **helpers/clob_executor.py**: Live order execution via py-clob-client (PAPER/LIVE modes)
- **terminal_ui.py**: Rich terminal UI with orderbook depth, signal module, entry/TP/SL

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

0. **Fly.io ONLY - DO NOT USE RAILWAY**: Railway has been permanently deprecated for this project. All deployments use Fly.io exclusively. Ignore any legacy Railway configuration files (railway.toml, Procfile, requirements-railway.toml). The worker is `fly_worker.py`, deploy with `fly deploy`.

1. **Share-based PnL reward**: `(exit - entry) × (dollars / entry)` matches actual binary market economics. 4.5x better ROI than probability-based rewards.

2. **Sparse rewards only**: Reward only on position close. Intermediate shaping rewards (Phase 1) caused policy collapse via reward hacking.

3. **High entropy coefficient (0.10)**: Prevents premature policy collapse. Values ≤0.05 caused entropy to drop to 0.36.

4. **Multi-asset single policy**: Same network trades all 4 assets, learning generalizable patterns vs overfitting.

5. **MLX on Apple Silicon**: Real-time PPO updates without cloud GPU.

## Model Files

- `rl_model.safetensors` / `rl_model_stats.npz`: Trained actor/critic weights and reward normalization stats
- `logs/trades_*.csv`: Trade records (entry/exit prices, PnL, duration)
- `logs/updates_*.csv`: PPO metrics per update (losses, entropy, KL)

## Live Trading Setup

1. Copy `.env.example` to `.env`
2. Add your private key and funder address
3. For EOA wallets: Set token allowances (one-time, see .env.example)
4. Run: `python terminal_ui.py --live`

```bash
# Environment variables for live trading
POLYMARKET_PRIVATE_KEY=your_key_here
POLYMARKET_FUNDER_ADDRESS=0xYourAddress
POLYMARKET_SIGNATURE_TYPE=0  # 0=EOA, 1=Email/Magic, 2=Browser
```

### Profit Transfer System

The profit transfer system automatically moves accumulated profits from the hot wallet to a cold wallet for security. It runs as a background task in the Fly.io worker (`fly_worker.py`).

**Configuration** (`.env`):
```bash
# Cold wallet address (where profits are transferred)
COLD_WALLET_ADDRESS=0xYourColdWalletAddressHere

# Transfer triggers (hybrid mode)
PROFIT_TRANSFER_THRESHOLD=100                # Transfer when profit >= $100
PROFIT_TRANSFER_MAX_INTERVAL_HOURS=24       # OR after 24 hours

# Polygon RPC endpoint (required for transfers)
# Get free API key from https://alchemy.com → Create App → Polygon PoS
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_API_KEY

# Safety limits
MINIMUM_TRANSFER_AMOUNT=10                   # Don't transfer less than $10
MAXIMUM_SINGLE_TRANSFER=10000                # Safety cap per transfer

# Retry configuration
MAX_TRANSFER_RETRIES=3                       # Retry failed transfers
TRANSFER_RETRY_BACKOFF_SECONDS=60            # Initial backoff (60s → 120s → 240s)

# Testing mode (set to 'false' for real transfers)
PROFIT_TRANSFER_DRY_RUN=true                 # true = log only, false = real transfers
```

**How it works**:
- Monitors balance every 60 seconds
- Calculates available profit: `current_balance - initial_balance - open_positions`
- Triggers transfer when: `profit >= threshold` OR `time >= max_interval_hours`
- Keeps 100% of initial balance as working capital (only transfers profits)
- Retries failed transfers with exponential backoff (60s → 120s → 240s)
- All transfers logged to database `transfers` table for audit trail
- Discord alerts sent on successful transfers with PolygonScan link

**Testing**:
1. Set `PROFIT_TRANSFER_DRY_RUN=true` in Fly.io secrets
2. Deploy with `fly deploy -c fly.worker.toml`
3. Watch logs with `fly logs -a cross-market-state-fusion`
4. When confident, set `PROFIT_TRANSFER_DRY_RUN=false` for real transfers

**Fly.io Deployment**:
```bash
# Set all secrets
fly secrets set POLYMARKET_PRIVATE_KEY=xxx -a cross-market-state-fusion
fly secrets set POLYMARKET_FUNDER_ADDRESS=0x... -a cross-market-state-fusion
fly secrets set TRADING_MODE=live -a cross-market-state-fusion
# ... (see .env.example for all required vars)

# Deploy
fly deploy -c fly.worker.toml

# View logs
fly logs -a cross-market-state-fusion
```

**IMPORTANT**: You need a Polygon RPC URL (not Ethereum mainnet). Polymarket operates on Polygon network.

## Debugging

Watch for entropy collapse (healthy ~1.0, collapsed <0.5). If entropy drops:
- Increase entropy_coef in strategies/rl_mlx.py
- Check reward signal isn't gameable
- Verify buffer/trade win rate aren't diverging (indicates reward hacking)
