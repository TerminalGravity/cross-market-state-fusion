# Training Journal: RL on Polymarket

This documents training a PPO agent to trade 15-minute binary prediction markets. The experiment ran on December 29, 2025 over ~2 hours of live market data.

---

## The Experiment

**Goal**: Can an RL agent learn to predict short-term crypto price direction by observing cross-exchange data?

**Setup**:
- 4 concurrent markets (BTC, ETH, SOL, XRP)
- 15-minute binary outcomes (UP or DOWN)
- Live data from Binance + Polymarket
- Paper trading with $10 base capital

**Result**: 109% ROI over 72 PPO updates, but with important caveats about what this proves.

---

## Why This Market?

Polymarket's 15-minute crypto markets have interesting properties:

1. **Binary resolution** - Market pays $1 or $0 based on whether price went up. No partial outcomes.

2. **Known end time** - Unlike continuous trading, you know exactly when resolution happens. Changes the decision problem.

3. **Cross-exchange lag** - Polymarket orderbook lags Binance by seconds. If you see Binance move, you can sometimes bet on Polymarket before the book adjusts.

4. **Sparse reward** - You only learn if you were right at resolution. No intermediate feedback during the 15-minute window.

This creates a clean RL problem: observe state, take action, wait for binary outcome.

---

## Data Fusion

The agent combines three data streams:

```
Binance Spot WSS     → Price returns (1m, 5m, 10m)
Binance Futures WSS  → Order flow, CVD, large trades
Polymarket CLOB WSS  → Bid/ask spread, orderbook imbalance
```

This creates an 18-dimensional state:

| Category | Features |
|----------|----------|
| Momentum | 1m/5m/10m returns |
| Order flow | L1/L5 imbalance, trade flow, CVD acceleration |
| Microstructure | Spread %, trade intensity, large trade flag |
| Volatility | 5m vol, vol expansion ratio |
| Position | Has position, side, PnL, time remaining |
| Regime | Vol regime, trend regime |

The hypothesis: combining underlying asset dynamics (Binance) with prediction market microstructure (Polymarket) gives exploitable signal.

---

## Training: Two Phases

### Phase 1: Shaped Rewards (Failed)

**Updates**: 1-36
**Duration**: ~52 minutes
**Trades**: 1,545

Started with a reward function that included shaping bonuses:

```python
reward = pnl_delta * 0.1           # Actual PnL (scaled down)
reward -= 0.001                    # Transaction cost
reward += 0.002 * momentum_aligned # Bonus for trading with momentum
reward += 0.001 * size_multiplier  # Bonus for larger positions
```

**What happened**: Entropy collapsed from 1.09 to 0.36. The policy became nearly deterministic, fixating on a single action.

**Why it failed**: The shaping rewards were similar magnitude to the actual PnL signal. The agent learned to collect bonuses (trade with momentum, use large sizes) without actually being profitable. Buffer win rate showed 90%+ but actual trade win rate was 20%.

**Lesson**: Shaping rewards can backfire when they're gameable. The agent optimized the reward function, not the underlying goal.

| Update | Entropy | PnL | Win Rate |
|--------|---------|-----|----------|
| 1 | 1.09 | $3.25 | 19.8% |
| 10 | 0.79 | $4.40 | 17.3% |
| 20 | 0.40 | $1.37 | 19.4% |
| 36 | 0.36 | $3.90 | 20.2% |

### Phase 2: Pure PnL (Worked)

**Updates**: 37-72
**Duration**: ~52 minutes
**Trades**: 3,330

Switched to pure realized PnL:

```python
def reward(position_close):
    return (exit_price - entry_price) * size  # That's it
```

Also doubled entropy coefficient (0.05 → 0.10) to prevent collapse.

**What happened**: Entropy recovered to 1.05 (near maximum for 3 actions). PnL grew steadily to $10.93, representing 109% ROI on the $10 base.

| Update | Entropy | PnL | Win Rate |
|--------|---------|-----|----------|
| 1 | 0.68 | $5.20 | 33.3% |
| 10 | 1.06 | $9.55 | 22.9% |
| 20 | 1.05 | $5.85 | 21.1% |
| 36 | 1.05 | $10.93 | 21.2% |

**Observation**: Win rate settled at ~21%, below random (33%). But the agent is profitable because binary markets have asymmetric payoffs. When you buy at prob=0.40, a win pays $0.60 and a loss costs $0.40.

---

## What Changed

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Reward | PnL delta + shaping bonuses | Pure realized PnL |
| Entropy coef | 0.05 | 0.10 |
| Actions | 7 (variable sizing) | 3 (fixed 50%) |
| Final entropy | 0.36 (collapsed) | 1.05 (healthy) |
| Final PnL | $3.90 | $10.93 |

Key changes:

1. **Removed shaping rewards** - No more momentum bonuses or sizing bonuses. Just pay the agent when it makes money.

2. **Doubled entropy coefficient** - Stronger incentive to explore. Prevented the policy from collapsing to a single action.

3. **Simplified action space** - Reduced from 7 actions (HOLD + 3 buy sizes + 3 sell sizes) to 3 actions (HOLD, BUY, SELL). Let the model learn when to trade before learning how much.

---

## Technical Implementation

### PPO with MLX

```python
# GAE advantage estimation
advantages = compute_gae(rewards, values, dones, gamma=0.99, lambda_=0.95)
returns = advantages + values

# Normalize advantages
advantages = (advantages - mean) / (std + 1e-8)

# Clipped policy loss
ratio = new_prob / old_prob
surr1 = ratio * advantage
surr2 = clip(ratio, 1-0.2, 1+0.2) * advantage
policy_loss = -min(surr1, surr2).mean()

# Value loss + entropy bonus
value_loss = MSE(values, returns)
entropy = -(probs * log(probs)).sum(-1).mean()
loss = policy_loss + 0.5 * value_loss - 0.10 * entropy
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Buffer size | 512 experiences |
| Batch size | 64 |
| Epochs per update | 10 |
| Actor LR | 1e-4 |
| Critic LR | 3e-4 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip epsilon | 0.2 |
| Entropy coef | 0.10 |

---

## What This Proves

1. **RL can learn from sparse binary rewards** - The agent improved despite only getting feedback every 15 minutes at market resolution.

2. **Pure PnL > shaped rewards** - Shaping was gameable. Sparse but honest signal worked better.

3. **Entropy coefficient matters** - 0.05 caused collapse; 0.10 maintained exploration.

4. **Low win rate can be profitable** - 21% wins, 109% ROI. Asymmetric payoffs change the math.

5. **Multi-source fusion provides signal** - Combining Binance and Polymarket data gave the agent something to learn from.

## What This Doesn't Prove

1. **Live edge** - Paper trading ignores latency, slippage, fees, and market impact. Real performance would be worse.

2. **Statistical significance** - 2 hours isn't enough. Could be variance. Need weeks of out-of-sample testing.

3. **Scalability** - $5 positions are invisible. At size, the agent's orders would move the market.

4. **Durability** - If this edge exists, it will get arbitraged away as others exploit it.

---

## Files

```
experiments/03_polymarket/
├── run.py                 # Main trading engine
├── dashboard.py           # Real-time visualization
├── strategies/
│   ├── base.py           # Action/State definitions
│   └── rl_mlx.py         # PPO implementation
├── helpers/
│   ├── polymarket_api.py # Market data
│   ├── binance_wss.py    # Price streaming
│   └── orderbook_wss.py  # Orderbook streaming
├── logs/
│   ├── trades_*.csv      # Trade history
│   └── updates_*.csv     # PPO metrics
├── rl_model.safetensors  # Model weights
└── rl_model_stats.npz    # Reward normalization
```

---

*December 29, 2025*
