# Polymarket Trading MCP Architecture Design

## Overview

Three specialized MCP servers that provide complete Claude Code control over the trading system:

```
                    Claude Code
                         │
     ┌───────────────────┼───────────────────┐
     │                   │                   │
     ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   TRADER     │  │   INFRA      │  │   ALPHA      │
│ (Live APIs)  │  │ (Internal)   │  │ (Strategy)   │
└──────────────┘  └──────────────┘  └──────────────┘
     │                   │                   │
     ▼                   ▼                   ▼
 Polymarket         PostgreSQL          Simulation
 Polygon           Log Files            Backtesting
 Binance/OKX       Sessions             Optimization
```

---

## Server 1: `polymarket_trader_mcp.py` (Source of Truth)

**Purpose**: All live data from external APIs. This is the ground truth.

### Tools

#### Market Discovery
```python
@mcp.tool()
async def discover_markets() -> dict:
    """
    Find active 15-minute UP/DOWN markets for BTC, ETH, SOL, XRP.
    Returns: condition_ids, token_up/down, end_times, current_prices
    """

@mcp.tool()
async def get_market_metadata(condition_id: str) -> dict:
    """
    Full market details: question, outcomes, resolution source, trading volume.
    """
```

#### Live Orderbook
```python
@mcp.tool()
async def get_orderbook(token_id: str, depth: int = 10) -> dict:
    """
    Live orderbook from Polymarket CLOB.
    Returns: bids, asks, spread, mid_price, imbalance metrics
    """

@mcp.tool()
async def get_orderbook_metrics(token_id: str) -> dict:
    """
    Computed orderbook health metrics:
    - Spread percentage
    - L1/L5 imbalance
    - Bid/ask volume ratio
    - Liquidity depth at price levels
    """
```

#### Trading History (Polymarket CLOB)
```python
@mcp.tool()
async def get_clob_trades(limit: int = 100) -> dict:
    """All trades from authenticated Polymarket account."""

@mcp.tool()
async def get_clob_orders() -> dict:
    """Current open orders on CLOB."""

@mcp.tool()
async def get_clob_positions() -> dict:
    """Current token holdings with unrealized PnL."""

@mcp.tool()
async def get_redeemable_positions() -> dict:
    """Resolved positions that can be converted to USDC."""
```

#### Polygon Blockchain
```python
@mcp.tool()
async def get_wallet_balance() -> dict:
    """USDC + MATIC balances from Polygon RPC."""

@mcp.tool()
async def get_wallet_transactions(limit: int = 20) -> dict:
    """On-chain transaction history from Polygonscan."""

@mcp.tool()
async def get_usdc_transfers(limit: int = 20) -> dict:
    """USDC ERC-20 transfers in/out of wallet."""

@mcp.tool()
async def get_transaction_details(tx_hash: str) -> dict:
    """Full transaction details + decoded input data."""
```

#### CEX Futures Data
```python
@mcp.tool()
async def get_futures_snapshot(asset: str = "BTC") -> dict:
    """
    Binance/OKX Futures data:
    - Mark price, index price, funding rate
    - Multi-timeframe returns (1m, 5m, 10m, 1h)
    - CVD (cumulative volume delta)
    - Open interest
    - Recent liquidations
    - Trade intensity
    """

@mcp.tool()
async def get_cross_exchange_state() -> dict:
    """
    Aggregate state across all data sources for all assets.
    Used to compute MarketState(18) features.
    """
```

#### Analytics (Computed from CLOB data)
```python
@mcp.tool()
async def calculate_expected_value() -> dict:
    """EV by entry price bucket with Wilson confidence intervals."""

@mcp.tool()
async def analyze_spread_impact(spread_pct: float = 1.5) -> dict:
    """Decompose gross alpha vs execution costs."""

@mcp.tool()
async def get_risk_metrics() -> dict:
    """Sharpe, Sortino, Max Drawdown, VaR, Expected Shortfall."""

@mcp.tool()
async def analyze_taker_maker() -> dict:
    """Execution quality: taker vs maker ratio and costs."""

@mcp.tool()
async def analyze_hold_duration() -> dict:
    """Duration vs profitability analysis."""

@mcp.tool()
async def counterfactual_analysis() -> dict:
    """What-if scenarios on historical trades."""
```

---

## Server 2: `polymarket_infra_mcp.py` (Internal Records)

**Purpose**: Database sessions, paper trades, system health, logs.

### Tools

#### Session Management
```python
@mcp.tool()
async def create_session(mode: str = "paper", trade_size: float = 50) -> dict:
    """Create new trading session. Returns session_id."""

@mcp.tool()
async def get_active_session() -> dict:
    """Get currently running session with stats."""

@mcp.tool()
async def get_session_history(limit: int = 10) -> dict:
    """List recent sessions with performance summary."""

@mcp.tool()
async def end_session(session_id: str) -> dict:
    """Gracefully end a session and compute final stats."""

@mcp.tool()
async def get_session_checkpoint(session_id: str) -> dict:
    """Get checkpoint data for crash recovery."""
```

#### Paper Trade Records
```python
@mcp.tool()
async def get_paper_trades(session_id: str = None, limit: int = 100) -> dict:
    """
    Get paper trades from database (NOT live CLOB).
    Includes: entry/exit prices, action_probs, market_state, pnl
    """

@mcp.tool()
async def get_open_paper_positions(session_id: str = None) -> dict:
    """Paper positions not yet closed."""

@mcp.tool()
async def get_paper_trade_stats(session_id: str = None) -> dict:
    """Aggregate stats: win rate, total PnL, avg duration, best/worst."""

@mcp.tool()
async def compare_paper_vs_live() -> dict:
    """
    Reconciliation: Compare paper records to CLOB actual trades.
    Identifies discrepancies in fill prices, timing, etc.
    """
```

#### Transfer & Profit Tracking
```python
@mcp.tool()
async def get_transfer_history() -> dict:
    """All profit transfers to cold wallet with tx hashes."""

@mcp.tool()
async def get_profit_metrics() -> dict:
    """
    Current profit status:
    - Balance vs initial deposit
    - Unrealized PnL in open positions
    - Available for transfer
    - Time since last transfer
    """

@mcp.tool()
async def simulate_transfer(amount: float) -> dict:
    """Dry-run transfer calculation without executing."""
```

#### Alerts & Notifications
```python
@mcp.tool()
async def get_alert_history(limit: int = 50) -> dict:
    """Discord alerts sent by the system."""

@mcp.tool()
async def send_alert(alert_type: str, message: str) -> dict:
    """Manually send Discord alert."""
```

#### System Health
```python
@mcp.tool()
async def get_system_health() -> dict:
    """
    Comprehensive health check:
    - Database connection status
    - WebSocket connections (Polymarket, Binance, OKX)
    - Proxy status (HiFi residential)
    - Worker process status
    - Last heartbeat
    """

@mcp.tool()
async def get_proxy_health() -> dict:
    """HiFi SOCKS5 proxy connection quality metrics."""

@mcp.tool()
async def get_websocket_status() -> dict:
    """Status of all WebSocket connections."""
```

#### Log Analysis
```python
@mcp.tool()
async def analyze_trade_logs(log_file: str = None) -> dict:
    """Parse CSV trade logs for analysis."""

@mcp.tool()
async def analyze_update_logs(log_file: str = None) -> dict:
    """Parse PPO update logs for training diagnostics."""
```

#### Safety Controls
```python
@mcp.tool()
async def get_kill_switch_status() -> dict:
    """Check if trading is halted."""

@mcp.tool()
async def get_loss_tracker_status() -> dict:
    """Current drawdown vs limits."""

@mcp.tool()
async def get_position_limits() -> dict:
    """Current position sizes vs limits per asset."""
```

---

## Server 3: `polymarket_alpha_mcp.py` (Strategic Intelligence) 🚀

**Purpose**: The brain - simulation, backtesting, optimization, regime detection.

### Tools

#### Signal Backtesting
```python
@mcp.tool()
async def backtest_signal_config(
    entry_low: float = 0.35,
    entry_high: float = 0.55,
    take_profit: float = 0.30,
    stop_loss: float = 0.05,
    lookback_hours: int = 24
) -> dict:
    """
    Backtest signal parameters against historical trade data.

    Returns:
    - Simulated PnL
    - Win rate
    - Avg hold duration
    - Max drawdown
    - Comparison to actual performance
    """

@mcp.tool()
async def optimize_signal_params(
    param_space: dict,
    objective: str = "sharpe"
) -> dict:
    """
    Grid search over signal parameters to find optimal config.

    Args:
        param_space: {"entry_low": [0.30, 0.35, 0.40], "tp": [0.25, 0.30, 0.35]}
        objective: "sharpe", "pnl", "win_rate", "sortino"

    Returns:
        Best params + performance surface heatmap data
    """
```

#### Monte Carlo Simulation
```python
@mcp.tool()
async def simulate_strategy(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    trades_per_day: int,
    days: int = 30,
    simulations: int = 1000,
    initial_capital: float = 1000
) -> dict:
    """
    Monte Carlo simulation of strategy outcomes.

    Returns:
    - Expected final capital (median, 5th/95th percentile)
    - Probability of ruin (capital < 50%)
    - Max drawdown distribution
    - Time to double capital (median)
    - Confidence intervals on daily PnL
    """

@mcp.tool()
async def stress_test(
    scenario: str = "flash_crash",
    capital: float = 1000
) -> dict:
    """
    Stress test strategy under extreme conditions.

    Scenarios:
    - flash_crash: 10% adverse move in 1 minute
    - liquidity_crisis: Spread widens to 5%
    - correlation_break: All assets move together
    - api_outage: Delayed fills by 30s

    Returns: Expected loss, recovery time, recommendations
    """
```

#### Regime Detection
```python
@mcp.tool()
async def detect_market_regime(asset: str = "BTC") -> dict:
    """
    Classify current market regime using recent data.

    Regimes:
    - trending_up: Consistent positive returns
    - trending_down: Consistent negative returns
    - ranging: Low directional movement
    - volatile: High variance, no trend
    - breakout: Transitioning between regimes

    Returns: regime, confidence, recommended strategy adjustment
    """

@mcp.tool()
async def get_regime_history(hours: int = 24) -> dict:
    """Historical regime transitions with timestamps."""

@mcp.tool()
async def analyze_regime_performance() -> dict:
    """
    Performance breakdown by regime.
    Shows which regimes the strategy performs best/worst in.
    """
```

#### Capital Optimization
```python
@mcp.tool()
async def calculate_kelly_criterion() -> dict:
    """
    Optimal bet sizing using Kelly Criterion.

    Kelly % = (p * b - q) / b
    where p = win rate, b = win/loss ratio, q = 1-p

    Returns:
    - Full Kelly fraction
    - Half Kelly (recommended for safety)
    - Current bet size vs optimal
    - Risk of ruin at different fractions
    """

@mcp.tool()
async def optimize_capital_allocation() -> dict:
    """
    Optimal capital split across BTC/ETH/SOL/XRP.

    Based on:
    - Historical Sharpe per asset
    - Correlation matrix
    - Current regime per asset

    Returns: Recommended allocation percentages
    """

@mcp.tool()
async def calculate_optimal_position_size(
    capital: float,
    max_drawdown_tolerance: float = 0.20
) -> dict:
    """
    Position size that limits drawdown to tolerance.
    Uses historical volatility and VaR.
    """
```

#### Feature Importance
```python
@mcp.tool()
async def analyze_feature_importance() -> dict:
    """
    Which MarketState features actually drive profitable trades?

    Methods:
    - Correlation with PnL
    - Mutual information
    - Feature values in winning vs losing trades

    Returns: Ranked feature importance with insights
    """

@mcp.tool()
async def analyze_entry_conditions() -> dict:
    """
    What market conditions precede profitable entries?

    Analyzes:
    - Binance momentum before entry
    - Order flow imbalance
    - Spread conditions
    - Volatility state

    Returns: Optimal entry condition profiles
    """
```

#### Lead-Lag Analysis
```python
@mcp.tool()
async def analyze_information_flow() -> dict:
    """
    Which market leads price discovery?

    Tests:
    - Binance → Polymarket lag (should be positive)
    - Cross-asset lead (BTC leads alts?)
    - Time of day effects

    Returns: Lag estimates in seconds, confidence intervals
    """

@mcp.tool()
async def detect_arbitrage_opportunities() -> dict:
    """
    Current UP/DOWN price discrepancies.

    For binary markets: UP + DOWN should ≈ 1.00
    Deviations indicate arbitrage opportunities.

    Returns: Mispriced markets with expected profit
    """
```

#### RL Policy Analysis
```python
@mcp.tool()
async def compare_model_checkpoints(
    checkpoint_a: str,
    checkpoint_b: str
) -> dict:
    """
    Compare two model checkpoints on historical data.

    Returns:
    - Action distribution differences
    - Performance comparison
    - Policy divergence (KL)
    """

@mcp.tool()
async def analyze_policy_entropy() -> dict:
    """
    Is the policy exploring enough?

    Entropy too low = stuck in local optimum
    Entropy too high = random actions

    Returns: Entropy trend, recommendations
    """

@mcp.tool()
async def replay_decision(trade_id: str) -> dict:
    """
    Replay a specific trading decision.

    Shows:
    - Full market state at decision time
    - Action probabilities from policy
    - Value estimate
    - What actually happened
    - Alternative action outcomes
    """

@mcp.tool()
async def analyze_reward_distribution() -> dict:
    """
    How is reward signal distributed?

    Checks for:
    - Reward hacking (high reward but bad outcome)
    - Sparse reward issues
    - Temporal credit assignment problems

    Returns: Reward statistics and red flags
    """
```

#### Execution Optimization
```python
@mcp.tool()
async def estimate_slippage(size_dollars: float, token_id: str) -> dict:
    """
    Estimate slippage for a given order size.

    Uses current orderbook to simulate fill.

    Returns:
    - Expected fill price
    - Slippage in cents and percentage
    - Recommendation (market vs limit)
    """

@mcp.tool()
async def analyze_execution_timing() -> dict:
    """
    When do orders get best fills?

    Analyzes:
    - Time to market expiry
    - Time of day
    - Market activity level

    Returns: Optimal execution windows
    """

@mcp.tool()
async def calculate_maker_savings() -> dict:
    """
    How much would switching to maker orders save?

    Based on historical taker fills and current spreads.
    """
```

#### Strategy Generation
```python
@mcp.tool()
async def generate_trading_rules() -> dict:
    """
    Use historical data to generate rule-based strategies.

    Example output:
    - "BUY when returns_5m > 0.5% AND ob_imbalance > 0.3"
    - "SELL when time_remaining < 60s AND pnl > 5%"

    Returns: Rules with backtested performance
    """

@mcp.tool()
async def evaluate_rule_combination(rules: list) -> dict:
    """
    Evaluate a combination of trading rules.

    Args:
        rules: ["returns_5m > 0.005", "ob_imbalance > 0.3"]

    Returns: Combined rule performance
    """
```

---

## Usage Examples

### Complete Market Analysis Flow
```
Claude: Let me analyze the current market state and recommend actions.

1. discover_markets() -> Find active 15m markets
2. get_cross_exchange_state() -> Snapshot all data sources
3. detect_market_regime() -> Classify current conditions
4. calculate_kelly_criterion() -> Optimal position size
5. estimate_slippage(50, token_id) -> Check execution cost
6. get_risk_metrics() -> Current risk exposure

Recommendation: BTC is in 'trending_up' regime with 0.65 Kelly fraction.
Optimal position: $32.50 with expected slippage of 0.8%.
Current Sharpe: 1.34, suggesting positive edge.
```

### Strategy Optimization Flow
```
Claude: Let me optimize your entry parameters.

1. backtest_signal_config(current_params) -> Baseline performance
2. optimize_signal_params(grid) -> Grid search
3. simulate_strategy(optimized_params) -> Monte Carlo validation
4. stress_test("flash_crash") -> Risk check

Result: Moving entry_low from 0.35 to 0.40 improves Sharpe by 23%.
Monte Carlo shows 94% probability of positive returns over 30 days.
Under flash crash: Expected max loss 8.2% vs current 12.1%.
```

### Debugging Poor Performance
```
Claude: Performance has degraded. Let me diagnose.

1. analyze_regime_performance() -> Check regime fit
2. analyze_feature_importance() -> Feature drift?
3. compare_paper_vs_live() -> Execution issues?
4. analyze_spread_impact() -> Hidden costs?
5. analyze_policy_entropy() -> Policy collapse?

Diagnosis: Entropy dropped to 0.42 (target: 0.85+).
Policy collapsed to always HOLD. Recommend increasing entropy_coef.
Also: spread costs ate 34% of gross alpha. Consider maker orders.
```

---

## Implementation Priority

### Phase 1 (This PR)
- [x] polymarket_trader_mcp.py - Merge existing servers
- [ ] polymarket_infra_mcp.py - Database + health tools

### Phase 2 (Next)
- [ ] polymarket_alpha_mcp.py - Monte Carlo, Kelly, Regime detection

### Phase 3 (Future)
- [ ] Live execution tools (place_order, cancel_order)
- [ ] Autonomous trading mode with Claude approval
- [ ] Real-time alerting integration

---

## Configuration

### .mcp.json
```json
{
  "mcpServers": {
    "polymarket-trader": {
      "command": "uv",
      "args": ["run", "python", "polymarket_trader_mcp.py"]
    },
    "polymarket-infra": {
      "command": "uv",
      "args": ["run", "python", "polymarket_infra_mcp.py"]
    },
    "polymarket-alpha": {
      "command": "uv",
      "args": ["run", "python", "polymarket_alpha_mcp.py"]
    }
  }
}
```
