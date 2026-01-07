---
task: "Achieve profitable RL agent trades deployed on Fly.io. Make at least $40 profit in LIVE trading to recover previous losses."
completion_promise: "Live trading has achieved cumulative profit of at least $40"
max_iterations: 50
current_iteration: 3
---

# Ralph Loop: Profitable RL Agent Trading

## Goal
Make at least $40 profit in LIVE trading to recover previous losses.

## Progress Log

### Iteration 1
- Added spread-cost to reward function (critical missing piece from nikshep's strategy)
- Deployed updated code to Fly.io
- Analyzed EV by price bucket: ONLY 0-20% entries are profitable!
- Need to restrict entries to low-price zones and add funds to wallet

### Iteration 2
- Tightened entry price filter: MAX_ENTRY_PRICE 35% → 20%
- Reduced TRADE_SIZE from $10 to $5 (wallet only has $6.26 USDC)
- Made confidence filter ADAPTIVE based on price zone
- User frustrated: "24 hours ago it was making money unrestricted"
- User showed screenshot of winning trades at 30-72¢ entry prices

### Iteration 3
- **STRIPPED ALL RESTRICTIVE FILTERS** per user request
- Only kept: market expiry check (15% time remaining)
- Removed: spread filter, confidence filter, exit slippage check, entry price filter
- User added $20 more USDC (~$29 total before losses)
- First LIVE trades executed:
  - XRP UP @ 60.5% for $7.50 → STOP-LOSS TRIGGERED (-38%) = **-$4.28**
  - BTC quick scalp → **+$0.75**
  - ETH UP @ 1% entry → market collapsed to 6.5% = **LOSS**
- Reduced TRADE_SIZE from $7.50 → $4 via Fly secrets
- **WALLET DEPLETED**: $1.78 USDC remaining
- **34+ consecutive LIVE order failures** ("not enough balance / allowance")
- 18 expired positions all worthless (UP bets that resolved DOWN)

**LIVE Performance This Session:**
- PnL: **-$0.77** (7 trades, 14% win rate)
- Historical cumulative: ~**-$103** from expired positions

## Key Metrics to Track
- LIVE trading PnL (target: +$40)
- Current: ~-$103 historical + -$0.77 recent = **~-$104 total**
- Win rate: 14% LIVE, 27% paper
- Entry price distribution: unrestricted (all zones)

## Critical Blockers
1. **CAPITAL DEPLETED** - $1.78 USDC, cannot place $2 minimum orders
2. **Minimum share requirement** - Polymarket requires 5 shares/order
   - $2 @ 50% = 4 shares (REJECTED)
   - $2 @ 40% = 5 shares (ACCEPTED)
3. **Systematic UP bias** - Bot bought UP on all assets, markets resolved DOWN
4. **18 expired worthless positions** - All UP bets that lost

## Current Status
- Filters: STRIPPED (unrestricted trading per user request)
- Bot: RUNNING but cannot execute LIVE trades (insufficient balance)
- Paper trading: Working fine (-$0.51 to -$0.90 PnL)
- **BLOCKER: Need user to add more USDC to continue LIVE trading**

## To Continue
User must add more USDC to wallet: `0x5C1be81bfCD92451733f4AaC7207FD4c3a818dE1`
Recommended minimum: $20-50 to allow meaningful position sizing
