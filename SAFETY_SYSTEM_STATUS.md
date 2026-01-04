# Safety System Analysis & Fixes

**Date**: 2026-01-03
**Issue**: XRP position expired unmanaged, causing -$25 loss
**Status**: Enhanced logging added, ready for testing

---

## What Happened

**Timeline from Polymarket screenshot:**
- **2 hours ago**: Bot opened 3 positions (BTC, ETH, XRP - $25 each)
- **1 hour ago**: XRP market expired → Position LOST (-$25)
- **Now**: BTC + ETH markets expired → Won but only due to luck
- **Result**: $98.93 → $81.75 (-$17.18 / -17%)

**Root Cause**: Safety system FAILED to close positions before market expiry

---

## Existing Safety Mechanisms (Already Implemented!)

**DISCOVERY**: The codebase ALREADY HAS comprehensive safety systems in `railway_worker.py`:

### 1. Position Safety Loop (`_position_safety_loop()`)
- **Location**: `railway_worker.py:793-882`
- **Runs**: Every 10 seconds in background
- **Checks**: All open positions for timeout conditions

### 2. Timeout Detection (`_should_emergency_close()`)
- **Location**: `railway_worker.py:884-914`
- **Triggers**:
  1. **Market Expiry**: Force-close at T-2 minutes before `market.end_time`
  2. **Max Age**: Force-close if position >14 minutes old (for 15-min markets)
  3. **Stale Data**: Force-close if orderbook data >60s old

### 3. Emergency Exit (`_emergency_close_position()`)
- **Location**: `railway_worker.py:916-1006`
- **Actions**:
  - Gets current price from orderbook
  - Places SELL order via executor
  - Records to database with `exit_reason="emergency:..."`
  - Updates PnL stats
  - Sends Discord alert

### 4. Orderbook Health Monitor (`_check_orderbook_health()`)
- **Location**: `railway_worker.py:1008-1043`
- **Monitors**: Orderbook data freshness per market
- **Threshold**: 60s without update = stale
- **Action**: Sets `system_healthy=False`, blocks new trades

---

## Why It Failed (Hypotheses)

The safety system EXISTS but didn't trigger. Possible reasons:

### Hypothesis 1: Market `end_time` Incorrect
- Safety loop checks: `time_to_expiry < 120s`
- If `market.end_time` is wrong, timeout won't trigger
- **Test**: Check if `get_15m_markets()` returns correct `end_time`

### Hypothesis 2: Position Not in `live_positions` Dict
- Safety loop iterates: `for cid, pos in self.live_positions.items()`
- If position not added to dict, won't be checked
- **Test**: Verify positions are added after `place_market_order()`

### Hypothesis 3: Safety Loop Not Running
- Safety loop added to `stream_tasks` at line 1163
- If exception during startup, task may not start
- **Test**: Look for `[SAFETY] Position safety loop started` in logs

### Hypothesis 4: Emergency Close Failed Silently
- Timeout detected but `_emergency_close_position()` threw exception
- Position state not cleared, no alert sent
- **Test**: Check logs for `[SAFETY] Emergency close failed`

### Hypothesis 5: `market_expiry_time` Not Set in Database
- Database field `market_expiry_time` may be NULL
- Safety system uses this for age calculation
- **Test**: Query database for recent trades

---

## Enhancements Added

### 1. Enhanced Logging (`railway_worker.py:793-882`)

**Before**:
```python
logger.warning(f"[SAFETY] Emergency close LIVE {pos.asset}: {reason}")
```

**After**:
```python
# Startup confirmation
logger.info("[SAFETY] Position safety loop started")

# Critical timeout trigger
logger.critical(f"[SAFETY] 🚨 TIMEOUT TRIGGERED: LIVE {pos.asset} - {reason}")

# Periodic health check (every minute)
logger.debug(f"[SAFETY] LIVE {pos.asset} healthy: T-{time_to_expiry/60:.1f}min to expiry, age={age:.0f}s")

# Summary stats
logger.info(f"[SAFETY] Checked {live_checked} LIVE + {paper_checked} PAPER positions | Closed: {live_closed} LIVE + {paper_closed} PAPER")

# Error details
logger.error(f"[SAFETY] Position safety check error: {e}", exc_info=True)
```

**Benefits**:
- ✅ Confirms safety loop is running
- ✅ Shows why positions are/aren't force-closed
- ✅ Tracks position health (time to expiry, age)
- ✅ Full exception tracebacks for debugging

### 2. Database Migration (`db/migrations/002_add_position_health.sql`)

**New Columns in `trades` table**:
```sql
force_closed BOOLEAN DEFAULT FALSE
force_close_reason VARCHAR(50)  -- 'timeout', 'health_check', 'manual', 'max_duration'
position_age_seconds INTEGER
market_expiry_time TIMESTAMPTZ
```

**New Table: `health_events`**:
```sql
CREATE TABLE health_events (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    event_type VARCHAR(20),  -- 'degraded', 'critical', 'recovered'
    asset VARCHAR(10),
    staleness_seconds NUMERIC(6, 2),
    action_taken VARCHAR(50),  -- 'stopped_new_positions', 'emergency_exit'
    open_positions INTEGER,
    affected_trades UUID[]
);
```

**Benefits**:
- ✅ Audit trail for all force-closes
- ✅ Track why positions were closed
- ✅ Identify patterns (which assets timeout most)
- ✅ Orderbook health history

### 3. Diagnostic Script (`scripts/check_safety_system.py`)

**Checks**:
- Recent sessions and PnL
- Open positions and their age
- Time to expiry for each position
- Whether safety triggers should fire
- Recent completed trades and exit reasons
- Health events history (if migration applied)
- Current active markets

**Usage**:
```bash
python scripts/check_safety_system.py
```

---

## New Components Created (Not Yet Integrated)

These were designed by the code architect agent but discovered existing implementations:

1. **`helpers/orderbook_health.py`** (NEW)
   - Standalone health monitor
   - Tracks last update per market
   - Can be used to replace existing `_check_orderbook_health()`

2. **`helpers/position_monitor.py`** (NEW)
   - Standalone position timeout enforcer
   - Background task with callbacks
   - More modular than inline `_position_safety_loop()`

3. **`helpers/emergency_exit.py`** (NEW)
   - Standalone emergency exit executor
   - Retry logic with exponential backoff
   - Discord alerts on failure

**Decision**: Keep existing inline implementation for now, enhance with logging. These modules can replace the inline code later if needed.

---

## Next Steps (Testing & Debugging)

### Step 1: Run Diagnostic Script
```bash
cd /Users/jackfelke/Projects/cross-market-state-fusion
python scripts/check_safety_system.py
```

**What to look for**:
- Are there open positions with age >840s?
- Are there positions with `time_to_expiry < 120s`?
- Do recent trades show `exit_reason="emergency:..."`?
- Are there `health_events` records?

### Step 2: Check Recent Logs
```bash
tail -200 /tmp/worker_with_allowances.log | grep -E "\[SAFETY\]|TIMEOUT|emergency"
```

**What to look for**:
- `[SAFETY] Position safety loop started` - confirms running
- `[SAFETY] TIMEOUT TRIGGERED` - confirms timeouts detected
- `[SAFETY] Emergency close failed` - confirms exit failures
- No safety messages = loop not running or no positions

### Step 3: Test with New Position
**Start the bot and watch for safety logs**:
```bash
cd /Users/jackfelke/Projects/cross-market-state-fusion
python railway_worker.py > /tmp/safety_test.log 2>&1 &

# Watch logs live
tail -f /tmp/safety_test.log | grep --line-buffered -E "\[SAFETY\]|LIVE|Position"
```

**Expected output (every 10 seconds)**:
```
[SAFETY] Position safety loop started
[SAFETY] Checked 2 LIVE + 0 PAPER positions | Closed: 0 LIVE + 0 PAPER
[SAFETY] LIVE BTC healthy: T-12.3min to expiry, age=180s
```

**When position approaches expiry**:
```
[SAFETY] 🚨 TIMEOUT TRIGGERED: LIVE BTC - market_expiry (expires in 119s)
[LIVE] Emergency sell executed: matched
[SAFETY] Checked 1 LIVE + 0 PAPER positions | Closed: 1 LIVE + 0 PAPER
```

### Step 4: Apply Database Migration (if needed)
```bash
# Check if migration needed
psql "$DATABASE_URL" -c "SELECT column_name FROM information_schema.columns WHERE table_name = 'trades' AND column_name = 'force_closed';"

# If no results, apply migration
psql "$DATABASE_URL" -f db/migrations/002_add_position_health.sql
```

---

## Success Criteria

After testing, you should see:

1. **✅ Safety loop running**: `[SAFETY] Position safety loop started` in logs
2. **✅ Periodic checks**: Safety summary logged every minute
3. **✅ Timeout detection**: Positions force-closed at T-2 minutes
4. **✅ Discord alerts**: Emergency close notifications sent
5. **✅ Database records**: `force_closed=true`, `force_close_reason` populated
6. **✅ No expirations**: Zero positions held to market close

---

## If Safety System Still Fails

If after enhanced logging, positions still expire:

1. **Check market end_time accuracy**:
   ```python
   from helpers.polymarket_api import get_15m_markets
   markets = get_15m_markets()
   for m in markets:
       print(f"{m.asset}: ends at {m.end_time} (in {(m.end_time - datetime.now(timezone.utc)).total_seconds()/60:.1f} min)")
   ```

2. **Verify position tracking**:
   - Add debug logs in `execute_action_dual_mode()` to confirm positions are added to `live_positions` dict

3. **Test emergency close manually**:
   ```python
   # In Python REPL with worker running
   await worker.emergency_close_all(reason="manual_test")
   ```

4. **Replace with standalone monitors**:
   - Integrate `helpers/orderbook_health.py`
   - Integrate `helpers/position_monitor.py`
   - Integrate `helpers/emergency_exit.py`

---

## Files Modified

1. **`railway_worker.py`**: Enhanced `_position_safety_loop()` with verbose logging
2. **`db/migrations/002_add_position_health.sql`**: Added health tracking schema (NEW)
3. **`scripts/check_safety_system.py`**: Diagnostic tool (NEW)
4. **`helpers/orderbook_health.py`**: Standalone health monitor (NEW, not integrated)
5. **`helpers/position_monitor.py`**: Standalone position monitor (NEW, not integrated)
6. **`helpers/emergency_exit.py`**: Standalone emergency exit (NEW, not integrated)

---

## Conclusion

The safety system EXISTS and SHOULD work, but clearly failed for your XRP position.

**Most likely cause**: `market.end_time` was incorrect OR position not added to `live_positions` dict.

**Next action**: Run diagnostic script and check logs to confirm which hypothesis is correct.

The enhanced logging will reveal exactly WHY the safety system didn't trigger, allowing us to fix the root cause.
