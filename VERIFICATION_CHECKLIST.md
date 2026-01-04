# 🚨 Post-Deployment Verification Checklist

**Date**: 2026-01-04
**Commit**: dead80b
**Priority**: CRITICAL

---

## Deployment Status

✅ **Code Committed**: Safety system committed to git (commit dead80b)
✅ **Code Pushed**: Pushed to GitHub origin/master
✅ **Database Migration**: Applied to production database
🚨 **Fly.io Deployment**: Manual deployment required

---

## Verification Steps

### Step 1: Deploy to Fly.io

Deploy the updated worker:
```bash
fly deploy --config fly.worker.toml
```

Look for successful deployment:
```
✓ Deployed successfully
✓ Service started
```

### Step 2: Verify Safety Loop Started

Check logs for safety system startup:
```bash
fly logs --app cross-market-state-fusion | grep "\[SAFETY\]"
```

**Expected output within 10 seconds of startup:**
```
[SAFETY] Position safety loop started
```

**If NOT present**: Safety loop failed to start - check for errors!

### Step 3: Verify Background Tasks Running

Check logs for periodic safety checks:
```bash
fly logs --app cross-market-state-fusion | grep "\[SAFETY\]"
```

**Expected output every ~60 seconds:**
```
[SAFETY] Checked 0 LIVE + 0 PAPER positions | Closed: 0 LIVE + 0 PAPER
```

### Step 4: Wait for Next Position

When bot opens a position, watch for:

**While position is open:**
```
[SAFETY] LIVE BTC healthy: T-12.5min to expiry, age=120s
[SAFETY] LIVE BTC healthy: T-11.5min to expiry, age=180s
...
```

**At T-2 minutes before market expiry:**
```
[SAFETY] 🚨 TIMEOUT TRIGGERED: LIVE BTC - market_expiry (expires in 119s)
[LIVE] Emergency sell executed: matched
```

### Step 5: Verify Database Recording

After first forced exit, check database:
```python
python scripts/check_safety_system.py
```

Look for:
```
ETH BUY:
  Exit Reason: emergency:timeout
  ✓ Emergency close triggered (safety system worked!)
```

---

## Success Criteria

All of these MUST be true:

- ✅ `[SAFETY] Position safety loop started` appears in logs at startup
- ✅ Periodic `[SAFETY] Checked X positions` messages every minute
- ✅ Position health logs show time to expiry counting down
- ✅ Timeout trigger fires at T-2 minutes
- ✅ Emergency SELL order executes successfully
- ✅ Database shows `force_closed=true`, `force_close_reason='timeout'`
- ✅ Discord alert sent for emergency close
- ✅ Zero positions held until market expiration

---

## Failure Scenarios

### Scenario 1: No Safety Loop Startup Message

**Problem**: Safety loop not starting
**Check**:
```bash
fly logs --app cross-market-state-fusion | grep "error\|exception\|traceback" -i
```
**Possible causes**:
- Exception during safety loop initialization
- Missing dependency
- Database connection failure

### Scenario 2: Safety Loop Starts But Doesn't Trigger

**Problem**: Position held to expiry despite safety loop running
**Check**:
- Is position added to `live_positions` dict?
- Is `market.end_time` correct?
- Are there exceptions in safety check loop?

**Debug**:
```python
# In logs, look for:
[SAFETY] LIVE {asset}: Market not found
[SAFETY] Position safety check error: {exception}
```

### Scenario 3: Timeout Triggers But SELL Fails

**Problem**: Emergency close fails with "not enough balance"
**This is the SECOND root cause from ROOT_CAUSE_AND_FIXES.md**

Evidence from previous failure:
```
PolyApiException[status_code=400, error_message={'error': 'not enough balance / allowance'}]
```

**Possible fixes needed**:
1. Verify CTF approval status before sell
2. Check token_id consistency between buy/sell
3. Add retry logic with settlement delay
4. Verify shares calculation: `shares = size / entry_price`

---

## Monitoring Commands

### Watch Live Logs
```bash
# All logs (live tail)
fly logs --app cross-market-state-fusion

# Safety system only
fly logs --app cross-market-state-fusion | grep "\[SAFETY\]"

# Position activity
fly logs --app cross-market-state-fusion | grep "LIVE\|BUY\|SELL"
```

### Check Recent Trades
```bash
uv run python scripts/check_previous_trades.py
```

### Check Safety System Status
```bash
uv run python scripts/check_safety_system.py
```

### Check Database Health
```bash
uv run python -c "
from db.connection import Database
import asyncio

async def check():
    db = Database()
    await db.connect()

    # Check health_events
    events = await db.fetch('SELECT * FROM health_events ORDER BY created_at DESC LIMIT 5')
    print(f'Recent health events: {len(events)}')

    # Check forced closes
    forced = await db.fetch('SELECT COUNT(*) as count FROM trades WHERE force_closed = true')
    print(f'Forced closes: {forced[0][\"count\"]}')

    await db.close()

asyncio.run(check())
"
```

---

## Next Steps if Safety Works

Once verified working:
1. ✅ Monitor for 24-48 hours with logging
2. ✅ Verify at least 2-3 successful timeout closes
3. ✅ Confirm database recording works
4. ✅ Test emergency close manually if needed
5. ✅ Review and adjust timeout thresholds if needed (currently T-2 min)

---

## Next Steps if SELL Orders Still Fail

If timeout triggers but SELL fails with "not enough balance":
1. Investigate token_id consistency (ROOT_CAUSE_AND_FIXES.md Fix #1)
2. Check CTF approval status (Fix #1 item 2)
3. Add share balance verification (Fix #1 item 3)
4. Implement retry with delay (Fix #1 item 4)

---

## Contact

If issues persist, check:
- Fly.io logs: `fly logs --app cross-market-state-fusion`
- Database: `scripts/check_safety_system.py`
- Root cause doc: `ROOT_CAUSE_AND_FIXES.md`
- Deployment doc: `DEPLOYMENT_REQUIRED.md`

---

**Last Updated**: 2026-01-04 03:30 UTC
**Deployment Commit**: dead80b
**Database Migration**: 002 applied ✅
