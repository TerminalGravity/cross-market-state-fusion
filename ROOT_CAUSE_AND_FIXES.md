# Root Cause Analysis & Fixes

**Date**: 2026-01-04
**Investigation**: Why positions expired unmanaged causing -$17.18 loss

---

## Root Causes Identified

### 1. Exit Orders Failed - "Not Enough Balance" (Primary)

**Timeline**:
- 23:01:55 - ETH BUY @ 44% - **MATCHED** ✅
- 23:02:54 - BTC BUY @ 65.5% - **MATCHED** ✅
- 23:16:48 - BTC SELL attempt - **FAILED** ❌ `not enough balance / allowance`
- 23:16:49 - ETH SELL attempt - **FAILED** ❌ `not enough balance / allowance`
- Markets expired ~23:16 with positions still open

**Evidence**:
```
[CLOB] LIVE Market BUY $25.00 on ETH: matched (order_id=0xfdd45ef...)
[CLOB] LIVE Market BUY $25.00 on BTC: matched (order_id=0xc4db8ad...)

# 15 minutes later...
PolyApiException[status_code=400, error_message={'error': 'not enough balance / allowance'}]
[LIVE] Order failed for BTC: no response
[LIVE] Order failed for ETH: no response
```

**Why BUY succeeded but SELL failed?**

The shares were purchased ("matched" status) but couldn't be sold 15 minutes later. Possible reasons:

1. **Token ID mismatch**: Bot might be trying to sell DIFFERENT tokens than what was bought
   - BUY uses token_up (e.g., "BTC UP" token)
   - SELL should use SAME token_id
   - Need to verify `pos.token_id` matches the bought tokens

2. **Share quantity incorrect**: Shares calculation might be wrong
   - Code: `shares_to_sell = pos.size / pos.entry_price`
   - For BTC: 25 / 0.655 = 38.17 shares
   - But what if pos.entry_price changed or is stale?

3. **Polymarket settlement delay**: Shares might not be immediately sellable after buy
   - Need to wait for on-chain settlement?
   - Check Polymarket docs for required delay

4. **CTF approval revoked**: Approval might have expired
   - We set approvals initially with `setApprovalForAll`
   - But could Polymarket revoke them?
   - Need to check approval status before sell

### 2. Safety System Didn't Run (Secondary)

**Evidence**:
- ZERO safety logs in `/tmp/worker_live_trading.log`
- No `[SAFETY]` messages
- No `emergency` close attempts
- No `timeout` triggers

**Why safety loop didn't run?**

Checking the code at `railway_worker.py:1163`:

```python
stream_tasks = [
    ...
    asyncio.create_task(self._position_safety_loop()),  # Should be here
]
```

Need to verify:
1. Was `_position_safety_loop()` actually added to stream_tasks in OLD code?
2. Did the safety loop crash during startup?
3. Are there exceptions preventing it from running?

**Expected behavior if safety loop was working:**
- At 23:14 (T-2min before 23:16 expiry): `[SAFETY] TIMEOUT TRIGGERED`
- Emergency SELL order placed
- Even if SELL fails, position cleared from tracking to prevent double-close

### 3. Database Out of Sync

**Evidence**:
```
ETH UP: STILL OPEN in database (exit_time=NULL)
BTC UP: STILL OPEN in database (exit_time=NULL)
```

But Polymarket shows these markets as RESOLVED (user can claim $57.82 winnings).

**Why database wasn't updated?**

When SELL orders fail, the code in `railway_worker.py:957` catches exception but doesn't update database:

```python
except Exception as e:
    logger.error(f"[{mode.upper()}] Emergency sell order failed: {e}")
    # Continue anyway to update state - position expiring is worse than tracking error
```

Comment says "continue anyway" but there's no database update in the except block!

The database update only happens BEFORE the sell attempt (lines 963-970), so if sell fails, database is never updated.

---

## Required Fixes

### Fix 1: Investigate & Fix Exit Order Failures

**Priority**: CRITICAL
**File**: `helpers/clob_executor.py`

**Actions**:

1. **Add token_id verification before sell**:
   ```python
   # Before placing sell order
   logger.info(f"[CLOB] Selling shares: token_id={token_id[:20]}..., amount={amount:.2f} shares")

   # Verify token_id hasn't changed
   if token_id != original_buy_token_id:
       logger.error(f"[CLOB] TOKEN MISMATCH: Trying to sell {token_id} but bought {original_buy_token_id}")
   ```

2. **Check CTF approval status before sell**:
   ```python
   # Query on-chain: is CTF still approved for this contract?
   is_approved = ctf_contract.isApprovedForAll(wallet_address, polymarket_contract)
   if not is_approved:
       logger.error("[CLOB] CTF approval missing - need to re-approve")
   ```

3. **Add share balance check before sell**:
   ```python
   # Query actual share balance from blockchain
   actual_shares = ctf_contract.balanceOf(wallet_address, token_id)
   logger.info(f"[CLOB] Wallet has {actual_shares} shares, trying to sell {amount}")

   if actual_shares < amount:
       logger.warning(f"[CLOB] Insufficient shares: have {actual_shares}, need {amount}")
   ```

4. **Retry sell with delay** (in case of settlement lag):
   ```python
   for attempt in range(3):
       try:
           response = self.client.post_order(signed_order, OrderType.FOK)
           break
       except PolyApiException as e:
           if 'not enough balance' in str(e) and attempt < 2:
               logger.warning(f"[CLOB] Balance error, retrying in 5s (attempt {attempt+1}/3)")
               await asyncio.sleep(5)
           else:
               raise
   ```

### Fix 2: Ensure Safety Loop Runs

**Priority**: CRITICAL
**File**: `railway_worker.py`

**Actions**:

1. **Verify safety loop added to stream_tasks** (line 1163):
   ```python
   stream_tasks = [
       asyncio.create_task(self.orderbook_streamer.stream()),
       asyncio.create_task(self.price_streamer.stream()),
       asyncio.create_task(self.futures_streamer.stream()),
       asyncio.create_task(self._checkpoint_loop()),
       asyncio.create_task(self._metrics_loop()),
       asyncio.create_task(self._daily_summary_loop()),
       asyncio.create_task(self._position_safety_loop()),  # MUST BE HERE
   ]
   ```

2. **Add startup confirmation** (already done in enhanced version):
   ```python
   async def _position_safety_loop(self) -> None:
       logger.info("[SAFETY] Position safety loop started")  # Confirms running
       ...
   ```

3. **Add exception handling to prevent crash**:
   ```python
   async def _position_safety_loop(self) -> None:
       logger.info("[SAFETY] Position safety loop started")
       while self.running:
           try:
               await asyncio.sleep(10)
               # ... check logic ...
           except asyncio.CancelledError:
               break  # Graceful shutdown
           except Exception as e:
               logger.error(f"[SAFETY] Loop error: {e}", exc_info=True)
               # DON'T break - keep loop running even on errors!
   ```

### Fix 3: Update Database Even When Sell Fails

**Priority**: HIGH
**File**: `railway_worker.py`

**Change emergency close logic** (line ~950):

**Before**:
```python
try:
    order = executor.place_market_order(...)
    logger.info(f"Emergency sell executed: {order.status}")
except Exception as e:
    logger.error(f"Emergency sell order failed: {e}")
    # Continue anyway - but NO database update!

# Database update only if no exception
if pos.trade_id:
    await self.db.record_trade_close(...)
```

**After**:
```python
sell_succeeded = False
try:
    order = executor.place_market_order(...)
    sell_succeeded = True
    logger.info(f"Emergency sell executed: {order.status}")
except Exception as e:
    logger.error(f"Emergency sell order failed: {e}")
    # Continue to update database anyway

# ALWAYS update database, even if sell failed
if pos.trade_id:
    await self.db.record_trade_close(
        trade_id=pos.trade_id,
        exit_price=exit_price,
        exit_binance_price=binance_price,
        exit_reason=f"emergency:{reason}{'_sell_failed' if not sell_succeeded else ''}",
        pnl=pnl,
        duration_seconds=duration
    )
```

This ensures database is updated even if Polymarket sell fails, preventing out-of-sync state.

### Fix 4: Add Position Timeout to Database

**Priority**: MEDIUM
**File**: `railway_worker.py`

When recording position open, store market expiry time:

```python
await self.db.record_trade_open(
    ...,
    market_expiry_time=market.end_time  # NEW FIELD
)
```

This allows safety system to use database as source of truth for expiry times, even if `market` object is stale or unavailable.

---

## Testing Plan

### Test 1: Verify Safety Loop Runs

**Objective**: Confirm safety loop starts and runs every 10 seconds

**Steps**:
1. Deploy code with enhanced logging
2. Start bot: `python railway_worker.py`
3. Check logs for: `[SAFETY] Position safety loop started`
4. Wait 60 seconds
5. Check for periodic: `[SAFETY] Checked X LIVE + Y PAPER positions`

**Success criteria**: Safety messages appear every 10-60 seconds

### Test 2: Test Position Timeout

**Objective**: Confirm positions are force-closed at T-2 minutes

**Steps**:
1. Wait for bot to open LIVE position
2. Watch logs for position open time
3. Calculate T-2min = (market.end_time - 120 seconds)
4. At T-2min, should see: `[SAFETY] 🚨 TIMEOUT TRIGGERED`
5. Verify emergency SELL order placed
6. Check database: `force_closed=true`, `force_close_reason='timeout'`

**Success criteria**: Position closed before market expiry

### Test 3: Verify Sell Orders Work

**Objective**: Confirm shares can be sold after buying

**Steps**:
1. Monitor next LIVE BUY order
2. Check Polymarket wallet for shares received
3. Wait 30 seconds (allow settlement)
4. Manually trigger sell or wait for strategy signal
5. Verify sell succeeds with "matched" status

**Success criteria**: Sell order succeeds without "not enough balance" error

### Test 4: Test Database Sync

**Objective**: Confirm database updates even when sells fail

**Steps**:
1. Open position
2. Simulate sell failure (disconnect network?)
3. Check database: should show `exit_time`, `exit_reason='emergency:..._sell_failed'`
4. Position should be removed from `live_positions` dict

**Success criteria**: Database always updated, even on sell failure

---

## Immediate Next Steps

1. **Deploy enhanced code** with better logging
2. **Run diagnostic**: `python scripts/check_safety_system.py`
3. **Start bot with logging**: `python railway_worker.py > /tmp/safety_test.log 2>&1 &`
4. **Monitor logs**: `tail -f /tmp/safety_test.log | grep --line-buffered "\[SAFETY\]"`
5. **Wait for position**: Watch for BUY order
6. **Verify safety triggers**: Should see timeout at T-2min
7. **Check sell success**: Verify exit orders complete
8. **Inspect database**: Confirm positions closed

---

## Success Metrics

After fixes:
- ✅ Safety loop confirmed running (startup message in logs)
- ✅ Positions force-closed at T-2min (before expiry)
- ✅ SELL orders succeed (no "not enough balance" errors)
- ✅ Database always in sync (exit_time set even if sell fails)
- ✅ Zero positions expire unmanaged
- ✅ Discord alerts sent on emergency closes

---

## Open Questions

1. **Why did allowances work for BUY but not SELL?**
   - Need to check if CTF approval is still active
   - Query: `ctf.isApprovedForAll(wallet, polymarket_contract)`

2. **Is there a Polymarket settlement delay?**
   - Check docs: Can shares be sold immediately after buy?
   - Or need to wait for on-chain confirmation?

3. **Are token_ids consistent between buy/sell?**
   - Log token_id on both buy and sell
   - Verify they match

4. **Why was safety loop not in old logs?**
   - Check git history: When was `_position_safety_loop()` added?
   - Was it missing in deployed version?
