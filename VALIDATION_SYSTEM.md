# Trade Validation System

## Overview

This system provides **cryptographic proof** that "live" trades actually execute on Polymarket, solving the trust problem where the bot could claim profits without real trades.

## Problem Solved

**Before:**
- System claimed $160.63 "live" profit
- Actual wallet had $98.93 USDC
- **$62 discrepancy** - no way to verify trades were real
- Database had no audit trail linking to Polymarket orders

**After:**
- Every trade records Polymarket CLOB order ID
- Trades can be verified against Polymarket API
- Wallet balance monitored and reconciled
- Full audit trail in validation_log table
- MCP server for external monitoring

## Architecture

```
Trading Bot (railway_worker.py)
    ↓
CLOB Executor (places orders)
    ↓ (captures order_id + full response)
Database (trades table with order tracking)
    ↓
Validator (verifies against Polymarket)
    ↓
Validation Log + Balance Snapshots
    ↓
MCP Server (external monitoring)
```

## Database Schema

### New Tables

1. **`trades` table additions:**
   - `order_id` - Polymarket CLOB order ID
   - `execution_type` - 'paper' or 'live'
   - `fill_status` - Order fill status (matched, rejected, etc.)
   - `clob_response` - Full CLOB API response (JSONB)
   - `verified` - Boolean flag for validated trades
   - `verified_at` - Timestamp of verification

2. **`validation_log`:**
   - Tracks all validation attempts
   - Links to trade_id
   - Stores validation results and errors

3. **`balance_snapshots`:**
   - Regular wallet balance checks
   - Compares to reported PnL
   - Flags discrepancies

### Views

- `unverified_live_trades` - All live trades pending validation
- `validation_summary` - Validation stats by execution type
- `balance_discrepancies` - Balance mismatches

## Usage

### 1. Run MCP Server (For Monitoring)

```bash
# Start the MCP server
fastmcp run trading_mcp_server.py
```

Add to your Claude Code MCP config (`~/.claude.json` or `.mcp.json`):

```json
{
  "mcpServers": {
    "trading-monitor": {
      "command": "fastmcp",
      "args": ["run", "trading_mcp_server.py"],
      "cwd": "/Users/jackfelke/Projects/cross-market-state-fusion",
      "env": {
        "DATABASE_URL": "postgresql://...",
        "POLYGON_RPC_URL": "https://polygon-mainnet.g.alchemy.com/...",
        "POLYMARKET_FUNDER_ADDRESS": "0x..."
      }
    }
  }
}
```

### 2. Query Trading Status

Via Claude Code with MCP server running:

```
You: "Check trading status"
Claude: [Uses get_trading_status MCP tool]

You: "Show unverified live trades"
Claude: [Uses get_unverified_trades MCP tool]

You: "Validate all live trades"
Claude: [Uses validate_live_trades MCP tool]

You: "Check wallet balance"
Claude: [Uses get_wallet_balance MCP tool]
```

### 3. Manual Validation

Run standalone validation:

```bash
# Validate all trades from last 24 hours
python helpers/polymarket_validator.py
```

### 4. Cron Job Validation

Add to crontab for hourly validation:

```bash
# Validate trades every hour
0 * * * * cd /path/to/project && python helpers/polymarket_validator.py >> logs/validation.log 2>&1
```

## Validation Process

### Order ID Validation

1. **Trade Executed** → CLOB executor captures `order_id` from API response
2. **Stored in DB** → `order_id` saved with trade record
3. **Verification** → Validator queries Polymarket API for order status
4. **Result Logged** → Success/failure recorded in `validation_log`
5. **Trade Marked** → `verified=TRUE` if successful

### Balance Reconciliation

1. **Snapshot Taken** → Query actual USDC balance on Polygon
2. **Compare to Reports** → Check against system's reported PnL
3. **Calculate Discrepancy** → Flag if difference > $1
4. **Record Snapshot** → Store in `balance_snapshots` table
5. **Alert if Critical** → Log warning if discrepancy > $10

## MCP Server Tools

| Tool | Purpose |
|------|---------|
| `get_trading_status()` | Current session, PnL, positions |
| `get_wallet_balance()` | Real on-chain USDC balance |
| `validate_live_trades(hours)` | Verify recent live trades |
| `get_unverified_trades()` | List trades pending validation |
| `get_balance_discrepancies()` | Show balance mismatches |
| `get_recent_trades(limit, mode)` | View trade history |
| `run_full_validation()` | Comprehensive validation check |

## Security Benefits

✅ **Cryptographic Proof** - Order IDs link to real Polymarket orders
✅ **Audit Trail** - Full CLOB API responses stored
✅ **Balance Verification** - On-chain wallet checks prevent lying
✅ **External Monitoring** - MCP server allows independent verification
✅ **Automated Validation** - Cron jobs ensure continuous auditing

## Deployment Checklist

- [ ] Database migration applied (001_add_order_tracking.sql)
- [ ] Environment variables set (DATABASE_URL, POLYGON_RPC_URL, etc.)
- [ ] MCP server configured in Claude Code
- [ ] Validation cron job set up
- [ ] Initial balance snapshot recorded
- [ ] Test validation with sample trades

## Troubleshooting

### "No order_id recorded"

- Check that `create_executor(live=True)` is being called
- Verify POLYMARKET_PRIVATE_KEY and POLYMARKET_FUNDER_ADDRESS are set
- Check logs for CLOB API errors

### Balance discrepancies

- Verify POLYGON_RPC_URL is working
- Check wallet address is correct
- Account for gas fees (small discrepancies OK)
- Verify initial balance was recorded correctly

### Unverified trades piling up

- Check that validator is running (cron job or manual)
- Verify CLOB client has access to order history API
- Check for API rate limits or connectivity issues

## Next Steps

1. **Blockchain Validation** - Verify transactions on Polygon explorer
2. **Real-time Alerts** - Discord notifications for validation failures
3. **Dashboard** - Web UI for validation status
4. **Historical Analysis** - Compare all past trades against API

## Files Modified/Created

### Modified
- `db/connection.py` - Added order tracking parameters
- `helpers/clob_executor.py` - Capture order IDs and responses
- `railway_worker.py` - Pass order info to database
- `helpers/discord.py` - Enhanced alert formatting

### Created
- `db/migrations/001_add_order_tracking.sql` - Schema migration
- `helpers/polymarket_validator.py` - Validation logic
- `trading_mcp_server.py` - FastMCP monitoring server
- `VALIDATION_SYSTEM.md` - This documentation

## Example Validation Output

```json
{
  "validation_time": "2026-01-03T21:30:00Z",
  "total_trades": 15,
  "verified": 12,
  "failed": 3,
  "errors": [
    {
      "trade_id": "550e8400-e29b-41d4-a716-446655440000",
      "order_id": "0x1234...",
      "error": "Order not found in CLOB API"
    }
  ]
}
```

---

**Status:** ✅ Implemented
**Version:** 1.0
**Date:** 2026-01-03
