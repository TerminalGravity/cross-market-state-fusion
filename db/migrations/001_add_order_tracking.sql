-- Migration: Add Order Tracking to Trades Table
-- Purpose: Enable validation of live trades against Polymarket CLOB API
-- Date: 2026-01-03

-- Add order tracking columns to trades table
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS order_id VARCHAR(100),
ADD COLUMN IF NOT EXISTS fill_status VARCHAR(20),
ADD COLUMN IF NOT EXISTS execution_type VARCHAR(10) DEFAULT 'paper'
    CHECK (execution_type IN ('paper', 'live')),
ADD COLUMN IF NOT EXISTS clob_response JSONB,
ADD COLUMN IF NOT EXISTS verified BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS verified_at TIMESTAMPTZ;

-- Add indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades(order_id);
CREATE INDEX IF NOT EXISTS idx_trades_execution_type ON trades(execution_type);
CREATE INDEX IF NOT EXISTS idx_trades_verified ON trades(verified) WHERE verified = FALSE;

-- Add comments for documentation
COMMENT ON COLUMN trades.order_id IS 'Polymarket CLOB order ID from API response';
COMMENT ON COLUMN trades.fill_status IS 'Order fill status: matched, partial, rejected, etc.';
COMMENT ON COLUMN trades.execution_type IS 'Trade execution type: paper (simulated) or live (real)';
COMMENT ON COLUMN trades.clob_response IS 'Full CLOB API response for audit trail';
COMMENT ON COLUMN trades.verified IS 'Whether trade has been verified against Polymarket API';
COMMENT ON COLUMN trades.verified_at IS 'Timestamp of last verification check';

-- =============================================================================
-- VALIDATION LOG TABLE
-- Track all validation attempts and results
-- =============================================================================
CREATE TABLE IF NOT EXISTS validation_log (
    id SERIAL PRIMARY KEY,
    trade_id UUID NOT NULL REFERENCES trades(id) ON DELETE CASCADE,

    -- Validation details
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    validation_type VARCHAR(30) NOT NULL
        CHECK (validation_type IN ('order_api', 'balance_check', 'blockchain', 'manual')),

    -- Results
    is_valid BOOLEAN NOT NULL,
    validation_data JSONB,  -- API response, balance snapshots, etc.
    error_message TEXT,

    -- Metadata
    validator_version VARCHAR(50),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_validation_trade ON validation_log(trade_id);
CREATE INDEX IF NOT EXISTS idx_validation_time ON validation_log(validated_at DESC);
CREATE INDEX IF NOT EXISTS idx_validation_type ON validation_log(validation_type);

-- =============================================================================
-- BALANCE SNAPSHOTS TABLE
-- Track wallet balance over time for reconciliation
-- =============================================================================
CREATE TABLE IF NOT EXISTS balance_snapshots (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,

    -- Balance details
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    usdc_balance DECIMAL(12, 4) NOT NULL,

    -- Context
    reported_pnl DECIMAL(12, 4),  -- What the system reports
    balance_delta DECIMAL(12, 4),  -- Change since last snapshot
    discrepancy DECIMAL(12, 4),  -- Difference between balance and reported PnL

    -- Source
    source VARCHAR(20) NOT NULL
        CHECK (source IN ('polygon_rpc', 'clob_api', 'manual')),

    -- Notes
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_balance_session ON balance_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_balance_time ON balance_snapshots(snapshot_at DESC);

-- =============================================================================
-- HELPER VIEWS FOR VALIDATION
-- =============================================================================

-- Unverified live trades
CREATE OR REPLACE VIEW unverified_live_trades AS
SELECT
    t.*,
    s.mode as session_mode,
    NOW() - t.entry_time as age
FROM trades t
JOIN sessions s ON t.session_id = s.id
WHERE t.execution_type = 'live'
  AND (t.verified IS NULL OR t.verified = FALSE)
ORDER BY t.entry_time DESC;

-- Trade validation summary
CREATE OR REPLACE VIEW validation_summary AS
SELECT
    t.execution_type,
    COUNT(*) as total_trades,
    SUM(CASE WHEN t.verified = TRUE THEN 1 ELSE 0 END) as verified_count,
    SUM(CASE WHEN t.verified = FALSE OR t.verified IS NULL THEN 1 ELSE 0 END) as unverified_count,
    SUM(t.pnl) FILTER (WHERE t.verified = TRUE) as verified_pnl,
    SUM(t.pnl) FILTER (WHERE t.verified = FALSE OR t.verified IS NULL) as unverified_pnl
FROM trades t
WHERE t.pnl IS NOT NULL
GROUP BY t.execution_type;

-- Balance discrepancy view
CREATE OR REPLACE VIEW balance_discrepancies AS
SELECT
    bs.*,
    ABS(bs.discrepancy) as abs_discrepancy,
    CASE
        WHEN ABS(bs.discrepancy) > 10 THEN 'critical'
        WHEN ABS(bs.discrepancy) > 5 THEN 'warning'
        ELSE 'ok'
    END as severity
FROM balance_snapshots bs
ORDER BY bs.snapshot_at DESC;

-- Update schema version
COMMENT ON EXTENSION "uuid-ossp" IS 'Schema version 2 - Added order tracking and validation (2026-01-03)';
