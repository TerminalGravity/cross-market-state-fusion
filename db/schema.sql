-- Polymarket Paper Trading Database Schema
-- For Railway PostgreSQL deployment

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- SESSIONS TABLE
-- Tracks trading sessions with checkpoint data for crash recovery
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Timing
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,

    -- Configuration
    mode VARCHAR(10) NOT NULL DEFAULT 'paper' CHECK (mode IN ('paper', 'live')),
    status VARCHAR(20) NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'stopped', 'crashed')),
    trade_size DECIMAL(10, 2) NOT NULL DEFAULT 50.00,
    model_version VARCHAR(100),
    config JSONB,  -- Hyperparameters, thresholds, etc.

    -- Accumulated stats
    total_pnl DECIMAL(12, 4) DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,

    -- Crash recovery checkpoint
    last_checkpoint TIMESTAMPTZ,
    checkpoint_data JSONB  -- Serialized positions, market states
);

CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at DESC);

-- =============================================================================
-- TRADES TABLE
-- Individual trade records with full context
-- =============================================================================
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    -- Market identification
    condition_id VARCHAR(100) NOT NULL,
    asset VARCHAR(10) NOT NULL,  -- BTC, ETH, SOL, XRP

    -- Entry
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DECIMAL(8, 6) NOT NULL,  -- Polymarket prob 0.01-0.99
    entry_binance_price DECIMAL(20, 8),
    side VARCHAR(10) NOT NULL CHECK (side IN ('UP', 'DOWN')),
    size_dollars DECIMAL(10, 2) NOT NULL,

    -- Exit (NULL while position is open)
    exit_time TIMESTAMPTZ,
    exit_price DECIMAL(8, 6),
    exit_binance_price DECIMAL(20, 8),
    exit_reason VARCHAR(20),  -- signal, expiry, stop_loss, take_profit, manual

    -- Results
    pnl DECIMAL(12, 4),
    shares DECIMAL(16, 8),  -- size_dollars / entry_price
    duration_seconds INTEGER,

    -- Context at entry
    time_remaining_at_entry DECIMAL(5, 4),  -- 0-1 fraction of 15 min
    action_probs JSONB,  -- {hold: 0.2, buy: 0.7, sell: 0.1}
    market_state JSONB,  -- Full 18-dim feature snapshot

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_session ON trades(session_id);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_condition ON trades(condition_id);

-- =============================================================================
-- METRICS TABLE
-- Time-series performance snapshots (every minute)
-- =============================================================================
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- PnL tracking
    cumulative_pnl DECIMAL(12, 4),
    hourly_pnl DECIMAL(12, 4),
    daily_pnl DECIMAL(12, 4),

    -- Trade stats
    trades_today INTEGER DEFAULT 0,
    win_rate_today DECIMAL(5, 4),
    avg_trade_pnl DECIMAL(10, 4),

    -- Positions
    open_position_count INTEGER DEFAULT 0,
    total_exposure DECIMAL(10, 2) DEFAULT 0,

    -- Market data snapshot
    active_markets INTEGER,
    markets_data JSONB  -- {BTC: {prob: 0.52, spread: 0.02}, ...}
);

CREATE INDEX IF NOT EXISTS idx_metrics_session_time ON metrics(session_id, recorded_at DESC);

-- =============================================================================
-- ALERTS TABLE
-- Discord notification log for audit and debugging
-- =============================================================================
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,

    sent_at TIMESTAMPTZ DEFAULT NOW(),
    alert_type VARCHAR(30) NOT NULL,  -- trade_open, trade_close, daily_summary, error, recovery
    payload JSONB NOT NULL,

    -- Discord response
    discord_message_id VARCHAR(50),
    status VARCHAR(20) DEFAULT 'sent' CHECK (status IN ('sent', 'failed', 'rate_limited'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_sent_at ON alerts(sent_at DESC);

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- Active session view
CREATE OR REPLACE VIEW active_session AS
SELECT * FROM sessions
WHERE status = 'running'
ORDER BY started_at DESC
LIMIT 1;

-- Recent trades view
CREATE OR REPLACE VIEW recent_trades AS
SELECT
    t.*,
    s.mode,
    s.trade_size as session_trade_size
FROM trades t
JOIN sessions s ON t.session_id = s.id
WHERE t.exit_time IS NOT NULL
ORDER BY t.exit_time DESC
LIMIT 100;

-- Daily performance view
CREATE OR REPLACE VIEW daily_performance AS
SELECT
    DATE(entry_time AT TIME ZONE 'UTC') as trade_date,
    asset,
    COUNT(*) as trade_count,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(pnl) as total_pnl,
    AVG(pnl) as avg_pnl,
    MIN(pnl) as worst_trade,
    MAX(pnl) as best_trade
FROM trades
WHERE pnl IS NOT NULL
GROUP BY DATE(entry_time AT TIME ZONE 'UTC'), asset
ORDER BY trade_date DESC, asset;

-- =============================================================================
-- CLEANUP FUNCTION (optional - for data retention)
-- =============================================================================
CREATE OR REPLACE FUNCTION cleanup_old_data(days_to_keep INTEGER DEFAULT 30)
RETURNS void AS $$
BEGIN
    -- Delete old metrics (keep summary data longer)
    DELETE FROM metrics
    WHERE recorded_at < NOW() - (days_to_keep || ' days')::INTERVAL;

    -- Delete old alerts
    DELETE FROM alerts
    WHERE sent_at < NOW() - (days_to_keep || ' days')::INTERVAL;

    -- Old sessions and trades are kept for analysis
END;
$$ LANGUAGE plpgsql;
