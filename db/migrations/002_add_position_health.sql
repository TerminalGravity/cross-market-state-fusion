-- Migration 002: Add Position Health Tracking
--
-- Adds columns and tables to track position lifecycle health:
-- - Force-close reasons (timeout, health, emergency)
-- - Position age monitoring
-- - Health check events
--
-- Prevents the failure mode where positions expire unmanaged.

BEGIN;

-- ============================================================================
-- 1. Add health tracking columns to trades table
-- ============================================================================

ALTER TABLE trades ADD COLUMN IF NOT EXISTS
    force_closed BOOLEAN DEFAULT FALSE;

ALTER TABLE trades ADD COLUMN IF NOT EXISTS
    force_close_reason VARCHAR(50) CHECK (force_close_reason IN (
        'timeout',      -- Exited at T-2min before market expiry
        'health_check', -- Orderbook stale, emergency exit
        'manual',       -- Manual emergency exit
        'max_duration'  -- Position held too long
    ));

ALTER TABLE trades ADD COLUMN IF NOT EXISTS
    position_age_seconds INTEGER;

ALTER TABLE trades ADD COLUMN IF NOT EXISTS
    market_expiry_time TIMESTAMPTZ;

-- Add comment explaining force_close_reason
COMMENT ON COLUMN trades.force_close_reason IS
'Why position was force-closed: timeout (T-2min), health_check (stale data), manual, max_duration';

-- ============================================================================
-- 2. Create health_events table
-- ============================================================================

CREATE TABLE IF NOT EXISTS health_events (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Event details
    event_type VARCHAR(20) NOT NULL CHECK (event_type IN (
        'degraded',   -- Orderbook 30-60s stale
        'critical',   -- Orderbook >60s stale
        'recovered'   -- Orderbook back to healthy
    )),

    -- Affected asset
    asset VARCHAR(10) NOT NULL,

    -- Staleness info
    staleness_seconds NUMERIC(6, 2),

    -- Actions taken
    action_taken VARCHAR(50), -- 'stopped_new_positions', 'emergency_exit', 'none'

    -- Context
    open_positions INTEGER DEFAULT 0,
    affected_trades UUID[]  -- IDs of trades force-closed
);

CREATE INDEX idx_health_events_asset ON health_events(asset);
CREATE INDEX idx_health_events_type ON health_events(event_type);
CREATE INDEX idx_health_events_time ON health_events(created_at DESC);

COMMENT ON TABLE health_events IS
'Tracks orderbook health issues and actions taken to prevent unmanaged position expiry';

-- ============================================================================
-- 3. Create view for position health summary
-- ============================================================================

CREATE OR REPLACE VIEW position_health_summary AS
SELECT
    session_id,
    asset,
    COUNT(*) FILTER (WHERE force_closed = TRUE) as force_closed_count,
    COUNT(*) FILTER (WHERE force_close_reason = 'timeout') as timeout_exits,
    COUNT(*) FILTER (WHERE force_close_reason = 'health_check') as health_exits,
    COUNT(*) FILTER (WHERE force_close_reason = 'max_duration') as duration_exits,
    AVG(position_age_seconds) FILTER (WHERE force_closed = TRUE) as avg_force_close_age,
    COUNT(*) as total_trades
FROM trades
WHERE exit_time IS NOT NULL
GROUP BY session_id, asset;

COMMENT ON VIEW position_health_summary IS
'Summary of position health metrics per session and asset';

-- ============================================================================
-- 4. Create function to record health event
-- ============================================================================

CREATE OR REPLACE FUNCTION record_health_event(
    p_event_type VARCHAR(20),
    p_asset VARCHAR(10),
    p_staleness_seconds NUMERIC,
    p_action_taken VARCHAR(50),
    p_open_positions INTEGER,
    p_affected_trades UUID[]
) RETURNS INTEGER AS $$
DECLARE
    v_event_id INTEGER;
BEGIN
    INSERT INTO health_events (
        event_type,
        asset,
        staleness_seconds,
        action_taken,
        open_positions,
        affected_trades
    ) VALUES (
        p_event_type,
        p_asset,
        p_staleness_seconds,
        p_action_taken,
        p_open_positions,
        p_affected_trades
    ) RETURNING id INTO v_event_id;

    RETURN v_event_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION record_health_event IS
'Helper function to record orderbook health events';

-- ============================================================================
-- 5. Add index for fast position age queries
-- ============================================================================

CREATE INDEX idx_trades_open_by_age ON trades(entry_time)
    WHERE exit_time IS NULL;

COMMENT ON INDEX idx_trades_open_by_age IS
'Fast lookup of open positions ordered by age (for timeout checks)';

-- ============================================================================
-- Migration metadata
-- ============================================================================

-- Track migration
INSERT INTO schema_migrations (version, description)
VALUES (2, 'Add position health tracking and force-close monitoring')
ON CONFLICT (version) DO NOTHING;

COMMIT;

-- ============================================================================
-- Verification queries (for manual testing)
-- ============================================================================

-- Check new columns exist
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'trades'
--   AND column_name IN ('force_closed', 'force_close_reason', 'position_age_seconds', 'market_expiry_time');

-- Check health_events table
-- SELECT COUNT(*) as health_events_count FROM health_events;

-- Check view works
-- SELECT * FROM position_health_summary LIMIT 5;
