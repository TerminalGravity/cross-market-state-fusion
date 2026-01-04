-- Phase 5 Monitoring Queries
-- Use these to track Phase 5 performance vs baseline

-- ==================================================
-- 1. Current Session Performance
-- ==================================================

-- Overall session stats
SELECT
    s.id as session_id,
    s.created_at as started_at,
    s.mode,
    COUNT(t.id) as total_trades,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN t.pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate_pct,
    ROUND(SUM(t.pnl)::numeric, 2) as total_pnl,
    ROUND(AVG(t.pnl)::numeric, 4) as avg_pnl_per_trade,
    ROUND(AVG(EXTRACT(EPOCH FROM (t.exit_time - t.entry_time)) / 60)::numeric, 1) as avg_duration_min
FROM sessions s
LEFT JOIN trades t ON t.session_id = s.id
WHERE s.id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
GROUP BY s.id, s.created_at, s.mode;

-- ==================================================
-- 2. Per-Asset Breakdown
-- ==================================================

SELECT
    asset,
    COUNT(*) as trades,
    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate_pct,
    ROUND(SUM(pnl)::numeric, 2) as total_pnl,
    ROUND(AVG(pnl)::numeric, 4) as avg_pnl,
    ROUND(MIN(pnl)::numeric, 2) as worst_loss,
    ROUND(MAX(pnl)::numeric, 2) as best_win
FROM trades
WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
GROUP BY asset
ORDER BY total_pnl DESC;

-- ==================================================
-- 3. Trading Action Distribution (Phase 5 = Sparse Policy)
-- ==================================================

-- Expect MOSTLY HOLD with Phase 5 (low entropy)
SELECT
    side,
    COUNT(*) as count,
    ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 2) as percentage
FROM trades
WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
GROUP BY side;

-- ==================================================
-- 4. Recent Trades (Last 20)
-- ==================================================

SELECT
    created_at,
    asset,
    side,
    ROUND(entry_price::numeric, 4) as entry,
    ROUND(exit_price::numeric, 4) as exit,
    ROUND(pnl::numeric, 2) as pnl,
    ROUND(EXTRACT(EPOCH FROM (exit_time - entry_time)) / 60, 1) as duration_min,
    CASE WHEN force_closed THEN force_close_reason ELSE 'strategy' END as exit_reason
FROM trades
WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
ORDER BY created_at DESC
LIMIT 20;

-- ==================================================
-- 5. PnL Over Time (Hourly Buckets)
-- ==================================================

SELECT
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as trades,
    ROUND(SUM(pnl)::numeric, 2) as hour_pnl,
    ROUND(SUM(SUM(pnl)) OVER (ORDER BY DATE_TRUNC('hour', created_at))::numeric, 2) as cumulative_pnl
FROM trades
WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
GROUP BY DATE_TRUNC('hour', created_at)
ORDER BY hour;

-- ==================================================
-- 6. Safety System Activity
-- ==================================================

-- Check for force-closed positions
SELECT
    asset,
    force_close_reason,
    COUNT(*) as count,
    ROUND(AVG(pnl)::numeric, 2) as avg_pnl,
    ROUND(AVG(position_age_seconds)::numeric, 0) as avg_age_sec
FROM trades
WHERE
    session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
    AND force_closed = true
GROUP BY asset, force_close_reason;

-- ==================================================
-- 7. Compare Last Two Sessions (Phase 5 vs Baseline)
-- ==================================================

WITH recent_sessions AS (
    SELECT
        s.id,
        s.created_at,
        s.mode,
        COUNT(t.id) as trades,
        ROUND(SUM(t.pnl)::numeric, 2) as total_pnl,
        ROUND(AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
        ROW_NUMBER() OVER (ORDER BY s.created_at DESC) as session_rank
    FROM sessions s
    LEFT JOIN trades t ON t.session_id = s.id
    GROUP BY s.id, s.created_at, s.mode
)
SELECT
    CASE WHEN session_rank = 1 THEN 'Phase 5 (Current)' ELSE 'Baseline (Previous)' END as session,
    mode,
    created_at,
    trades,
    total_pnl,
    win_rate,
    CASE
        WHEN session_rank = 1 THEN NULL
        ELSE ROUND((LEAD(total_pnl) OVER (ORDER BY session_rank DESC) - total_pnl) / NULLIF(total_pnl, 0) * 100, 1)
    END as pnl_improvement_pct
FROM recent_sessions
WHERE session_rank <= 2
ORDER BY session_rank;

-- ==================================================
-- 8. Entry Price Distribution (Phase 5 = Favor Extremes)
-- ==================================================

-- Phase 5 should favor entries near 0 or 1 (asymmetric payoffs)
SELECT
    CASE
        WHEN entry_price < 0.20 THEN '0.00-0.20 (Very Low)'
        WHEN entry_price < 0.40 THEN '0.20-0.40 (Low)'
        WHEN entry_price < 0.60 THEN '0.40-0.60 (Mid)'
        WHEN entry_price < 0.80 THEN '0.60-0.80 (High)'
        ELSE '0.80-1.00 (Very High)'
    END as price_range,
    COUNT(*) as trades,
    ROUND(AVG(pnl)::numeric, 4) as avg_pnl,
    ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate
FROM trades
WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
GROUP BY 1
ORDER BY 1;

-- ==================================================
-- 9. Health Events (Safety System Logs)
-- ==================================================

SELECT
    created_at,
    event_type,
    asset,
    severity,
    details
FROM health_events
WHERE session_id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1)
ORDER BY created_at DESC
LIMIT 20;

-- ==================================================
-- 10. Phase 5 Specific Checks
-- ==================================================

-- Check entropy stays healthy (should be ~0.03 with Phase 5)
-- This would require logging PPO metrics to database (not currently done)
-- For now, monitor via application logs

-- Memory usage check (rough estimate from session duration)
SELECT
    EXTRACT(EPOCH FROM (NOW() - created_at)) / 3600 as hours_running,
    CASE
        WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) > 86400 THEN 'OK: 24h+ runtime'
        WHEN EXTRACT(EPOCH FROM (NOW() - created_at)) > 43200 THEN 'OK: 12h+ runtime'
        ELSE 'Short runtime (< 12h)'
    END as stability_check
FROM sessions
WHERE id = (SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1);

