-- Dual-Mode Trading Schema Extensions
-- Adds views for side-by-side paper/live comparison

-- Active dual-mode sessions view
-- Matches paper and live sessions started on the same date
CREATE OR REPLACE VIEW dual_mode_sessions AS
SELECT
    p.id as paper_session_id,
    l.id as live_session_id,
    p.started_at,
    p.ended_at,
    p.status,
    p.total_pnl as paper_pnl,
    l.total_pnl as live_pnl,
    p.trade_count as paper_trades,
    l.trade_count as live_trades,
    p.win_count as paper_wins,
    l.win_count as live_wins,
    CASE
        WHEN p.trade_count > 0 THEN ROUND(p.win_count::DECIMAL / p.trade_count * 100, 1)
        ELSE 0
    END as paper_win_rate,
    CASE
        WHEN l.trade_count > 0 THEN ROUND(l.win_count::DECIMAL / l.trade_count * 100, 1)
        ELSE 0
    END as live_win_rate
FROM sessions p
LEFT JOIN sessions l ON DATE(p.started_at) = DATE(l.started_at)
    AND l.mode = 'live'
WHERE p.mode = 'paper'
ORDER BY p.started_at DESC;

-- Execution quality comparison view
-- Analyzes fill times and slippage by mode and asset
CREATE OR REPLACE VIEW execution_quality AS
SELECT
    s.mode,
    t.asset,
    COUNT(*) as total_fills,
    AVG(ABS(t.exit_price - t.entry_price)) as avg_slippage,
    AVG(t.duration_seconds) as avg_fill_time_seconds,
    MIN(t.duration_seconds) as min_fill_time,
    MAX(t.duration_seconds) as max_fill_time
FROM trades t
JOIN sessions s ON t.session_id = s.id
WHERE t.exit_time IS NOT NULL
  AND t.exit_time > NOW() - INTERVAL '24 hours'
GROUP BY s.mode, t.asset
ORDER BY s.mode, t.asset;

-- Current active sessions by mode
-- Quick lookup for running paper and live sessions
CREATE OR REPLACE VIEW active_sessions_by_mode AS
SELECT
    mode,
    id as session_id,
    started_at,
    total_pnl,
    trade_count,
    win_count,
    CASE
        WHEN trade_count > 0 THEN ROUND(win_count::DECIMAL / trade_count * 100, 1)
        ELSE 0
    END as win_rate,
    last_checkpoint
FROM sessions
WHERE status = 'running'
ORDER BY mode, started_at DESC;

-- Performance comparison by hour
-- Useful for comparing performance trends over time
CREATE OR REPLACE VIEW hourly_performance_comparison AS
SELECT
    s.mode,
    DATE_TRUNC('hour', m.recorded_at) as hour,
    AVG(m.cumulative_pnl) as avg_pnl,
    AVG(m.win_rate_today) as avg_win_rate,
    AVG(m.total_exposure) as avg_exposure,
    MAX(m.trades_today) as trades_in_hour
FROM metrics m
JOIN sessions s ON m.session_id = s.id
WHERE m.recorded_at > NOW() - INTERVAL '24 hours'
GROUP BY s.mode, DATE_TRUNC('hour', m.recorded_at)
ORDER BY hour DESC, s.mode;

-- Trade outcome comparison
-- Analyzes win/loss distribution by mode
CREATE OR REPLACE VIEW trade_outcomes_by_mode AS
SELECT
    s.mode,
    COUNT(*) as total_trades,
    COUNT(CASE WHEN t.pnl > 0 THEN 1 END) as wins,
    COUNT(CASE WHEN t.pnl < 0 THEN 1 END) as losses,
    COUNT(CASE WHEN t.pnl = 0 THEN 1 END) as breakeven,
    AVG(t.pnl) as avg_pnl,
    MIN(t.pnl) as worst_trade,
    MAX(t.pnl) as best_trade,
    STDDEV(t.pnl) as pnl_stddev
FROM trades t
JOIN sessions s ON t.session_id = s.id
WHERE t.exit_time IS NOT NULL
  AND t.exit_time > NOW() - INTERVAL '24 hours'
GROUP BY s.mode
ORDER BY s.mode;
