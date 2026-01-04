#!/usr/bin/env python3
"""
Phase 5 Performance Comparison Script

Compares the most recent session (Phase 5) with the previous session (baseline)
to quantify the performance improvement from temporal architecture.

Usage:
    uv run python scripts/compare_performance.py

Requires DATABASE_URL environment variable.
"""

import os
import sys
import asyncio
from datetime import datetime

sys.path.insert(0, ".")
from db.connection import DatabaseConnection


async def compare_sessions():
    """Compare last two sessions and display performance metrics."""

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set. Set it in .env or export it.")
        return

    db = DatabaseConnection(db_url)
    await db.connect()

    try:
        # Get last two sessions with trade stats
        query = """
        WITH session_stats AS (
            SELECT
                s.id,
                s.created_at,
                s.mode,
                COUNT(t.id) as total_trades,
                SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN t.pnl < 0 THEN 1 ELSE 0 END) as losses,
                ROUND(AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 2) as win_rate,
                ROUND(SUM(t.pnl)::numeric, 2) as total_pnl,
                ROUND(AVG(t.pnl)::numeric, 4) as avg_pnl,
                ROUND(AVG(EXTRACT(EPOCH FROM (t.exit_time - t.entry_time)) / 60)::numeric, 1) as avg_duration_min,
                ROW_NUMBER() OVER (ORDER BY s.created_at DESC) as rank
            FROM sessions s
            LEFT JOIN trades t ON t.session_id = s.id
            GROUP BY s.id, s.created_at, s.mode
        )
        SELECT * FROM session_stats WHERE rank <= 2 ORDER BY rank;
        """

        sessions = await db.fetch(query)

        if len(sessions) < 2:
            print("❌ Need at least 2 sessions to compare. Current sessions:", len(sessions))
            return

        current = sessions[0]
        previous = sessions[1]

        print("\n" + "=" * 70)
        print("PHASE 5 PERFORMANCE COMPARISON")
        print("=" * 70)

        # Session info
        print(f"\n📊 CURRENT SESSION (Phase 5)")
        print(f"   Started: {current['created_at']}")
        print(f"   Mode: {current['mode']}")
        print(f"   Trades: {current['total_trades']}")

        print(f"\n📊 PREVIOUS SESSION (Baseline)")
        print(f"   Started: {previous['created_at']}")
        print(f"   Mode: {previous['mode']}")
        print(f"   Trades: {previous['total_trades']}")

        # Performance comparison
        print("\n" + "-" * 70)
        print("PERFORMANCE METRICS")
        print("-" * 70)

        def calc_improvement(current_val, previous_val):
            if previous_val == 0:
                return "N/A"
            pct = ((current_val - previous_val) / abs(previous_val)) * 100
            return f"{pct:+.1f}%"

        # PnL
        pnl_improvement = calc_improvement(current['total_pnl'], previous['total_pnl'])
        print(f"\nTotal PnL:")
        print(f"   Previous: ${previous['total_pnl']:.2f}")
        print(f"   Current:  ${current['total_pnl']:.2f}")
        print(f"   Change:   {pnl_improvement}")

        # Win rate
        wr_improvement = calc_improvement(current['win_rate'], previous['win_rate'])
        print(f"\nWin Rate:")
        print(f"   Previous: {previous['win_rate']:.1f}%")
        print(f"   Current:  {current['win_rate']:.1f}%")
        print(f"   Change:   {wr_improvement}")

        # Avg PnL per trade
        avg_pnl_improvement = calc_improvement(current['avg_pnl'], previous['avg_pnl'])
        print(f"\nAvg PnL per Trade:")
        print(f"   Previous: ${previous['avg_pnl']:.4f}")
        print(f"   Current:  ${current['avg_pnl']:.4f}")
        print(f"   Change:   {avg_pnl_improvement}")

        # Trade volume
        trade_improvement = calc_improvement(current['total_trades'], previous['total_trades'])
        print(f"\nTrade Volume:")
        print(f"   Previous: {previous['total_trades']} trades")
        print(f"   Current:  {current['total_trades']} trades")
        print(f"   Change:   {trade_improvement}")

        # Avg duration
        dur_improvement = calc_improvement(current['avg_duration_min'], previous['avg_duration_min'])
        print(f"\nAvg Trade Duration:")
        print(f"   Previous: {previous['avg_duration_min']:.1f} min")
        print(f"   Current:  {current['avg_duration_min']:.1f} min")
        print(f"   Change:   {dur_improvement}")

        # Per-asset breakdown for current session
        print("\n" + "-" * 70)
        print("CURRENT SESSION - PER ASSET")
        print("-" * 70)

        asset_query = """
        SELECT
            asset,
            COUNT(*) as trades,
            ROUND(SUM(pnl)::numeric, 2) as total_pnl,
            ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100, 1) as win_rate
        FROM trades
        WHERE session_id = $1
        GROUP BY asset
        ORDER BY total_pnl DESC;
        """

        assets = await db.fetch(asset_query, current['id'])

        if assets:
            print(f"\n{'Asset':<8} {'Trades':<8} {'PnL':<12} {'Win Rate':<10}")
            print("-" * 50)
            for asset in assets:
                print(f"{asset['asset']:<8} {asset['trades']:<8} ${asset['total_pnl']:<11.2f} {asset['win_rate']:<9.1f}%")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        if current['total_pnl'] > previous['total_pnl']:
            improvement_x = current['total_pnl'] / abs(previous['total_pnl']) if previous['total_pnl'] != 0 else 0
            print(f"\n✅ PHASE 5 IMPROVEMENT: {improvement_x:.1f}x PnL vs baseline")
            print(f"   Phase 5 achieved ${current['total_pnl']:.2f} vs ${previous['total_pnl']:.2f}")
        else:
            print(f"\n⚠️  Phase 5 underperforming: ${current['total_pnl']:.2f} vs ${previous['total_pnl']:.2f}")
            print("   Consider:")
            print("   - Check if temporal encoder is loading correctly")
            print("   - Verify entropy hasn't collapsed (check logs)")
            print("   - Ensure Phase 5 model weights are in use")
            print("   - Compare market conditions (volatility, liquidity)")

        # Expectations
        print("\n📈 EXPECTED PHASE 5 PERFORMANCE (from upstream):")
        print("   - 10-15x improvement in ROI")
        print("   - Win rate: ~23% (similar to baseline)")
        print("   - More HOLD actions (sparse policy)")
        print("   - Better momentum/trend following")

        print("\n" + "=" * 70 + "\n")

    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(compare_sessions())
