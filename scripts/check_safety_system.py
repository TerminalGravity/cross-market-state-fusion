#!/usr/bin/env python3
"""
Safety System Diagnostic Tool

Checks if the position safety system would have prevented
the position expiry failure.

Usage:
    python scripts/check_safety_system.py
"""
import asyncio
import os
import sys
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import Database
from helpers.polymarket_api import get_15m_markets


async def main():
    print("\n" + "="*60)
    print("Safety System Diagnostic")
    print("="*60 + "\n")

    # Connect to database
    db = Database()
    await db.connect()

    # Get recent sessions
    recent = await db.fetch("""
        SELECT id, mode, status, started_at, ended_at, total_pnl, trade_count
        FROM sessions
        WHERE started_at > NOW() - INTERVAL '24 hours'
        ORDER BY started_at DESC
        LIMIT 5
    """)

    print("Recent Sessions:")
    for session in recent:
        print(f"  {session['mode']:6} | {session['status']:10} | "
              f"PnL: ${session['total_pnl']:7.2f} | "
              f"Trades: {session['trade_count']:3} | "
              f"Started: {session['started_at']}")

    # Check for recent live session
    live_session = await db.fetchrow("""
        SELECT id FROM sessions
        WHERE mode = 'live' AND status = 'running'
        ORDER BY started_at DESC
        LIMIT 1
    """)

    if live_session:
        session_id = str(live_session['id'])
        print(f"\n✓ Found running LIVE session: {session_id[:8]}...\n")

        # Check open positions
        open_trades = await db.get_open_trades(session_id)
        print(f"Open Positions: {len(open_trades)}")

        if open_trades:
            print("\n" + "-"*60)
            for trade in open_trades:
                age = (datetime.now(timezone.utc) - trade['entry_time']).total_seconds()
                print(f"\n{trade['asset']} {trade['side']}:")
                print(f"  Entry Time: {trade['entry_time']}")
                print(f"  Entry Price: {trade['entry_price']*100:.1f}%")
                print(f"  Size: ${trade['size_dollars']}")
                print(f"  Age: {age:.0f}s ({age/60:.1f}min)")

                if trade.get('market_expiry_time'):
                    time_to_expiry = (trade['market_expiry_time'] - datetime.now(timezone.utc)).total_seconds()
                    print(f"  Time to Expiry: {time_to_expiry:.0f}s ({time_to_expiry/60:.1f}min)")

                    # Check if safety system should trigger
                    if time_to_expiry < 120:  # 2 minutes
                        print(f"  ⚠️  SAFETY TRIGGER: T-{time_to_expiry:.0f}s (should force-close!)")
                    elif age > 840:  # 14 minutes
                        print(f"  ⚠️  SAFETY TRIGGER: Age {age:.0f}s > 840s (should force-close!)")
                    else:
                        print(f"  ✓ Healthy (no timeout)")
                else:
                    print(f"  ⚠️  WARNING: No market_expiry_time set in database!")
            print("-"*60)

        # Check recent closed trades
        print("\n\nRecent Completed Trades (last 10):")
        recent_trades = await db.get_recent_trades(session_id, limit=10)

        for trade in recent_trades:
            duration = trade.get('duration_seconds', 0)
            pnl = trade.get('pnl', 0)
            reason = trade.get('exit_reason', 'unknown')

            print(f"\n{trade['asset']} {trade['side']}:")
            print(f"  Entry: {trade['entry_price']*100:.1f}% → Exit: {trade.get('exit_price', 0)*100:.1f}%")
            print(f"  PnL: ${pnl:+.2f} | Duration: {duration}s ({duration/60:.1f}min)")
            print(f"  Exit Reason: {reason}")

            # Check if this was an emergency close
            if 'emergency' in reason:
                print(f"  ✓ Emergency close triggered (safety system worked!)")
            elif duration > 840:  # >14 min
                print(f"  ❌ Position held {duration}s - safety system FAILED!")

    else:
        print("\n⚠️  No running LIVE session found")

    # Check health events table (if it exists)
    try:
        health_events = await db.fetch("""
            SELECT event_type, asset, staleness_seconds, action_taken, created_at
            FROM health_events
            ORDER BY created_at DESC
            LIMIT 10
        """)

        if health_events:
            print("\n\nRecent Health Events:")
            print("-"*60)
            for event in health_events:
                print(f"{event['created_at']}: {event['event_type']:10} | "
                      f"{event['asset']:4} | "
                      f"staleness={event['staleness_seconds']}s | "
                      f"action={event['action_taken']}")
            print("-"*60)
    except Exception as e:
        print(f"\n⚠️  health_events table not found (migration not applied): {e}")

    # Get current markets
    print("\n\nCurrent 15-Min Markets:")
    try:
        markets = get_15m_markets()
        now = datetime.now(timezone.utc)

        for market in markets:
            time_to_close = (market.end_time - now).total_seconds()
            print(f"  {market.asset}: closes in {time_to_close/60:.1f}min ({market.question[:50]}...)")
    except Exception as e:
        print(f"  Error fetching markets: {e}")

    await db.close()

    print("\n" + "="*60)
    print("Diagnostic Complete")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
