#!/usr/bin/env python3
import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, '.')
from db.connection import Database

async def check():
    db = Database()
    await db.connect()

    # Get the stopped session with 3 trades
    session = await db.fetchrow("""
        SELECT id FROM sessions
        WHERE mode = 'live' AND trade_count = 3 AND status = 'stopped'
        ORDER BY started_at DESC LIMIT 1
    """)

    if session:
        session_id = str(session['id'])
        print(f'Session: {session_id[:8]}...\n')

        # Get all trades
        trades = await db.fetch("""
            SELECT asset, side, entry_price, exit_price, pnl, duration_seconds,
                   exit_reason, entry_time, exit_time
            FROM trades
            WHERE session_id = $1
            ORDER BY entry_time
        """, session_id)

        print(f'Found {len(trades)} trades:\n')
        for t in trades:
            dur_min = t['duration_seconds'] / 60 if t['duration_seconds'] else 0
            exit_pct = (t['exit_price'] * 100) if t.get('exit_price') is not None else 0
            pnl = t.get('pnl') if t.get('pnl') is not None else 0

            status = "CLOSED" if t.get('exit_time') else "STILL OPEN"
            print(f"{t['asset']} {t['side']}: Entry={t['entry_price']*100:.1f}% Exit={exit_pct:.1f}% PnL=${pnl:.2f} [{status}]")
            print(f"  Duration: {dur_min:.1f}min | Exit reason: {t.get('exit_reason', 'N/A')}")
            print(f"  Entry: {t['entry_time']}")
            if t.get('exit_time'):
                print(f"  Exit:  {t['exit_time']}")
            print()

    await db.close()

asyncio.run(check())
