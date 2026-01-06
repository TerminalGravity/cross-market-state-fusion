#!/usr/bin/env python3
"""Check open positions in database."""
import asyncio
import asyncpg
import os

async def main():
    db_url = os.environ.get('DATABASE_URL')
    conn = await asyncpg.connect(db_url)

    # Get latest live session
    session = await conn.fetchrow('''
        SELECT id::text, mode FROM sessions
        WHERE mode = 'live'
        ORDER BY started_at DESC LIMIT 1
    ''')

    if session:
        print(f'Live session: {session["id"]}')

        # Get open positions (exit_time IS NULL)
        rows = await conn.fetch('''
            SELECT id, asset, side, size_dollars::float, entry_time
            FROM trades
            WHERE session_id = $1 AND exit_time IS NULL
        ''', session['id'])

        print(f'Open positions in DB: {len(rows)}')
        for r in rows:
            print(f'  ID={r["id"]}: {r["asset"]} {r["side"]} ${r["size_dollars"]:.2f}')

        # If BTC position exists, close it (dust fill cleanup)
        for r in rows:
            if r["asset"] == "BTC":
                print(f'\nClosing phantom BTC position ID={r["id"]}...')
                await conn.execute('''
                    UPDATE trades
                    SET exit_time = NOW(),
                        exit_price = 0.5,
                        pnl = 0,
                        duration_seconds = 0,
                        force_closed = true,
                        force_close_reason = 'dust_fill_cleanup'
                    WHERE id = $1
                ''', r["id"])
                print(f'  Closed successfully')

    await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
